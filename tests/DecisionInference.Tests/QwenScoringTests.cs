using DecisionInference.TensorSharp;
using TensorSharp.Runtime;
using static DecisionInference.Tests.ContractTests;

namespace DecisionInference.Tests;

public sealed class QwenScoringTests
{
    [Fact]
    public void AliasIds_UsesEveryDistinctSingleTokenAToZWithoutSpecialTokens()
    {
        var tokenizer = new FakeTokenizer();
        var ids = QwenScoring.AliasIds(tokenizer);
        Assert.Equal(Enumerable.Range(65, 26), ids);
        Assert.Equal(Enumerable.Range(65, 26).Select(c => (((char)c).ToString(), false)), tokenizer.Calls);
        ids[0] = 999;
        Assert.Equal(65, QwenScoring.AliasIds(new FakeTokenizer())[0]);
    }

    [Theory]
    [InlineData("duplicate")]
    [InlineData("multi")]
    [InlineData("negative")]
    [InlineData("empty")]
    public void AliasIds_RejectsDuplicateMultitokenNegativeOrMissingTokens(string defect)
    {
        var tokenizer = new FakeTokenizer
        {
            Encoder = text => text == "Z" ? defect switch
            {
                "duplicate" => [65], "multi" => [90, 91], "negative" => [-1], _ => []
            } : [text[0]]
        };
        Assert.Throws<NotSupportedException>(() => QwenScoring.AliasIds(tokenizer));
        Assert.Equal(26, tokenizer.Calls.Count);
        Assert.All(tokenizer.Calls, call => Assert.False(call.AddSpecial));
    }

    [Fact]
    public void Prepare_ExactLiteralPromptUsesDescriptionsNotOpaqueIdsAndOwnsTokens()
    {
        List<int> encoded = [101, 202, 303];
        var tokenizer = new FakeTokenizer { Encoder = _ => encoded };
        var question = new ChoiceDecisionQuestion("opaque <|question", "Pick a color.",
            [new("e\u0301 <think>", "red"), new("\u00e9 </think>", "blue")]);
        var tokens = QwenScoring.Prepare(tokenizer, Json("\"state\\nline\""), question, 3);
        const string expected = "<|im_start|>system\nPick a color.<|im_end|>\n" +
            "<|im_start|>user\nstate\nline\n\nA. red\nB. blue<|im_end|>\n" +
            "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        Assert.Equal((expected, false), Assert.Single(tokenizer.Calls));
        Assert.Equal([101, 202, 303], tokens);
        encoded[0] = -1;
        encoded.Add(404);
        Assert.Equal([101, 202, 303], tokens);
        Assert.DoesNotContain("opaque", tokenizer.Calls[0].Text, StringComparison.Ordinal);
    }

    [Fact]
    public void Prepare_DefaultInstructionAndStructuredStateUseLiteralRecipe()
    {
        var tokenizer = new FakeTokenizer { Encoder = _ => [7, 8] };
        var question = new ChoiceDecisionQuestion("c", null, [new("a", "first"), new("b", "second")]);
        int[] tokens = QwenScoring.Prepare(tokenizer, Json("""{"x":1,"y":["a"]}"""), question, 2);
        const string expected = "<|im_start|>system\nChoose one option.<|im_end|>\n" +
            "<|im_start|>user\n{\"x\":1,\"y\":[\"a\"]}\n\nA. first\nB. second<|im_end|>\n" +
            "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        Assert.Equal((expected, false), Assert.Single(tokenizer.Calls));
        Assert.Equal([7, 8], tokens);
    }

    [Theory]
    [InlineData(0, 3, false)]
    [InlineData(1, 1, true)]
    [InlineData(3, 3, true)]
    [InlineData(4, 3, false)]
    [InlineData(1, 0, false)]
    public void Prepare_EnforcesNonemptyInclusiveContextBoundWithoutTruncation(int tokenCount, int limit, bool accepted)
    {
        var tokenizer = new FakeTokenizer { Encoder = _ => Enumerable.Range(10, tokenCount).ToList() };
        if (accepted)
            Assert.Equal(Enumerable.Range(10, tokenCount), QwenScoring.Prepare(tokenizer, Json("\"state\""), Choice(), limit));
        else
            Assert.Throws<ArgumentException>(() => QwenScoring.Prepare(tokenizer, Json("\"state\""), Choice(), limit));
        Assert.False(Assert.Single(tokenizer.Calls).AddSpecial);
    }

    [Theory]
    [InlineData(1, false)]
    [InlineData(2, true)]
    [InlineData(26, true)]
    [InlineData(27, false)]
    public void Prepare_EnforcesTwoThroughTwentySixOptions(int count, bool accepted)
    {
        var tokenizer = new FakeTokenizer { Encoder = _ => [1, 2] };
        var question = new ChoiceDecisionQuestion("c", "pick", Enumerable.Range(0, count).Select(i => new DecisionCandidate($"opaque{i}", $"option{i}")));
        if (accepted)
        {
            Assert.Equal([1, 2], QwenScoring.Prepare(tokenizer, Json("\"state\""), question, 10));
            string prompt = Assert.Single(tokenizer.Calls).Text;
            Assert.Contains("A. option0\nB. option1", prompt, StringComparison.Ordinal);
            Assert.Contains($"{(char)('A' + count - 1)}. option{count - 1}<|im_end|>", prompt, StringComparison.Ordinal);
            Assert.DoesNotContain("opaque", prompt, StringComparison.Ordinal);
        }
        else
        {
            Assert.Throws<NotSupportedException>(() => QwenScoring.Prepare(tokenizer, Json("\"state\""), question, 10));
            Assert.Empty(tokenizer.Calls);
        }
    }

    [Theory]
    [InlineData("<|", "state")]
    [InlineData("<think>", "state")]
    [InlineData("</think>", "state")]
    [InlineData("<|", "instruction")]
    [InlineData("<think>", "instruction")]
    [InlineData("</think>", "instruction")]
    [InlineData("<|", "option")]
    [InlineData("<think>", "option")]
    [InlineData("</think>", "option")]
    public void Prepare_RejectsEveryReservedMarkerInEveryRenderedPartBeforeTokenization(string marker, string part)
    {
        var tokenizer = new FakeTokenizer();
        var state = System.Text.Json.JsonSerializer.SerializeToElement(part == "state" ? marker : "state");
        var question = new ChoiceDecisionQuestion("c", part == "instruction" ? marker : "pick",
            [new("a", part == "option" ? marker : "first"), new("b", "second")]);
        Assert.Throws<NotSupportedException>(() => QwenScoring.Prepare(tokenizer, state, question, 1000));
        Assert.Empty(tokenizer.Calls);
    }

    [Fact]
    public void Normalize_IsStableForHugeLogitsAndReturnsIndependentNonmutatingDistribution()
    {
        double[] logits = [10000, 10001, 9999];
        var first = QwenScoring.Normalize(logits);
        var second = QwenScoring.Normalize(logits);
        Assert.Equal(new double[] { 10000, 10001, 9999 }, logits);
        Assert.Equal(.24472847105479764, first[0], 12);
        Assert.Equal(.6652409557748218, first[1], 12);
        Assert.Equal(.09003057317038046, first[2], 12);
        Assert.Equal(1, first.Sum(), 12);
        first[0] = -1;
        Assert.Equal(.24472847105479764, second[0], 12);
        Assert.Equal(10000, logits[0]);
        Assert.Equal(new double[] { 1, 0 }, QwenScoring.Normalize([double.MaxValue, -double.MaxValue]));
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public void Normalize_RejectsNonfiniteValidAliasLogits(double value) =>
        Assert.Throws<DecisionProtocolException>(() => QwenScoring.Normalize([0, value]));

    [Fact]
    public void Normalize_RejectsEmptyAndHandlesTiesWithUniformProbabilities()
    {
        Assert.Throws<DecisionProtocolException>(() => QwenScoring.Normalize([]));
        Assert.Equal(new double[] { .5, .5 }, QwenScoring.Normalize([-10000, -10000]));
    }

    [Fact]
    public void RawAnswer_ExposesReadOnlyManagedEvidenceFromTransferredArrays()
    {
        // Internal construction consumes already-owned provider arrays. This does not simulate
        // model execution or claim the constructor clones arbitrary externally shared arrays.
        var raw = new QwenRawAnswer("ABC", [65, 66, 67], [10000, 10001, 9999]);
        Assert.Equal("tensorsharp-qwen", raw.Provider);
        Assert.True(raw.ConditionalOnAnswerTokens);
        Assert.Equal("ABC", raw.Aliases);
        Assert.Equal(new[] { 65, 66, 67 }, raw.TokenIds);
        Assert.Equal(new[] { 10000d, 10001, 9999 }, raw.Logits);
        Assert.Throws<NotSupportedException>(() => ((IList<int>)raw.TokenIds)[0] = -1);
        Assert.Throws<NotSupportedException>(() => ((IList<double>)raw.Logits)[0] = -1);
        Assert.Equal(65, raw.TokenIds[0]);
        Assert.Equal(10000, raw.Logits[0]);
    }

    // Implements the real package interface. No fake model or claims about KV reset/concurrency.
    private sealed class FakeTokenizer : ITokenizer
    {
        public Func<string, List<int>> Encoder { get; init; } = text => text.Select(c => (int)c).ToList();
        public List<(string Text, bool AddSpecial)> Calls { get; } = [];
        public string[] Vocab => [];
        public int BosTokenId => 0;
        public int[] EosTokenIds => [1];
        public int VocabSize => 1000;
        public List<int> Encode(string text, bool addSpecial = true) { Calls.Add((text, addSpecial)); return Encoder(text); }
        public string Decode(List<int> tokens) => throw new NotSupportedException();
        public void AppendTokenBytes(int tokenId, List<byte> bytes) => throw new NotSupportedException();
        public bool IsEos(int tokenId) => tokenId == 1;
        public int LookupToken(string token) => throw new NotSupportedException();
    }
}
