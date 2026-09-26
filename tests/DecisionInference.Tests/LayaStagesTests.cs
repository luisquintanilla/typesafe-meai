using System.Buffers;
using System.Numerics.Tensors;
using DecisionInference.Laya;
using Microsoft.ML.Tokenizers;
using static DecisionInference.Tests.ContractTests;

namespace DecisionInference.Tests;

public sealed class LayaStagesTests
{
    internal static LayaProfile Profile(int max = 512, int head = 256) =>
        new(max, head, new Dictionary<string, float> { ["choice"] = 1, ["score"] = 1, ["noul"] = 1 });
    internal static LayaTokenMetadata Tokens() => new(1, 2, 3, 0, ["[CLS]", "[SEP]", "[MASK]", "[PAD]"]);
    internal static PrepareDecisionInputs Prepare(CharacterTokenizer? tokenizer = null, LayaProfile? profile = null) =>
        new(profile ?? Profile(), tokenizer ?? new(), Tokens());
    internal static DecisionRequest Mixed() => new(Json("{}"),
    [
        new ChoiceDecisionQuestion("opaque [MASK] choice", "pick", [new("opaque-A", "red"), new("opaque-B", "blue")]),
        new ScoreDecisionQuestion("score", "rate", ["low", "mid", "high"]),
        new BinaryDecisionQuestion("binary", "b", "t", "f")
    ]);

    [Fact]
    public void Prepare_ExactFramingPaddingMasksTypesAndInternalAliasesExcludeOpaqueIds()
    {
        var tokenizer = new CharacterTokenizer();
        var batch = Prepare(tokenizer).Prepare(Mixed(), TestContext.Current.CancellationToken);
        long[][] expected =
        [
            [1, .. Chars("choice question: pick"), 2, 3, .. Chars(" 0: red"), 3, .. Chars(" 1: blue"), 2, .. Chars("{}"), 2],
            [1, .. Chars("score question: rate"), 2, 3, .. Chars(" level 0: low"), 3, .. Chars(" level 1: mid"), 3, .. Chars(" level 2: high"), 2, .. Chars("{}"), 2],
            [1, .. Chars("noul question: b"), 2, 3, .. Chars(" false: f"), 3, .. Chars(" true: t"), 2, .. Chars("{}"), 2]
        ];
        Assert.Equal(3, batch.BatchSize);
        Assert.Equal(expected.Max(row => row.Length), batch.SequenceLength);
        Assert.Equal(3, batch.MarkerWidth);
        Assert.Equal(new nint[] { 3, batch.SequenceLength }, batch.InputIds.Lengths.ToArray());
        Assert.Equal(new nint[] { 3, 3 }, batch.MarkerPositions.Lengths.ToArray());
        for (int row = 0; row < 3; row++)
        {
            for (int i = 0; i < batch.SequenceLength; i++)
            {
                Assert.Equal(i < expected[row].Length ? expected[row][i] : 0, batch.InputIds[row, i]);
                Assert.Equal(i < expected[row].Length ? 1L : 0L, batch.AttentionMask[row, i]);
            }
            int[] positions = Enumerable.Range(0, expected[row].Length).Where(i => expected[row][i] == 3).ToArray();
            for (int i = 0; i < 3; i++)
            {
                Assert.Equal(i < positions.Length ? positions[i] : 0, batch.MarkerPositions[row, i]);
                Assert.Equal(i < positions.Length, batch.MarkerMask[row, i]);
            }
            Assert.Equal(new long[] { 0, 1, 2 }[row], batch.QuestionTypes[row]);
        }
        Assert.Equal(new[]
        {
            "{}", " 0: red", " 1: blue", "choice question: pick",
            " level 0: low", " level 1: mid", " level 2: high", "score question: rate",
            " false: f", " true: t", "noul question: b"
        }, tokenizer.Calls);
        Assert.DoesNotContain(tokenizer.Calls, text => text.Contains("opaque", StringComparison.Ordinal));
    }

    private static long[] Chars(string text) => text.Select(c => (long)c + 100).ToArray();

    [Theory]
    [InlineData("state", 129, false)]
    [InlineData("remaining-state", 98, true)]
    [InlineData("remaining-state", 99, false)]
    [InlineData("head", 54, true)]
    [InlineData("head", 55, false)]
    [InlineData("option", 49, true)]
    [InlineData("option", 50, false)]
    [InlineData("aggregate-options", 21, true)]
    [InlineData("aggregate-options", 22, false)]
    public void Prepare_EnforcesEveryTokenBudgetWithoutTruncation(string part, int size, bool accepted)
    {
        // Controlled standard tokenizer isolates exact budgets: head=20, option=2 each,
        // remaining state=128-(20+2*(2+1)+4)=98. Head limit=60.
        // For option length 49, use head limit=116 (100 option+16 reserved).
        var tokenizer = new CharacterTokenizer
        {
            Encode = text =>
            {
                int length = text == "state" ? (part is "state" or "remaining-state" ? size : 1)
                    : text.StartsWith("choice question:", StringComparison.Ordinal) ? (part == "head" ? size : part is "option" or "aggregate-options" ? 16 : 20)
                    : part is "option" or "aggregate-options" ? size : 2;
                return Enumerable.Repeat(100, length).ToArray();
            }
        };
        var stage = Prepare(tokenizer, Profile(128, part == "option" ? 116 : 60));
        if (accepted)
        {
            var batch = stage.Prepare(Request(Choice()), TestContext.Current.CancellationToken);
            int head = part == "head" ? size : part is "option" or "aggregate-options" ? 16 : 20;
            int options = 2 * ((part is "option" or "aggregate-options" ? size : 2) + 1);
            int state = part is "state" or "remaining-state" ? size : 1;
            Assert.Equal(head + options + state + 4, batch.SequenceLength);
            Assert.Equal(2, batch.MarkerWidth);
            Assert.Equal(2, batch.InputIds[0, batch.SequenceLength - 1]);
        }
        else
            Assert.Throws<ArgumentException>(() => stage.Prepare(Request(Choice()), TestContext.Current.CancellationToken));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(61)]
    public void Prepare_RejectsUnsupportedOptionCardinalityBeforeEncodingOptions(int count)
    {
        var tokenizer = new CharacterTokenizer();
        var question = new ChoiceDecisionQuestion("c", "", Enumerable.Range(0, count).Select(i => new DecisionCandidate(i.ToString(), "")));
        Assert.Throws<ArgumentException>(() => Prepare(tokenizer, Profile(128, 60)).Prepare(Request(question), TestContext.Current.CancellationToken));
        Assert.Equal(["state"], tokenizer.Calls);
    }

    [Theory]
    [InlineData("state")]
    [InlineData("instructions")]
    [InlineData("option")]
    [InlineData("binary")]
    [InlineData("rubric")]
    public void Prepare_RejectsReservedMarkersInEveryCallerTextPosition(string part)
    {
        var request = new DecisionRequest(Json(part == "state" ? "\"[MASK]\"" : "\"state\""),
        [
            part == "binary" ? new BinaryDecisionQuestion("b", "", "[MASK]", "")
            : part == "rubric" ? new ScoreDecisionQuestion("s", "", ["[MASK]", "ok"])
            : new ChoiceDecisionQuestion("c", part == "instructions" ? "[MASK]" : "", [new("a", part == "option" ? "[MASK]" : ""), new("b", "")])
        ]);
        Assert.Throws<NotSupportedException>(() => Prepare().Prepare(request, TestContext.Current.CancellationToken));
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void Prepare_RejectsNegativeOrFramingTokenIdsFromTokenizer(int id)
    {
        var tokenizer = new CharacterTokenizer { Encode = _ => [id] };
        Assert.Throws<NotSupportedException>(() => Prepare(tokenizer).Prepare(Request(Choice()), TestContext.Current.CancellationToken));
        Assert.Equal(["state"], tokenizer.Calls);
    }

    [Theory]
    [InlineData(2, .5f)]
    [InlineData(3, 2f)]
    [InlineData(5, 2f)]
    [InlineData(6, 3f)]
    [InlineData(10, 3f)]
    [InlineData(11, 5f)]
    public void Profile_UsesCardinalityBucketsAndClampsWithoutChangingSource(int count, float expected)
    {
        var source = new Dictionary<string, float>
        {
            ["choice"] = 1, ["score"] = .1f, ["noul"] = 6,
            ["choice:2"] = .1f, ["choice:3-5"] = 2, ["choice:6-10"] = 3, ["choice:11+"] = 10
        };
        var profile = new LayaProfile(512, 256, source);
        source["choice:2"] = 4;
        Assert.Equal(expected, profile.Temperature(DecisionKind.Choice, count));
        Assert.Equal(.5f, profile.Temperature(DecisionKind.Score, count));
        Assert.Equal(5f, profile.Temperature(DecisionKind.Binary, count));
        Assert.Equal(.1f, profile.Temperatures["choice:2"]);
        Assert.Equal(10f, profile.Temperatures["choice:11+"]);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(2)]
    [InlineData(4)]
    public void ProfileLoad_RequiresExactlyThreeBaseTemperatures(int count)
    {
        using var file = new TemporaryFixture(System.Text.Encoding.UTF8.GetBytes(
            $$$"""{"max_len":128,"head_max_len":60,"temperature":[{{{string.Join(",", Enumerable.Repeat("1", count))}}}],"temperature_by_options":{}}"""));
        Assert.Throws<InvalidDataException>(() => LayaProfile.Load(file.Path));
    }

    [Fact]
    public void ProfileLoad_MapsChoiceScoreBinaryAndBucketOverrides()
    {
        using var file = new TemporaryFixture(System.Text.Encoding.UTF8.GetBytes(
            """{"max_len":128,"head_max_len":60,"temperature":[1,2,3],"temperature_by_options":{"choice:2":0.2,"score:11+":8}}"""));
        var profile = LayaProfile.Load(file.Path);
        Assert.Equal(128, profile.MaxLength);
        Assert.Equal(60, profile.HeadMaxLength);
        Assert.Equal(.5f, profile.Temperature(DecisionKind.Choice, 2));
        Assert.Equal(1f, profile.Temperature(DecisionKind.Choice, 3));
        Assert.Equal(2f, profile.Temperature(DecisionKind.Score, 10));
        Assert.Equal(5f, profile.Temperature(DecisionKind.Score, 11));
        Assert.Equal(3f, profile.Temperature(DecisionKind.Binary, 2));
    }

    [Fact]
    public void Decode_StableLargeLogitsIgnorePaddedNaNAndPreserveTrueIndexScoreAndEvidence()
    {
        var batch = Prepare().Prepare(Mixed(), TestContext.Current.CancellationToken);
        var logits = Tensor.Create<float>([10000, 10001, float.NaN, 0, 0, 0, -10000, 10000, float.NaN], [3, 3]);
        var action = Tensor.Create<float>([.25f, .75f, .5f, .5f, .75f, .25f], [3, 2]);
        var result = new DecodeDecisions(Profile()).Decode(batch, new(logits, action));
        logits[0, 0] = -99;
        action[0, 0] = 1;
        var choice = Assert.IsType<ChoiceDecisionAnswer>(result.Answers[0]);
        Assert.Equal("opaque-B", choice.SelectedCandidateId);
        Assert.Equal(.2689414213699951, choice.Probabilities["opaque-A"], 12);
        Assert.Equal(.7310585786300049, choice.Probabilities["opaque-B"], 12);
        Assert.Equal(1, choice.Probabilities.Values.Sum(), 12);
        var score = Assert.IsType<ScoreDecisionAnswer>(result.Answers[1]);
        Assert.All(score.Probabilities, p => Assert.Equal(1d / 3, p, 12));
        Assert.Equal(1, score.ExpectedIndex, 12);
        Assert.Null(score.NativeScore);
        Assert.Equal(1, Assert.IsType<BinaryDecisionAnswer>(result.Answers[2]).TrueProbability);
        Assert.Equal(new[] { 10000d, 10001d }, Assert.IsType<LayaRawAnswer>(choice.RawRepresentation).Logits);
        Assert.Equal(new[] { .25, .75 }, Assert.IsType<LayaRawAnswer>(choice.RawRepresentation).ActionProbabilities);
        var scoreRaw = Assert.IsType<LayaRawAnswer>(score.RawRepresentation);
        var binaryRaw = Assert.IsType<LayaRawAnswer>(result.Answers[2].RawRepresentation);
        logits[1, 0] = 17;
        logits[2, 0] = 23;
        action[1, 0] = 1;
        action[2, 0] = 1;
        Assert.Equal(new[] { 0d, 0, 0 }, scoreRaw.Logits);
        Assert.Equal(new[] { -10000d, 10000 }, binaryRaw.Logits);
        Assert.Equal(new[] { .5, .5 }, scoreRaw.ActionProbabilities);
        Assert.Equal(new[] { .75, .25 }, binaryRaw.ActionProbabilities);
        foreach (var answer in result.Answers)
        {
            var raw = Assert.IsType<LayaRawAnswer>(answer.RawRepresentation);
            Assert.Equal("laya", raw.Provider);
            Assert.Equal(1, raw.Temperature);
            Assert.Throws<NotSupportedException>(() => ((IList<double>)raw.Logits)[0] = -1);
            Assert.Throws<NotSupportedException>(() => ((IList<double>)raw.ActionProbabilities)[0] = -1);
        }
        long tokens = 0;
        for (int r = 0; r < batch.BatchSize; r++)
            for (int i = 0; i < batch.SequenceLength; i++) tokens += batch.AttentionMask[r, i];
        Assert.Equal(tokens, result.Usage!.InputTokenCount);
        Assert.Equal(LayaProfile.ModelId, result.ModelId);
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void Decode_RejectsNonfiniteValidOptionLogits(float invalid)
    {
        var batch = Prepare().Prepare(Request(Choice()), TestContext.Current.CancellationToken);
        Assert.Throws<DecisionProtocolException>(() => new DecodeDecisions(Profile()).Decode(batch,
            new(Tensor.Create<float>([0, invalid], [1, 2]), Tensor.Create<float>([.5f, .5f], [1, 2]))));
    }

    [Fact]
    public void Decode_AppliesPerKindNonunitTemperatureBeforeSoftmax()
    {
        var profile = new LayaProfile(512, 256, new Dictionary<string, float> { ["choice"] = 2, ["score"] = 4, ["noul"] = .5f });
        var batch = Prepare(profile: profile).Prepare(Mixed(), TestContext.Current.CancellationToken);
        var output = new LayaModelOutputs(
            Tensor.Create<float>([0, 2, float.NaN, 0, 4, 8, 0, 1, float.NaN], [3, 3]),
            Tensor.Create<float>([.5f, .5f, .5f, .5f, .5f, .5f], [3, 2]));
        var response = new DecodeDecisions(profile).Decode(batch, output);
        var choice = Assert.IsType<ChoiceDecisionAnswer>(response.Answers[0]);
        Assert.Equal(.7310585786300049, choice.Probabilities["opaque-B"], 12);
        Assert.Equal(2, Assert.IsType<LayaRawAnswer>(choice.RawRepresentation).Temperature);
        var score = Assert.IsType<ScoreDecisionAnswer>(response.Answers[1]);
        Assert.Equal(.09003057317038046, score.Probabilities[0], 12);
        Assert.Equal(.24472847105479764, score.Probabilities[1], 12);
        Assert.Equal(.6652409557748218, score.Probabilities[2], 12);
        Assert.Equal(1.5752103826044412, score.ExpectedIndex, 12);
        Assert.Equal(4, Assert.IsType<LayaRawAnswer>(score.RawRepresentation).Temperature);
        var binary = Assert.IsType<BinaryDecisionAnswer>(response.Answers[2]);
        Assert.Equal(.8807970779778823, binary.TrueProbability, 12);
        Assert.Equal(.5, Assert.IsType<LayaRawAnswer>(binary.RawRepresentation).Temperature);
    }

    [Theory]
    [InlineData(.5000009f, true)]
    [InlineData(.4999991f, true)]
    [InlineData(.5000011f, false)]
    [InlineData(.4999989f, false)]
    public void Decode_ActionSumUsesOneMicroAbsoluteTolerance(float second, bool accepted)
    {
        var batch = Prepare().Prepare(Request(Choice()), TestContext.Current.CancellationToken);
        var output = new LayaModelOutputs(Tensor.Create<float>([0, 1], [1, 2]), Tensor.Create<float>([.5f, second], [1, 2]));
        if (accepted)
        {
            var result = new DecodeDecisions(Profile()).Decode(batch, output);
            Assert.Equal(.7310585786300049, Assert.IsType<ChoiceDecisionAnswer>(result.Answers[0]).Probabilities["A"], 12);
            Assert.Equal((double)second, Assert.IsType<LayaRawAnswer>(result.Answers[0].RawRepresentation).ActionProbabilities[1]);
        }
        else Assert.Throws<DecisionProtocolException>(() => new DecodeDecisions(Profile()).Decode(batch, output));
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(-1f)]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    public void Profile_RejectsNonpositiveOrNonfiniteTemperatures(float invalid)
    {
        Assert.Throws<ArgumentException>(() => new LayaProfile(512, 256,
            new Dictionary<string, float> { ["choice"] = invalid, ["score"] = 1, ["noul"] = 1 }));
        Assert.Throws<ArgumentException>(() => new LayaProfile(512, 256,
            new Dictionary<string, float> { ["choice"] = 1, ["score"] = 1, ["noul"] = 1, ["choice:2"] = invalid }));
    }

    [Fact]
    public void TokenMetadata_RejectsInvalidIdsAndOwnsReservedTokenCollection()
    {
        Assert.Throws<ArgumentException>(() => new LayaTokenMetadata(-1, 2, 3, 0, []));
        Assert.Throws<ArgumentException>(() => new LayaTokenMetadata(1, 1, 3, 0, []));
        Assert.Throws<ArgumentException>(() => new LayaTokenMetadata(1, 2, 3, 0, [""]));
        string[] reserved = ["[MASK]"];
        var metadata = new LayaTokenMetadata(1, 2, 3, 0, reserved);
        reserved[0] = "changed";
        Assert.Equal(["[MASK]"], metadata.ReservedTokens);
        Assert.Equal(3, metadata.Mask);
    }

    [Theory]
    [InlineData(-.1f, 1.1f)]
    [InlineData(.4f, .4f)]
    [InlineData(.6f, .6f)]
    [InlineData(float.NaN, .5f)]
    [InlineData(float.PositiveInfinity, 0)]
    public void Decode_RejectsInvalidAlreadyNormalizedActionHead(float first, float second)
    {
        var batch = Prepare().Prepare(Request(Choice()), TestContext.Current.CancellationToken);
        Assert.Throws<DecisionProtocolException>(() => new DecodeDecisions(Profile()).Decode(batch,
            new(Tensor.Create<float>([0, 1], [1, 2]), Tensor.Create<float>([first, second], [1, 2]))));
    }

    [Theory]
    [InlineData(2, 1, 1, 2)]
    [InlineData(1, 3, 1, 2)]
    [InlineData(1, 2, 2, 1)]
    [InlineData(1, 2, 1, 3)]
    public void Decode_RejectsRowAndWidthShapeMismatch(int rows, int width, int actionRows, int actionWidth)
    {
        var batch = Prepare().Prepare(Request(Choice()), TestContext.Current.CancellationToken);
        Assert.Throws<DecisionProtocolException>(() => new DecodeDecisions(Profile()).Decode(batch,
            new(Tensor.Create(new float[rows * width], [rows, width]), Tensor.Create(new float[actionRows * actionWidth], [actionRows, actionWidth]))));
    }
}

// Standard package seam, not a new runtime interface. IDs 100+c provide an independent framing oracle.
internal sealed class CharacterTokenizer : Tokenizer
{
    public List<string> Calls { get; } = [];
    public Func<string, int[]> Encode { get; init; } = text => text.Select(c => c + 100).ToArray();
    protected override EncodeResults<int> EncodeToIds(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings)
    {
        string value = text ?? textSpan.ToString();
        Calls.Add(value);
        return new() { Tokens = Encode(value), CharsConsumed = value.Length, NormalizedText = value };
    }
    protected override EncodeResults<EncodedToken> EncodeToTokens(string? text, ReadOnlySpan<char> textSpan, EncodeSettings settings) =>
        throw new NotSupportedException("This test seam exercises EncodeToIds only.");
    public override OperationStatus Decode(IEnumerable<int> ids, Span<char> destination, out int idsConsumed, out int charsWritten) =>
        throw new NotSupportedException("No production decoding is expected.");
}

internal sealed class TemporaryFixture : IDisposable
{
    public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"decision-fixture-{Guid.NewGuid():N}.bin");
    public TemporaryFixture(byte[] bytes) => File.WriteAllBytes(Path, bytes);
    public void Dispose() => File.Delete(Path); // Only the uniquely owned, untracked fixture.
}
