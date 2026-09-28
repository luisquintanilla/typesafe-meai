using System.Text.Json;
using System.Text.Json.Nodes;
using DecisionInference.Julia;
using Tokenizers.HuggingFace.Tokenizer;

namespace DecisionInference.Tests;

/// <summary>The established native tokenizer, with a tiny lossless vocabulary, not a tokenizer mock.</summary>
internal sealed class JuliaTestFixture : IDisposable
{
    internal string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"julia-{Guid.NewGuid():N}.json");

    internal JuliaTestFixture(Action<JsonObject>? mutate = null)
    {
        string[] framing = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>"];
        var vocab = new Dictionary<string, int>();
        foreach (string token in framing) vocab.Add(token, vocab.Count);
        for (int i = 0; i < 256; i++) vocab.Add($"<0x{i:X2}>", vocab.Count);
        for (int i = 33; i <= 126; i++) vocab.Add(((char)i).ToString(), vocab.Count);
        vocab.Add("\u2581", vocab.Count);
        var json = JsonSerializer.SerializeToNode(new
        {
            version = "1.0", truncation = (object?)null, padding = (object?)null,
            added_tokens = framing.Select((content, id) => new
            {
                id, content, single_word = false, lstrip = false, rstrip = false, normalized = false, special = true
            }),
            normalizer = new { type = "Replace", pattern = new { String = " " }, content = "\u2581" },
            pre_tokenizer = new { type = "Metaspace", replacement = "\u2581", prepend_scheme = "always", split = true },
            post_processor = (object?)null,
            decoder = new { type = "ByteFallback" },
            model = new
            {
                type = "BPE", dropout = (object?)null, unk_token = "<unk>",
                continuing_subword_prefix = (object?)null, end_of_word_suffix = (object?)null,
                fuse_unk = true, byte_fallback = true, ignore_merges = false, vocab, merges = Array.Empty<string>()
            }
        })!.AsObject();
        mutate?.Invoke(json);
        File.WriteAllText(Path, json.ToJsonString());
    }

    internal PrepareDecisionInputs Prepare() => new(Path);

    internal long[] Encode(string text)
    {
        using var tokenizer = Tokenizer.FromFile(Path);
        return tokenizer.Encode(text, addSpecialTokens: false).Single().Ids.Select(id => (long)id).ToArray();
    }

    internal static DecisionRequest Mixed() => new(ContractTests.Json("""{"state":"é"}"""),
    [
        new ChoiceDecisionQuestion("choice", "pick", [new(" a ", "red"), new("A", "blue")]),
        new ScoreDecisionQuestion("score", "rate", ["low", "mid", "high"]),
        new BinaryDecisionQuestion("binary", "decide", "yes", "no")
    ]);

    internal static long[] Row(JuliaInputBatch batch, int row) =>
        Enumerable.Range(0, batch.SequenceLength).Select(i => batch.InputIds[row, i]).ToArray();

    internal static void AssertRow(JuliaInputBatch batch, int row, long[] expected, long type, int[] positions)
    {
        Assert.Equal(expected, Row(batch, row).Take(expected.Length));
        Assert.All(Row(batch, row).Skip(expected.Length), value => Assert.Equal(0, value));
        Assert.Equal(0, batch.SequenceLength % 8);
        for (int i = 0; i < batch.SequenceLength; i++)
            Assert.Equal(i < expected.Length ? 1L : 0L, batch.AttentionMask[row, i]);
        for (int i = 0; i < batch.MarkerWidth; i++)
        {
            Assert.Equal(i < positions.Length ? positions[i] : 0, batch.MarkerPositions[row, i]);
            Assert.Equal(i < positions.Length, batch.MarkerMask[row, i]);
        }
        Assert.Equal(type, batch.QuestionTypes[row]);
    }

    public void Dispose() => File.Delete(Path);
}
