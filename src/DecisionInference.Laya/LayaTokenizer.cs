using System.Text;
using System.Text.Json;
using Microsoft.ML.Tokenizers;

namespace DecisionInference.Laya;

public sealed class LayaTokenMetadata
{
    public int Cls { get; }
    public int Sep { get; }
    public int Mask { get; }
    public int Pad { get; }
    public IReadOnlyList<string> ReservedTokens { get; }
    public LayaTokenMetadata(int cls, int sep, int mask, int pad, IEnumerable<string> reservedTokens)
    {
        ArgumentNullException.ThrowIfNull(reservedTokens);
        int[] ids = [cls, sep, mask, pad];
        if (ids.Any(i => i < 0) || ids.Distinct().Count() != 4) throw new ArgumentException("Special IDs must be distinct and nonnegative.");
        var reserved = reservedTokens.ToArray();
        if (reserved.Any(string.IsNullOrEmpty)) throw new ArgumentException("Reserved tokens must not be empty.");
        (Cls, Sep, Mask, Pad) = (cls, sep, mask, pad);
        ReservedTokens = Array.AsReadOnly(reserved);
    }
}

/// <summary>Selected Laya BPE configuration, not a tokenizer implementation or general HF graph loader.</summary>
public static class LayaTokenizer
{
    public static (Tokenizer Tokenizer, LayaTokenMetadata Metadata) Load(string tokenizerJsonPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(tokenizerJsonPath));
        var root = doc.RootElement;
        var model = root.GetProperty("model");
        var pre = root.GetProperty("pre_tokenizer");
        if (model.GetProperty("type").GetString() != "BPE" ||
            root.GetProperty("normalizer").GetProperty("type").GetString() != "NFC" ||
            pre.GetProperty("type").GetString() != "ByteLevel" || pre.GetProperty("add_prefix_space").GetBoolean() ||
            !pre.GetProperty("trim_offsets").GetBoolean() || !pre.GetProperty("use_regex").GetBoolean() ||
            model.GetProperty("byte_fallback").GetBoolean() || model.GetProperty("ignore_merges").GetBoolean() ||
            model.GetProperty("fuse_unk").GetBoolean() ||
            model.GetProperty("dropout").ValueKind != JsonValueKind.Null ||
            model.GetProperty("unk_token").ValueKind != JsonValueKind.Null ||
            model.GetProperty("continuing_subword_prefix").ValueKind != JsonValueKind.Null ||
            model.GetProperty("end_of_word_suffix").ValueKind != JsonValueKind.Null)
            throw new NotSupportedException("Only the pinned Laya NFC/ByteLevel BPE configuration is supported.");

        Dictionary<string, int> special = new(StringComparer.Ordinal);
        List<string> reserved = [];
        foreach (var entry in root.GetProperty("added_tokens").EnumerateArray())
        {
            string text = entry.GetProperty("content").GetString()!;
            special.Add(text, entry.GetProperty("id").GetInt32());
            if (entry.GetProperty("special").GetBoolean() || entry.GetProperty("lstrip").GetBoolean() ||
                entry.GetProperty("rstrip").GetBoolean() || entry.GetProperty("single_word").GetBoolean())
                reserved.Add(text);
        }
        var metadata = new LayaTokenMetadata(special["[CLS]"], special["[SEP]"], special["[MASK]"], special["[PAD]"], reserved);
        var vocabulary = model.GetProperty("vocab").EnumerateObject().Select(p => KeyValuePair.Create(p.Name, p.Value.GetInt32())).ToArray();
        var merges = model.GetProperty("merges").EnumerateArray().Select(m =>
        {
            if (m.ValueKind == JsonValueKind.String) return m.GetString()!;
            var pair = m.EnumerateArray().ToArray();
            if (pair.Length != 2) throw new InvalidDataException("Expected a two-token BPE merge.");
            return $"{pair[0].GetString()} {pair[1].GetString()}";
        }).ToArray();
        return (BpeTokenizer.Create(new BpeOptions(vocabulary)
        {
            Merges = merges, SpecialTokens = special, ByteLevel = true,
            PreTokenizer = RobertaPreTokenizer.Instance, Normalizer = new NfcNormalizer()
        }), metadata);
    }

    private sealed class NfcNormalizer : Normalizer
    {
        public override string Normalize(string original) => original.Normalize(NormalizationForm.FormC);
        public override string Normalize(ReadOnlySpan<char> original) => original.ToString().Normalize(NormalizationForm.FormC);
    }
}
