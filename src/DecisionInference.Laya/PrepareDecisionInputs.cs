using System.Globalization;
using System.Numerics.Tensors;
using System.Text.Json;
using Microsoft.ML.Tokenizers;

namespace DecisionInference.Laya;

/// <summary>Owned, per-input buffers. Public inspection uses standard read-only tensor spans.</summary>
public sealed class LayaInputBatch
{
    public DecisionInput Input { get; }
    public int BatchSize => Input.Questions.Count;
    public int SequenceLength { get; }
    public int MarkerWidth { get; }
    internal long[] Ids { get; }
    internal long[] Attention { get; }
    internal long[] Positions { get; }
    internal bool[] Masks { get; }
    internal long[] Types { get; }
    internal int[] OptionCounts { get; }
    public ReadOnlyTensorSpan<long> InputIds => new(Ids, [BatchSize, SequenceLength]);
    public ReadOnlyTensorSpan<long> AttentionMask => new(Attention, [BatchSize, SequenceLength]);
    public ReadOnlyTensorSpan<long> MarkerPositions => new(Positions, [BatchSize, MarkerWidth]);
    public ReadOnlyTensorSpan<bool> MarkerMask => new(Masks, [BatchSize, MarkerWidth]);
    public ReadOnlyTensorSpan<long> QuestionTypes => new(Types, [BatchSize]);

    internal LayaInputBatch(DecisionInput input, int length, int width, long[] ids, long[] attention,
        long[] positions, bool[] masks, long[] types, int[] optionCounts)
        => (Input, SequenceLength, MarkerWidth, Ids, Attention, Positions, Masks, Types, OptionCounts) =
            (input, length, width, ids, attention, positions, masks, types, optionCounts);
}

// Laya framing adapted from PrepareDecisionInputs at e9596f6; truncation/rewrite paths deliberately removed.
public sealed class PrepareDecisionInputs(LayaProfile profile, Tokenizer tokenizer, LayaTokenMetadata metadata)
{
    private readonly LayaProfile _profile = profile ?? throw new ArgumentNullException(nameof(profile));
    private readonly Tokenizer _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
    private readonly LayaTokenMetadata _metadata = metadata ?? throw new ArgumentNullException(nameof(metadata));
    internal int PadTokenId => _metadata.Pad;

    public LayaInputBatch Prepare(DecisionInput input, CancellationToken cancellationToken = default) =>
        PrepareCore(input, null, cancellationToken);

    internal LayaInputBatch PrepareBounded(DecisionInput input, long availableTensorBytes, CancellationToken cancellationToken) =>
        PrepareCore(input, availableTensorBytes, cancellationToken);

    private LayaInputBatch PrepareCore(DecisionInput input, long? availableTensorBytes, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(input);
        cancellationToken.ThrowIfCancellationRequested();
        var state = Encode(Render(input.State), _profile.MaxLength, "state");
        var rows = new List<(int[] Ids, int[] Positions, long Type)>();
        foreach (var q in input.Questions)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string[] options = q switch
            {
                ChoiceDecisionQuestion c => c.Candidates.Select((c, i) => $"{i.ToString(CultureInfo.InvariantCulture)}: {c.Description}").ToArray(),
                ScoreDecisionQuestion s => s.Rubric.Select((r, i) => $"level {i.ToString(CultureInfo.InvariantCulture)}: {r}").ToArray(),
                BinaryDecisionQuestion b => [
                    $"false: {b.FalseDescription ?? "no, the statement does not hold"}",
                    $"true: {b.TrueDescription ?? "yes, the statement holds"}"],
                _ => throw new NotSupportedException("Unknown question kind.")
            };
            if (options.Length < 2 || options.Length > _profile.HeadMaxLength)
                throw new ArgumentException($"Laya question '{q.Id}' requires 2..{_profile.HeadMaxLength} options; received {options.Length}.");
            var optionIds = options.Select(o => Encode(" " + o, 49, "option")).ToArray();
            int optionLength = optionIds.Sum(o => o.Length + 1);
            // The reference reserves at least 16 head tokens before considering option compression.
            int instructionBudget = _profile.HeadMaxLength - optionLength;
            if (instructionBudget < 16) throw new ArgumentException($"Laya question '{q.Id}' options use {optionLength}/{_profile.HeadMaxLength} head tokens; at least 16 instruction tokens are required.");
            var head = Encode($"{LayaProfile.KindName(q.Kind)} question: {q.Instructions}", instructionBudget, "instructions");
            if ((long)head.Length + optionLength + state.Length + 4 > _profile.MaxLength)
                throw new ArgumentException($"Laya question '{q.Id}' framed tokens {head.Length + optionLength + state.Length + 4} exceed context budget {_profile.MaxLength}; state tokens={state.Length}.");
            List<int> ids = [_metadata.Cls, .. head, _metadata.Sep];
            List<int> positions = [];
            foreach (var option in optionIds) { positions.Add(ids.Count); ids.Add(_metadata.Mask); ids.AddRange(option); }
            ids.Add(_metadata.Sep);
            ids.AddRange(state);
            ids.Add(_metadata.Sep);
            rows.Add((ids.ToArray(), positions.ToArray(), q.Kind switch { DecisionKind.Choice => 0, DecisionKind.Score => 1, _ => 2 }));
        }
        int length = rows.Max(r => r.Ids.Length), width = rows.Max(r => r.Positions.Length);
        if (availableTensorBytes is { } available && LayaBatchCollator.InputBytes(rows.Count, length, width) > available)
            throw new ArgumentException("Laya native batch exceeds the 64 MiB managed tensor-buffer budget.");
        var tokens = new long[checked(rows.Count * length)];
        Array.Fill(tokens, _metadata.Pad);
        var attention = new long[tokens.Length];
        var markers = new long[checked(rows.Count * width)];
        var masks = new bool[markers.Length];
        for (int row = 0; row < rows.Count; row++)
        {
            for (int i = 0; i < rows[row].Ids.Length; i++) { tokens[row * length + i] = rows[row].Ids[i]; attention[row * length + i] = 1; }
            for (int i = 0; i < rows[row].Positions.Length; i++) { markers[row * width + i] = rows[row].Positions[i]; masks[row * width + i] = true; }
        }
        return new(input, length, width, tokens, attention, markers, masks,
            rows.Select(r => r.Type).ToArray(), rows.Select(r => r.Positions.Length).ToArray());
    }

    private int[] Encode(string text, int budget, string part)
    {
        if (_metadata.ReservedTokens.Any(t => text.Contains(t, StringComparison.Ordinal)))
            throw new NotSupportedException($"Reserved token input in {part} cannot be preserved by this framing.");
        var ids = _tokenizer.EncodeToIds(text).ToArray();
        if (ids.Length > budget) throw new ArgumentException($"Laya {part} token budget exceeded ({ids.Length}>{budget}); truncation is disabled.");
        if (ids.Any(i => i < 0 || i == _metadata.Cls || i == _metadata.Sep || i == _metadata.Mask || i == _metadata.Pad))
            throw new NotSupportedException("Tokenizer emitted reserved framing IDs from caller text.");
        return ids;
    }

    internal static string Render(JsonElement value) => value.ValueKind switch
    {
        JsonValueKind.Undefined => "", JsonValueKind.String => value.GetString()!, _ => value.GetRawText()
    };
}
