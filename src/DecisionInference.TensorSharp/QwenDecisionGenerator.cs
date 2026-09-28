using System.Numerics.Tensors;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace DecisionInference.TensorSharp;

/// <summary>Owned managed answer-token scores, copied before Qwen resets or reuses its buffers.</summary>
public sealed class QwenRawAnswer
{
    public string Provider => "tensorsharp-qwen";
    public bool ConditionalOnAnswerTokens => true;
    public string Aliases { get; }
    public IReadOnlyList<int> TokenIds { get; }
    public IReadOnlyList<double> Logits { get; }

    internal QwenRawAnswer(string aliases, int[] tokenIds, double[] logits)
        => (Aliases, TokenIds, Logits) = (aliases, Array.AsReadOnly(tokenIds), Array.AsReadOnly(logits));
}

/// <summary>Observed forward count and effective context limit for one Qwen result.</summary>
public sealed record QwenRawResult(int ForwardCalls, int ContextLength);

/// <summary>Exclusive access to one Qwen model. Borrowers must not use or dispose the model during calls.</summary>
/// <remarks>Calls are synchronous and serialized; cancellation is checked around, not during, ForwardRefill.</remarks>
public sealed class QwenDecisionGenerator : IDecisionGenerator
{
    public const string ModelSha256 = "7f4030143c1c477224c5434f8272c662a8b042079a0a584f0a27a1684fe2e1fa";
    public const string PinnedModelId = "Qwen3-0.6B-Q4_K_M@" + ModelSha256;
    private readonly ModelBase _model;
    private readonly bool _ownsModel;
    private readonly SemaphoreSlim _gate = new(1);
    private readonly int[] _aliases;
    private readonly DecisionGeneratorMetadata _metadata;
    private readonly string _modelId;
    private bool _disposed;

    public QwenDecisionGenerator(ModelBase model, string modelId, bool ownsModel = false)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        if (model is not Qwen3Model || model.MaxContextLength <= 0)
            throw new ArgumentException("A loaded Qwen3 model with an effective context bound is required.");
        _aliases = QwenScoring.AliasIds(model.Tokenizer);
        _modelId = modelId;
        _metadata = new("tensorsharp-qwen", defaultModelId: modelId, supportedKinds: [DecisionKind.Choice]);
        (_model, _ownsModel) = (model, ownsModel);
    }

    /// <summary>Loads the hash-pinned local GGUF on CPU. The host configures MAX_CONTEXT before this call.</summary>
    public static QwenDecisionGenerator Load(string ggufPath)
    {
        using (var stream = File.OpenRead(ggufPath))
            if (!Convert.ToHexString(SHA256.HashData(stream)).Equals(ModelSha256, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("GGUF does not match the pinned Qwen3 0.6B Q4_K_M SHA256.");
        var model = ModelBase.Create(ggufPath, BackendType.Cpu);
        try { return new(model, PinnedModelId, ownsModel: true); }
        catch { model.Dispose(); throw; }
    }

    public Task<DecisionResult> GenerateAsync(DecisionInput input, DecisionGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ArgumentNullException.ThrowIfNull(input);
        var model = options?.ModelId;
        if (model is not null)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(model);
            if (!string.Equals(model, _modelId, StringComparison.Ordinal))
                throw new NotSupportedException("This local generator cannot switch its loaded model.");
        }
        if (input.Questions.Any(q => q.Kind != DecisionKind.Choice))
            throw new NotSupportedException("The batch contains an unsupported decision kind.");
        _gate.Wait(cancellationToken);
        try
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            // Validate every prompt before changing KV state or running the first forward.
            var prompts = input.Questions.Cast<ChoiceDecisionQuestion>().Select(q =>
                QwenScoring.Prepare(_model.Tokenizer, input.State, q, _model.MaxContextLength)).ToArray();
            var answers = new List<DecisionAnswer>();
            long inputTokens = 0;
            for (int row = 0; row < prompts.Length; row++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                _model.ResetKVCache();
                try
                {
                    // ForwardRefill has no cancellation API. Do not abandon this synchronous operation.
                    var raw = _model.ForwardRefill(prompts[row]);
                    var q = (ChoiceDecisionQuestion)input.Questions[row];
                    var logits = new double[q.Candidates.Count];
                    for (int i = 0; i < logits.Length; i++)
                    {
                        if (_aliases[i] >= raw.Length) throw new DecisionProtocolException("Answer token is outside the output vocabulary.");
                        logits[i] = raw[_aliases[i]];
                    }
                    cancellationToken.ThrowIfCancellationRequested();
                    var probabilities = QwenScoring.Normalize(logits);
                    answers.Add(new ChoiceDecisionAnswer(q.Id, q.Candidates[Array.IndexOf(probabilities, probabilities.Max())].Id,
                        q.Candidates.Select((c, i) => KeyValuePair.Create(c.Id, probabilities[i])))
                        { RawRepresentation = new QwenRawAnswer(QwenScoring.Aliases[..q.Candidates.Count], _aliases[..q.Candidates.Count], logits) });
                    inputTokens += prompts[row].Length;
                }
                finally { _model.ResetKVCache(); }
            }
            cancellationToken.ThrowIfCancellationRequested();
            return Task.FromResult(new DecisionResult(input, _modelId, answers, new() { InputTokenCount = inputTokens })
                { RawRepresentation = new QwenRawResult(prompts.Length, _model.MaxContextLength) });
        }
        finally { _gate.Release(); }
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null) return null;
        return serviceType == typeof(DecisionGeneratorMetadata) ? _metadata : serviceType.IsInstanceOfType(this) ? this : null;
    }

    public void Dispose()
    {
        _gate.Wait();
        try { if (!_disposed) { _disposed = true; if (_ownsModel) _model.Dispose(); } }
        finally { _gate.Release(); }
    }
}

internal static class QwenScoring
{
    internal const string Aliases = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    internal static int[] AliasIds(ITokenizer tokenizer)
    {
        var result = Aliases.Select(a =>
        {
            var ids = tokenizer.Encode(a.ToString(), addSpecial: false);
            if (ids.Count != 1 || ids[0] < 0) throw new NotSupportedException("Every answer alias must be one valid token.");
            return ids[0];
        }).ToArray();
        if (result.Distinct().Count() != result.Length) throw new NotSupportedException("Answer aliases must have distinct token IDs.");
        return result;
    }

    internal static int[] Prepare(ITokenizer tokenizer, JsonElement state, ChoiceDecisionQuestion question, int maxContext)
    {
        if (question.Candidates.Count is < 2 or > 26) throw new NotSupportedException("Qwen Choice supports 2 through 26 candidates.");
        string text = Render(state);
        string instruction = question.Instructions;
        string choices = string.Join("\n", question.Candidates.Select((c, i) => $"{Aliases[i]}. {c.Description}"));
        foreach (var part in new[] { text, instruction, choices })
            if (part.Contains("<|", StringComparison.Ordinal) || part.Contains("<think>", StringComparison.Ordinal) ||
                part.Contains("</think>", StringComparison.Ordinal))
                throw new NotSupportedException("Caller text must not contain reserved Qwen prompt markers.");
        string prompt = $"<|im_start|>system\n{instruction}<|im_end|>\n" +
            $"<|im_start|>user\n{text}\n\n{choices}<|im_end|>\n" +
            "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        var ids = tokenizer.Encode(prompt, addSpecial: false).ToArray();
        if (ids.Length == 0 || ids.Length > maxContext) throw new ArgumentException("Qwen prompt exceeds the effective context budget; truncation is disabled.");
        return ids;
    }

    internal static double[] Normalize(double[] logits)
    {
        if (logits.Length == 0 || logits.Any(x => !double.IsFinite(x))) throw new DecisionProtocolException("Nonfinite answer-token logit.");
        var values = new double[logits.Length];
        TensorPrimitives.Subtract<double>(logits, TensorPrimitives.Max<double>(logits), values);
        TensorPrimitives.Exp<double>(values, values);
        TensorPrimitives.Divide<double>(values, TensorPrimitives.Sum<double>(values), values);
        return values;
    }

    private static string Render(JsonElement value) => value.ValueKind == JsonValueKind.String ? value.GetString()! : value.GetRawText();
}
