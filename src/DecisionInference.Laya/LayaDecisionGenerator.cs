using Microsoft.ML.OnnxRuntime;

namespace DecisionInference.Laya;

/// <summary>Serializes synchronous local calls and disposal; ORT cancellation uses RunOptions.Terminate.</summary>
public sealed class LayaDecisionGenerator : IDecisionGenerator
{
    private readonly PrepareDecisionInputs _prepare;
    private readonly ScoreOnnxDecisionModel _score;
    private readonly DecodeDecisions _decode;
    private readonly bool _ownsScorer;
    private readonly SemaphoreSlim _gate = new(1);
    private bool _disposed;
    private static readonly DecisionGeneratorMetadata Metadata = new("laya", defaultModelId: LayaProfile.ModelId,
        supportedKinds: [DecisionKind.Binary, DecisionKind.Choice, DecisionKind.Score]);

    public LayaDecisionGenerator(PrepareDecisionInputs prepare, ScoreOnnxDecisionModel score,
        DecodeDecisions decode, bool ownsScorer = false)
    {
        _prepare = prepare ?? throw new ArgumentNullException(nameof(prepare));
        _score = score ?? throw new ArgumentNullException(nameof(score));
        _decode = decode ?? throw new ArgumentNullException(nameof(decode));
        _ownsScorer = ownsScorer;
    }

    public LayaDecisionGenerator(string modelPath, string tokenizerJsonPath, string profilePath, SessionOptions? options = null)
    {
        var profile = LayaProfile.Load(profilePath);
        var (tokenizer, metadata) = LayaTokenizer.Load(tokenizerJsonPath);
        _prepare = new(profile, tokenizer, metadata);
        _decode = new(profile);
        _score = ScoreOnnxDecisionModel.Load(modelPath, options);
        _ownsScorer = true;
    }

    public Task<DecisionResult> GenerateAsync(DecisionInput input, DecisionGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(input);
        var model = options?.ModelId;
        if (model is not null)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(model);
            if (!string.Equals(model, LayaProfile.ModelId, StringComparison.Ordinal))
                throw new NotSupportedException("This local generator cannot switch its loaded model.");
        }
        if (input.Questions.Any(q => Metadata.SupportedKinds?.Contains(q.Kind) != true))
            throw new NotSupportedException("The batch contains an unsupported decision kind.");
        _gate.Wait(cancellationToken);
        try
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            var batch = _prepare.Prepare(input, cancellationToken);
            var output = _score.Score(batch, cancellationToken);
            var result = _decode.Decode(batch, output);
            cancellationToken.ThrowIfCancellationRequested();
            return Task.FromResult(result);
        }
        finally { _gate.Release(); }
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null) return null;
        return serviceType == typeof(DecisionGeneratorMetadata) ? Metadata : serviceType.IsInstanceOfType(this) ? this : null;
    }

    public void Dispose()
    {
        _gate.Wait();
        try { if (!_disposed) { _disposed = true; if (_ownsScorer) _score.Dispose(); } }
        finally { _gate.Release(); }
    }
}
