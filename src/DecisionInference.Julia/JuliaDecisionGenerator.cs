using Microsoft.ML.OnnxRuntime;

namespace DecisionInference.Julia;

/// <summary>Native Julia-1 composition using the same public prepare, score and decode stages.</summary>
public sealed class JuliaDecisionGenerator : IDecisionGenerator
{
    public const string ModelId = "SupersonicLabs/Julia-1-ONNX";
    private readonly PrepareDecisionInputs _prepare;
    private readonly ScoreOnnxDecisionModel _score;
    private readonly DecodeDecisions _decode;
    private readonly bool _ownsStages;
    private readonly SemaphoreSlim _gate = new(1);
    private bool _disposed;
    private static readonly DecisionGeneratorMetadata Metadata = new("julia", defaultModelId: ModelId,
        supportedKinds: [DecisionKind.Binary, DecisionKind.Choice, DecisionKind.Score]);

    public JuliaDecisionGenerator(PrepareDecisionInputs prepare, ScoreOnnxDecisionModel score, DecodeDecisions decode, bool ownsStages = false)
    {
        _prepare = prepare ?? throw new ArgumentNullException(nameof(prepare));
        _score = score ?? throw new ArgumentNullException(nameof(score));
        _decode = decode ?? throw new ArgumentNullException(nameof(decode));
        _ownsStages = ownsStages;
    }

    public JuliaDecisionGenerator(string modelPath, string tokenizerJsonPath, SessionOptions? sessionOptions = null)
    {
        _prepare = new(tokenizerJsonPath);
        try { _score = ScoreOnnxDecisionModel.Load(modelPath, sessionOptions); }
        catch { _prepare.Dispose(); throw; }
        _decode = new();
        _ownsStages = true;
    }

    public Task<DecisionResult> GenerateAsync(DecisionInput input, DecisionGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (options?.ModelId is { } model && !string.Equals(model, ModelId, StringComparison.Ordinal))
            throw new NotSupportedException("This local Julia generator cannot switch its loaded model.");
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
        try
        {
            if (!_disposed)
            {
                _disposed = true;
                if (_ownsStages) { _score.Dispose(); _prepare.Dispose(); }
            }
        }
        finally { _gate.Release(); }
    }
}
