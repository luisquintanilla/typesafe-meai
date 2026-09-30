using Microsoft.ML.OnnxRuntime;

namespace DecisionInference.Laya;

/// <summary>Local construction settings, not per-call decision generation options.</summary>
public sealed class LayaDecisionGeneratorOptions
{
    public required string ModelPath { get; init; }
    public required string TokenizerJsonPath { get; init; }
    public required string ProfilePath { get; init; }

    /// <summary>Configures temporary options once before loading. Do not retain or dispose them.</summary>
    public Action<SessionOptions>? ConfigureSession { get; init; }
}
