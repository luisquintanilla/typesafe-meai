using DecisionInference.Onnx;
using Microsoft.ML.OnnxRuntime;
using System.Numerics.Tensors;

namespace DecisionInference.Laya;

public sealed record LayaModelOutputs(Tensor<float> Logits, Tensor<float> ActionProbabilities);

/// <summary>Direct ORT execution; returns managed copies after releasing all native call buffers.</summary>
public sealed class ScoreOnnxDecisionModel : IDisposable
{
    private readonly MarkerOnnxRunner _runner;

    public ScoreOnnxDecisionModel(InferenceSession session, bool ownsSession = false)
    {
        ArgumentNullException.ThrowIfNull(session);
        MarkerOnnxRunner.ValidateInputs(session, "Laya");
        foreach (string name in new[] { "logits", "act_probs" })
        {
            if (!session.OutputMetadata.TryGetValue(name, out var output)) throw new ArgumentException($"Missing {name} output.");
            MarkerOnnxRunner.ValidateMetadata(name, output, typeof(float), 2, "Laya");
            if (name == "act_probs" && output.Dimensions[1] >= 0 && output.Dimensions[1] != 2)
                throw new ArgumentException("act_probs must have width 2.");
        }
        _runner = new(session, ownsSession, "Laya");
    }

    public static ScoreOnnxDecisionModel Load(string modelPath, SessionOptions? sessionOptions = null)
    {
        if (!File.Exists(modelPath)) throw new FileNotFoundException("An explicit local Laya model is required.", modelPath);
        InferenceSession session;
        if (sessionOptions is null) { using var defaults = new SessionOptions(); session = new(modelPath, defaults); }
        else session = new(modelPath, sessionOptions);
        try { return new(session, ownsSession: true); }
        catch { session.Dispose(); throw; }
    }

    public LayaModelOutputs Score(LayaInputBatch batch, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(batch);
        var outputs = _runner.Run(new(batch.BatchSize, batch.SequenceLength, batch.MarkerWidth,
            batch.Ids, batch.Attention, batch.Positions, batch.Masks, batch.Types),
            [new("logits", batch.MarkerWidth, "Laya output"), new("act_probs", 2, "Laya output")], this, cancellationToken);
        return new(outputs[0], outputs[1]);
    }

    internal LayaModelOutputs ScoreCombined(LayaCombinedBatch batch, CancellationToken cancellationToken)
    {
        var outputs = _runner.Run(batch.Tensors,
            [new("logits", batch.Tensors.Width, "Laya output"), new("act_probs", 2, "Laya output")], this, cancellationToken);
        return new(outputs[0], outputs[1]);
    }

    public void Dispose() => _runner.Dispose();
}
