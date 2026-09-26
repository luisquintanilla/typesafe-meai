using Microsoft.ML.OnnxRuntime;
using TensorElementType = Microsoft.ML.OnnxRuntime.Tensors.TensorElementType;
using System.Numerics.Tensors;

namespace DecisionInference.Laya;

public sealed record LayaModelOutputs(Tensor<float> Logits, Tensor<float> ActionProbabilities);

/// <summary>Direct ORT execution; returns managed copies after releasing all native call buffers.</summary>
public sealed class ScoreOnnxDecisionModel : IDisposable
{
    private readonly InferenceSession _session;
    private readonly bool _ownsSession;
    private readonly SemaphoreSlim _gate = new(1);
    private bool _disposed;
    private static readonly string[] Inputs = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"];

    public ScoreOnnxDecisionModel(InferenceSession session, bool ownsSession = false)
    {
        ArgumentNullException.ThrowIfNull(session);
        if (!session.InputMetadata.Keys.ToHashSet(StringComparer.Ordinal).SetEquals(Inputs))
            throw new ArgumentException("Laya requires exactly five named inputs.");
        foreach (string name in Inputs)
            ValidateMetadata(name, session.InputMetadata[name], name == "marker_mask" ? typeof(bool) : typeof(long), name == "qtype" ? 1 : 2);
        foreach (string name in new[] { "logits", "act_probs" })
        {
            if (!session.OutputMetadata.TryGetValue(name, out var output)) throw new ArgumentException($"Missing {name} output.");
            ValidateMetadata(name, output, typeof(float), 2);
            if (name == "act_probs" && output.Dimensions[1] >= 0 && output.Dimensions[1] != 2)
                throw new ArgumentException("act_probs must have width 2.");
        }
        (_session, _ownsSession) = (session, ownsSession);
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
        _gate.Wait(cancellationToken);
        try
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            long[] sequence = [batch.BatchSize, batch.SequenceLength], markers = [batch.BatchSize, batch.MarkerWidth], types = [batch.BatchSize];
            foreach (var input in _session.InputMetadata)
            {
                var shape = input.Key == "qtype" ? types : input.Key is "marker_pos" or "marker_mask" ? markers : sequence;
                if (input.Value.Dimensions.Where((d, i) => d >= 0 && d != shape[i]).Any())
                    throw new ArgumentException($"Batch does not fit static graph input {input.Key}.");
            }
            using var ids = OrtValue.CreateTensorValueFromMemory(batch.Ids, sequence);
            using var attention = OrtValue.CreateTensorValueFromMemory(batch.Attention, sequence);
            using var positions = OrtValue.CreateTensorValueFromMemory(batch.Positions, markers);
            using var masks = OrtValue.CreateTensorValueFromMemory(batch.Masks, markers);
            using var qtypes = OrtValue.CreateTensorValueFromMemory(batch.Types, types);
            using var options = new RunOptions();
            using var registration = cancellationToken.Register(() => options.Terminate = true);
            try
            {
                using var outputs = _session.Run(options, new Dictionary<string, OrtValue>
                {
                    ["input_ids"] = ids, ["attention_mask"] = attention, ["marker_pos"] = positions,
                    ["marker_mask"] = masks, ["qtype"] = qtypes
                }, ["logits", "act_probs"]);
                cancellationToken.ThrowIfCancellationRequested();
                return new(Read(outputs[0], batch.BatchSize, batch.MarkerWidth), Read(outputs[1], batch.BatchSize, 2));
            }
            catch (OnnxRuntimeException error) when (cancellationToken.IsCancellationRequested)
            {
                throw new OperationCanceledException("ONNX inference was cancelled.", error, cancellationToken);
            }
        }
        finally { _gate.Release(); }
    }

    public void Dispose()
    {
        _gate.Wait();
        try { if (!_disposed) { _disposed = true; if (_ownsSession) _session.Dispose(); } }
        finally { _gate.Release(); }
    }

    private static Tensor<float> Read(OrtValue value, int rows, int width)
    {
        var info = value.GetTensorTypeAndShape();
        if (info.ElementDataType != TensorElementType.Float || !info.Shape.SequenceEqual(new long[] { rows, width }))
            throw new DecisionProtocolException("Unexpected Laya output shape or type.");
        return Tensor.Create(value.GetTensorDataAsSpan<float>().ToArray(), [rows, width]);
    }

    private static void ValidateMetadata(string name, NodeMetadata value, Type elementType, int rank)
    {
        if (!value.IsTensor || value.ElementType != elementType || value.Dimensions.Length != rank)
            throw new ArgumentException($"Invalid Laya graph tensor {name}.");
    }
}
