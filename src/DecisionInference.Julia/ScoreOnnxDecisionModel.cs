using Microsoft.ML.OnnxRuntime;
using System.Numerics.Tensors;
using TensorElementType = Microsoft.ML.OnnxRuntime.Tensors.TensorElementType;

namespace DecisionInference.Julia;

/// <summary>Julia's logits-only ORT stage. Native buffers are released before returning managed tensors.</summary>
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
        if (!session.InputMetadata.Keys.ToHashSet(StringComparer.Ordinal).SetEquals(Inputs) ||
            !session.OutputMetadata.Keys.SequenceEqual(["logits"]))
            throw new ArgumentException("Julia requires exactly five named inputs and only the logits output.");
        foreach (string name in Inputs)
            ValidateMetadata(name, session.InputMetadata[name], name == "marker_mask" ? typeof(bool) : typeof(long), name == "qtype" ? 1 : 2);
        ValidateMetadata("logits", session.OutputMetadata["logits"], typeof(float), 2);
        (_session, _ownsSession) = (session, ownsSession);
    }

    public static ScoreOnnxDecisionModel Load(string modelPath, SessionOptions? sessionOptions = null)
    {
        if (!File.Exists(modelPath)) throw new FileNotFoundException("An explicit local Julia model is required.", modelPath);
        InferenceSession session;
        if (sessionOptions is null) { using var defaults = new SessionOptions(); session = new(modelPath, defaults); }
        else session = new(modelPath, sessionOptions);
        try { return new(session, ownsSession: true); }
        catch { session.Dispose(); throw; }
    }

    public Tensor<float> Score(JuliaInputBatch batch, CancellationToken cancellationToken = default)
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
                    throw new ArgumentException($"Batch does not fit static Julia graph input {input.Key}.");
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
                }, ["logits"]);
                cancellationToken.ThrowIfCancellationRequested();
                var info = outputs[0].GetTensorTypeAndShape();
                if (info.ElementDataType != TensorElementType.Float || !info.Shape.SequenceEqual(markers))
                    throw new DecisionProtocolException("Unexpected Julia logits shape or type.");
                var logits = Tensor.Create(outputs[0].GetTensorDataAsSpan<float>().ToArray(), [batch.BatchSize, batch.MarkerWidth]);
                for (int row = 0; row < batch.BatchSize; row++)
                    for (int i = 0; i < batch.OptionCounts[row]; i++)
                        if (!float.IsFinite(logits[row, i])) throw new DecisionProtocolException("Nonfinite Julia valid-option logit.");
                return logits;
            }
            catch (OnnxRuntimeException error) when (cancellationToken.IsCancellationRequested)
            {
                throw new OperationCanceledException("Julia ONNX inference was cancelled.", error, cancellationToken);
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

    private static void ValidateMetadata(string name, NodeMetadata value, Type type, int rank)
    {
        if (!value.IsTensor || value.ElementType != type || value.Dimensions.Length != rank)
            throw new ArgumentException($"Invalid Julia graph tensor {name}.");
    }
}
