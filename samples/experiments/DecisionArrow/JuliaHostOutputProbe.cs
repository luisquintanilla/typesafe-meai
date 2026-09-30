using DecisionInference;
using DecisionInference.Julia;
using Microsoft.ML.OnnxRuntime;

internal static class JuliaHostOutputProbe
{
    internal static object Check(InferenceSession session, JuliaInputBatch batch, CancellationToken token)
    {
        long[] sequence = [batch.BatchSize, batch.SequenceLength], markers = [batch.BatchSize, batch.MarkerWidth];
        using var ids = OrtValue.CreateTensorValueFromMemory(batch.Ids, sequence);
        using var attention = OrtValue.CreateTensorValueFromMemory(batch.Attention, sequence);
        using var positions = OrtValue.CreateTensorValueFromMemory(batch.Positions, markers);
        using var masks = OrtValue.CreateTensorValueFromMemory(batch.Masks, markers);
        using var qtypes = OrtValue.CreateTensorValueFromMemory(batch.Types, [batch.BatchSize]);
        using var options = new RunOptions();
        using var registration = token.Register(() => options.Terminate = true);
        using var outputs = session.Run(options, new Dictionary<string, OrtValue>
        {
            ["input_ids"] = ids, ["attention_mask"] = attention, ["marker_pos"] = positions,
            ["marker_mask"] = masks, ["qtype"] = qtypes
        }, ["logits"]);
        token.ThrowIfCancellationRequested();
        using var memory = outputs[0].GetTensorMemoryInfo();
        var shape = outputs[0].GetTensorTypeAndShape();
        if (memory.Name != "Cpu" || shape.ElementDataType != Microsoft.ML.OnnxRuntime.Tensors.TensorElementType.Float ||
            !shape.Shape.SequenceEqual(new long[] { batch.BatchSize, batch.MarkerWidth }))
            throw new DecisionProtocolException("Default synchronized Julia Run did not return the expected host-readable FP32 output.");
        var values = outputs[0].GetTensorDataAsSpan<float>();
        for (int row = 0; row < batch.BatchSize; row++)
            for (int i = 0; i < batch.OptionCounts[row]; i++)
                if (!float.IsFinite(values[row * batch.MarkerWidth + i]))
                    throw new DecisionProtocolException("Nonfinite Julia host-probe valid lane.");
        return new
        {
            name = memory.Name, deviceId = memory.Id, memoryType = memory.GetMemoryType().ToString(),
            allocatorType = memory.GetAllocatorType().ToString(), deviceMemoryType = memory.GetDeviceMemoryType().ToString(),
            shape = shape.Shape, fp32 = true, hostReadableBeforeSpan = true,
            policy = "Original default synchronized Run; no IOBinding/GPU pointer borrow; one separate readiness call"
        };
    }
}
