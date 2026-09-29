using DecisionInference.Onnx;
using System.Numerics.Tensors;

namespace DecisionInference.Laya;

internal sealed record LayaCombinedBatch(MarkerInputs Tensors, IReadOnlyList<LayaInputBatch> Inputs, int[] StateOffsets)
{
    internal long ManagedTensorBytes => LayaBatchCollator.EstimatedBytes(Inputs);
    internal LayaModelOutputs Slice(LayaModelOutputs outputs, int state)
    {
        var input = Inputs[state];
        var logits = new float[checked(input.BatchSize * input.MarkerWidth)];
        var actions = new float[checked(input.BatchSize * 2)];
        for (int question = 0; question < input.BatchSize; question++)
        {
            int source = StateOffsets[state] + question;
            for (int option = 0; option < input.MarkerWidth; option++)
                logits[question * input.MarkerWidth + option] = outputs.Logits[source, option];
            for (int action = 0; action < 2; action++)
                actions[question * 2 + action] = outputs.ActionProbabilities[source, action];
        }
        return new(Tensor.Create(logits, [input.BatchSize, input.MarkerWidth]), Tensor.Create(actions, [input.BatchSize, 2]));
    }
}

internal static class LayaBatchCollator
{
    internal const long MaximumManagedTensorBytes = 64L * 1024 * 1024;

    internal static long InputBytes(int rows, int length, int width) => checked(
        (long)rows * length * sizeof(long) * 2 + (long)rows * width * (sizeof(long) + sizeof(bool)) +
        (long)rows * (sizeof(long) + sizeof(int)));

    internal static LayaCombinedBatch Collate(IReadOnlyList<LayaInputBatch> inputs, int padTokenId, CancellationToken token)
    {
        if (inputs.Count == 0) throw new ArgumentException("At least one prepared state is required.");
        int rows = checked(inputs.Sum(b => b.BatchSize));
        int length = inputs.Max(b => b.SequenceLength), width = inputs.Max(b => b.MarkerWidth);
        if (EstimatedBytes(inputs) > MaximumManagedTensorBytes)
            throw new ArgumentException("Laya native batch exceeds the 64 MiB managed tensor-buffer budget.");
        var ids = new long[checked(rows * length)];
        System.Array.Fill(ids, padTokenId);
        var attention = new long[ids.Length];
        var positions = new long[checked(rows * width)];
        var masks = new bool[positions.Length];
        var types = new long[rows];
        var offsets = new int[inputs.Count];
        int target = 0;
        for (int state = 0; state < inputs.Count; state++)
        {
            token.ThrowIfCancellationRequested();
            var input = inputs[state];
            offsets[state] = target;
            for (int question = 0; question < input.BatchSize; question++, target++)
            {
                System.Array.Copy(input.Ids, question * input.SequenceLength, ids, target * length, input.SequenceLength);
                System.Array.Copy(input.Attention, question * input.SequenceLength, attention, target * length, input.SequenceLength);
                System.Array.Copy(input.Positions, question * input.MarkerWidth, positions, target * width, input.MarkerWidth);
                System.Array.Copy(input.Masks, question * input.MarkerWidth, masks, target * width, input.MarkerWidth);
                types[target] = input.Types[question];
            }
        }
        return new(new(rows, length, width, ids, attention, positions, masks, types), inputs, offsets);
    }

    internal static long EstimatedBytes(IReadOnlyList<LayaInputBatch> inputs)
    {
        int rows = checked(inputs.Sum(b => b.BatchSize));
        int length = inputs.Max(b => b.SequenceLength), width = inputs.Max(b => b.MarkerWidth);
        long preparedBytes = inputs.Sum(b => InputBytes(b.BatchSize, b.SequenceLength, b.MarkerWidth));
        long combinedBytes = InputBytes(rows, length, width);
        // Both combined outputs and simultaneously owned per-state output slices count toward the bound.
        long outputBytes = checked((long)rows * (width + 2L) * sizeof(float) * 2);
        return checked(preparedBytes + combinedBytes + outputBytes);
    }

    internal static bool SameQuestions(DecisionInput first, DecisionInput second) =>
        first.Questions.Count == second.Questions.Count && first.Questions.Zip(second.Questions).All(pair =>
        {
            var (a, b) = pair;
            if (a.Id != b.Id || a.Kind != b.Kind || a.Instructions != b.Instructions) return false;
            return (a, b) switch
            {
                (BinaryDecisionQuestion x, BinaryDecisionQuestion y) => x.TrueDescription == y.TrueDescription && x.FalseDescription == y.FalseDescription,
                (ChoiceDecisionQuestion x, ChoiceDecisionQuestion y) => x.Candidates.Count == y.Candidates.Count &&
                    x.Candidates.Zip(y.Candidates).All(c => c.First.Id == c.Second.Id && c.First.Description == c.Second.Description),
                (ScoreDecisionQuestion x, ScoreDecisionQuestion y) => x.Rubric.SequenceEqual(y.Rubric, StringComparer.Ordinal),
                _ => false
            };
        });
}
