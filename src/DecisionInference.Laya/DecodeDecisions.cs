using System.Numerics.Tensors;

namespace DecisionInference.Laya;

/// <summary>Owned managed raw values for one Laya answer; no tensors or native handles are retained.</summary>
public sealed class LayaRawAnswer
{
    public string Provider => "laya";
    public double Temperature { get; }
    public IReadOnlyList<double> Logits { get; }
    public IReadOnlyList<double> ActionProbabilities { get; }

    internal LayaRawAnswer(double temperature, double[] logits, double[] actionProbabilities)
        => (Temperature, Logits, ActionProbabilities) = (temperature, Array.AsReadOnly(logits), Array.AsReadOnly(actionProbabilities));
}

public sealed class DecodeDecisions(LayaProfile profile)
{
    private readonly LayaProfile _profile = profile ?? throw new ArgumentNullException(nameof(profile));

    public DecisionResult Decode(LayaInputBatch inputs, LayaModelOutputs outputs)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(outputs);
        if (!outputs.Logits.Lengths.SequenceEqual(new nint[] { inputs.BatchSize, inputs.MarkerWidth }) ||
            !outputs.ActionProbabilities.Lengths.SequenceEqual(new nint[] { inputs.BatchSize, 2 }))
            throw new DecisionProtocolException("Laya input/output shapes disagree.");
        var answers = new List<DecisionAnswer>();
        for (int row = 0; row < inputs.BatchSize; row++)
        {
            var q = inputs.Input.Questions[row];
            int count = inputs.OptionCounts[row];
            double[] logits = new double[count];
            for (int i = 0; i < count; i++) logits[i] = outputs.Logits[row, i];
            if (logits.Any(l => !double.IsFinite(l))) throw new DecisionProtocolException("Nonfinite valid-option logit.");
            double temperature = _profile.Temperature(q.Kind, count);
            double[] probabilities = new double[count];
            TensorPrimitives.Subtract<double>(logits, TensorPrimitives.Max<double>(logits), probabilities);
            TensorPrimitives.Divide<double>(probabilities, temperature, probabilities);
            TensorPrimitives.Exp<double>(probabilities, probabilities);
            TensorPrimitives.Divide<double>(probabilities, TensorPrimitives.Sum<double>(probabilities), probabilities);
            double[] action = [outputs.ActionProbabilities[row, 0], outputs.ActionProbabilities[row, 1]];
            if (action.Any(p => !double.IsFinite(p) || p is < 0 or > 1) || Math.Abs(action.Sum() - 1) > DecisionAnswer.DistributionTolerance)
                throw new DecisionProtocolException("Malformed already-normalized action head.");
            DecisionAnswer answer = q switch
            {
                BinaryDecisionQuestion => new BinaryDecisionAnswer(q.Id, probabilities[1]),
                ChoiceDecisionQuestion c => new ChoiceDecisionAnswer(q.Id, c.Candidates[Array.IndexOf(probabilities, probabilities.Max())].Id,
                    c.Candidates.Select((c, i) => KeyValuePair.Create(c.Id, probabilities[i]))),
                ScoreDecisionQuestion => new ScoreDecisionAnswer(q.Id, probabilities),
                _ => throw new NotSupportedException()
            };
            answer.RawRepresentation = new LayaRawAnswer(temperature, logits, action);
            answers.Add(answer);
        }
        return new(inputs.Input, LayaProfile.ModelId, answers, new() { InputTokenCount = inputs.Attention.Sum() });
    }
}
