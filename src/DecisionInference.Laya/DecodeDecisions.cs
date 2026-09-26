using System.Numerics.Tensors;
using System.Text.Json;

namespace DecisionInference.Laya;

public sealed class DecodeDecisions(LayaProfile profile)
{
    private readonly LayaProfile _profile = profile ?? throw new ArgumentNullException(nameof(profile));

    public DecisionResponse Decode(LayaInputBatch inputs, LayaModelOutputs outputs)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(outputs);
        if (!outputs.Logits.Lengths.SequenceEqual(new nint[] { inputs.BatchSize, inputs.MarkerWidth }) ||
            !outputs.ActionProbabilities.Lengths.SequenceEqual(new nint[] { inputs.BatchSize, 2 }))
            throw new DecisionProtocolException("Laya input/output shapes disagree.");
        var answers = new List<DecisionAnswer>();
        for (int row = 0; row < inputs.BatchSize; row++)
        {
            var q = inputs.Request.Questions[row];
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
            var evidence = JsonSerializer.SerializeToElement(new { provider = "laya", temperature, logits, actionProbabilities = action });
            answers.Add(q switch
            {
                BinaryDecisionQuestion => new BinaryDecisionAnswer(q.Id, probabilities[1], evidence),
                ChoiceDecisionQuestion c => new ChoiceDecisionAnswer(q.Id, c.Candidates[Array.IndexOf(probabilities, probabilities.Max())].Id,
                    c.Candidates.Select((c, i) => KeyValuePair.Create(c.Id, probabilities[i])), evidence),
                ScoreDecisionQuestion => new ScoreDecisionAnswer(q.Id, probabilities, providerEvidence: evidence),
                _ => throw new NotSupportedException()
            });
        }
        return new(inputs.Request, LayaProfile.ModelId, answers, new() { InputTokenCount = inputs.Attention.Sum() });
    }
}
