using System.Numerics.Tensors;

namespace DecisionInference.Julia;

/// <summary>Owned uncalibrated Julia evidence; no display suppression or action head.</summary>
public sealed class JuliaRawAnswer
{
    public IReadOnlyList<double> Logits { get; }
    internal JuliaRawAnswer(double[] logits) => Logits = Array.AsReadOnly(logits);
}

public sealed class DecodeDecisions
{
    public DecisionResult Decode(JuliaInputBatch inputs, Tensor<float> logits)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(logits);
        ValidateShape(inputs, logits);
        var answers = new List<DecisionAnswer>();
        for (int row = 0; row < inputs.BatchSize; row++)
        {
            var q = inputs.Input.Questions[row];
            double[] values = new double[inputs.OptionCounts[row]];
            double[] probabilities = new double[values.Length];
            Probabilities(logits, row, values, probabilities);
            DecisionAnswer answer = q switch
            {
                BinaryDecisionQuestion => new BinaryDecisionAnswer(q.Id, probabilities[1]),
                ChoiceDecisionQuestion c => new ChoiceDecisionAnswer(q.Id, c.Candidates[Array.IndexOf(probabilities, probabilities.Max())].Id,
                    c.Candidates.Select((c, i) => KeyValuePair.Create(c.Id, probabilities[i]))),
                ScoreDecisionQuestion => new ScoreDecisionAnswer(q.Id, probabilities),
                _ => throw new NotSupportedException()
            };
            answer.RawRepresentation = new JuliaRawAnswer(values);
            answers.Add(answer);
        }
        return new(inputs.Input, JuliaDecisionGenerator.ModelId, answers);
    }

    internal void DecodeToSink(JuliaInputBatch inputs, Tensor<float> logits, IJuliaNumericSink sink)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(logits);
        ArgumentNullException.ThrowIfNull(sink);
        ValidateShape(inputs, logits);
        var offsets = new int[checked(inputs.BatchSize + 1)];
        for (int q = 0; q < inputs.BatchSize; q++) offsets[q + 1] = checked(offsets[q] + inputs.OptionCounts[q]);
        var raw = new double[offsets[^1]];
        var probabilities = new double[raw.Length];
        var selected = new int[inputs.BatchSize];
        var scores = new double[checked(inputs.BatchSize * 3)];
        for (int q = 0; q < inputs.BatchSize; q++)
        {
            int start = offsets[q], length = offsets[q + 1] - start;
            var values = probabilities.AsSpan(start, length);
            Probabilities(logits, q, raw.AsSpan(start, length), values);
            selected[q] = -1;
            switch (inputs.Input.Questions[q])
            {
                case BinaryDecisionQuestion:
                    DecisionObservationValidation.Binary(values[1], DecisionPrecision.HighPrecision);
                    break;
                case ChoiceDecisionQuestion:
                    DecisionObservationValidation.Distribution(values, DecisionPrecision.HighPrecision);
                    int first = 0;
                    for (int i = 1; i < values.Length; i++) if (values[i] > values[first]) first = i;
                    selected[q] = first;
                    break;
                case ScoreDecisionQuestion:
                    var observation = DecisionObservationValidation.Score(values, null, DecisionPrecision.HighPrecision);
                    scores[q * 3] = observation.Expected;
                    scores[q * 3 + 1] = observation.Lower;
                    scores[q * 3 + 2] = observation.Upper;
                    break;
                default: throw new NotSupportedException();
            }
        }
        sink.WriteFrame(inputs.Input, selected, offsets, probabilities, scores);
    }

    private static void ValidateShape(JuliaInputBatch inputs, Tensor<float> logits)
    {
        if (logits.Lengths.Length != 2 || logits.Lengths[0] != inputs.BatchSize || logits.Lengths[1] != inputs.MarkerWidth)
            throw new DecisionProtocolException("Julia input/output shapes disagree.");
    }

    private static void Probabilities(Tensor<float> logits, int row, Span<double> raw, Span<double> probabilities)
    {
        for (int i = 0; i < raw.Length; i++)
        {
            raw[i] = logits[row, i];
            if (!double.IsFinite(raw[i])) throw new DecisionProtocolException("Nonfinite Julia valid-option logit.");
        }
        TensorPrimitives.Subtract<double>(raw, TensorPrimitives.Max<double>(raw), probabilities);
        TensorPrimitives.Exp<double>(probabilities, probabilities);
        TensorPrimitives.Divide<double>(probabilities, TensorPrimitives.Sum<double>(probabilities), probabilities);
    }
}
