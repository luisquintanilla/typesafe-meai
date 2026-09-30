using System.Collections.ObjectModel;
using System.Text.Json.Serialization;
using Microsoft.Extensions.AI;

namespace DecisionInference;

public sealed class DecisionProtocolException(string message) : Exception(message);

/// <summary>Numerical representation of reported probabilities, not model calibration or accuracy.</summary>
public enum DecisionPrecision
{
    HighPrecision,
    /// <summary>Nearest four-decimal observations; interval endpoints conservatively include rounding ties.</summary>
    FourDecimalPlaces,
    /// <summary>Nearest two-decimal observations; interval endpoints conservatively include rounding ties.</summary>
    TwoDecimalPlaces
}

/// <summary>An immutable portable decision payload with mutable, provider-specific extension metadata.</summary>
public abstract class DecisionAnswer
{
    public string QuestionId { get; }
    public DecisionKind Kind { get; }
    public DecisionPrecision Precision { get; }
    /// <summary>Gets or sets provider-specific raw data, excluded from ordinary JSON serialization.</summary>
    /// <remarks>Arbitrary assigned references are not cloned; providers document their snapshot ownership.</remarks>
    [JsonIgnore]
    public object? RawRepresentation { get; set; }
    /// <summary>Gets or sets mutable extension metadata; referenced values are not deep-copied.</summary>
    public AdditionalPropertiesDictionary? AdditionalProperties { get; set; }

    private protected DecisionAnswer(string questionId, DecisionKind kind, DecisionPrecision precision)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(questionId);
        if (!Enum.IsDefined(precision)) throw new ArgumentOutOfRangeException(nameof(precision));
        QuestionId = questionId;
        Kind = kind;
        Precision = precision;
    }

    public const double DistributionTolerance = 1e-6;
}

public sealed class BinaryDecisionAnswer : DecisionAnswer
{
    public double TrueProbability { get; }
    public BinaryDecisionAnswer(string questionId, double trueProbability, DecisionPrecision precision = DecisionPrecision.HighPrecision)
        : base(questionId, DecisionKind.Binary, precision)
    {
        DecisionObservationValidation.Binary(trueProbability, precision);
        TrueProbability = trueProbability;
    }
}

public sealed class ChoiceDecisionAnswer : DecisionAnswer
{
    public string SelectedCandidateId { get; }
    public IReadOnlyDictionary<string, double> Probabilities { get; }
    public ChoiceDecisionAnswer(string questionId, string selectedCandidateId,
        IEnumerable<KeyValuePair<string, double>> probabilities, DecisionPrecision precision = DecisionPrecision.HighPrecision)
        : base(questionId, DecisionKind.Choice, precision)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(selectedCandidateId);
        ArgumentNullException.ThrowIfNull(probabilities);
        Dictionary<string, double> copy = new(StringComparer.Ordinal);
        foreach (var p in probabilities)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(p.Key);
            if (!copy.TryAdd(p.Key, p.Value)) throw new DecisionProtocolException("Duplicate candidate ID.");
        }
        DecisionObservationValidation.Distribution(copy.Values.ToArray(), precision);
        if (!copy.ContainsKey(selectedCandidateId)) throw new DecisionProtocolException("Unknown selected candidate.");
        SelectedCandidateId = selectedCandidateId;
        Probabilities = new ReadOnlyDictionary<string, double>(copy);
    }
}

public sealed class ScoreDecisionAnswer : DecisionAnswer
{
    public IReadOnlyList<double> Probabilities { get; }
    /// <summary>Sum of index times observed probability. For rounded observations this is only an estimate and may exceed the rubric range.</summary>
    public double ExpectedIndex { get; }
    /// <summary>Lower feasible latent expectation for rounded observations; high precision uses the retained validation tolerance.</summary>
    public double ExpectedIndexLowerBound { get; }
    /// <summary>Upper feasible latent expectation, not a confidence interval or accuracy guarantee.</summary>
    public double ExpectedIndexUpperBound { get; }
    public double? NativeScore { get; }
    public ScoreDecisionAnswer(string questionId, IEnumerable<double> probabilities,
        double? nativeScore = null, DecisionPrecision precision = DecisionPrecision.HighPrecision)
        : base(questionId, DecisionKind.Score, precision)
    {
        ArgumentNullException.ThrowIfNull(probabilities);
        var copy = probabilities.ToArray();
        var observation = DecisionObservationValidation.Score(copy, nativeScore, precision);
        ExpectedIndex = observation.Expected;
        ExpectedIndexLowerBound = observation.Lower;
        ExpectedIndexUpperBound = observation.Upper;
        NativeScore = nativeScore;
        Probabilities = Array.AsReadOnly(copy);
    }
}

internal static class DecisionRounding
{
    internal static (decimal Lower, decimal Upper) Interval(double value, int maximum, DecisionPrecision precision)
    {
        if (!double.IsFinite(value) || value < 0 || value > maximum)
            throw new DecisionProtocolException("Rounded value must be finite and inside its domain.");
        var (digits, halfUnit) = precision switch
        {
            DecisionPrecision.TwoDecimalPlaces => (2, 0.005m),
            DecisionPrecision.FourDecimalPlaces => (4, 0.00005m),
            _ => throw new ArgumentOutOfRangeException(nameof(precision))
        };
        decimal observed = decimal.Round((decimal)value, digits, MidpointRounding.ToEven);
        // A decimal-grid JSON number has one canonical nearest-double representation.
        if (value != (double)observed)
            throw new DecisionProtocolException(precision == DecisionPrecision.TwoDecimalPlaces
                ? "Value does not lie on the declared two-decimal grid."
                : "Value does not lie on the declared four-decimal grid.");
        return (Math.Max(0m, observed - halfUnit), Math.Min(maximum, observed + halfUnit));
    }

    internal static (decimal Lower, decimal Upper) ExpectedBounds(ReadOnlySpan<double> values, DecisionPrecision precision)
    {
        Span<(decimal Lower, decimal Upper)> intervals = values.Length <= 64
            ? stackalloc (decimal, decimal)[values.Length] : new (decimal, decimal)[values.Length];
        decimal baseMass = 0, baseIndex = 0;
        for (int i = 0; i < values.Length; i++)
        {
            var interval = Interval(values[i], 1, precision);
            intervals[i] = interval;
            baseMass += interval.Lower;
            baseIndex += i * interval.Lower;
        }
        return (Extremum(intervals, baseMass, baseIndex, false), Extremum(intervals, baseMass, baseIndex, true));

        static decimal Extremum(ReadOnlySpan<(decimal Lower, decimal Upper)> intervals, decimal baseMass, decimal baseIndex, bool descending)
        {
            decimal remaining = 1 - baseMass, result = baseIndex;
            for (int step = 0; step < intervals.Length && remaining > 0; step++)
            {
                int i = descending ? intervals.Length - 1 - step : step;
                var interval = intervals[i];
                decimal take = Math.Min(remaining, interval.Upper - interval.Lower);
                result += i * take;
                remaining -= take;
            }
            return result;
        }
    }
}

/// <summary>Validated, ordered answers and copied usage, with mutable raw data and extension properties.</summary>
public sealed class DecisionResult
{
    public string ModelId { get; }
    public IReadOnlyList<DecisionAnswer> Answers { get; }
    public UsageDetails? Usage { get; }
    /// <summary>Gets or sets provider-specific raw data, excluded from ordinary JSON serialization.</summary>
    /// <remarks>Arbitrary assigned references are not cloned; providers document their snapshot ownership.</remarks>
    [JsonIgnore]
    public object? RawRepresentation { get; set; }
    /// <summary>Gets or sets mutable extension metadata; referenced values are not deep-copied.</summary>
    public AdditionalPropertiesDictionary? AdditionalProperties { get; set; }

    public DecisionResult(DecisionInput input, string modelId, IEnumerable<DecisionAnswer> answers,
        UsageDetails? usage = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        ArgumentNullException.ThrowIfNull(answers);
        var copy = answers.ToArray();
        if (copy.Length != input.Questions.Count || copy.Any(a => a is null) ||
            copy.Select(a => a.QuestionId).Distinct(StringComparer.Ordinal).Count() != copy.Length)
            throw new DecisionProtocolException("Missing, extra, duplicate or null answers.");
        var map = copy.ToDictionary(a => a.QuestionId, StringComparer.Ordinal);
        foreach (var q in input.Questions)
        {
            if (!map.TryGetValue(q.Id, out var answer) || answer.Kind != q.Kind)
                throw new DecisionProtocolException("Answer ID or kind mismatch.");
            if (q is ChoiceDecisionQuestion c && answer is ChoiceDecisionAnswer a &&
                !a.Probabilities.Keys.ToHashSet(StringComparer.Ordinal).SetEquals(c.Candidates.Select(c => c.Id)))
                throw new DecisionProtocolException("Incomplete or unexpected candidate distribution.");
            if (q is ScoreDecisionQuestion s && answer is ScoreDecisionAnswer b && b.Probabilities.Count != s.Rubric.Count)
                throw new DecisionProtocolException("Incomplete or unexpected rubric distribution.");
        }
        if (usage is { } u && (u.InputTokenCount < 0 || u.OutputTokenCount < 0 || u.TotalTokenCount < 0 ||
            (u.InputTokenCount is { } inputTokens && u.OutputTokenCount is { } output &&
                (inputTokens > long.MaxValue - output || (u.TotalTokenCount is { } total && total != inputTokens + output)))))
            throw new DecisionProtocolException("Invalid observed usage.");
        ModelId = modelId;
        Answers = Array.AsReadOnly(input.Questions.Select(q => map[q.Id]).ToArray());
        if (usage is not null)
        {
            Usage = new();
            Usage.Add(usage);
        }
    }
}
