using System.Collections.ObjectModel;
using System.Text.Json.Serialization;
using Microsoft.Extensions.AI;

namespace DecisionInference;

public sealed class DecisionProtocolException(string message) : Exception(message);

/// <summary>An immutable portable decision payload with mutable, provider-specific extension metadata.</summary>
public abstract class DecisionAnswer
{
    public string QuestionId { get; }
    public DecisionKind Kind { get; }
    /// <summary>Gets or sets provider-specific raw data, excluded from ordinary JSON serialization.</summary>
    /// <remarks>Arbitrary assigned references are not cloned; providers document their snapshot ownership.</remarks>
    [JsonIgnore]
    public object? RawRepresentation { get; set; }
    /// <summary>Gets or sets mutable extension metadata; referenced values are not deep-copied.</summary>
    public AdditionalPropertiesDictionary? AdditionalProperties { get; set; }

    private protected DecisionAnswer(string questionId, DecisionKind kind)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(questionId);
        QuestionId = questionId;
        Kind = kind;
    }

    public const double DistributionTolerance = 1e-6;
    internal static void Unit(double value)
    {
        if (!double.IsFinite(value) || value is < 0 or > 1)
            throw new DecisionProtocolException("Probability must be finite and in [0,1].");
    }
    internal static void Distribution(IEnumerable<double> values)
    {
        double sum = 0;
        foreach (double p in values) { Unit(p); sum += p; }
        if (Math.Abs(sum - 1) > DistributionTolerance)
            throw new DecisionProtocolException("Distribution sum outside absolute tolerance 1e-6.");
    }
}

public sealed class BinaryDecisionAnswer : DecisionAnswer
{
    public double TrueProbability { get; }
    public BinaryDecisionAnswer(string questionId, double trueProbability)
        : base(questionId, DecisionKind.Binary)
    {
        Unit(trueProbability);
        TrueProbability = trueProbability;
    }
}

public sealed class ChoiceDecisionAnswer : DecisionAnswer
{
    public string SelectedCandidateId { get; }
    public IReadOnlyDictionary<string, double> Probabilities { get; }
    public ChoiceDecisionAnswer(string questionId, string selectedCandidateId,
        IEnumerable<KeyValuePair<string, double>> probabilities)
        : base(questionId, DecisionKind.Choice)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(selectedCandidateId);
        ArgumentNullException.ThrowIfNull(probabilities);
        Dictionary<string, double> copy = new(StringComparer.Ordinal);
        foreach (var p in probabilities)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(p.Key);
            if (!copy.TryAdd(p.Key, p.Value)) throw new DecisionProtocolException("Duplicate candidate ID.");
        }
        Distribution(copy.Values);
        if (!copy.ContainsKey(selectedCandidateId)) throw new DecisionProtocolException("Unknown selected candidate.");
        SelectedCandidateId = selectedCandidateId;
        Probabilities = new ReadOnlyDictionary<string, double>(copy);
    }
}

public sealed class ScoreDecisionAnswer : DecisionAnswer
{
    public IReadOnlyList<double> Probabilities { get; }
    public double ExpectedIndex { get; }
    public double? NativeScore { get; }
    public ScoreDecisionAnswer(string questionId, IEnumerable<double> probabilities,
        double? nativeScore = null)
        : base(questionId, DecisionKind.Score)
    {
        ArgumentNullException.ThrowIfNull(probabilities);
        var copy = probabilities.ToArray();
        if (copy.Length < 2) throw new DecisionProtocolException("An ordinal distribution requires at least two levels.");
        Distribution(copy);
        ExpectedIndex = copy.Select((p, i) => i * p).Sum();
        if (nativeScore is { } n && (!double.IsFinite(n) || n < 0 || n > copy.Length - 1 ||
            Math.Abs(n - ExpectedIndex) > DistributionTolerance * (copy.Length - 1)))
            throw new DecisionProtocolException("Native score disagrees with the distribution's expected index.");
        NativeScore = nativeScore;
        Probabilities = Array.AsReadOnly(copy);
    }
}

/// <summary>Validated, ordered answers and copied usage, with mutable raw data and extension properties.</summary>
public sealed class DecisionResponse
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

    public DecisionResponse(DecisionRequest request, string modelId, IEnumerable<DecisionAnswer> answers,
        UsageDetails? usage = null)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        ArgumentNullException.ThrowIfNull(answers);
        var copy = answers.ToArray();
        if (copy.Length != request.Questions.Count || copy.Any(a => a is null) ||
            copy.Select(a => a.QuestionId).Distinct(StringComparer.Ordinal).Count() != copy.Length)
            throw new DecisionProtocolException("Missing, extra, duplicate or null answers.");
        var map = copy.ToDictionary(a => a.QuestionId, StringComparer.Ordinal);
        foreach (var q in request.Questions)
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
            (u.InputTokenCount is { } input && u.OutputTokenCount is { } output &&
                (input > long.MaxValue - output || (u.TotalTokenCount is { } total && total != input + output)))))
            throw new DecisionProtocolException("Invalid observed usage.");
        ModelId = modelId;
        Answers = Array.AsReadOnly(request.Questions.Select(q => map[q.Id]).ToArray());
        if (usage is not null)
        {
            Usage = new();
            Usage.Add(usage);
        }
    }
}
