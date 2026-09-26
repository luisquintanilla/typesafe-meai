using System.Text.Json;

namespace DecisionInference;

public sealed class DecisionRequest
{
    public JsonElement State { get; }
    public IReadOnlyList<DecisionQuestion> Questions { get; }

    public DecisionRequest(JsonElement state, IEnumerable<DecisionQuestion> questions)
    {
        if (state.ValueKind == JsonValueKind.Undefined) throw new ArgumentException("State must be explicit JSON.", nameof(state));
        ArgumentNullException.ThrowIfNull(questions);
        var copy = questions.ToArray();
        if (copy.Length == 0 || copy.Any(q => q is null) ||
            copy.Select(q => q.Id).Distinct(StringComparer.Ordinal).Count() != copy.Length)
            throw new ArgumentException("Questions must be nonempty with unique, nonnull IDs.", nameof(questions));
        State = state.Clone();
        Questions = Array.AsReadOnly(copy);
    }
}

public abstract class DecisionQuestion
{
    public string Id { get; }
    public DecisionKind Kind { get; }
    public string? Instructions { get; }

    private protected DecisionQuestion(string id, DecisionKind kind, string? instructions)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        Id = id;
        Kind = kind;
        Instructions = instructions;
    }
}

public sealed class BinaryDecisionQuestion : DecisionQuestion
{
    public string? TrueDescription { get; }
    public string? FalseDescription { get; }

    public BinaryDecisionQuestion(string id, string? instructions = null,
        string? trueDescription = null, string? falseDescription = null)
        : base(id, DecisionKind.Binary, instructions)
    {
        TrueDescription = trueDescription;
        FalseDescription = falseDescription;
    }
}

public sealed class DecisionCandidate
{
    public string Id { get; }
    public string Description { get; }

    public DecisionCandidate(string id, string description)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        ArgumentNullException.ThrowIfNull(description);
        Id = id;
        Description = description;
    }
}

public sealed class ChoiceDecisionQuestion : DecisionQuestion
{
    public IReadOnlyList<DecisionCandidate> Candidates { get; }

    public ChoiceDecisionQuestion(string id, string? instructions, IEnumerable<DecisionCandidate> candidates)
        : base(id, DecisionKind.Choice, instructions)
    {
        ArgumentNullException.ThrowIfNull(candidates);
        var copy = candidates.ToArray();
        if (copy.Length == 0 || copy.Any(c => c is null) ||
            copy.Select(c => c.Id).Distinct(StringComparer.Ordinal).Count() != copy.Length)
            throw new ArgumentException("Candidates must be nonempty with unique, nonnull IDs.");
        Candidates = Array.AsReadOnly(copy);
    }
}

public sealed class ScoreDecisionQuestion : DecisionQuestion
{
    public IReadOnlyList<string> Rubric { get; }

    public ScoreDecisionQuestion(string id, string? instructions, IEnumerable<string> rubric)
        : base(id, DecisionKind.Score, instructions)
    {
        ArgumentNullException.ThrowIfNull(rubric);
        var copy = rubric.Select(r => r ?? throw new ArgumentException("Rubric descriptions must be explicit.")).ToArray();
        if (copy.Length < 2) throw new ArgumentException("An ordinal rubric requires at least two levels.");
        Rubric = Array.AsReadOnly(copy);
    }
}
