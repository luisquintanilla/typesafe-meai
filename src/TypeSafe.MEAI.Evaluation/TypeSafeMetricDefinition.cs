using Microsoft.Extensions.AI.Evaluation;

namespace TypeSafe.MEAI.Evaluation;

/// <summary>A named native metric with an optional explicit interpretation policy.</summary>
public sealed record TypeSafeMetricDefinition
{
    /// <summary>Name returned in EvaluationMetricNames and evaluation results.</summary>
    public string Name { get; }
    /// <summary>Immutable native judgment semantics.</summary>
    public TypeSafeQuestion Question { get; }
    /// <summary>Explicit host interpretation. Null does not imply success.</summary>
    public Func<TypeSafeAnswer, EvaluationMetricInterpretation>? Interpret { get; }

    /// <summary>Creates a named metric without inventing ratings or explanations.</summary>
    public TypeSafeMetricDefinition(string name, TypeSafeQuestion question, Func<TypeSafeAnswer, EvaluationMetricInterpretation>? interpret = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        Name = name;
        Question = question ?? throw new ArgumentNullException(nameof(question));
        Interpret = interpret;
    }
}
