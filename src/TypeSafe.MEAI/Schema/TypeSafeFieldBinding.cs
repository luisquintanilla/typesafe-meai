using System.Collections.ObjectModel;
using System.Text.Json;

namespace TypeSafe.MEAI;

/// <summary>Explicit inclusive Boolean bounds with an open abstention interval.</summary>
public sealed record TypeSafeBooleanPolicy
{
    /// <summary>Accept false at or below this bound.</summary>
    public double FalseMaximum { get; }
    /// <summary>Accept true at or above this bound.</summary>
    public double TrueMinimum { get; }
    /// <summary>Requires 0 &lt;= falseMaximum &lt; trueMinimum &lt;= 1.</summary>
    public TypeSafeBooleanPolicy(double falseMaximum, double trueMinimum)
    {
        if (!double.IsFinite(falseMaximum) || !double.IsFinite(trueMinimum) ||
            falseMaximum < 0 || falseMaximum >= trueMinimum || trueMinimum > 1)
            throw new ArgumentException("Boolean policy requires 0 <= falseMaximum < trueMinimum <= 1.");
        FalseMaximum = falseMaximum;
        TrueMinimum = trueMinimum;
    }
}

/// <summary>Immutable task semantics and explicit per-leaf acceptance policy.</summary>
public sealed class TypeSafeFieldBinding
{
    /// <summary>Judgment instructions; a nonempty schema description is the documented fallback.</summary>
    public string? Instructions { get; }
    /// <summary>Required for Boolean leaves.</summary>
    public TypeSafeBooleanPolicy? BooleanPolicy { get; }
    /// <summary>Required server-confidence threshold for Choice and Score, including explicit zero to accept all.</summary>
    public double? MinimumConfidence { get; }
    /// <summary>Exact enum-label descriptions for Choice.</summary>
    public IReadOnlyDictionary<string, JsonElement>? Choices { get; }
    /// <summary>Ordered Score levels; the output is a fractional index, not an extracted quantity.</summary>
    public IReadOnlyList<JsonElement>? Rubric { get; }
    /// <summary>Optional deterministic validator, returning a rejection reason or null.</summary>
    public Func<JsonElement, string?>? Validate { get; }

    /// <summary>Snapshots semantics and policies for one escaped JSON Pointer leaf.</summary>
    public TypeSafeFieldBinding(string? instructions = null, TypeSafeBooleanPolicy? booleanPolicy = null,
        double? minimumConfidence = null, IEnumerable<KeyValuePair<string, JsonElement>>? choices = null,
        IEnumerable<JsonElement>? rubric = null, Func<JsonElement, string?>? validate = null)
    {
        if (minimumConfidence is { } c && (!double.IsFinite(c) || c < 0 || c > 1))
            throw new ArgumentException("Confidence threshold must be finite in [0,1].");
        Instructions = instructions;
        BooleanPolicy = booleanPolicy;
        MinimumConfidence = minimumConfidence;
        Choices = choices is null ? null : new ReadOnlyDictionary<string, JsonElement>(choices.ToDictionary(p => p.Key, p => p.Value.Clone(), StringComparer.Ordinal));
        Rubric = rubric is null ? null : Array.AsReadOnly(rubric.Select(v => v.Clone()).ToArray());
        Validate = validate;
    }
}

/// <summary>A rejected field or whole-object policy result.</summary>
/// <param name="Pointer">Escaped JSON Pointer; empty means root/application validation.</param>
/// <param name="Reason">Explicit reason, never a fabricated model explanation.</param>
public sealed record TypeSafeRejection(string Pointer, string Reason);

/// <summary>An unsupported or invalid schema at a specific path.</summary>
public sealed class TypeSafeSchemaException : ArgumentException
{
    /// <summary>Actionable escaped JSON Pointer into the schema.</summary>
    public string Pointer { get; }
    /// <summary>Creates a path-specific schema failure.</summary>
    public TypeSafeSchemaException(string pointer, string reason) : base($"Schema '{pointer}': {reason}") => Pointer = pointer;
}
