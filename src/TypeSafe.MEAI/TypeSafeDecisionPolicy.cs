namespace TypeSafe.MEAI;

/// <summary>Explicit client-level acceptance policy, inherited by Boolean/enum/Score fields and tool selection.
/// These thresholds are host policy, not model calibration claims.</summary>
public sealed record TypeSafeDecisionPolicy
{
    /// <summary>Inclusive false/true bounds with an open abstention interval.</summary>
    public TypeSafeBooleanPolicy Boolean { get; }
    /// <summary>Inclusive server confidence threshold for Choice, Score and tool selection; zero explicitly accepts all valid confidences.</summary>
    public double MinimumConfidence { get; }
    /// <summary>Defines explicit defaults once; fields may override them.</summary>
    public TypeSafeDecisionPolicy(TypeSafeBooleanPolicy boolean, double minimumConfidence)
    {
        Boolean = boolean ?? throw new ArgumentNullException(nameof(boolean));
        if (!double.IsFinite(minimumConfidence) || minimumConfidence is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(minimumConfidence));
        MinimumConfidence = minimumConfidence;
    }
}
