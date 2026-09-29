namespace DecisionInference.Jeeves;

/// <summary>An explicit snapshot of all four pinned Jeeves reasoning options, not a partial override.</summary>
/// <remarks>
/// Null or plain DecisionGenerationOptions omit the options object and use server defaults.
/// This type sends every field, including a null threshold to clear a server default.
/// ModelId remains only a request/echo label. AdditionalProperties input keys are unsupported.
/// </remarks>
public sealed class JeevesGenerationOptions : DecisionGenerationOptions
{
    public JeevesGenerationOptions() { }

    private JeevesGenerationOptions(JeevesGenerationOptions other) : base(other)
        => (Think, MaxThink, NoThinkThreshold, ReturnReasoning) =
            (other.Think, other.MaxThink, other.NoThinkThreshold, other.ReturnReasoning);

    public bool Think { get; set; } = true;
    /// <summary>Maximum generated reasoning tokens per question; context limits can reduce this cap.</summary>
    public int MaxThink { get; set; } = 2560;
    /// <summary>Skip thinking when the maximum no-think probability reaches this threshold, not the returned confidence.</summary>
    public double? NoThinkThreshold { get; set; }
    /// <summary>Opt in to sensitive diagnostic text; it is not a guaranteed faithful explanation.</summary>
    public bool ReturnReasoning { get; set; }

    public override JeevesGenerationOptions Clone() => new(this);
}
