namespace DecisionInference.OllamaSharp;

/// <summary>Per-call model selection and residency for the native SDK operation.</summary>
public sealed class OllamaSharpGenerationOptions : DecisionGenerationOptions
{
    public OllamaSharpGenerationOptions() { }

    private OllamaSharpGenerationOptions(OllamaSharpGenerationOptions other) : base(other)
        => KeepAliveSeconds = other.KeepAliveSeconds;

    /// <summary>Null omits residency; zero unloads after the call, negative retains, positive retains for that many seconds.</summary>
    /// <remarks>Sent as an invariant signed duration string with an "s" suffix. Not a request deadline or model-management operation.</remarks>
    public int? KeepAliveSeconds { get; set; }

    public override OllamaSharpGenerationOptions Clone() => new(this);
}
