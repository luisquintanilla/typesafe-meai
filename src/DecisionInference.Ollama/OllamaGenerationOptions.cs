namespace DecisionInference.Ollama;

/// <summary>Per-call local model selection and residency, not generation controls or a request timeout.</summary>
public sealed class OllamaGenerationOptions : DecisionGenerationOptions
{
    public OllamaGenerationOptions() { }

    private OllamaGenerationOptions(OllamaGenerationOptions other) : base(other)
        => KeepAliveSeconds = other.KeepAliveSeconds;

    /// <summary>Null omits keep_alive (server setting); zero unloads after the call, negative retains indefinitely, positive retains for that many seconds.</summary>
    /// <remarks>Integer seconds are an intentional subset of Ollama's duration syntax. No daemon is managed by this option.</remarks>
    public int? KeepAliveSeconds { get; set; }

    public override OllamaGenerationOptions Clone() => new(this);
}
