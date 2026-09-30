namespace DecisionInference.Jeeves;

/// <summary>Owned provider evidence stored at JeevesDecisionGenerator.MetadataKey, not portable confidence.</summary>
public sealed class JeevesResultMetadata
{
    /// <summary>Server-reported inference milliseconds; excludes lock wait, serialization and client round-trip work.</summary>
    public double LatencyMilliseconds { get; }
    /// <summary>Immutable snapshots keyed by exact caller question IDs; null unless explicitly requested.</summary>
    public IReadOnlyDictionary<string, JeevesReasoning>? Reasoning { get; }

    internal JeevesResultMetadata(double latencyMilliseconds, IReadOnlyDictionary<string, JeevesReasoning>? reasoning)
        => (LatencyMilliseconds, Reasoning) = (latencyMilliseconds, reasoning);
}

/// <summary>Owned diagnostic reasoning. Text is never logged by this library; closing tokens are omitted from text.</summary>
public sealed class JeevesReasoning
{
    public bool Thought { get; }
    /// <summary>Whether generation reached the closing token rather than exhausting its cap.</summary>
    public bool Closed { get; }
    /// <summary>Generated chain tokens, including a closing token when present; not the text's retokenized length.</summary>
    public long Tokens { get; }
    public string Text { get; }

    internal JeevesReasoning(bool thought, bool closed, long tokens, string text)
        => (Thought, Closed, Tokens, Text) = (thought, closed, tokens, text);
}
