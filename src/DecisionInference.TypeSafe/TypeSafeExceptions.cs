namespace DecisionInference.TypeSafe;

public sealed class TypeSafeApiException : HttpRequestException
{
    public string? RequestId { get; }
    /// <summary>Opt-in diagnostic data; may contain sensitive information. Never log automatically.</summary>
    public string DiagnosticBody { get; }
    internal TypeSafeApiException(System.Net.HttpStatusCode status, string? requestId, string body)
        : base($"TypeSafe HTTP operation failed with status {(int)status}.", null, status)
        => (RequestId, DiagnosticBody) = (requestId, body);
}

public sealed class TypeSafeBodyLimitException()
    : Exception("TypeSafe response exceeded the configured buffering limit.");
