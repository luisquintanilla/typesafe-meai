using System.Net;

namespace TypeSafe.MEAI;

/// <summary>An HTTP failure with opt-in bounded diagnostics, never response text in its message.</summary>
public sealed class TypeSafeApiException : HttpRequestException
{
    /// <summary>Server request ID, if supplied.</summary>
    public string? RequestId { get; }
    /// <summary>Explicitly accessed diagnostic body. May contain sensitive data.</summary>
    public string DiagnosticBody { get; }
    internal TypeSafeApiException(HttpStatusCode status, string? requestId, string body)
        : base($"TypeSafe HTTP operation failed with status {(int)status}.", null, status)
    {
        RequestId = requestId;
        DiagnosticBody = body;
    }
}

/// <summary>A configured body limit was exceeded; this is not a transient HTTP failure.</summary>
public sealed class TypeSafeBodyLimitException : Exception
{
    /// <summary>Creates a body-size failure.</summary>
    public TypeSafeBodyLimitException() : base("TypeSafe response exceeded the configured buffering limit.") { }
}
