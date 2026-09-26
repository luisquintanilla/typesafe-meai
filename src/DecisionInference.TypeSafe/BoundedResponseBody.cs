namespace DecisionInference.TypeSafe;

// Adapted from TypeSafe.MEAI/Transport/BoundedResponseHandler.cs at bc2f307.
internal static class BoundedResponseBody
{
    internal static async Task<byte[]> BufferAsync(HttpResponseMessage response, int limit, CancellationToken cancellationToken)
    {
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (response.Content.Headers.ContentLength > limit) throw new TypeSafeBodyLimitException();
            if (response.Content is BufferedContent buffered)
            {
                if (buffered.Bytes.Length > limit) throw new TypeSafeBodyLimitException();
                return buffered.Bytes;
            }
            // StreamContent subclasses can drop cancellation in LoadIntoBufferAsync.
            await using var source = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
            using var buffer = new MemoryStream();
            byte[] chunk = new byte[Math.Min(limit, 81920)];
            int read;
            while ((read = await source.ReadAsync(chunk, cancellationToken).ConfigureAwait(false)) != 0)
            {
                if (buffer.Length + read > limit) throw new TypeSafeBodyLimitException();
                await buffer.WriteAsync(chunk.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
            }
            cancellationToken.ThrowIfCancellationRequested();
            var replacement = new BufferedContent(buffer.ToArray());
            foreach (var header in response.Content.Headers) replacement.Headers.TryAddWithoutValidation(header.Key, header.Value);
            response.Content.Dispose();
            response.Content = replacement;
            return replacement.Bytes;
        }
        catch (IOException)
        {
            response.Dispose();
            throw new HttpRequestException(HttpRequestError.ResponseEnded, "TypeSafe response body read failed.");
        }
        catch
        {
            response.Dispose();
            throw;
        }
    }

    private sealed class BufferedContent(byte[] bytes) : ByteArrayContent(bytes)
    {
        internal byte[] Bytes { get; } = bytes;
    }
}

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
