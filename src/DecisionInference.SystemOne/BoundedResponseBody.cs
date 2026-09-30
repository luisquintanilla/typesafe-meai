namespace DecisionInference.SystemOne;

// Adapted from TypeSafe.MEAI/Transport/BoundedResponseHandler.cs at bc2f307.
internal static class BoundedResponseBody
{
    internal static async Task<byte[]> BufferAsync(HttpResponseMessage response, int limit,
        Func<Exception> limitError, CancellationToken cancellationToken)
    {
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (response.Content.Headers.ContentLength > limit) throw limitError();
            if (response.Content is BufferedContent buffered)
            {
                if (buffered.Bytes.Length > limit) throw limitError();
                return buffered.Bytes;
            }
            // StreamContent subclasses can drop cancellation in LoadIntoBufferAsync.
            await using var source = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
            using var buffer = new MemoryStream();
            byte[] chunk = new byte[Math.Min(limit, 81920)];
            int read;
            while ((read = await source.ReadAsync(chunk, cancellationToken).ConfigureAwait(false)) != 0)
            {
                if (buffer.Length + read > limit) throw limitError();
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
            throw new HttpRequestException(HttpRequestError.ResponseEnded, "System One response body read failed.");
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
