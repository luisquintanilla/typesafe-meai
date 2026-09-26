namespace TypeSafe.MEAI;

internal sealed class BoundedResponseHandler(int limit) : DelegatingHandler
{
    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        var response = await base.SendAsync(request, cancellationToken).ConfigureAwait(false);
        await BufferAsync(response, limit, cancellationToken).ConfigureAwait(false);
        return response;
    }

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
            // Legacy StreamContent subclasses can drop cancellation in LoadIntoBufferAsync.
            // Keep forwarding our token to every stream read instead.
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
