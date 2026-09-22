using System.Net;

namespace TypeSafe.MEAI;

internal sealed class BoundedResponseHandler(int limit) : DelegatingHandler
{
    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        var response = await base.SendAsync(request, cancellationToken).ConfigureAwait(false);
        await BufferAsync(response, limit, cancellationToken).ConfigureAwait(false);
        return response;
    }

    internal static async Task BufferAsync(HttpResponseMessage response, int limit, CancellationToken cancellationToken)
    {
        try
        {
            if (response.Content.Headers.ContentLength > limit) throw new TypeSafeBodyLimitException();
            await using var source = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
            using var buffer = new MemoryStream();
            byte[] chunk = new byte[Math.Min(limit, 81920)];
            int read;
            while ((read = await source.ReadAsync(chunk, cancellationToken).ConfigureAwait(false)) != 0)
            {
                if (buffer.Length + read > limit) throw new TypeSafeBodyLimitException();
                await buffer.WriteAsync(chunk.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
            }
            var replacement = new ByteArrayContent(buffer.ToArray());
            foreach (var header in response.Content.Headers) replacement.Headers.TryAddWithoutValidation(header.Key, header.Value);
            response.Content.Dispose();
            response.Content = replacement;
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
}
