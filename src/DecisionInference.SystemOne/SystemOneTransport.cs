using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace DecisionInference.SystemOne;

// Shared native HTTP mechanics, not an inference pipeline or public provider framework.
internal sealed class SystemOneTransport : IDisposable
{
    private readonly HttpClient _http;
    private readonly Uri _endpoint;
    private readonly string? _apiKey;
    private readonly int _maximumBytes;
    private readonly bool _ownsHttp;
    private readonly Func<HttpStatusCode, string?, string, Exception> _httpError;
    private readonly Func<Exception> _bodyLimit;
    private int _disposed;

    internal SystemOneTransport(HttpClient http, Uri endpoint, string? apiKey, int maximumBytes, bool ownsHttp,
        Func<HttpStatusCode, string?, string, Exception> httpError, Func<Exception> bodyLimit)
        => (_http, _endpoint, _apiKey, _maximumBytes, _ownsHttp, _httpError, _bodyLimit) =
            (http, endpoint, apiKey, maximumBytes, ownsHttp, httpError, bodyLimit);

    internal void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    internal async Task<(JsonElement Body, string? RequestId)> PostAsync(string path, JsonElement body,
        CancellationToken cancellationToken, int? maximumRequestBytes = null)
    {
        ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();
        if (_apiKey is null && _http.DefaultRequestHeaders.Authorization is not null)
            throw new InvalidOperationException("Use an explicit provider API key rather than ambient HttpClient authorization.");
        string json = body.GetRawText();
        if (maximumRequestBytes is { } limit && Encoding.UTF8.GetByteCount(json) > limit)
            throw new ArgumentException("System One request exceeds the server's body limit.");
        var expectedUri = new Uri(_endpoint, path);
        using var message = new HttpRequestMessage(HttpMethod.Post, expectedUri);
        if (_apiKey is not null) message.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);
        message.Content = new StringContent(json, Encoding.UTF8, "application/json");
        using var response = await _http.SendAsync(message, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
        var bytes = await BoundedResponseBody.BufferAsync(response, _maximumBytes, _bodyLimit, cancellationToken).ConfigureAwait(false);
        string? requestId = response.Headers.TryGetValues("x-request-id", out var ids) ? ids.FirstOrDefault() :
            response.Headers.TryGetValues("x-typesafe-request-id", out ids) ? ids.FirstOrDefault() : null;
        if (!response.IsSuccessStatusCode)
            throw _httpError(response.StatusCode, requestId, Encoding.UTF8.GetString(bytes));
        if (response.RequestMessage?.RequestUri is { } actual && actual != expectedUri)
            throw new DecisionProtocolException("Automatic redirects are not supported; configure HttpClient to disable them.");
        string text;
        try { text = new UTF8Encoding(false, true).GetString(bytes); }
        catch (DecoderFallbackException) { throw new DecisionProtocolException("Response is not valid UTF-8."); }
        cancellationToken.ThrowIfCancellationRequested();
        try
        {
            using var document = JsonDocument.Parse(text);
            SystemOneProtocol.UniqueProperties(document.RootElement);
            return (document.RootElement.Clone(), requestId);
        }
        catch (JsonException) { throw new DecisionProtocolException("Malformed native response JSON."); }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0 && _ownsHttp) _http.Dispose();
    }

    internal static void ValidateEndpoint(Uri endpoint, bool allowLoopbackHttp)
    {
        ArgumentNullException.ThrowIfNull(endpoint);
        bool loopback = endpoint.IsAbsoluteUri &&
            IPAddress.TryParse(endpoint.IdnHost.Trim('[', ']'), out var address) && IPAddress.IsLoopback(address);
        if (!endpoint.IsAbsoluteUri ||
            (endpoint.Scheme != Uri.UriSchemeHttps && !(allowLoopbackHttp && loopback && endpoint.Scheme == Uri.UriSchemeHttp)) ||
            endpoint.AbsolutePath != "/" || endpoint.Query.Length != 0 || endpoint.Fragment.Length != 0 || endpoint.UserInfo.Length != 0)
            throw new ArgumentException("Endpoint must be an HTTPS origin or explicitly allowed literal-loopback HTTP origin, without credentials, path, query or fragment.");
    }
}
