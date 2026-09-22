using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace TypeSafe.MEAI;

internal sealed class TypeSafeHttpTransport
{
    private readonly HttpClient _http;
    private readonly TypeSafeClientOptions _options;
    private readonly string _key;

    internal TypeSafeHttpTransport(HttpClient http, TypeSafeClientOptions options, string key)
    {
        _http = http;
        _options = options.Validate();
        ArgumentException.ThrowIfNullOrWhiteSpace(key);
        if (key.Any(char.IsWhiteSpace)) throw new ArgumentException("API key must not contain whitespace.");
        _key = key;
    }

    internal async Task<string> SendAsync(string path, JsonElement? body, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        using var request = new HttpRequestMessage(body is null ? HttpMethod.Get : HttpMethod.Post, new Uri(_options.Endpoint, path));
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _key);
        if (body is not null) request.Content = new StringContent(body.Value.GetRawText(), Encoding.UTF8, "application/json");
        using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
        // Factory-created pipelines buffer inside resilience. This also bounds defensive direct-client use.
        await BoundedResponseHandler.BufferAsync(response, _options.MaximumResponseBytes, cancellationToken).ConfigureAwait(false);
        var text = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
            throw new TypeSafeApiException(response.StatusCode,
                response.Headers.TryGetValues("x-request-id", out var ids) ? ids.FirstOrDefault() : null, text);
        return text;
    }
}
