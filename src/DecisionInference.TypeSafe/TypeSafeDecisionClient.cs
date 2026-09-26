using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace DecisionInference.TypeSafe;

public sealed record TypeSafeDecisionOptions
{
    public Uri Endpoint { get; init; } = new("https://api.typesafe.ai");
    public string ModelId { get; init; } = "jev-1.13.0";
    public int MaximumResponseBytes { get; init; } = 4 * 1024 * 1024;
    internal void Validate()
    {
        if (!Endpoint.IsAbsoluteUri || Endpoint.Scheme != Uri.UriSchemeHttps || Endpoint.AbsolutePath != "/" ||
            Endpoint.Query.Length != 0 || Endpoint.Fragment.Length != 0 || Endpoint.UserInfo.Length != 0)
            throw new ArgumentException("Endpoint must be an HTTPS origin without path, credentials, query, or fragment.");
        ArgumentException.ThrowIfNullOrWhiteSpace(ModelId);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(MaximumResponseBytes);
    }
}

/// <summary>Thin System One transport. The host owns redirects, resilience and a deadline covering body reads.</summary>
public sealed class TypeSafeDecisionClient : IDecisionClient
{
    private readonly HttpClient _http;
    private readonly TypeSafeDecisionOptions _options;
    private readonly string _apiKey;
    private readonly bool _ownsHttp;
    private readonly DecisionClientMetadata _metadata;
    private int _disposed;

    public TypeSafeDecisionClient(HttpClient httpClient, string apiKey,
        TypeSafeDecisionOptions? options = null, bool ownsHttpClient = false)
    {
        ArgumentNullException.ThrowIfNull(httpClient);
        ArgumentException.ThrowIfNullOrWhiteSpace(apiKey);
        if (apiKey.Any(char.IsWhiteSpace)) throw new ArgumentException("API key must not contain whitespace.");
        _options = options ?? new();
        _options.Validate();
        _http = httpClient;
        _apiKey = apiKey;
        _ownsHttp = ownsHttpClient;
        _metadata = new("typesafe", _options.ModelId, DecisionKind.Binary, DecisionKind.Choice, DecisionKind.Score);
    }

    public async Task<DecisionResponse> GetResponseAsync(DecisionRequest request, DecisionOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        cancellationToken.ThrowIfCancellationRequested();
        _metadata.Validate(request, options);
        var body = SystemOneProtocol.Encode(request, options?.ModelId ?? _options.ModelId);
        using var message = new HttpRequestMessage(HttpMethod.Post, new Uri(_options.Endpoint, "/v1/systemone"));
        message.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);
        message.Content = new StringContent(body.GetRawText(), Encoding.UTF8, "application/json");
        using var response = await _http.SendAsync(message, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
        var bytes = await BoundedResponseBody.BufferAsync(response, _options.MaximumResponseBytes, cancellationToken).ConfigureAwait(false);
        if (!response.IsSuccessStatusCode)
            throw new TypeSafeApiException(response.StatusCode,
                response.Headers.TryGetValues("x-request-id", out var ids) ? ids.FirstOrDefault() : null,
                Encoding.UTF8.GetString(bytes));
        string text;
        try { text = new UTF8Encoding(false, true).GetString(bytes); }
        catch (DecoderFallbackException) { throw new DecisionProtocolException("Response is not valid UTF-8."); }
        cancellationToken.ThrowIfCancellationRequested();
        return SystemOneProtocol.Decode(request, text);
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null) return null;
        return serviceType == typeof(DecisionClientMetadata) ? _metadata : serviceType.IsInstanceOfType(this) ? this : null;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0 && _ownsHttp) _http.Dispose();
    }
}
