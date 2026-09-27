using DecisionInference.SystemOne;

namespace DecisionInference.TypeSafe;

public sealed record TypeSafeDecisionOptions
{
    public Uri Endpoint { get; init; } = new("https://api.typesafe.ai");
    public string ModelId { get; init; } = "jev-1.13.0";
    public int MaximumResponseBytes { get; init; } = 4 * 1024 * 1024;
    internal void Validate()
    {
        SystemOneTransport.ValidateEndpoint(Endpoint, allowLoopbackHttp: false);
        ArgumentException.ThrowIfNullOrWhiteSpace(ModelId);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(MaximumResponseBytes);
    }
}

/// <summary>Thin System One transport. The host owns redirects, resilience and a deadline covering body reads.</summary>
public sealed class TypeSafeDecisionClient : IDecisionClient
{
    private readonly SystemOneTransport _transport;
    private readonly TypeSafeDecisionOptions _options;
    private readonly DecisionClientMetadata _metadata;

    public TypeSafeDecisionClient(HttpClient httpClient, string apiKey,
        TypeSafeDecisionOptions? options = null, bool ownsHttpClient = false)
    {
        ArgumentNullException.ThrowIfNull(httpClient);
        ArgumentException.ThrowIfNullOrWhiteSpace(apiKey);
        if (apiKey.Any(char.IsWhiteSpace)) throw new ArgumentException("API key must not contain whitespace.");
        _options = options ?? new();
        _options.Validate();
        _transport = new(httpClient, _options.Endpoint, apiKey, _options.MaximumResponseBytes, ownsHttpClient,
            (status, requestId, body) => new TypeSafeApiException(status, requestId, body), () => new TypeSafeBodyLimitException());
        _metadata = new("typesafe", _options.Endpoint, _options.ModelId);
    }

    public async Task<DecisionResponse> GetResponseAsync(DecisionRequest request, DecisionOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        _transport.ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();
        ArgumentNullException.ThrowIfNull(request);
        var model = options?.ModelId ?? _options.ModelId;
        ArgumentException.ThrowIfNullOrWhiteSpace(model);
        var wire = SystemOneProtocol.Encode(request, model);
        var response = await _transport.PostAsync("/v1/systemone", wire.Body, cancellationToken).ConfigureAwait(false);
        var result = SystemOneProtocol.Decode(wire, response.Body, DecisionPrecision.HighPrecision);
        if (response.RequestId is not null) result.AdditionalProperties!["systemone.requestId"] = response.RequestId;
        return result;
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null) return null;
        return serviceType == typeof(DecisionClientMetadata) ? _metadata : serviceType.IsInstanceOfType(this) ? this : null;
    }

    public void Dispose() => _transport.Dispose();
}
