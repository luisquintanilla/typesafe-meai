using System.Text;
using System.Text.Json;
using DecisionInference.SystemOne;

namespace DecisionInference.Ollaya;

/// <summary>Connection settings, not daemon lifecycle or model-management options.</summary>
public sealed record OllayaDecisionOptions
{
    public Uri Endpoint { get; init; } = new("http://127.0.0.1:11435");
    public string? ModelId { get; init; }
    public int MaximumResponseBytes { get; init; } = 4 * 1024 * 1024;
}

/// <summary>Ollaya's nontruncating System One endpoint, with per-call concrete-model discovery.</summary>
/// <remarks>
/// The caller owns redirects, cookies, resilience and deadlines on the injected HttpClient.
/// Disable automatic redirects. No model is pulled or loaded by management calls here.
/// Cancellation stops awaiting HTTP work; it does not promise interruption of server inference.
/// </remarks>
public sealed class OllayaDecisionClient : IDecisionClient
{
    private const int MaximumRequestBytes = 8 * 1024 * 1024;
    private readonly SystemOneTransport _transport;
    private readonly DecisionClientMetadata _metadata;

    public OllayaDecisionClient(HttpClient httpClient, OllayaDecisionOptions? options = null,
        string? apiKey = null, bool ownsHttpClient = false)
    {
        ArgumentNullException.ThrowIfNull(httpClient);
        options ??= new();
        SystemOneTransport.ValidateEndpoint(options.Endpoint, allowLoopbackHttp: true);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.MaximumResponseBytes);
        if (options.ModelId is not null) ArgumentException.ThrowIfNullOrWhiteSpace(options.ModelId);
        if (apiKey is not null && (string.IsNullOrWhiteSpace(apiKey) || apiKey.Any(char.IsWhiteSpace)))
            throw new ArgumentException("An explicit API key must be nonempty and contain no whitespace.", nameof(apiKey));
        if (httpClient.DefaultRequestHeaders.Authorization is not null)
            throw new ArgumentException("Ollaya requires isolated HttpClient authorization; supply its key explicitly.", nameof(httpClient));
        _metadata = new("ollaya", options.Endpoint, options.ModelId);
        _transport = new(httpClient, options.Endpoint, apiKey, options.MaximumResponseBytes, ownsHttpClient,
            (status, requestId, body) => new OllayaApiException(status, requestId, body), () => new OllayaBodyLimitException());
    }

    public async Task<DecisionResponse> GetResponseAsync(DecisionRequest request, DecisionOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        _transport.ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();
        ArgumentNullException.ThrowIfNull(request);
        string? model = options?.ModelId ?? _metadata.DefaultModelId;
        ArgumentException.ThrowIfNullOrWhiteSpace(model);
        if (request.State.ValueKind is not (JsonValueKind.String or JsonValueKind.Object or JsonValueKind.Array))
            throw new ArgumentException("Ollaya state must be a JSON string, object or array.", nameof(request));
        if (request.Questions.Count > 256) throw new ArgumentException("Ollaya supports at most 256 questions.", nameof(request));
        if (request.Questions.OfType<ChoiceDecisionQuestion>().Any(q => q.Candidates.Count < 2))
            throw new ArgumentException("Ollaya Choice requires at least two candidates.", nameof(request));
        var wire = SystemOneProtocol.Encode(request, model);
        if (Encoding.UTF8.GetByteCount(wire.Body.GetRawText()) > MaximumRequestBytes)
            throw new ArgumentException("Ollaya request exceeds its 8 MiB body limit.", nameof(request));
        var discovery = await DiscoverAsync(model, cancellationToken).ConfigureAwait(false);
        var profile = OllayaProfile.Read(discovery);
        profile.Validate(request);
        wire = SystemOneProtocol.Encode(profile.Prepare(request), model) with { Request = request };
        var response = await _transport.PostAsync("/v1/systemone", wire.Body, cancellationToken, MaximumRequestBytes).ConfigureAwait(false);
        var result = SystemOneProtocol.Decode(wire, response.Body, DecisionPrecision.FourDecimalPlaces);
        if (!string.Equals(result.ModelId, model, StringComparison.Ordinal))
        {
            var actual = await DiscoverAsync(result.ModelId, cancellationToken).ConfigureAwait(false);
            var actualProfile = OllayaProfile.Read(actual);
            actualProfile.Validate(request);
            if (actualProfile != profile)
                throw new DecisionProtocolException("The answering model's profile differs from the selected model's discovery.");
            result.AdditionalProperties!["ollaya.actualModelInfo"] = actual;
        }
        result.AdditionalProperties!["ollaya.modelInfo"] = discovery;
        result.AdditionalProperties["ollaya.wireRequest"] = wire.Body;
        if (response.RequestId is not null) result.AdditionalProperties["systemone.requestId"] = response.RequestId;
        cancellationToken.ThrowIfCancellationRequested();
        return result;
    }

    private async Task<JsonElement> DiscoverAsync(string model, CancellationToken cancellationToken)
        => (await _transport.PostAsync("/api/show", JsonSerializer.SerializeToElement(new { model }), cancellationToken).ConfigureAwait(false)).Body;

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null) return null;
        return serviceType == typeof(DecisionClientMetadata) ? _metadata : serviceType.IsInstanceOfType(this) ? this : null;
    }

    public void Dispose() => _transport.Dispose();
}

public sealed class OllayaApiException : HttpRequestException
{
    public string? RequestId { get; }
    public string? Code { get; }
    /// <summary>Owned, opt-in diagnostic text. Never log automatically; it may contain sensitive data.</summary>
    public string DiagnosticBody { get; }
    internal OllayaApiException(System.Net.HttpStatusCode status, string? requestId, string body)
        : base($"Ollaya HTTP operation failed with status {(int)status}.", null, status)
    {
        (RequestId, DiagnosticBody) = (requestId, body);
        try
        {
            using var json = JsonDocument.Parse(body);
            if (json.RootElement.ValueKind == JsonValueKind.Object &&
                json.RootElement.TryGetProperty("code", out var code) && code.ValueKind == JsonValueKind.String)
                Code = code.GetString();
        }
        catch (JsonException) { /* Malformed error bodies must not erase the HTTP status. */ }
    }
}

public sealed class OllayaBodyLimitException() : Exception("Ollaya response exceeded the configured buffering limit.");
