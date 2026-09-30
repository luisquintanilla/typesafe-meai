using System.Text.Json;
using DecisionInference.SystemOne;
using OllamaSharp;
using NativeResponse = OllamaSharp.Models.SystemOneResponse;

namespace DecisionInference.OllamaSharp;

/// <summary>Adapts an ordinary OllamaSharp client to native typed decisions without chat projection.</summary>
/// <remarks>
/// Captures the default model and endpoint at construction; never mutates SelectedModel.
/// RawRepresentation is an owned SDK DTO-derived JsonElement snapshot, not original response JSON.
/// The caller owns endpoint safety, HTTP policy, serializer settings, buffering and deadlines.
/// Missing scalar fields and discarded source properties cannot be recovered from SDK DTOs.
/// Do not mutate shared SDK configuration or dispose borrowed resources during a call.
/// </remarks>
public sealed class OllamaSharpDecisionGenerator : IDecisionGenerator
{
    private readonly IOllamaApiClient _client;
    private readonly IDisposable? _ownedClient;
    private readonly DecisionGeneratorMetadata _metadata;
    private int _disposed;

    /// <summary>Wraps the supplied native SDK client, borrowing it unless ownership is explicitly transferred.</summary>
    /// <remarks>Ownership requires IDisposable. Disposing an owned SDK client does not separately dispose its borrowed HttpClient.</remarks>
    public OllamaSharpDecisionGenerator(IOllamaApiClient client, bool ownsClient = false)
    {
        ArgumentNullException.ThrowIfNull(client);
        if (ownsClient && client is not IDisposable)
            throw new ArgumentException("An owned OllamaSharp client must implement IDisposable.", nameof(client));
        string? model = client.SelectedModel;
        if (model is { Length: > 0 }) ArgumentException.ThrowIfNullOrWhiteSpace(model);
        _client = client;
        _ownedClient = ownsClient ? (IDisposable)client : null;
        _metadata = new("ollamasharp", client.Uri, string.IsNullOrEmpty(model) ? null : model,
            [DecisionKind.Binary, DecisionKind.Choice, DecisionKind.Score]);
    }

    public async Task<DecisionResult> GenerateAsync(DecisionInput input, DecisionGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();
        ArgumentNullException.ThrowIfNull(input);
        DecisionGenerationOptions? snapshot = options switch
        {
            null => null,
            OllamaSharpGenerationOptions typed => typed.Clone(),
            _ when options.GetType() == typeof(DecisionGenerationOptions) => options.Clone(),
            _ => throw new NotSupportedException("Unsupported generation options type for OllamaSharp.")
        };
        if (snapshot?.AdditionalProperties is { Count: > 0 })
            throw new NotSupportedException("OllamaSharp does not accept AdditionalProperties input keys.");
        string? model = snapshot?.ModelId ?? _metadata.DefaultModelId;
        ArgumentException.ThrowIfNullOrWhiteSpace(model);
        OllamaDecisionProtocol.ValidateInput(input);
        var wire = SystemOneProtocol.Encode(input, model, OllamaDecisionProtocol.MaximumScoreLevels);
        var request = OllamaSharpSystemOneMapper.Request(wire, (snapshot as OllamaSharpGenerationOptions)?.KeepAliveSeconds);
        NativeResponse response;
        try
        {
            response = await _client.SystemOneAsync(request, cancellationToken).ConfigureAwait(false);
        }
        catch (JsonException)
        {
            throw new DecisionProtocolException("Malformed OllamaSharp native response.");
        }
        cancellationToken.ThrowIfCancellationRequested();
        var raw = OllamaSharpSystemOneMapper.Response(response);
        var result = SystemOneProtocol.Decode(wire, raw, DecisionPrecision.HighPrecision);
        OllamaDecisionProtocol.ValidateResult(input, model, result);
        result.AdditionalProperties!["ollamasharp.rawRepresentationKind"] = "sdk-dto-snapshot";
        cancellationToken.ThrowIfCancellationRequested();
        return result;
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null) return null;
        if (serviceType == typeof(DecisionGeneratorMetadata)) return _metadata;
        if (serviceType == typeof(IOllamaApiClient)) return _client;
        return serviceType == typeof(IDecisionGenerator) || serviceType == typeof(OllamaSharpDecisionGenerator) ? this : null;
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0) _ownedClient?.Dispose();
    }
}
