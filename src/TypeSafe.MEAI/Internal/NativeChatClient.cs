using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;

namespace TypeSafe.MEAI;

/// <summary>A constrained native Jev provider; not a conversational or generative model.</summary>
internal sealed class NativeChatClient : IChatClient
{
    private readonly HttpClient _http;
    private readonly bool _owns;
    private readonly TypeSafeClientOptions _options;
    private readonly TypeSafeHttpTransport _transport;
    private int _disposed;

    /// <summary>Creates a provider borrowing HttpClient unless ownership is transferred. Direct hosts must supply a
    /// nonredirecting resilience/buffering handler chain or a caller cancellation deadline covering body reads;
    /// HttpClient.Timeout alone stops at headers. Use keyed registration for the built-in budgets.</summary>
    public NativeChatClient(HttpClient httpClient, TypeSafeClientOptions options, string apiKey, bool ownsHttpClient = false)
    {
        _http = httpClient;
        _options = options.Validate();
        _transport = new(httpClient, options, apiKey);
        _owns = ownsHttpClient;
    }

    /// <inheritdoc/>
    public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        cancellationToken.ThrowIfCancellationRequested();
        var request = NativeRequest.Create(messages, options, _options);
        return SendAsync(request, cancellationToken);
    }

    private async Task<ChatResponse> SendAsync(NativeRequest request, CancellationToken ct)
    {
        var json = await _transport.SendAsync("/v1/systemone", request.Body, ct).ConfigureAwait(false);
        var content = new JevResultContent(json);
        content.Native.ValidateCorrelation(request.Questions);
        var native = content.Native;
        return new ChatResponse(new ChatMessage(ChatRole.Assistant, [content]))
        {
            ModelId = native.Model,
            FinishReason = ChatFinishReason.Stop,
            Usage = new() { InputTokenCount = native.InputTokens, OutputTokenCount = native.OutputTokens, TotalTokenCount = native.TotalTokens }
        };
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages,
        ChatOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var response = await GetResponseAsync(messages, options, cancellationToken).ConfigureAwait(false);
        foreach (var update in response.ToChatResponseUpdates())
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return update;
        }
    }

    /// <inheritdoc/>
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null) return null;
        if (serviceType == typeof(ChatClientMetadata)) return new ChatClientMetadata("typesafe", _options.Endpoint, _options.ModelId);
        return serviceType.IsInstanceOfType(this) ? this : null;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0 && _owns) _http.Dispose();
    }
}
