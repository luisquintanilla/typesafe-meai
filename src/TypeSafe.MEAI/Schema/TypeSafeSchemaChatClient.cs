using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.AI;

namespace TypeSafe.MEAI;

/// <summary>Bounded schema projection outside the decorated native/cache pipeline.</summary>
public sealed class TypeSafeSchemaChatClient : DelegatingChatClient
{
    private readonly IReadOnlyDictionary<string, TypeSafeFieldBinding> _bindings;
    private readonly Func<JsonElement, string?>? _validate;

    /// <summary>Creates an adapter over a decorated native client and snapshots leaf bindings.</summary>
    public TypeSafeSchemaChatClient(IChatClient decoratedNativeClient, IReadOnlyDictionary<string, TypeSafeFieldBinding> bindings,
        Func<JsonElement, string?>? validate = null) : base(decoratedNativeClient)
    {
        _bindings = new Dictionary<string, TypeSafeFieldBinding>(bindings, StringComparer.Ordinal);
        _validate = validate;
    }

    /// <inheritdoc/>
    public override async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        NativeRequest.ValidateCapabilities(options, schema: true);
        if (options?.ResponseFormat is not ChatResponseFormatJson { Schema: { } schema })
            throw new NotSupportedException("Use the JSON-schema response-format typed-output path; prompt-appending mode is unsupported.");
        var plan = TypeSafeSchemaCompiler.Compile(schema, _bindings, options.Instructions);
        ChatOptions nativeOptions = NativeRequest.SnapshotOptions(options, schema: true);
        nativeOptions.Instructions = null;
        nativeOptions.ResponseFormat = ChatResponseFormat.Json;
        nativeOptions = nativeOptions.WithTypeSafeQuestions(plan.Questions);
        // Validate and own the one-message input before entering any decorated native pipeline.
        var state = NativeRequest.ReadState(messages);
        var chat = await InnerClient.GetResponseAsync([state.ToMessage()], nativeOptions, cancellationToken).ConfigureAwait(false);
        var evidence = chat.GetTypeSafeResponse(plan.Questions);
        var projection = plan.Project(evidence);
        if (projection.Rejections.Count != 0) throw new TypeSafeAbstentionException(evidence, projection.Rejections);
        JsonElement value = projection.Value!.Value;
        if (_validate?.Invoke(value) is { } reason)
            throw new TypeSafeAbstentionException(evidence, [new("", reason)]);
        return new ChatResponse(new ChatMessage(ChatRole.Assistant, value.GetRawText()))
        {
            ModelId = evidence.Model, FinishReason = ChatFinishReason.Stop,
            Usage = new() { InputTokenCount = evidence.InputTokens, OutputTokenCount = evidence.OutputTokens, TotalTokenCount = evidence.TotalTokens },
            AdditionalProperties = new()
            {
                ["typesafe.v1.evidence"] = evidence.Json,
                ["typesafe.v1.evidence.questions"] = JsonSerializer.SerializeToElement(plan.Questions.ToDictionary(q => q.Id, q => q.ToJson(), StringComparer.Ordinal))
            }
        };
    }

    /// <inheritdoc/>
    public override async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var response = await GetResponseAsync(messages, options, cancellationToken).ConfigureAwait(false);
        foreach (var update in response.ToChatResponseUpdates())
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return update;
        }
    }
}
