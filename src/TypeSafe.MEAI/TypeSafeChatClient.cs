using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.AI;

namespace TypeSafe.MEAI;

/// <summary>One constrained MEAI provider for native batches, finite typed decisions and selection-only tools.
/// This is not a free-form generative client. Use ordinary MEAI APIs and explicit client acceptance policies.</summary>
public sealed class TypeSafeChatClient : IChatClient
{
    /// <summary>Serializable native question object in ChatOptions.AdditionalProperties for explicit batching.</summary>
    public const string QuestionsKey = "typesafe.v1.questions";
    private readonly IChatClient _native;
    private readonly TypeSafeClientOptions _settings;
    private int _disposed;
    internal static JsonSerializerOptions ContentJson
    {
        get
        {
            var json = new JsonSerializerOptions(AIJsonUtilities.DefaultOptions);
            json.AddAIContentType<JevResultContent>(JevResultContent.TypeDiscriminator);
            return json;
        }
    }

    /// <summary>Creates a provider borrowing HttpClient unless ownership is transferred. Direct hosts must supply
    /// nonredirecting handlers and a caller deadline covering body reads; use keyed registration for the supported resilience/cache pipeline.</summary>
    public TypeSafeChatClient(HttpClient httpClient, TypeSafeClientOptions options, string apiKey, bool ownsHttpClient = false)
        : this(new NativeChatClient(httpClient, options, apiKey, ownsHttpClient), options) { }

    internal TypeSafeChatClient(IChatClient native, TypeSafeClientOptions options)
    {
        _native = native;
        _settings = options.Validate();
    }

    /// <inheritdoc />
    public async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        cancellationToken.ThrowIfCancellationRequested();
        var request = NativeRequest.SnapshotOptions(options, schema: true);
        bool hasNative = request.AdditionalProperties!.ContainsKey(QuestionsKey);
        bool hasSchema = request.ResponseFormat is ChatResponseFormatJson { Schema: not null };
        bool hasTools = request.Tools is not null;
        if ((hasNative ? 1 : 0) + (hasSchema ? 1 : 0) + (hasTools ? 1 : 0) != 1)
            throw new NotSupportedException("Specify exactly one of finite JSON-schema output, tools, or explicit native questions. Free-form generation and ambiguous combinations are unsupported.");
        if (!hasSchema && request.ResponseFormat is not null && request.ResponseFormat is not ChatResponseFormatJson { Schema: null })
            throw new NotSupportedException("Native/tool decisions do not support text or other response formats.");
        if (!hasTools && (request.ToolMode is not null || request.AllowMultipleToolCalls is not null))
            throw new NotSupportedException("Tool options require a tool request.");

        var state = MessageProjection.State(messages);
        string? instructions = request.Instructions;
        request.Instructions = null;
        if (hasTools)
            return await ToolSelection.RunAsync(this, state, request, instructions, _settings, cancellationToken).ConfigureAwait(false);
        if (hasSchema)
        {
            var format = (ChatResponseFormatJson)request.ResponseFormat!;
            var plan = TypeSafeSchemaCompiler.Compile(format.Schema!.Value,
                _settings.Fields, instructions, _settings.Policy, format.SchemaDescription);
            request.ResponseFormat = ChatResponseFormat.Json;
            var evidence = await BatchAsync(state, plan.Questions, request, "projection", cancellationToken).ConfigureAwait(false);
            evidence.AdditionalProperties!["typesafe.v1.fields"] = JsonSerializer.SerializeToElement(plan.Fields.ToDictionary(f => f.Question.Id, f => f.Pointer));
            var projection = plan.Project(evidence.Native);
            if (projection.Rejections.Count != 0) throw new TypeSafeAbstentionException([evidence], projection.Rejections);
            if (_settings.ValidateResult?.Invoke(projection.Value!.Value) is { } reason)
                throw new TypeSafeAbstentionException([evidence], [new("", reason)]);
            return Response([new TextContent(projection.Value!.Value.GetRawText()), evidence], [evidence]);
        }
        var payload = (JsonElement)request.AdditionalProperties![QuestionsKey]!;
        TypeSafeJson.UniqueProperties(payload);
        var questions = payload.EnumerateObject().Select(p => TypeSafeQuestion.Parse(p.Name, p.Value)).ToArray();
        if (instructions is not null)
            questions = questions.Select(question =>
            {
                var node = JsonNode.Parse(question.ToJson().GetRawText())!.AsObject();
                node["instructions"] = JsonSerializer.SerializeToNode(new { taskInstructions = instructions, questionInstructions = question.ToJson().TryGetProperty("instructions", out var original) ? original : JsonSerializer.SerializeToElement<object?>(null) });
                return TypeSafeQuestion.Parse(question.Id, JsonSerializer.SerializeToElement(node));
            }).ToArray();
        var native = await BatchAsync(state, questions, request, "native", cancellationToken).ConfigureAwait(false);
        return Response([native], [native]);
    }

    internal async Task<JevResultContent> BatchAsync(TypeSafeState state, IEnumerable<TypeSafeQuestion> questions, ChatOptions options,
        string role, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var questionArray = questions.ToArray();
        var nativeOptions = options.WithTypeSafeQuestions(questionArray);
        nativeOptions.Tools = null;
        nativeOptions.ToolMode = null;
        nativeOptions.AllowMultipleToolCalls = null;
        nativeOptions.ResponseFormat = ChatResponseFormat.Json;
        var chat = await _native.GetResponseAsync([state.ToMessage()], nativeOptions, cancellationToken).ConfigureAwait(false);
        cancellationToken.ThrowIfCancellationRequested();
        var evidence = chat.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>().Single();
        evidence.Native.ValidateCorrelation(questionArray);
        var json = ContentJson;
        var owned = JsonSerializer.Deserialize<JevResultContent>(JsonSerializer.Serialize(evidence, json), json)!;
        owned.AdditionalProperties ??= new();
        owned.AdditionalProperties["typesafe.v1.batch-role"] = role;
        return owned;
    }

    internal static ChatResponse Response(IList<AIContent> contents, IReadOnlyList<JevResultContent> evidence, bool toolCall = false) =>
        new(new ChatMessage(ChatRole.Assistant, contents))
        {
            ModelId = evidence.LastOrDefault()?.Model,
            FinishReason = toolCall ? ChatFinishReason.ToolCalls : ChatFinishReason.Stop,
            Usage = evidence.Count == 0 ? null : new()
            {
                InputTokenCount = evidence.Sum(e => e.InputTokens), OutputTokenCount = evidence.Sum(e => e.OutputTokens),
                TotalTokenCount = evidence.Sum(e => e.TotalTokens)
            }
        };

    /// <inheritdoc />
    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var response = await GetResponseAsync(messages, options, cancellationToken).ConfigureAwait(false);
        foreach (var update in response.ToChatResponseUpdates())
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return update;
        }
    }

    /// <inheritdoc />
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        return serviceKey is null && serviceType.IsInstanceOfType(this) ? this :
            serviceKey is null && serviceType == typeof(ChatClientMetadata) ? new ChatClientMetadata("typesafe", _settings.Endpoint, _settings.ModelId) : null;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0) _native.Dispose();
    }
}
