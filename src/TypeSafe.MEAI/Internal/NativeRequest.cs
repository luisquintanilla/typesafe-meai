using System.Text.Json;
using Microsoft.Extensions.AI;

namespace TypeSafe.MEAI;

internal sealed record NativeRequest(TypeSafeState State, TypeSafeQuestion[] Questions, ChatOptions Options)
{
    internal const string QuestionsKey = "typesafe.v1.questions";
    internal const string ScopeKey = "typesafe.v1.cache-scope";

    internal JsonElement Body => JsonSerializer.SerializeToElement(new
    {
        state = State.Value, model = Options.ModelId,
        questions = Questions.ToDictionary(q => q.Id, q => q.ToJson(), StringComparer.Ordinal)
    });

    internal static NativeRequest Create(IEnumerable<ChatMessage> messages, ChatOptions? options, TypeSafeClientOptions settings)
    {
        ValidateCapabilities(options);
        var state = ReadState(messages);
        return CreateWithState(state, options, settings);
    }

    internal static TypeSafeState ReadState(IEnumerable<ChatMessage> messages)
    {
        var array = messages.ToArray();
        if (array.Length != 1 || array[0].Role != ChatRole.User || array[0].Contents.Count != 1)
            throw new NotSupportedException("Native inference requires exactly one user message with one explicit state content.");
        return array[0].Contents[0] switch
        {
            TextContent text => TypeSafeState.FromText(text.Text),
            DataContent data when string.Equals(data.MediaType, "application/json", StringComparison.OrdinalIgnoreCase) =>
                TypeSafeState.Parse(new System.Text.UTF8Encoding(false, true).GetString(data.Data.Span)),
            _ => throw new NotSupportedException("Native state must be literal text or owned application/json data.")
        };
    }

    private static NativeRequest CreateWithState(TypeSafeState state, ChatOptions? options, TypeSafeClientOptions settings)
    {
        if (options?.AdditionalProperties?.TryGetValue(QuestionsKey, out var payload) != true)
            throw new ArgumentException("Native questions are required.");
        var json = payload is JsonElement element ? element.Clone() : JsonSerializer.SerializeToElement(payload);
        TypeSafeJson.UniqueProperties(json);
        var questions = json.EnumerateObject().Select(p => TypeSafeQuestion.Parse(p.Name, p.Value)).ToArray();
        if (questions.Length == 0) throw new ArgumentException("At least one question is required.");
        var model = options.ModelId ?? settings.ModelId;
        ArgumentException.ThrowIfNullOrWhiteSpace(model);
        var properties = SnapshotOptions(options).AdditionalProperties!;
        properties[QuestionsKey] = json;
        properties[ScopeKey] = JsonSerializer.SerializeToElement(new
        {
            provider = "typesafe", protocol = 1, endpoint = settings.Endpoint.AbsoluteUri, account = settings.CacheNamespace
        });
        return new(state, questions, new ChatOptions { ModelId = model, ResponseFormat = ChatResponseFormat.Json, AdditionalProperties = properties });
    }

    internal static ChatOptions SnapshotOptions(ChatOptions? options, bool schema = false)
    {
        ValidateCapabilities(options, schema);
        var snapshot = options?.Clone() ?? new ChatOptions();
        snapshot.AdditionalProperties = new();
        if (options?.AdditionalProperties is { } properties)
            foreach (var p in properties)
            {
                if (p.Key.StartsWith("typesafe.", StringComparison.Ordinal) && p.Key != QuestionsKey && p.Key != ScopeKey)
                    throw new NotSupportedException("Unrecognized reserved TypeSafe option.");
                snapshot.AdditionalProperties[p.Key] = p.Value is JsonElement value ? value.Clone() : JsonSerializer.SerializeToElement(p.Value);
            }
        return snapshot;
    }

    internal static void ValidateCapabilities(ChatOptions? o, bool schema = false)
    {
        if (o is null) return;
        if (o.ConversationId is not null || (!schema && o.Instructions is not null) || o.Temperature is not null ||
            o.MaxOutputTokens is not null || o.TopP is not null || o.TopK is not null || o.FrequencyPenalty is not null ||
            o.PresencePenalty is not null || o.Seed is not null || o.Reasoning is not null || o.StopSequences is not null ||
            o.AllowMultipleToolCalls is not null || o.ToolMode is not null || o.Tools is not null ||
            o.AllowBackgroundResponses is not null || o.ContinuationToken is not null || o.RawRepresentationFactory is not null)
            throw new NotSupportedException("An explicitly supplied ChatOptions capability is unsupported by native decision inference.");
        if (!schema && o.ResponseFormat is not null && o.ResponseFormat is not ChatResponseFormatJson { Schema: null })
            throw new NotSupportedException("Native inference supports plain JSON format only, not text or JSON Schema.");
    }
}
