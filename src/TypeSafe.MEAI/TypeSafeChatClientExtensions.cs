using System.Text.Json;
using Microsoft.Extensions.AI;

namespace TypeSafe.MEAI;

/// <summary>Native helpers that always invoke the supplied decorated IChatClient.</summary>
public static class TypeSafeChatClientExtensions
{
    /// <summary>Clones options and attaches snapshotted, serializable native questions.</summary>
    public static ChatOptions WithTypeSafeQuestions(this ChatOptions? options, IEnumerable<TypeSafeQuestion> questions)
    {
        Dictionary<string, JsonElement> map = new(StringComparer.Ordinal);
        foreach (var question in questions)
            if (!map.TryAdd(question.Id, question.ToJson())) throw new ArgumentException("Duplicate question ID.");
        if (map.Count == 0) throw new ArgumentException("At least one question is required.");
        var clone = options?.Clone() ?? new ChatOptions();
        clone.AdditionalProperties = options?.AdditionalProperties is { } properties ? new(properties) : new();
        clone.AdditionalProperties[NativeRequest.QuestionsKey] = JsonSerializer.SerializeToElement(map);
        return clone;
    }

    /// <summary>Submits one batch through decorated middleware and decodes ordinary response content.</summary>
    public static async Task<TypeSafeResponse> GetTypeSafeResponseAsync(this IChatClient client, TypeSafeState state,
        IEnumerable<TypeSafeQuestion> questions, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var snapshot = questions.ToArray();
        var request = options.WithTypeSafeQuestions(snapshot);
        var response = await client.GetResponseAsync([state.ToMessage()], request, cancellationToken).ConfigureAwait(false);
        return response.GetTypeSafeResponse(snapshot);
    }

    /// <summary>Decodes canonical standard content without depending on RawRepresentation.</summary>
    public static TypeSafeResponse GetTypeSafeResponse(this ChatResponse response, IEnumerable<TypeSafeQuestion> questions)
    {
        if (response.Messages.Count != 1 || response.Messages[0].Role != ChatRole.Assistant ||
            response.Messages[0].Contents.Count != 1 || response.Messages[0].Contents[0] is not TextContent text)
            throw new TypeSafeProtocolException("Expected a single assistant JSON TextContent.");
        return TypeSafeResponse.Parse(text.Text, questions);
    }
}
