using System.Text.Json;
using Microsoft.Extensions.AI;

namespace TypeSafe.MEAI;

/// <summary>Internal native option construction, not a consumer invocation surface.</summary>
internal static class TypeSafeChatClientExtensions
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

}
