using System.Text;
using System.Text.Json;
using Microsoft.Extensions.AI;

namespace TypeSafe.MEAI;

internal static class MessageProjection
{
    internal static TypeSafeState State(IEnumerable<ChatMessage> messages)
    {
        var snapshot = messages.ToArray();
        if (snapshot.Length == 0) throw new ArgumentException("At least one task/state message is required.", nameof(messages));
        if (snapshot.Length == 1 && snapshot[0].Role == ChatRole.User && snapshot[0].Contents.Count == 1)
        {
            var content = snapshot[0].Contents[0];
            if (content is TextContent text) return TypeSafeState.FromText(text.Text);
            return TypeSafeState.FromJson(Json(content));
        }
        return TypeSafeState.FromJson(JsonSerializer.SerializeToElement(new
        {
            conversation = snapshot.Select(message => new
            {
                role = message.Role.Value,
                author = message.AuthorName,
                contents = message.Contents.Select(content => content is TextContent text ?
                    new { type = "text", value = JsonSerializer.SerializeToElement(text.Text) } :
                    new { type = "json", value = Json(content) }).ToArray()
            }).ToArray()
        }));
    }

    internal static JsonElement Json(AIContent content)
    {
        if (content is not DataContent data || !(string.Equals(data.MediaType, "application/json", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(data.MediaType, JevResultContent.ContentMediaType, StringComparison.OrdinalIgnoreCase)))
            throw new NotSupportedException("TypeSafe accepts text, application/json data and Jev evidence only; it does not infer from other modalities or tool protocol contents.");
        using var document = JsonDocument.Parse(new UTF8Encoding(false, true).GetString(data.Data.Span));
        TypeSafeJson.ValidateEntry(document.RootElement);
        return document.RootElement.Clone();
    }
}
