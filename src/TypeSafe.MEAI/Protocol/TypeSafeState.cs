using System.Text;
using System.Text.Json;
using Microsoft.Extensions.AI;

namespace TypeSafe.MEAI;

/// <summary>An owned explicit state; literal text is never implicitly parsed.</summary>
internal sealed class TypeSafeState
{
    private readonly JsonElement _value;
    private readonly bool _text;

    private TypeSafeState(JsonElement value, bool text)
    {
        TypeSafeJson.ValidateEntry(value);
        _value = value.Clone();
        _text = text;
    }

    /// <summary>Creates literal string state.</summary>
    public static TypeSafeState FromText(string text) =>
        new(JsonSerializer.SerializeToElement(text ?? throw new ArgumentNullException(nameof(text))), true);

    /// <summary>Copies structured JSON state, including JSON string or null.</summary>
    public static TypeSafeState FromJson(JsonElement value) => new(value, false);

    /// <summary>Parses and owns structured JSON state.</summary>
    public static TypeSafeState Parse(string json)
    {
        using var document = JsonDocument.Parse(json);
        return FromJson(document.RootElement);
    }

    /// <summary>Creates the one-message native request representation.</summary>
    public ChatMessage ToMessage() => new(ChatRole.User, [
        _text ? new TextContent(_value.GetString()!) :
            new DataContent(Encoding.UTF8.GetBytes(_value.GetRawText()), "application/json")
    ]);

    internal JsonElement Value => _value;
}
