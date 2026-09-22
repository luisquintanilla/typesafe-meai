using System.Text.Json;

namespace TypeSafe.MEAI;

/// <summary>Owned JSON values used by the native protocol.</summary>
public static class TypeSafeJson
{
    /// <summary>Serializes and owns a structured entry, including an explicit null.</summary>
    public static JsonElement Entry<T>(T value)
    {
        var json = JsonSerializer.SerializeToElement(value);
        ValidateEntry(json);
        return json;
    }

    internal static void ValidateEntry(JsonElement value)
    {
        if (value.ValueKind is not (JsonValueKind.String or JsonValueKind.Object or JsonValueKind.Array or JsonValueKind.Null))
            throw new ArgumentException("An entry must be string, object, array, or null.");
    }

    internal static void UniqueProperties(JsonElement value)
    {
        if (value.ValueKind == JsonValueKind.Object)
        {
            HashSet<string> names = new(StringComparer.Ordinal);
            foreach (var p in value.EnumerateObject())
            {
                if (!names.Add(p.Name)) throw new JsonException("Duplicate JSON property.");
                UniqueProperties(p.Value);
            }
        }
        else if (value.ValueKind == JsonValueKind.Array)
        {
            foreach (var child in value.EnumerateArray()) UniqueProperties(child);
        }
    }
}
