using System.Collections.ObjectModel;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace TypeSafe.MEAI;

/// <summary>An immutable correlated native question.</summary>
public abstract record TypeSafeQuestion
{
    /// <summary>Ordinal correlation ID, not judgment instructions.</summary>
    public string Id { get; }
    /// <summary>Native discriminator.</summary>
    public string Type { get; }
    /// <summary>Owned instructions. Undefined means omitted; Null means explicitly null.</summary>
    public JsonElement Instructions { get; }

    private protected TypeSafeQuestion(string id, string type, JsonElement instructions)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);
        if (instructions.ValueKind != JsonValueKind.Undefined) TypeSafeJson.ValidateEntry(instructions);
        Id = id;
        Type = type;
        Instructions = instructions.ValueKind == JsonValueKind.Undefined ? default : instructions.Clone();
    }

    /// <summary>Returns an owned serializable wire question without its map key.</summary>
    public JsonElement ToJson()
    {
        JsonObject obj = new() { ["type"] = Type };
        if (Instructions.ValueKind != JsonValueKind.Undefined)
            obj["instructions"] = JsonNode.Parse(Instructions.GetRawText());
        AddCriteria(obj);
        return JsonSerializer.SerializeToElement(obj);
    }

    private protected abstract void AddCriteria(JsonObject obj);

    internal static TypeSafeQuestion Parse(string id, JsonElement json)
    {
        foreach (var property in json.EnumerateObject())
            if (property.Name is not ("type" or "instructions" or "criteria"))
                throw new ArgumentException("Unsupported top-level native question field. Use structured instructions for application context.");
        var instructions = json.TryGetProperty("instructions", out var i) ? i : default;
        return json.GetProperty("type").GetString() switch
        {
            "noul" => new NoulQuestion(id, instructions, json.TryGetProperty("criteria", out var c) ? c : default),
            "choice" => new ChoiceQuestion(id, instructions, json.GetProperty("criteria").EnumerateObject()
                .Select(p => KeyValuePair.Create(p.Name, p.Value))),
            "score" => new ScoreQuestion(id, instructions, json.GetProperty("criteria").EnumerateArray()),
            _ => throw new ArgumentException("Unknown question discriminator.")
        };
    }
}

/// <summary>A native yes-probability question, never a two-label Choice.</summary>
public sealed record NoulQuestion : TypeSafeQuestion
{
    /// <summary>Optional true/false descriptions; Undefined omits criteria.</summary>
    public JsonElement Criteria { get; }
    /// <summary>Creates a Noul question with snapshotted instructions and criteria.</summary>
    public NoulQuestion(string id, JsonElement instructions = default, JsonElement criteria = default)
        : base(id, "noul", instructions)
    {
        if (criteria.ValueKind is not (JsonValueKind.Undefined or JsonValueKind.Null or JsonValueKind.Object))
            throw new ArgumentException("Noul criteria must be a true/false object, null, or omitted.");
        if (criteria.ValueKind == JsonValueKind.Object)
        {
            TypeSafeJson.UniqueProperties(criteria);
            foreach (var p in criteria.EnumerateObject())
            {
                if (p.Name is not ("true" or "false")) throw new ArgumentException("Unknown Noul criterion.");
                TypeSafeJson.ValidateEntry(p.Value);
            }
        }
        Criteria = criteria.ValueKind == JsonValueKind.Undefined ? default : criteria.Clone();
    }
    private protected override void AddCriteria(JsonObject obj)
    {
        if (Criteria.ValueKind != JsonValueKind.Undefined) obj["criteria"] = JsonNode.Parse(Criteria.GetRawText());
    }
}

/// <summary>A closed-set native Choice question.</summary>
public sealed record ChoiceQuestion : TypeSafeQuestion
{
    /// <summary>Ordinal option names and owned descriptions.</summary>
    public IReadOnlyDictionary<string, JsonElement> Criteria { get; }
    /// <summary>Creates a question with 1 through 255 distinct options.</summary>
    public ChoiceQuestion(string id, JsonElement instructions, IEnumerable<KeyValuePair<string, JsonElement>> criteria)
        : base(id, "choice", instructions)
    {
        Dictionary<string, JsonElement> copy = new(StringComparer.Ordinal);
        foreach (var p in criteria)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(p.Key);
            TypeSafeJson.ValidateEntry(p.Value);
            if (!copy.TryAdd(p.Key, p.Value.Clone())) throw new ArgumentException("Duplicate Choice option.");
        }
        if (copy.Count is < 1 or > 255) throw new ArgumentException("Choice requires 1 through 255 options.");
        Criteria = new ReadOnlyDictionary<string, JsonElement>(copy);
    }
    private protected override void AddCriteria(JsonObject obj) => obj["criteria"] = JsonSerializer.SerializeToNode(Criteria);
}

/// <summary>A native probability-weighted fractional rubric index.</summary>
public sealed record ScoreQuestion : TypeSafeQuestion
{
    /// <summary>Owned levels in inference-affecting order.</summary>
    public IReadOnlyList<JsonElement> Criteria { get; }
    /// <summary>Creates a Score question with 2 through 10 levels.</summary>
    public ScoreQuestion(string id, JsonElement instructions, IEnumerable<JsonElement> criteria)
        : base(id, "score", instructions)
    {
        var copy = criteria.Select(c => { TypeSafeJson.ValidateEntry(c); return c.Clone(); }).ToArray();
        if (copy.Length is < 2 or > 10) throw new ArgumentException("Score requires 2 through 10 levels.");
        Criteria = Array.AsReadOnly(copy);
    }
    private protected override void AddCriteria(JsonObject obj) => obj["criteria"] = JsonSerializer.SerializeToNode(Criteria);
}
