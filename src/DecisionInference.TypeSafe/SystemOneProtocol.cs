using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace DecisionInference.TypeSafe;

// Native validation/serialization adapted from the frozen bc2f307 protocol, without chat projection.
internal static class SystemOneProtocol
{
    internal static JsonElement Encode(DecisionRequest request, string model)
    {
        Entry(request.State);
        var questions = new JsonObject();
        foreach (var q in request.Questions)
        {
            var body = new JsonObject { ["type"] = q.Kind switch
            {
                DecisionKind.Binary => "noul", DecisionKind.Choice => "choice", DecisionKind.Score => "score",
                _ => throw new NotSupportedException()
            }};
            if (q.Instructions is not null) body["instructions"] = q.Instructions;
            switch (q)
            {
                case BinaryDecisionQuestion b:
                    if (b.TrueDescription is not null || b.FalseDescription is not null)
                    {
                        var criteria = new JsonObject();
                        if (b.TrueDescription is not null) criteria["true"] = b.TrueDescription;
                        if (b.FalseDescription is not null) criteria["false"] = b.FalseDescription;
                        body["criteria"] = criteria;
                    }
                    break;
                case ChoiceDecisionQuestion c:
                    if (c.Candidates.Count > 255) throw new ArgumentException("Jev Choice supports 1 through 255 candidates.");
                    var choices = new JsonObject();
                    foreach (var candidate in c.Candidates) choices.Add(candidate.Id, JsonValue.Create(candidate.Description));
                    body["criteria"] = choices;
                    break;
                case ScoreDecisionQuestion s:
                    if (s.Rubric.Count > 10) throw new ArgumentException("Jev Score supports 2 through 10 levels.");
                    body["criteria"] = new JsonArray(s.Rubric.Select(r => (JsonNode?)JsonValue.Create(r)).ToArray());
                    break;
            }
            questions.Add(q.Id, body);
        }
        return JsonSerializer.SerializeToElement(new { state = request.State, model, questions });
    }

    internal static DecisionResponse Decode(DecisionRequest request, string json)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            UniqueProperties(root);
            var model = root.GetProperty("model").GetString();
            if (string.IsNullOrWhiteSpace(model)) throw new DecisionProtocolException("Missing model.");
            var usage = root.GetProperty("usage");
            long input = usage.GetProperty("input_tokens").GetInt64(), output = usage.GetProperty("output_tokens").GetInt64();
            long sum = checked(input + output);
            if (input < 0 || output < 0 || (usage.TryGetProperty("total_tokens", out var t) && t.GetInt64() != sum))
                throw new DecisionProtocolException("Invalid token counts.");
            var answers = new List<DecisionAnswer>();
            foreach (var p in root.GetProperty("answers").EnumerateObject())
            {
                var a = p.Value;
                switch (a.GetProperty("type").GetString())
                {
                    case "noul":
                        answers.Add(new BinaryDecisionAnswer(p.Name, a.GetProperty("noul").GetDouble()) { RawRepresentation = a.Clone() });
                        break;
                    case "choice":
                        Unit(a.GetProperty("confidence").GetDouble());
                        answers.Add(new ChoiceDecisionAnswer(p.Name, a.GetProperty("choice").GetString()!,
                            a.GetProperty("probabilities").EnumerateObject().Select(p => KeyValuePair.Create(p.Name, p.Value.GetDouble())))
                            { RawRepresentation = a.Clone() });
                        break;
                    case "score":
                        Unit(a.GetProperty("confidence").GetDouble());
                        var probabilities = a.GetProperty("probabilities");
                        int count = probabilities.EnumerateObject().Count();
                        var keys = Enumerable.Range(0, count).Select(i => i.ToString(CultureInfo.InvariantCulture)).ToArray();
                        if (!probabilities.EnumerateObject().Select(p => p.Name).ToHashSet(StringComparer.Ordinal).SetEquals(keys) ||
                            !a.GetProperty("legend").EnumerateObject().Select(p => p.Name).ToHashSet(StringComparer.Ordinal).SetEquals(keys))
                            throw new DecisionProtocolException("Invalid ordinal indices.");
                        foreach (var entry in a.GetProperty("legend").EnumerateObject()) Entry(entry.Value);
                        answers.Add(new ScoreDecisionAnswer(p.Name, keys.Select(k => probabilities.GetProperty(k).GetDouble()),
                            a.GetProperty("score").GetDouble()) { RawRepresentation = a.Clone() });
                        break;
                    default: throw new DecisionProtocolException("Unknown answer kind.");
                }
            }
            return new DecisionResponse(request, model, answers,
                new() { InputTokenCount = input, OutputTokenCount = output, TotalTokenCount = sum })
                { RawRepresentation = root.Clone() };
        }
        catch (Exception e) when (e is JsonException or InvalidOperationException or KeyNotFoundException or
            FormatException or OverflowException or ArgumentException)
        {
            throw new DecisionProtocolException("Malformed native response envelope.");
        }
    }

    private static void Entry(JsonElement value)
    {
        if (value.ValueKind is not (JsonValueKind.String or JsonValueKind.Object or JsonValueKind.Array or JsonValueKind.Null))
            throw new ArgumentException("Jev entries must be strings, objects, arrays or null.");
        UniqueProperties(value);
    }
    private static void Unit(double n)
    {
        if (!double.IsFinite(n) || n is < 0 or > 1) throw new DecisionProtocolException("Invalid unit interval value.");
    }
    private static void UniqueProperties(JsonElement value)
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
            foreach (var child in value.EnumerateArray()) UniqueProperties(child);
    }
}
