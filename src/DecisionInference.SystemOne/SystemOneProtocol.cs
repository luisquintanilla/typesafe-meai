using System.Globalization;
using System.Collections.ObjectModel;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace DecisionInference.SystemOne;

internal sealed record SystemOneRequest(DecisionInput Input, JsonElement Body,
    IReadOnlyDictionary<string, string> QuestionIds,
    IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>> CandidateIds);

// Native validation/serialization adapted from the frozen bc2f307 protocol, without chat projection.
internal static class SystemOneProtocol
{
    internal static SystemOneRequest Encode(DecisionInput input, string model)
    {
        Entry(input.State);
        var questions = new JsonObject();
        Dictionary<string, string> questionIds = new(StringComparer.Ordinal);
        Dictionary<string, IReadOnlyDictionary<string, string>> candidateIds = new(StringComparer.Ordinal);
        for (int row = 0; row < input.Questions.Count; row++)
        {
            var q = input.Questions[row];
            string wireId = "q" + row.ToString(CultureInfo.InvariantCulture);
            questionIds.Add(wireId, q.Id);
            var body = new JsonObject { ["type"] = q.Kind switch
            {
                DecisionKind.Binary => "noul", DecisionKind.Choice => "choice", DecisionKind.Score => "score",
                _ => throw new NotSupportedException()
            }};
            body["instructions"] = q.Instructions;
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
                    if (c.Candidates.Count > 255) throw new ArgumentException("System One Choice supports at most 255 candidates.");
                    var choices = new JsonObject();
                    Dictionary<string, string> aliases = new(StringComparer.Ordinal);
                    for (int i = 0; i < c.Candidates.Count; i++)
                    {
                        string alias = i.ToString(CultureInfo.InvariantCulture);
                        choices.Add(alias, JsonValue.Create(c.Candidates[i].Description));
                        aliases.Add(alias, c.Candidates[i].Id);
                    }
                    candidateIds.Add(wireId, new ReadOnlyDictionary<string, string>(aliases));
                    body["criteria"] = choices;
                    break;
                case ScoreDecisionQuestion s:
                    if (s.Rubric.Count > 10) throw new ArgumentException("System One Score supports 2 through 10 levels.");
                    body["criteria"] = new JsonArray(s.Rubric.Select(r => (JsonNode?)JsonValue.Create(r)).ToArray());
                    break;
            }
            questions.Add(wireId, body);
        }
        return new(input, JsonSerializer.SerializeToElement(new { state = input.State, model, questions }),
            new ReadOnlyDictionary<string, string>(questionIds),
            new ReadOnlyDictionary<string, IReadOnlyDictionary<string, string>>(candidateIds));
    }

    internal static DecisionResult Decode(SystemOneRequest wire, JsonElement root, DecisionPrecision precision)
    {
        try
        {
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
                if (!wire.QuestionIds.TryGetValue(p.Name, out var questionId))
                    throw new DecisionProtocolException("Unknown wire question ID.");
                var a = p.Value;
                switch (a.GetProperty("type").GetString())
                {
                    case "noul":
                        answers.Add(new BinaryDecisionAnswer(questionId, a.GetProperty("noul").GetDouble(), precision) { RawRepresentation = a.Clone() });
                        break;
                    case "choice":
                        Unit(a.GetProperty("confidence").GetDouble());
                        if (!wire.CandidateIds.TryGetValue(p.Name, out var aliases) ||
                            !aliases.TryGetValue(a.GetProperty("choice").GetString()!, out var selected))
                            throw new DecisionProtocolException("Unknown selected wire candidate.");
                        var reported = a.GetProperty("probabilities");
                        if (!reported.EnumerateObject().Select(p => p.Name).ToHashSet(StringComparer.Ordinal).SetEquals(aliases.Keys))
                            throw new DecisionProtocolException("Incomplete or unexpected wire candidate distribution.");
                        answers.Add(new ChoiceDecisionAnswer(questionId, selected,
                            aliases.Select(p => KeyValuePair.Create(p.Value, reported.GetProperty(p.Key).GetDouble())), precision)
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
                        answers.Add(new ScoreDecisionAnswer(questionId, keys.Select(k => probabilities.GetProperty(k).GetDouble()),
                            a.GetProperty("score").GetDouble(), precision) { RawRepresentation = a.Clone() });
                        break;
                    default: throw new DecisionProtocolException("Unknown answer kind.");
                }
            }
            return new DecisionResult(wire.Input, model, answers,
                new() { InputTokenCount = input, OutputTokenCount = output, TotalTokenCount = sum })
            {
                RawRepresentation = root.Clone(),
                AdditionalProperties = new()
                {
                    ["systemone.questionIds"] = wire.QuestionIds,
                    ["systemone.candidateIds"] = wire.CandidateIds
                }
            };
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
            throw new ArgumentException("System One entries must be strings, objects, arrays or null.");
        UniqueProperties(value);
    }
    private static void Unit(double n)
    {
        if (!double.IsFinite(n) || n is < 0 or > 1) throw new DecisionProtocolException("Invalid unit interval value.");
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
            foreach (var child in value.EnumerateArray()) UniqueProperties(child);
    }
}
