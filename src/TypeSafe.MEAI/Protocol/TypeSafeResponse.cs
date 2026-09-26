using System.Collections.ObjectModel;
using System.Globalization;
using System.Text.Json;

namespace TypeSafe.MEAI;

/// <summary>A successful HTTP response violating the native contract.</summary>
public sealed class TypeSafeProtocolException : Exception
{
    /// <summary>Creates a sanitized protocol failure without response body text.</summary>
    public TypeSafeProtocolException(string message) : base(message) { }
}

/// <summary>A native answer retaining every additive field.</summary>
public abstract record TypeSafeAnswer
{
    /// <summary>Complete owned answer JSON.</summary>
    public JsonElement Json { get; }
    private protected TypeSafeAnswer(JsonElement json) => Json = json.Clone();
}

/// <summary>A validated native Noul answer.</summary>
public sealed record NoulAnswer : TypeSafeAnswer
{
    /// <summary>Unmodified probability of yes.</summary>
    public double Noul => Json.GetProperty("noul").GetDouble();
    internal NoulAnswer(JsonElement json) : base(json) { }
}

/// <summary>A validated native Choice answer.</summary>
public sealed record ChoiceAnswer : TypeSafeAnswer
{
    /// <summary>Server-selected label.</summary>
    public string Choice => Json.GetProperty("choice").GetString()!;
    /// <summary>Server confidence, not the winning probability.</summary>
    public double Confidence => Json.GetProperty("confidence").GetDouble();
    /// <summary>Complete distribution without normalization.</summary>
    public JsonElement Probabilities => Json.GetProperty("probabilities");
    internal ChoiceAnswer(JsonElement json) : base(json) { }
}

/// <summary>A validated native Score answer.</summary>
public sealed record ScoreAnswer : TypeSafeAnswer
{
    /// <summary>Unrounded fractional rubric index.</summary>
    public double Score => Json.GetProperty("score").GetDouble();
    /// <summary>Server confidence.</summary>
    public double Confidence => Json.GetProperty("confidence").GetDouble();
    /// <summary>Complete unmodified distribution.</summary>
    public JsonElement Probabilities => Json.GetProperty("probabilities");
    /// <summary>Original structured legend.</summary>
    public JsonElement Legend => Json.GetProperty("legend");
    internal ScoreAnswer(JsonElement json) : base(json) { }
}

/// <summary>The full native envelope, independent of RawRepresentation.</summary>
internal sealed class TypeSafeResponse
{
    /// <summary>Complete owned native response including unknown additive fields.</summary>
    public JsonElement Json { get; }
    /// <summary>Actual server model.</summary>
    public string Model => Json.GetProperty("model").GetString()!;
    /// <summary>Input tokens reported by the server.</summary>
    public long InputTokens => Json.GetProperty("usage").GetProperty("input_tokens").GetInt64();
    /// <summary>Output tokens reported by the server.</summary>
    public long OutputTokens => Json.GetProperty("usage").GetProperty("output_tokens").GetInt64();
    /// <summary>Native total, or input plus output when absent.</summary>
    public long TotalTokens => Json.GetProperty("usage").TryGetProperty("total_tokens", out var t) ? t.GetInt64() : checked(InputTokens + OutputTokens);
    /// <summary>All validated, case-sensitive answers.</summary>
    public IReadOnlyDictionary<string, TypeSafeAnswer> Answers { get; }

    private TypeSafeResponse(JsonElement json, Dictionary<string, TypeSafeAnswer> answers)
    {
        Json = json.Clone();
        Answers = new ReadOnlyDictionary<string, TypeSafeAnswer>(answers);
    }

    /// <summary>Returns a correlated answer or fails clearly on missing/wrong type.</summary>
    public T GetAnswer<T>(string id) where T : TypeSafeAnswer =>
        Answers.TryGetValue(id, out var value) && value is T typed ? typed :
            throw new TypeSafeProtocolException("Missing answer or mismatched requested answer type.");

    /// <summary>Parses and validates an envelope against the exact submitted questions.</summary>
    public static TypeSafeResponse Parse(string json, IEnumerable<TypeSafeQuestion>? questions = null)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            TypeSafeJson.UniqueProperties(root);
            if (string.IsNullOrWhiteSpace(root.GetProperty("model").GetString())) Fail("Missing model.");
            var usage = root.GetProperty("usage");
            long input = usage.GetProperty("input_tokens").GetInt64(), output = usage.GetProperty("output_tokens").GetInt64();
            if (input < 0 || output < 0) Fail("Invalid token counts.");
            long sum = checked(input + output);
            if (usage.TryGetProperty("total_tokens", out var total) && (total.GetInt64() < 0 || total.GetInt64() != sum))
                Fail("Invalid total token count.");
            var answers = root.GetProperty("answers");
            if (!answers.EnumerateObject().Any()) Fail("Native response has no answers.");
            Dictionary<string, TypeSafeAnswer> result = new(StringComparer.Ordinal);
            foreach (var p in answers.EnumerateObject())
            {
                if (string.IsNullOrWhiteSpace(p.Name)) Fail("Invalid answer ID.");
                var a = p.Value;
                TypeSafeAnswer answer;
                switch (a.GetProperty("type").GetString())
                {
                    case "noul":
                        Unit(a.GetProperty("noul"));
                        answer = new NoulAnswer(a);
                        break;
                    case "choice":
                        Unit(a.GetProperty("confidence"));
                        var probabilities = a.GetProperty("probabilities");
                        var labels = probabilities.EnumerateObject().Select(p => p.Name).ToArray();
                        if (labels.Length is < 1 or > 255 || labels.Any(string.IsNullOrWhiteSpace)) Fail("Invalid Choice labels.");
                        Distribution(probabilities, labels);
                        if (!labels.Contains(a.GetProperty("choice").GetString(), StringComparer.Ordinal)) Fail("Unknown selected label.");
                        answer = new ChoiceAnswer(a);
                        break;
                    case "score":
                        Unit(a.GetProperty("confidence"));
                        int count = a.GetProperty("probabilities").EnumerateObject().Count();
                        if (count is < 2 or > 10) Fail("Invalid Score cardinality.");
                        double value = a.GetProperty("score").GetDouble();
                        if (!double.IsFinite(value) || value < 0 || value > count - 1) Fail("Invalid Score range.");
                        var keys = Enumerable.Range(0, count).Select(i => i.ToString(CultureInfo.InvariantCulture)).ToArray();
                        Distribution(a.GetProperty("probabilities"), keys);
                        var legend = a.GetProperty("legend");
                        if (!legend.EnumerateObject().Select(p => p.Name).ToHashSet(StringComparer.Ordinal).SetEquals(keys))
                            Fail("Invalid Score legend indices.");
                        foreach (var entry in legend.EnumerateObject()) TypeSafeJson.ValidateEntry(entry.Value);
                        answer = new ScoreAnswer(a);
                        break;
                    default: throw new TypeSafeProtocolException("Unsupported question.");
                }
                result.Add(p.Name, answer);
            }
            var response = new TypeSafeResponse(root, result);
            if (questions is not null) response.ValidateCorrelation(questions);
            return response;
        }
        catch (Exception e) when (e is JsonException or InvalidOperationException or KeyNotFoundException or FormatException or OverflowException or ArgumentException)
        {
            throw new TypeSafeProtocolException("Malformed native response envelope.");
        }
    }

    internal void ValidateCorrelation(IEnumerable<TypeSafeQuestion> questions)
    {
        var expected = questions.ToDictionary(q => q.Id, StringComparer.Ordinal);
        if (Answers.Count != expected.Count) Fail("Answer count mismatch.");
        foreach (var pair in Answers)
        {
            if (!expected.TryGetValue(pair.Key, out var question)) Fail("Unknown answer ID.");
            if (pair.Value.Json.GetProperty("type").GetString() != question!.Type) Fail("Answer type mismatch.");
            if (question is ChoiceQuestion choice && pair.Value is ChoiceAnswer c)
                Distribution(c.Probabilities, choice.Criteria.Keys);
            if (question is ScoreQuestion score && pair.Value is ScoreAnswer s)
                Distribution(s.Probabilities, Enumerable.Range(0, score.Criteria.Count).Select(i => i.ToString(CultureInfo.InvariantCulture)));
        }
    }

    private static void Distribution(JsonElement json, IEnumerable<string> expected)
    {
        if (!json.EnumerateObject().Select(p => p.Name).ToHashSet(StringComparer.Ordinal).SetEquals(expected))
            Fail("Distribution labels mismatch.");
        decimal sum = 0;
        foreach (var p in json.EnumerateObject()) { Unit(p.Value); sum += p.Value.GetDecimal(); }
        if (Math.Abs(sum - 1m) > 0.000001m) Fail("Distribution sum outside absolute tolerance 1e-6.");
    }
    private static void Unit(JsonElement json)
    {
        double n = json.GetDouble();
        if (!double.IsFinite(n) || n is < 0 or > 1) Fail("Invalid unit interval value.");
    }
    private static void Fail(string message) => throw new TypeSafeProtocolException(message);
}
