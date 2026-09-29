using System.Collections.ObjectModel;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Apache.Arrow;
using Apache.Arrow.Types;

namespace DecisionInference.Arrow;

public sealed class DecisionArrowIdentity
{
    public string Provider { get; }
    public string ModelId { get; }
    public IReadOnlyDictionary<string, string> AssetHashes { get; }
    public string Preprocessing { get; }
    public string Decoder { get; }
    public IReadOnlyList<string> FeatureProjection { get; }

    public DecisionArrowIdentity(string provider, string modelId, IEnumerable<KeyValuePair<string, string>> assetHashes,
        string preprocessing, string decoder, IEnumerable<string> featureProjection)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(provider);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        ArgumentException.ThrowIfNullOrWhiteSpace(preprocessing);
        ArgumentException.ThrowIfNullOrWhiteSpace(decoder);
        ArgumentNullException.ThrowIfNull(assetHashes);
        ArgumentNullException.ThrowIfNull(featureProjection);
        var hashes = new SortedDictionary<string, string>(StringComparer.Ordinal);
        foreach (var item in assetHashes)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(item.Key);
            if (!IsHash(item.Value)) throw new ArgumentException("Asset hashes must be lowercase SHA-256.");
            hashes.Add(item.Key, item.Value);
        }
        if (hashes.Count == 0) throw new ArgumentException("Explicit asset identities are required.");
        var projection = featureProjection.ToArray();
        if (projection.Any(string.IsNullOrWhiteSpace) || projection.Distinct(StringComparer.Ordinal).Count() != projection.Length)
            throw new ArgumentException("Feature projection must contain unique, explicit coordinates.");
        Provider = provider;
        ModelId = modelId;
        AssetHashes = new ReadOnlyDictionary<string, string>(hashes);
        Preprocessing = preprocessing;
        Decoder = decoder;
        FeatureProjection = System.Array.AsReadOnly(projection);
    }

    internal static bool IsHash(string? value) => value is { Length: 64 } && value.All(c => c is >= '0' and <= '9' or >= 'a' and <= 'f');
}

/// <summary>Immutable semantic contract; package versions are provenance, not feature semantics.</summary>
public sealed class DecisionArrowSchema
{
    public const int Version = 1;
    public const string MetadataPrefix = "decision_arrow.";
    public IReadOnlyList<DecisionQuestion> Questions { get; }
    public DecisionPrecision Precision { get; }
    public DecisionArrowIdentity Identity { get; }
    public string CanonicalJson { get; }
    public string FeatureFingerprint { get; }
    public Schema Schema => BuildSchema();

    public DecisionArrowSchema(IEnumerable<DecisionQuestion> questions, DecisionPrecision precision, DecisionArrowIdentity identity)
    {
        ArgumentNullException.ThrowIfNull(questions);
        ArgumentNullException.ThrowIfNull(identity);
        if (!Enum.IsDefined(precision)) throw new ArgumentOutOfRangeException(nameof(precision));
        var copy = questions.ToArray();
        if (copy.Length == 0 || copy.Any(q => q is null || q.Id == "row_id") ||
            copy.Select(q => q.Id).Distinct(StringComparer.Ordinal).Count() != copy.Length)
            throw new ArgumentException("Questions must be nonempty, unique and must not use reserved row_id.");
        Questions = System.Array.AsReadOnly(copy);
        Precision = precision;
        Identity = identity;
        var validFeatures = copy.SelectMany(q => q switch
        {
            BinaryDecisionQuestion => [q.Id],
            ChoiceDecisionQuestion c => Enumerable.Range(0, c.Candidates.Count).Select(i => $"{q.Id}[{i}]"),
            ScoreDecisionQuestion s => Enumerable.Range(0, s.Rubric.Count).Select(i => $"{q.Id}[{i}]"),
            _ => throw new NotSupportedException("Unknown question kind.")
        }).ToHashSet(StringComparer.Ordinal);
        if (identity.FeatureProjection.Any(p => !validFeatures.Contains(p)))
            throw new ArgumentException("Feature projection refers to an unknown observation.");
        CanonicalJson = Json(writer =>
        {
            writer.WriteStartObject();
            writer.WriteNumber("schemaVersion", Version);
            writer.WriteString("numericRepresentation", "float64");
            writer.WriteString("precision", precision.ToString());
            writer.WritePropertyName("questions");
            writer.WriteStartArray();
            foreach (var q in copy) WriteQuestion(writer, q);
            writer.WriteEndArray();
            writer.WriteString("provider", identity.Provider);
            writer.WriteString("modelId", identity.ModelId);
            writer.WriteStartObject("assetHashes");
            foreach (var p in identity.AssetHashes) writer.WriteString(p.Key, p.Value);
            writer.WriteEndObject();
            writer.WriteString("preprocessing", identity.Preprocessing);
            writer.WriteString("decoder", identity.Decoder);
            writer.WriteStartArray("featureProjection");
            foreach (var p in identity.FeatureProjection) writer.WriteStringValue(p);
            writer.WriteEndArray();
            writer.WriteEndObject();
        });
        FeatureFingerprint = Hash(Encoding.UTF8.GetBytes(CanonicalJson));
    }

    public static DecisionArrowSchema FromCanonicalJson(string json)
    {
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;
        if (root.GetProperty("schemaVersion").GetInt32() != Version ||
            root.GetProperty("numericRepresentation").GetString() != "float64")
            throw new InvalidDataException("Unsupported decision Arrow schema version or numeric representation.");
        var result = new DecisionArrowSchema(ParseQuestions(root.GetProperty("questions")),
            Enum.Parse<DecisionPrecision>(root.GetProperty("precision").GetString()!, ignoreCase: false),
            new(root.GetProperty("provider").GetString()!, root.GetProperty("modelId").GetString()!,
                root.GetProperty("assetHashes").EnumerateObject().Select(p => KeyValuePair.Create(p.Name, p.Value.GetString()!)),
                root.GetProperty("preprocessing").GetString()!, root.GetProperty("decoder").GetString()!,
                root.GetProperty("featureProjection").EnumerateArray().Select(p => p.GetString()!)));
        if (!string.Equals(result.CanonicalJson, json, StringComparison.Ordinal))
            throw new InvalidDataException("Contract JSON is not canonical.");
        return result;
    }

    public static IReadOnlyList<DecisionQuestion> ParseQuestions(JsonElement questions) =>
        System.Array.AsReadOnly(questions.EnumerateArray().Select(q =>
        {
            string id = q.GetProperty("id").GetString()!, instructions = q.GetProperty("instructions").GetString()!;
            return q.GetProperty("kind").GetString() switch
            {
                "binary" => (DecisionQuestion)new BinaryDecisionQuestion(id, instructions,
                    q.TryGetProperty("trueDescription", out var t) ? t.GetString() : null,
                    q.TryGetProperty("falseDescription", out var f) ? f.GetString() : null),
                "choice" => new ChoiceDecisionQuestion(id, instructions, q.GetProperty("candidates").EnumerateArray()
                    .Select(c => new DecisionCandidate(c.GetProperty("id").GetString()!, c.GetProperty("description").GetString()!))),
                "score" => new ScoreDecisionQuestion(id, instructions, q.GetProperty("rubric").EnumerateArray().Select(r => r.GetString()!)),
                _ => throw new InvalidDataException("Unsupported question kind.")
            };
        }).ToArray());

    internal static string QuestionJson(DecisionQuestion question) => Json(w => WriteQuestion(w, question));
    internal static string Hash(ReadOnlySpan<byte> bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));
    private static string Json(Action<Utf8JsonWriter> write)
    {
        using var stream = new MemoryStream();
        using (var writer = new Utf8JsonWriter(stream)) write(writer);
        return Encoding.UTF8.GetString(stream.ToArray());
    }

    private static void WriteQuestion(Utf8JsonWriter writer, DecisionQuestion question)
    {
        writer.WriteStartObject();
        writer.WriteString("id", question.Id);
        writer.WriteString("kind", question.Kind.ToString().ToLowerInvariant());
        writer.WriteString("instructions", question.Instructions);
        switch (question)
        {
            case BinaryDecisionQuestion b:
                writer.WriteString("trueDescription", b.TrueDescription);
                writer.WriteString("falseDescription", b.FalseDescription);
                break;
            case ChoiceDecisionQuestion c:
                writer.WriteStartArray("candidates");
                foreach (var candidate in c.Candidates)
                {
                    writer.WriteStartObject();
                    writer.WriteString("id", candidate.Id);
                    writer.WriteString("description", candidate.Description);
                    writer.WriteEndObject();
                }
                writer.WriteEndArray();
                break;
            case ScoreDecisionQuestion s:
                writer.WriteStartArray("rubric");
                foreach (var level in s.Rubric) writer.WriteStringValue(level);
                writer.WriteEndArray();
                break;
            default: throw new NotSupportedException("Unknown question kind.");
        }
        writer.WriteEndObject();
    }

    private Schema BuildSchema()
    {
        Field Probabilities(int width) => new("probabilities",
            new FixedSizeListType(new Field("item", DoubleType.Default, false), width), false);
        var fields = new List<Field> { new("row_id", Int64Type.Default, false) };
        foreach (var q in Questions)
        {
            IArrowType type = q switch
            {
                BinaryDecisionQuestion => DoubleType.Default,
                ChoiceDecisionQuestion c => new StructType([new("selected_index", Int32Type.Default, false), Probabilities(c.Candidates.Count)]),
                ScoreDecisionQuestion s => new StructType([
                    new("expected_index", DoubleType.Default, false),
                    new("expected_index_lower_bound", DoubleType.Default, false),
                    new("expected_index_upper_bound", DoubleType.Default, false),
                    new("native_score", DoubleType.Default, true), Probabilities(s.Rubric.Count)]),
                _ => throw new NotSupportedException()
            };
            fields.Add(new(q.Id, type, false, new Dictionary<string, string>(StringComparer.Ordinal)
            {
                [MetadataPrefix + "definition"] = QuestionJson(q),
                [MetadataPrefix + "precision"] = Precision.ToString()
            }));
        }
        return new(fields, new Dictionary<string, string>(StringComparer.Ordinal)
        {
            [MetadataPrefix + "version"] = Version.ToString(System.Globalization.CultureInfo.InvariantCulture),
            [MetadataPrefix + "contract"] = CanonicalJson,
            [MetadataPrefix + "feature_fingerprint"] = FeatureFingerprint
        });
    }
}
