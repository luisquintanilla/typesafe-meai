using System.Collections.ObjectModel;
using System.Globalization;
using System.Numerics;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace TypeSafe.MEAI;

/// <summary>A bounded JSON-schema compiler; consumes MEAI schemas, never reflects over CLR types.</summary>
internal static class TypeSafeSchemaCompiler
{
    private static readonly HashSet<string> Keywords = new(StringComparer.Ordinal)
    {
        "type", "properties", "required", "additionalProperties", "enum", "$ref", "$defs",
        "minimum", "maximum", "title", "description", "$comment", "examples", "default", "$schema"
    };

    /// <summary>Compiles finite objects and Boolean/enum/explicitly rubric-bound Score leaves.</summary>
    public static TypeSafeCompiledSchema Compile(JsonElement schema, IReadOnlyDictionary<string, TypeSafeFieldOptions> bindings,
        string? sharedInstructions = null, TypeSafeDecisionPolicy? policy = null, string? schemaDescription = null) =>
        CompileCore(schema, bindings, sharedInstructions, null, null, policy, schemaDescription);

    internal static TypeSafeCompiledSchema CompileCore(JsonElement schema, IReadOnlyDictionary<string, TypeSafeFieldOptions> bindings,
        string? sharedInstructions, IReadOnlyDictionary<string, JsonElement>? fixedValues, IReadOnlySet<string>? omitted,
        TypeSafeDecisionPolicy? policy = null, string? schemaDescription = null)
    {
        schema = schema.Clone();
        Audit(schema, "", 0);
        var compiler = new Compiler(schema, bindings, sharedInstructions, fixedValues, omitted, policy);
        IReadOnlyList<JsonElement> descriptions = schemaDescription is null ? [] :
            [TypeSafeJson.Entry(new { source = "responseFormat", pointer = "", description = schemaDescription })];
        var root = compiler.Visit(schema, "", "", 0, new HashSet<string>(StringComparer.Ordinal), descriptions, required: true);
        if (fixedValues is null && compiler.Fields.Count == 0) throw new TypeSafeSchemaException("", "At least one decision leaf is required.");
        foreach (var path in bindings.Keys.Concat(fixedValues?.Keys ?? []).Concat(omitted ?? new HashSet<string>()))
            if (!compiler.Used.Contains(path)) throw new TypeSafeSchemaException(path, "Binding/fixed/omitted path is unused or conflicting.");
        return new(root, compiler.Fields);
    }

    private static void Audit(JsonElement schema, string path, int depth)
    {
        if (depth > 32) throw new TypeSafeSchemaException(path, "Maximum schema traversal depth is 32.");
        if (schema.ValueKind != JsonValueKind.Object) throw new TypeSafeSchemaException(path, "Only object-form schemas are supported.");
        HashSet<string> seen = new(StringComparer.Ordinal);
        foreach (var p in schema.EnumerateObject())
        {
            if (!seen.Add(p.Name)) throw new TypeSafeSchemaException(path, "Duplicate schema keyword.");
            if (!Keywords.Contains(p.Name)) throw new TypeSafeSchemaException(path + "/" + Escape(p.Name), "Unsupported keyword.");
            if (p.Name is "properties" or "$defs")
            {
                if (p.Value.ValueKind != JsonValueKind.Object) throw new TypeSafeSchemaException(path, $"{p.Name} must be an object.");
                HashSet<string> names = new(StringComparer.Ordinal);
                foreach (var child in p.Value.EnumerateObject())
                {
                    if (!names.Add(child.Name)) throw new TypeSafeSchemaException(path, "Duplicate property/definition.");
                    Audit(child.Value, path + "/" + p.Name + "/" + Escape(child.Name), depth + 1);
                }
            }
            if (p.Name == "$schema" && (p.Value.ValueKind != JsonValueKind.String ||
                p.Value.GetString() != "https://json-schema.org/draft/2020-12/schema"))
                throw new TypeSafeSchemaException(path + "/$schema", "Only JSON Schema 2020-12 is supported.");
            if (p.Name is "title" or "description" or "$comment" && p.Value.ValueKind != JsonValueKind.String)
                throw new TypeSafeSchemaException(path + "/" + p.Name, "Annotation must be a string.");
            if (p.Name == "examples" && p.Value.ValueKind != JsonValueKind.Array)
                throw new TypeSafeSchemaException(path + "/examples", "Examples must be an array.");
        }
    }

    internal static string Escape(string value) => value.Replace("~", "~0", StringComparison.Ordinal).Replace("/", "~1", StringComparison.Ordinal);

    private sealed class Compiler(JsonElement document, IReadOnlyDictionary<string, TypeSafeFieldOptions> bindings,
        string? shared, IReadOnlyDictionary<string, JsonElement>? fixedValues, IReadOnlySet<string>? omitted, TypeSafeDecisionPolicy? policy)
    {
        internal List<TypeSafeCompiledField> Fields { get; } = [];
        internal HashSet<string> Used { get; } = new(StringComparer.Ordinal);

        internal SchemaNode Visit(JsonElement schema, string pointer, string schemaPath, int depth, HashSet<string> references,
            IReadOnlyList<JsonElement> descriptions, bool required)
        {
            void Fail(string reason) => throw new TypeSafeSchemaException(schemaPath, reason);
            if (depth > 32) Fail("Maximum reference/object traversal depth is 32.");
            if (schema.ValueKind != JsonValueKind.Object) Fail("Reference must resolve to an object-form schema.");
            if (schema.TryGetProperty("$ref", out var reference))
            {
                if (reference.ValueKind != JsonValueKind.String) Fail("$ref must be a local reference string.");
                string name = reference.GetString()!;
                if (name != "#" && !name.StartsWith("#/", StringComparison.Ordinal)) Fail("Only local references are supported.");
                if (schema.EnumerateObject().Any(p => p.Name is not ("$ref" or "$defs" or "$schema" or "title" or "description" or "$comment" or "examples" or "default")))
                    Fail("Assertion siblings of $ref are not supported.");
                if (!references.Add(name)) Fail("Recursive reference.");
                var target = document;
                string decoded;
                try { decoded = Uri.UnescapeDataString(name[1..]); }
                catch (UriFormatException) { throw new TypeSafeSchemaException(schemaPath, "Invalid reference encoding."); }
                if (decoded.Length != 0)
                    foreach (var segment in decoded[1..].Split('/'))
                    {
                        for (int i = 0; i < segment.Length; i++)
                            if (segment[i] == '~' && (++i == segment.Length || segment[i] is not ('0' or '1')))
                                Fail("Invalid JSON Pointer escape.");
                        var token = segment.Replace("~1", "/", StringComparison.Ordinal).Replace("~0", "~", StringComparison.Ordinal);
                        if (target.ValueKind != JsonValueKind.Object || !target.TryGetProperty(token, out target))
                            Fail("Unresolved local reference.");
                    }
                if (target.ValueKind != JsonValueKind.Object) Fail("Reference must resolve to an object-form schema.");
                if (schema.TryGetProperty("description", out _) || schema.TryGetProperty("default", out _))
                {
                    var annotated = JsonNode.Parse(target.GetRawText())!.AsObject();
                    foreach (string annotation in new[] { "description", "default" })
                        if (schema.TryGetProperty(annotation, out var value)) annotated[annotation] = JsonNode.Parse(value.GetRawText());
                    target = JsonSerializer.SerializeToElement(annotated);
                }
                var node = Visit(target, pointer, schemaPath + "/$ref", depth + 1, references, descriptions, required);
                references.Remove(name);
                return node;
            }
            if (!schema.TryGetProperty("type", out var t) || t.ValueKind != JsonValueKind.String) Fail("One explicit nonnullable type is required.");
            string type = t.GetString()!;
            if (type is not ("object" or "boolean" or "string" or "number" or "integer")) Fail("Unsupported type.");
            if (type != "object" && schema.EnumerateObject().Any(p => p.Name is "properties" or "required" or "additionalProperties"))
                Fail("Object keywords on a nonobject leaf.");
            if (type != "string" && schema.TryGetProperty("enum", out _)) Fail("Only string enums are supported.");
            if (type is not ("number" or "integer") && (schema.TryGetProperty("minimum", out _) || schema.TryGetProperty("maximum", out _)))
                Fail("Numeric constraints on a nonnumeric field.");
            bindings.TryGetValue(pointer, out var binding);
            bool hasFixed = fixedValues?.ContainsKey(pointer) == true, omit = omitted?.Contains(pointer) == true;
            if ((hasFixed && omit) || ((hasFixed || omit) && binding is not null)) Fail("Conflicting inferred, fixed, or omitted binding.");
            if (hasFixed || omit)
            {
                if (type == "object") Fail("Only scalar fixed/omitted arguments are supported.");
                if (omit && (required || !schema.TryGetProperty("default", out _))) Fail("Omission requires an optional parameter with a declared default.");
                var value = omit ? schema.GetProperty("default") : fixedValues![pointer];
                ValidateScalar(schema, type, value, schemaPath);
                if (type == "integer" && !omit) value = ExactNumber.Parse(value, schemaPath).ToIntegerJson();
                Used.Add(pointer);
                return new SchemaNode(value.Clone(), omit);
            }
            if (!required && fixedValues is not null) Fail("Optional tool arguments require an explicit fixed/omit strategy.");
            if (type == "object")
            {
                if (schema.TryGetProperty("description", out var objectDescription))
                    descriptions = [.. descriptions, TypeSafeJson.Entry(new { source = "schema", pointer, description = objectDescription.GetString() })];
                if (binding is not null) Fail("Bindings must identify leaves, not objects.");
                if (schema.TryGetProperty("additionalProperties", out var closed) && closed.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
                    Fail("Arbitrary dictionary schemas are unsupported; only finite declared properties may be produced.");
                if (!schema.TryGetProperty("properties", out var props) || props.ValueKind != JsonValueKind.Object)
                    Fail("Object properties are required.");
                bool hasRequired = schema.TryGetProperty("required", out var requiredJson);
                if (hasRequired && requiredJson.ValueKind != JsonValueKind.Array) Fail("Required must be an array.");
                HashSet<string> names = new(StringComparer.Ordinal);
                if (hasRequired)
                    foreach (var r in requiredJson.EnumerateArray())
                        if (r.ValueKind != JsonValueKind.String || !names.Add(r.GetString()!)) Fail("Invalid or duplicate required entry.");
                if (names.Any(n => !props.TryGetProperty(n, out _))) Fail("Required property is undeclared.");
                List<KeyValuePair<string, SchemaNode>> children = [];
                foreach (var p in props.EnumerateObject())
                    children.Add(KeyValuePair.Create(p.Name, Visit(p.Value, pointer + "/" + Escape(p.Name),
                        schemaPath + "/properties/" + Escape(p.Name), depth + 1, references, descriptions, fixedValues is null || names.Contains(p.Name))));
                return new SchemaNode(children);
            }
            string? instructions = binding?.Instructions;
            if (instructions is null && schema.TryGetProperty("description", out var description)) instructions = description.GetString();
            instructions ??= $"Determine the value at '{pointer}' using the task and conversation in state.";
            if (string.IsNullOrWhiteSpace(instructions)) Fail("Meaningful instructions or a nonempty schema description are required.");
            var choices = binding?.Choices;
            if (type == "string" && choices is null)
                choices = Labels(schema, schemaPath).ToDictionary(label => label, _ => TypeSafeJson.Entry<object?>(null), StringComparer.Ordinal);
            binding = new TypeSafeFieldOptions(instructions,
                binding?.BooleanPolicy ?? (type == "boolean" ? policy?.Boolean : null),
                binding?.MinimumConfidence ?? (type != "boolean" ? policy?.MinimumConfidence : null),
                choices, binding?.Rubric, binding?.Validate);
            JsonElement instructionValue = descriptions.Count != 0 ?
                TypeSafeJson.Entry(new { sharedInstructions = shared, schemaDescriptions = descriptions, fieldInstructions = instructions }) :
                string.IsNullOrEmpty(shared) ? TypeSafeJson.Entry(instructions) :
                    TypeSafeJson.Entry(new { sharedInstructions = shared, fieldInstructions = instructions });
            if (Fields.Count >= 256) Fail("Maximum decision leaves is 256.");
            string id = "q" + Fields.Count.ToString("D4", System.Globalization.CultureInfo.InvariantCulture);
            TypeSafeQuestion question;
            if (type == "boolean")
            {
                if (binding.BooleanPolicy is null || binding.MinimumConfidence is not null || binding.Choices is not null || binding.Rubric is not null)
                    Fail("Boolean requires only an explicit false/true policy.");
                question = new NoulQuestion(id, instructionValue);
            }
            else if (type == "string")
            {
                string[] labels = Labels(schema, schemaPath);
                if (binding.MinimumConfidence is null || binding.BooleanPolicy is not null || binding.Rubric is not null ||
                    binding.Choices is null || !binding.Choices.Keys.ToHashSet(StringComparer.Ordinal).SetEquals(labels))
                    Fail("Choice requires exact enum descriptions and an explicit confidence threshold.");
                ValidateEntries(binding.Choices!.Values, schemaPath);
                question = new ChoiceQuestion(id, instructionValue, labels.Select(label => KeyValuePair.Create(label, binding.Choices![label])));
            }
            else if (type == "number")
            {
                if (binding.MinimumConfidence is null || binding.BooleanPolicy is not null || binding.Choices is not null ||
                    binding.Rubric is null || binding.Rubric.Count is < 2 or > 10)
                    Fail("Number requires explicit fractional-index Score semantics, 2-10 rubric levels, and a confidence threshold.");
                ValidateScalar(schema, type, JsonSerializer.SerializeToElement(0), schemaPath);
                ValidateScalar(schema, type, JsonSerializer.SerializeToElement(binding.Rubric!.Count - 1), schemaPath);
                ValidateEntries(binding.Rubric, schemaPath);
                question = new ScoreQuestion(id, instructionValue, binding.Rubric);
            }
            else { Fail("Integer Score projection is not supported; host-fixed integers are allowed for tools."); throw new InvalidOperationException(); }
            if (bindings.ContainsKey(pointer)) Used.Add(pointer);
            var field = new TypeSafeCompiledField(pointer, question, binding);
            Fields.Add(field);
            return new SchemaNode(field);
        }
    }

    private static void ValidateEntries(IEnumerable<JsonElement> entries, string path)
    {
        foreach (var entry in entries)
        {
            try { TypeSafeJson.ValidateEntry(entry); }
            catch (ArgumentException error) { throw new TypeSafeSchemaException(path, error.Message); }
        }
    }

    private static string[] Labels(JsonElement schema, string path)
    {
        if (!schema.TryGetProperty("enum", out var e) || e.ValueKind != JsonValueKind.Array)
            throw new TypeSafeSchemaException(path, "Free-form string inference is not supported.");
        if (e.EnumerateArray().Any(v => v.ValueKind != JsonValueKind.String))
            throw new TypeSafeSchemaException(path, "Enum labels must be strings.");
        string[] labels = e.EnumerateArray().Select(v => v.GetString()!).ToArray();
        if (labels.Length is < 1 or > 255 || labels.Any(string.IsNullOrWhiteSpace) || labels.Distinct(StringComparer.Ordinal).Count() != labels.Length)
            throw new TypeSafeSchemaException(path, "Enum requires 1-255 distinct nonempty labels.");
        return labels;
    }

    internal static void ValidateScalar(JsonElement schema, string type, JsonElement value, string path)
    {
        ExactNumber? number = type is "number" or "integer" ? ExactNumber.Parse(value, path) : null;
        bool valid = type switch
        {
            "string" => value.ValueKind == JsonValueKind.String,
            "boolean" => value.ValueKind is JsonValueKind.True or JsonValueKind.False,
            "number" => true,
            "integer" => number!.Value.IsInteger,
            _ => false
        };
        if (!valid) throw new TypeSafeSchemaException(path, "Fixed/default value has the wrong type or is nonfinite.");
        if (schema.TryGetProperty("enum", out _) && !Labels(schema, path).Contains(value.GetString(), StringComparer.Ordinal))
            throw new TypeSafeSchemaException(path, "Value is not an enum member.");
        foreach (string keyword in new[] { "minimum", "maximum" })
        {
            if (!schema.TryGetProperty(keyword, out var constraint)) continue;
            var bound = ExactNumber.Parse(constraint, path + "/" + keyword);
            int comparison = number!.Value.CompareTo(bound);
            if (keyword == "minimum" ? comparison < 0 : comparison > 0)
                throw new TypeSafeSchemaException(path, "The complete Score range or fixed value violates a numeric constraint; clamping is forbidden.");
        }
    }

    // Double is only a finite-range guard; constraints and integer-ness use exact JSON decimal arithmetic.
    private readonly record struct ExactNumber(BigInteger Coefficient, int Exponent)
    {
        internal bool IsInteger => Exponent >= 0 || Coefficient % BigInteger.Pow(10, -Exponent) == 0;

        internal JsonElement ToIntegerJson()
        {
            var integer = Exponent >= 0 ? Coefficient * BigInteger.Pow(10, Exponent) : Coefficient / BigInteger.Pow(10, -Exponent);
            using var document = JsonDocument.Parse(integer.ToString(CultureInfo.InvariantCulture));
            return document.RootElement.Clone();
        }

        internal int CompareTo(ExactNumber other)
        {
            int common = Math.Min(Exponent, other.Exponent);
            return (Coefficient * BigInteger.Pow(10, Exponent - common))
                .CompareTo(other.Coefficient * BigInteger.Pow(10, other.Exponent - common));
        }

        internal static ExactNumber Parse(JsonElement value, string path)
        {
            if (value.ValueKind != JsonValueKind.Number || !value.TryGetDouble(out var finite) || !double.IsFinite(finite))
                throw new TypeSafeSchemaException(path, "Expected a finite numeric value within the supported double range.");
            string raw = value.GetRawText();
            if (raw.Length > 1024) throw new TypeSafeSchemaException(path, "Numeric literal exceeds 1024 characters.");
            int exponent = 0, e = raw.IndexOfAny(['e', 'E']);
            if (e >= 0)
            {
                if (!int.TryParse(raw.AsSpan(e + 1), NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out exponent) ||
                    exponent is < -1024 or > 1024)
                    throw new TypeSafeSchemaException(path, "Decimal exponent must be within [-1024,1024].");
                raw = raw[..e];
            }
            int point = raw.IndexOf('.');
            if (point >= 0)
            {
                exponent -= raw.Length - point - 1;
                raw = raw.Remove(point, 1);
            }
            if (exponent is < -1024 or > 1024)
                throw new TypeSafeSchemaException(path, "Effective decimal exponent must be within [-1024,1024].");
            return new(BigInteger.Parse(raw, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture), exponent);
        }
    }
}

/// <summary>One immutable leaf-to-native-question mapping.</summary>
/// <param name="Pointer">Escaped output JSON Pointer.</param>
/// <param name="Question">Deterministic native question.</param>
/// <param name="Binding">Explicit semantic and acceptance policy.</param>
internal sealed record TypeSafeCompiledField(string Pointer, TypeSafeQuestion Question, TypeSafeFieldOptions Binding);

/// <summary>An immutable compiled schema, reusable across requests.</summary>
internal sealed class TypeSafeCompiledSchema
{
    private readonly SchemaNode _root;
    internal bool IsObjectRoot => _root.IsObject;
    /// <summary>All decision leaves and their collision-free question IDs.</summary>
    public IReadOnlyList<TypeSafeCompiledField> Fields { get; }
    /// <summary>Native inference questions, excluding projection-only thresholds.</summary>
    public IReadOnlyList<TypeSafeQuestion> Questions { get; }
    internal TypeSafeCompiledSchema(SchemaNode root, List<TypeSafeCompiledField> fields)
    {
        _root = root;
        Fields = fields.AsReadOnly();
        Questions = Array.AsReadOnly(fields.Select(f => f.Question).ToArray());
    }
    internal (JsonElement? Value, IReadOnlyList<TypeSafeRejection> Rejections) Project(TypeSafeResponse? evidence)
    {
        List<TypeSafeRejection> rejections = [];
        var result = _root.Project(evidence, rejections);
        return (rejections.Count == 0 ? JsonSerializer.SerializeToElement(result) : null, rejections.AsReadOnly());
    }
}

internal sealed class SchemaNode
{
    private readonly IReadOnlyList<KeyValuePair<string, SchemaNode>>? _children;
    private readonly TypeSafeCompiledField? _field;
    private readonly JsonElement _fixed;
    internal bool IsObject => _children is not null;
    internal bool Omit { get; }
    internal SchemaNode(IEnumerable<KeyValuePair<string, SchemaNode>> children) => _children = children.ToArray();
    internal SchemaNode(TypeSafeCompiledField field) => _field = field;
    internal SchemaNode(JsonElement value, bool omit) { _fixed = value; Omit = omit; }

    internal JsonNode? Project(TypeSafeResponse? evidence, List<TypeSafeRejection> rejected)
    {
        if (_children is not null)
        {
            JsonObject obj = new();
            foreach (var child in _children) if (!child.Value.Omit) obj[child.Key] = child.Value.Project(evidence, rejected);
            return obj;
        }
        if (_field is null) return Omit ? null : JsonNode.Parse(_fixed.GetRawText());
        var f = _field;
        TypeSafeAnswer answer = evidence!.Answers[f.Question.Id];
        JsonNode? value = answer switch
        {
            NoulAnswer n when n.Noul <= f.Binding.BooleanPolicy!.FalseMaximum => JsonValue.Create(false),
            NoulAnswer n when n.Noul >= f.Binding.BooleanPolicy!.TrueMinimum => JsonValue.Create(true),
            ChoiceAnswer c when c.Confidence >= f.Binding.MinimumConfidence => JsonValue.Create(c.Choice),
            ScoreAnswer s when s.Confidence >= f.Binding.MinimumConfidence => JsonValue.Create(s.Score),
            _ => null
        };
        if (value is null) rejected.Add(new(f.Pointer, "Native evidence is inside the abstention interval or below the explicit server-confidence threshold."));
        else if (f.Binding.Validate?.Invoke(JsonSerializer.SerializeToElement(value)) is { } reason)
            rejected.Add(new(f.Pointer, reason));
        return value;
    }
}
