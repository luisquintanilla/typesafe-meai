using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.AI;
using TypeSafe.MEAI.Tests.Fixtures;
using Xunit;

namespace TypeSafe.MEAI.Tests.Schema;

[Trait("Stage", "C")]
public sealed class TypeSafeSchemaCompilerTests
{
    [Fact]
    public void Compile_GeneratedRequiredNestedSchemaHasStableNativeMapping()
    {
        var bindings = SchemaFixture.Bindings();
        var schema = SchemaFixture.Schema;
        var first = TypeSafeSchemaCompiler.Compile(schema, bindings);
        var second = TypeSafeSchemaCompiler.Compile(schema, bindings);
        bindings.Clear();
        Assert.False(schema.GetProperty("additionalProperties").GetBoolean());
        Assert.Equal(["followup", "details"], schema.GetProperty("required").EnumerateArray().Select(v => v.GetString()));
        Assert.Equal(["/followup", "/details/destination", "/details/quality"], first.Fields.Select(f => f.Pointer));
        Assert.Equal(["q0000", "q0001", "q0002"], first.Questions.Select(q => q.Id));
        Assert.Equal(["noul", "choice", "score"], first.Questions.Select(q => q.Type));
        Assert.Equal(["Billing", "Technical"], Assert.IsType<ChoiceQuestion>(first.Questions[1]).Criteria.Keys);
        Assert.Equal(3, Assert.IsType<ScoreQuestion>(first.Questions[2]).Criteria.Count);
        for (int i = 0; i < first.Fields.Count; i++)
        {
            Assert.Equal(first.Fields[i].Pointer, second.Fields[i].Pointer);
            Assert.True(JsonElement.DeepEquals(first.Questions[i].ToJson(), second.Questions[i].ToJson()));
            Assert.False(string.IsNullOrWhiteSpace(first.Questions[i].Instructions.GetString()));
        }
    }

    [Fact]
    public void Compile_EscapedPointersAndSharedLocalReferencesRemainDistinct()
    {
        var schema = NativeFixture.Json("""
            {"type":"object","additionalProperties":false,
             "$defs":{"flag /~":{"type":"boolean","description":"Assess this synthetic flag."}},
             "properties":{"a/b":{"$ref":"#/$defs/flag%20~1~0"},"a":{"type":"object","additionalProperties":false,
               "properties":{"b":{"$ref":"#/$defs/flag%20~1~0"}},"required":["b"]},"a~b":{"type":"boolean"},"~1":{"type":"boolean"}},
             "required":["a/b","a","a~b","~1"]}
            """);
        string[] pointers = ["/a~1b", "/a/b", "/a~0b", "/~01"];
        var bindings = pointers.ToDictionary(p => p, p => SchemaFixture.Boolean(p is "/a~1b" or "/a/b" ? null : "Assess this synthetic flag."));
        var plan = TypeSafeSchemaCompiler.Compile(schema, bindings);
        Assert.Equal(pointers, plan.Fields.Select(f => f.Pointer));
        Assert.Equal(["q0000", "q0001", "q0002", "q0003"], plan.Questions.Select(q => q.Id));
        Assert.Equal(4, plan.Questions.Select(q => q.Id).Distinct(StringComparer.Ordinal).Count());
        Assert.All(plan.Questions, q => Assert.Equal("Assess this synthetic flag.", q.Instructions.GetString()));
        Assert.All(plan.Questions, q => Assert.IsType<NoulQuestion>(q));
    }

    public static TheoryData<string> UnsupportedKeywords => new(
        "$id", "$anchor", "$dynamicRef", "$dynamicAnchor", "definitions", "allOf", "anyOf", "oneOf", "not",
        "if", "then", "else", "dependentSchemas", "dependentRequired", "dependencies", "patternProperties",
        "propertyNames", "unevaluatedProperties", "unevaluatedItems", "items", "prefixItems", "additionalItems",
        "contains", "minContains", "maxContains", "minItems", "maxItems", "uniqueItems", "minProperties",
        "maxProperties", "minLength", "maxLength", "pattern", "format", "multipleOf", "exclusiveMinimum",
        "exclusiveMaximum", "const", "readOnly", "writeOnly", "deprecated", "contentEncoding", "contentMediaType",
        "contentSchema", "futureConstraint");

    [Theory]
    [MemberData(nameof(UnsupportedKeywords))]
    public async Task Schema_RejectsEveryUnlistedKeywordBeforeDecoratedHttp(string keyword)
    {
        string value = keyword switch
        {
            "allOf" or "anyOf" or "oneOf" or "prefixItems" => """[{"type":"boolean"}]""",
            "dependentSchemas" or "patternProperties" or "definitions" => """{"synthetic":{"type":"boolean"}}""",
            "dependentRequired" or "dependencies" => """{"synthetic":["other"]}""",
            "not" or "if" or "then" or "else" or "propertyNames" or "items" or "additionalItems" or
                "contains" or "contentSchema" or "unevaluatedProperties" or "unevaluatedItems" => """{"type":"boolean"}""",
            "minContains" or "maxContains" or "minItems" or "maxItems" or "minProperties" or "maxProperties" or
                "minLength" or "maxLength" or "multipleOf" or "exclusiveMinimum" or "exclusiveMaximum" => "1",
            "$id" => "\"https://example.invalid/schema\"",
            "$anchor" or "$dynamicAnchor" => "\"synthetic\"",
            "$dynamicRef" => "\"#synthetic\"",
            "format" => "\"email\"",
            "pattern" => "\"^[A-Z]+$\"",
            "contentEncoding" => "\"base64\"",
            "contentMediaType" => "\"application/json\"",
            _ => "true"
        };
        var leaf = new JsonObject { ["type"] = "boolean", [keyword] = JsonNode.Parse(value) };
        await AssertRejected(SchemaFixture.ObjectSchema(leaf.ToJsonString()), new() { ["/value"] = SchemaFixture.Boolean() },
            "/properties/value/" + keyword);
    }

    public static TheoryData<string, string, string> InvalidShapes => new()
    {
        { """{"type":"array"}""", "/properties/value", "bool" },
        { """{"type":"null"}""", "/properties/value", "bool" },
        { """{"type":["boolean","null"]}""", "/properties/value", "bool" },
        { """{}""", "/properties/value", "bool" },
        { """{"type":true}""", "/properties/value", "bool" },
        { """{"type":"integer"}""", "/properties/value", "score" },
        { """{"type":"string"}""", "/properties/value", "choice" },
        { """{"type":"string","enum":[1,"Billing"]}""", "/properties/value", "choice" },
        { """{"type":"string","enum":["Billing","Billing"]}""", "/properties/value", "choice" },
        { """{"type":"string","enum":[""]}""", "/properties/value", "choice" },
        { """{"type":"string","enum":{}}""", "/properties/value", "choice" },
        { """{"type":"number"}""", "/properties/value", "missing" },
        { """{"type":"boolean","enum":[true,false]}""", "/properties/value", "bool" },
        { """{"type":"boolean","minimum":0}""", "/properties/value", "bool" },
        { """{"type":"boolean","properties":{}}""", "/properties/value", "bool" },
        { """{"type":"boolean","required":[]}""", "/properties/value", "bool" },
        { """{"type":"boolean","additionalProperties":false}""", "/properties/value", "bool" },
        { """{"type":"boolean","$schema":"https://example.invalid/dialect"}""", "/properties/value/$schema", "bool" },
        { """{"type":"boolean","$schema":true}""", "/properties/value/$schema", "bool" },
        { """{"type":"boolean","title":1}""", "/properties/value/title", "bool" },
        { """{"type":"boolean","description":false}""", "/properties/value/description", "bool" },
        { """{"type":"boolean","$comment":{}}""", "/properties/value/$comment", "bool" },
        { """{"type":"boolean","examples":{}}""", "/properties/value/examples", "bool" },
        { """{"type":"number","minimum":0.01}""", "/properties/value", "score" },
        { """{"type":"number","maximum":1.99}""", "/properties/value", "score" },
        { """{"type":"number","minimum":"0"}""", "/properties/value/minimum", "score" },
        { """{"type":"number","maximum":1e999}""", "/properties/value/maximum", "score" },
        { """{"type":"boolean","type":"boolean"}""", "/properties/value", "bool" },
        { "false", "/properties/value", "bool" }
    };

    [Theory]
    [MemberData(nameof(InvalidShapes))]
    public async Task Schema_RejectsInvalidShapesAnnotationsAndConstraintsBeforeHttp(string leaf, string path, string kind)
    {
        Dictionary<string, TypeSafeFieldBinding> bindings = kind == "missing" ? [] :
            new() { ["/value"] = kind switch { "score" => SchemaFixture.Score(), "choice" => SchemaFixture.Choice(), _ => SchemaFixture.Boolean() } };
        await AssertRejected(SchemaFixture.ObjectSchema(leaf), bindings, path);
    }

    public static TheoryData<string, string> InvalidObjects => new()
    {
        { """{"type":"object","properties":{},"required":[]}""", "" },
        { """{"type":"object","additionalProperties":true,"properties":{},"required":[]}""", "" },
        { """{"type":"object","additionalProperties":{},"properties":{},"required":[]}""", "" },
        { """{"type":"object","additionalProperties":false,"properties":{"value":{"type":"boolean"}},"required":[]}""", "/properties/value" },
        { """{"type":"object","additionalProperties":false,"properties":{"value":{"type":"boolean"}}}""", "" },
        { """{"type":"object","additionalProperties":false,"properties":{"value":{"type":"boolean"}},"required":["value","value"]}""", "" },
        { """{"type":"object","additionalProperties":false,"properties":{"value":{"type":"boolean"}},"required":[1]}""", "" },
        { """{"type":"object","additionalProperties":false,"properties":{},"required":["missing"]}""", "" },
        { """{"type":"object","additionalProperties":false,"properties":[],"required":[]}""", "" },
        { """{"type":"object","additionalProperties":false,"properties":{"value":{"type":"boolean"},"value":{"type":"boolean"}},"required":["value"]}""", "" },
        { """{"type":"boolean","$defs":[]}""", "" }
    };

    [Theory]
    [MemberData(nameof(InvalidObjects))]
    public Task Schema_RejectsOpenOptionalAndMalformedObjectsBeforeHttp(string schema, string path) =>
        AssertRejected(NativeFixture.Json(schema), new() { ["/value"] = SchemaFixture.Boolean() }, path);

    [Theory]
    [InlineData("https://example.invalid/schema", false, false, "/properties/value")]
    [InlineData("#/$defs/missing", false, false, "/properties/value")]
    [InlineData("#/$defs/bad~2escape", false, false, "/properties/value")]
    [InlineData("#/$defs/loop", true, false, "/properties/value/$ref")]
    [InlineData("#/$defs/flag", false, true, "/properties/value")]
    [InlineData(null, false, false, "/properties/value")]
    public async Task Schema_RejectsExternalUnresolvedCyclicAndConflictingReferences(string? reference, bool cycle, bool sibling, string path)
    {
        JsonObject leaf = new() { ["$ref"] = reference };
        if (sibling) leaf["type"] = "boolean";
        var schema = JsonNode.Parse(SchemaFixture.ObjectSchema(leaf.ToJsonString()).GetRawText())!.AsObject();
        schema["$defs"] = new JsonObject
        {
            ["flag"] = new JsonObject { ["type"] = "boolean" },
            ["loop"] = cycle ? new JsonObject { ["$ref"] = "#/$defs/loop" } : new JsonObject { ["type"] = "boolean" }
        };
        await AssertRejected(JsonSerializer.SerializeToElement(schema), new() { ["/value"] = SchemaFixture.Boolean() }, path);
    }

    [Fact]
    public void Compile_AllowsInertAnnotationsDescriptionFallbackAndWholeScoreRange()
    {
        var schema = SchemaFixture.ObjectSchema("""
            {"type":"boolean","description":"Assess the described synthetic flag.","title":"Flag","$comment":"inert",
             "examples":[true,false],"default":true,"$schema":"https://json-schema.org/draft/2020-12/schema"}
            """);
        var fallback = TypeSafeSchemaCompiler.Compile(schema, new Dictionary<string, TypeSafeFieldBinding> { ["/value"] = SchemaFixture.Boolean(null) });
        Assert.Equal("Assess the described synthetic flag.", Assert.Single(fallback.Questions).Instructions.GetString());
        var explicitPlan = TypeSafeSchemaCompiler.Compile(schema,
            new Dictionary<string, TypeSafeFieldBinding> { ["/value"] = SchemaFixture.Boolean("Explicit approved predicate.") }, "Shared synthetic policy.");
        var instructions = Assert.Single(explicitPlan.Questions).Instructions;
        Assert.Equal("Shared synthetic policy.", instructions.GetProperty("sharedInstructions").GetString());
        Assert.Equal("Explicit approved predicate.", instructions.GetProperty("fieldInstructions").GetString());
        foreach (var constraints in new[] { """{"minimum":0,"maximum":2}""", """{"minimum":-1,"maximum":3}""" })
        {
            var leaf = JsonNode.Parse(constraints)!.AsObject();
            leaf["type"] = "number";
            var score = TypeSafeSchemaCompiler.Compile(SchemaFixture.ObjectSchema(leaf.ToJsonString()),
                new Dictionary<string, TypeSafeFieldBinding> { ["/value"] = SchemaFixture.Score() });
            Assert.Equal(3, Assert.IsType<ScoreQuestion>(Assert.Single(score.Questions)).Criteria.Count);
        }
    }

    [Theory]
    [InlineData("absent")]
    [InlineData("empty")]
    [InlineData("whitespace")]
    [InlineData("description-empty")]
    [InlineData("object-binding")]
    [InlineData("unused")]
    [InlineData("boolean-conflict")]
    [InlineData("boolean-missing-policy")]
    [InlineData("choice-missing-threshold")]
    [InlineData("choice-missing-descriptions")]
    [InlineData("choice-mismatched-labels")]
    [InlineData("choice-conflict")]
    [InlineData("score-conflict")]
    [InlineData("score-missing-threshold")]
    [InlineData("score-missing-rubric")]
    [InlineData("invalid-choice-description")]
    [InlineData("invalid-score-level")]
    public async Task Schema_RejectsMissingSemanticsAndConflictingBindingsBeforeHttp(string invalid)
    {
        string leaf = """{"type":"boolean"}""", path = "/properties/value";
        var binding = SchemaFixture.Boolean();
        switch (invalid)
        {
            case "absent": binding = SchemaFixture.Boolean(null); break;
            case "empty": binding = SchemaFixture.Boolean(""); break;
            case "whitespace": binding = SchemaFixture.Boolean(" "); break;
            case "description-empty": leaf = """{"type":"boolean","description":""}"""; binding = SchemaFixture.Boolean(null); break;
            case "boolean-conflict": binding = new("Assess flag.", new(0.2, 0.8), minimumConfidence: 0.8); break;
            case "boolean-missing-policy": binding = new("Assess flag."); break;
            case "choice-missing-threshold": case "choice-missing-descriptions": case "choice-mismatched-labels": case "choice-conflict": case "invalid-choice-description":
                leaf = """{"type":"string","enum":["Billing","Technical"]}""";
                var choices = SchemaFixture.Choice().Choices!;
                binding = invalid switch
                {
                    "choice-missing-threshold" => new("Select team.", choices: choices),
                    "choice-missing-descriptions" => new("Select team.", minimumConfidence: 0.8),
                    "choice-mismatched-labels" => SchemaFixture.Choice(labels: ["Billing"]),
                    "choice-conflict" => new("Select team.", booleanPolicy: new(0.2, 0.8), minimumConfidence: 0.8, choices: choices),
                    _ => new("Select team.", minimumConfidence: 0.8, choices: choices.Select(p =>
                        KeyValuePair.Create(p.Key, JsonSerializer.SerializeToElement(17))))
                };
                break;
            case "score-conflict": case "score-missing-threshold": case "score-missing-rubric": case "invalid-score-level":
                leaf = """{"type":"number"}""";
                binding = invalid switch
                {
                    "score-conflict" => new("Assess quality.", booleanPolicy: new(0.2, 0.8), minimumConfidence: 0.8, rubric: SchemaFixture.Score().Rubric),
                    "score-missing-threshold" => new("Assess quality.", rubric: SchemaFixture.Score().Rubric),
                    "score-missing-rubric" => new("Assess quality.", minimumConfidence: 0.8),
                    _ => SchemaFixture.Score(rubric: [JsonSerializer.SerializeToElement(true), TypeSafeJson.Entry("high")])
                };
                break;
        }
        Dictionary<string, TypeSafeFieldBinding> bindings = new() { ["/value"] = binding };
        if (invalid == "object-binding") { bindings[""] = SchemaFixture.Boolean(); path = ""; }
        if (invalid == "unused") { bindings["/unused"] = SchemaFixture.Boolean(); path = "/unused"; }
        await AssertRejected(SchemaFixture.ObjectSchema(leaf), bindings, path);
    }

    [Theory]
    [InlineData("choice", 0, false)]
    [InlineData("choice", 1, true)]
    [InlineData("choice", 255, true)]
    [InlineData("choice", 256, false)]
    [InlineData("score", 1, false)]
    [InlineData("score", 2, true)]
    [InlineData("score", 10, true)]
    [InlineData("score", 11, false)]
    public async Task Compile_EnforcesExactNativeCardinalityBounds(string type, int count, bool accepted)
    {
        var labels = Enumerable.Range(0, count).Select(i => "label-" + i).ToArray();
        var leaf = type == "choice"
            ? JsonSerializer.Serialize(new { type = "string", @enum = labels }) : """{"type":"number"}""";
        var bindings = new Dictionary<string, TypeSafeFieldBinding>
        {
            ["/value"] = type == "choice" ? SchemaFixture.Choice(labels: labels) : SchemaFixture.Score(rubric: labels.Select(TypeSafeJson.Entry))
        };
        var schema = SchemaFixture.ObjectSchema(leaf);
        if (!accepted) await AssertRejected(schema, bindings, "/properties/value");
        else
        {
            var question = Assert.Single(TypeSafeSchemaCompiler.Compile(schema, bindings).Questions);
            Assert.Equal(type, question.Type);
            Assert.Equal(count, type == "choice" ? Assert.IsType<ChoiceQuestion>(question).Criteria.Count : Assert.IsType<ScoreQuestion>(question).Criteria.Count);
        }
    }

    [Theory]
    [InlineData(32, true)]
    [InlineData(33, false)]
    public async Task Compile_EnforcesExactObjectTraversalDepth(int depth, bool accepted)
    {
        string json = """{"type":"boolean"}""";
        for (int i = 0; i < depth; i++)
            json = """{"type":"object","additionalProperties":false,"properties":{"n":LEAF},"required":["n"]}""".Replace("LEAF", json, StringComparison.Ordinal);
        using var document = JsonDocument.Parse(json, new JsonDocumentOptions { MaxDepth = 128 });
        string pointer = string.Concat(Enumerable.Repeat("/n", depth));
        var bindings = new Dictionary<string, TypeSafeFieldBinding> { [pointer] = SchemaFixture.Boolean() };
        if (!accepted) await AssertRejected(document.RootElement, bindings, string.Concat(Enumerable.Repeat("/properties/n", 33)));
        else
        {
            var field = Assert.Single(TypeSafeSchemaCompiler.Compile(document.RootElement, bindings).Fields);
            Assert.Equal(pointer, field.Pointer);
            Assert.Equal("q0000", field.Question.Id);
        }
    }

    [Theory]
    [InlineData(0, false)]
    [InlineData(256, true)]
    [InlineData(257, false)]
    public async Task Compile_EnforcesExactDecisionLeafLimit(int count, bool accepted)
    {
        var properties = Enumerable.Range(0, count).ToDictionary(i => $"p{i:D3}", _ => new { type = "boolean" });
        var schema = JsonSerializer.SerializeToElement(new { type = "object", additionalProperties = false, properties, required = properties.Keys.ToArray() });
        var bindings = properties.Keys.ToDictionary(name => "/" + name, _ => SchemaFixture.Boolean());
        if (!accepted) await AssertRejected(schema, bindings, count == 0 ? "" : "/properties/p256");
        else
        {
            var plan = TypeSafeSchemaCompiler.Compile(schema, bindings);
            Assert.Equal(256, plan.Fields.Count);
            Assert.Equal(256, plan.Questions.Select(q => q.Id).Distinct(StringComparer.Ordinal).Count());
            Assert.Equal("q0255", plan.Questions[^1].Id);
            Assert.Equal("/p255", plan.Fields[^1].Pointer);
        }
    }

    [Theory]
    [InlineData(32, true)]
    [InlineData(33, false)]
    public async Task Compile_EnforcesExactReferenceTraversalDepth(int depth, bool accepted)
    {
        JsonObject definitions = new();
        for (int i = 0; i < depth; i++)
            definitions["d" + i] = i == depth - 1 ? new JsonObject { ["type"] = "boolean" } :
                new JsonObject { ["$ref"] = "#/$defs/d" + (i + 1) };
        var schema = JsonSerializer.SerializeToElement(new JsonObject { ["$ref"] = "#/$defs/d0", ["$defs"] = definitions });
        var bindings = new Dictionary<string, TypeSafeFieldBinding> { [""] = SchemaFixture.Boolean() };
        if (!accepted) await AssertRejected(schema, bindings, string.Concat(Enumerable.Repeat("/$ref", 33)));
        else
        {
            var field = Assert.Single(TypeSafeSchemaCompiler.Compile(schema, bindings).Fields);
            Assert.Equal("", field.Pointer);
            Assert.IsType<NoulQuestion>(field.Question);
        }
    }

    [Fact]
    public async Task Schema_ReferenceDescriptionsOverlayThroughChainsWithoutMutatingDefinitions()
    {
        var schema = NativeFixture.Json("""
            {"type":"object","additionalProperties":false,
             "$defs":{"flag":{"type":"boolean","description":"Definition predicate."},
                      "alias":{"$ref":"#/$defs/flag","description":"Alias predicate."}},
             "properties":{"left":{"$ref":"#/$defs/alias","description":"Left sibling predicate."},
                           "right":{"$ref":"#/$defs/alias","description":"Right sibling predicate."},
                           "inherited":{"$ref":"#/$defs/alias"}},
             "required":["left","right","inherited"]}
            """);
        var bindings = new Dictionary<string, TypeSafeFieldBinding>
        {
            ["/left"] = SchemaFixture.Boolean(null),
            ["/right"] = SchemaFixture.Boolean(null),
            ["/inherited"] = SchemaFixture.Boolean(null)
        };
        using var handler = SchemaFixture.Handler();
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        using var adapter = new TypeSafeSchemaChatClient(spy, bindings);

        var response = await adapter.GetResponseAsync([SchemaFixture.State.ToMessage()], SchemaFixture.Options(schema),
            TestContext.Current.CancellationToken);

        var questions = NativeFixture.Json(Assert.Single(handler.Requests).Body!).GetProperty("questions");
        Assert.Equal(["Left sibling predicate.", "Right sibling predicate.", "Alias predicate."],
            questions.EnumerateObject().Select(q => q.Value.GetProperty("instructions").GetString()));
        var value = NativeFixture.Json(response.Text);
        Assert.Equal(["left", "right", "inherited"], value.EnumerateObject().Select(p => p.Name));
        Assert.All(value.EnumerateObject(), p => Assert.True(p.Value.GetBoolean()));
        Assert.Equal("Definition predicate.", schema.GetProperty("$defs").GetProperty("flag").GetProperty("description").GetString());
        Assert.Equal("Alias predicate.", schema.GetProperty("$defs").GetProperty("alias").GetProperty("description").GetString());
        Assert.Equal(3, response.GetTypeSafeEvidence().Answers.Count);
        Assert.Equal(1, spy.Calls);
        Assert.Equal(1, handler.Sends);
    }

    [Fact]
    public async Task Schema_RejectsWholeScoreRangeBeyondExactMaximumBeforeHttp()
    {
        var binding = SchemaFixture.Score(rubric:
            [TypeSafeJson.Entry("low"), TypeSafeJson.Entry("middle"), TypeSafeJson.Entry("high")]);
        await AssertRejected(
            SchemaFixture.ObjectSchema("""{"type":"number","maximum":1.9999999999999999}"""),
            new Dictionary<string, TypeSafeFieldBinding> { ["/value"] = binding },
            "/properties/value");
    }

    [Fact]
    public async Task Schema_ExplicitBindingInstructionsOverrideReferenceDescriptionWithoutChangingSchema()
    {
        var schema = NativeFixture.Json("""
            {"type":"object","additionalProperties":false,
             "$defs":{"flag":{"type":"boolean","description":"Definition predicate."}},
             "properties":{"fallback":{"$ref":"#/$defs/flag","description":"Use-site fallback predicate."},
                           "explicit":{"$ref":"#/$defs/flag","description":"Unused use-site predicate."}},
             "required":["fallback","explicit"]}
            """);
        string original = schema.GetRawText();
        var bindings = new Dictionary<string, TypeSafeFieldBinding>
        {
            ["/fallback"] = SchemaFixture.Boolean(null),
            ["/explicit"] = SchemaFixture.Boolean("Host binding predicate.")
        };
        using var handler = SchemaFixture.Handler();
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        using var adapter = new TypeSafeSchemaChatClient(spy, bindings);

        var response = await adapter.GetResponseAsync([SchemaFixture.State.ToMessage()],
            SchemaFixture.Options(schema, "Shared host policy."), TestContext.Current.CancellationToken);

        var questions = NativeFixture.Json(Assert.Single(handler.Requests).Body!).GetProperty("questions");
        Assert.Equal(["Use-site fallback predicate.", "Host binding predicate."],
            questions.EnumerateObject().Select(q => q.Value.GetProperty("instructions").GetProperty("fieldInstructions").GetString()));
        Assert.All(questions.EnumerateObject(), q =>
            Assert.Equal("Shared host policy.", q.Value.GetProperty("instructions").GetProperty("sharedInstructions").GetString()));
        var value = NativeFixture.Json(response.Text);
        Assert.Equal(["fallback", "explicit"], value.EnumerateObject().Select(p => p.Name));
        Assert.True(value.GetProperty("fallback").GetBoolean());
        Assert.True(value.GetProperty("explicit").GetBoolean());
        Assert.Equal(original, schema.GetRawText());
        Assert.Equal(1, spy.Calls);
        Assert.Equal(1, handler.Sends);
    }

    private static async Task AssertRejected(JsonElement schema, Dictionary<string, TypeSafeFieldBinding> bindings, string pointer)
    {
        using var handler = SchemaFixture.Handler();
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        using var adapter = new TypeSafeSchemaChatClient(spy, bindings);
        var exception = await Record.ExceptionAsync(() => adapter.GetResponseAsync(
            [SchemaFixture.State.ToMessage()], SchemaFixture.Options(schema), TestContext.Current.CancellationToken));
        Assert.Equal(0, spy.Calls);
        Assert.Equal(0, handler.Sends);
        var error = Assert.IsType<TypeSafeSchemaException>(exception);
        Assert.Equal(pointer, error.Pointer);
        Assert.Contains("Schema '" + pointer + "'", error.Message);
    }
}
