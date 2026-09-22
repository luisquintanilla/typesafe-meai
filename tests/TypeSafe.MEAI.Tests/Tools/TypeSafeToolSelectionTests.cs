using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.AI;
using TypeSafe.MEAI.Tests.Fixtures;
using TypeSafe.MEAI.Tests.Schema;
using Xunit;

namespace TypeSafe.MEAI.Tests.Tools;

[Trait("Stage", "C")]
public sealed class TypeSafeToolSelectionTests
{
    [Theory]
    [InlineData(0, false)]
    [InlineData(254, true)]
    [InlineData(255, false)]
    public async Task Select_Enforces254ToolsPlusReservedSentinel(int count, bool accepted)
    {
        int invocations = 0;
        var definitions = Enumerable.Range(0, count).Select(i => ToolFixture.FixedDefinition(
            ToolFixture.FixedFunction("tool_" + i, () => invocations++))).ToArray();
        using var handler = ToolFixture.Handler(confidence: 0.8);
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        if (!accepted)
        {
            Assert.Throws<ArgumentException>(() => new TypeSafeToolSelector(spy, definitions, 0.8));
            Assert.Equal(0, handler.Sends);
            Assert.Equal(0, spy.Calls);
        }
        else
        {
            var result = await new TypeSafeToolSelector(spy, definitions, 0.8).SelectAsync(SchemaFixture.State,
                cancellationToken: TestContext.Current.CancellationToken);
            Assert.Equal(TypeSafeToolSelectionStatus.Accepted, result.Status);
            Assert.Equal("tool_0", result.FunctionCall!.Name);
            var criteria = NativeFixture.Json(Assert.Single(handler.Requests).Body!).GetProperty("questions")
                .GetProperty("tool").GetProperty("criteria");
            Assert.Equal(255, criteria.EnumerateObject().Count());
            Assert.True(criteria.TryGetProperty(TypeSafeToolSelector.NoMatch, out _));
            Assert.Single(result.Evidence);
            Assert.Equal(0.8, result.Evidence[0].GetAnswer<ChoiceAnswer>("tool").Confidence);
            Assert.Equal(1, spy.Calls);
            Assert.Equal(1, handler.Sends);
        }
        Assert.Equal(0, invocations);
    }

    [Theory]
    [InlineData("name-empty")]
    [InlineData("name-space")]
    [InlineData("description-empty")]
    [InlineData("description-space")]
    [InlineData("reserved")]
    [InlineData("duplicate")]
    public void Configuration_RejectsEmptyReservedOrDuplicateDeclarations(string invalid)
    {
        int invocations = 0;
        var function = ToolFixture.FixedFunction("preview", () => invocations++);
        if (invalid == "name-empty") function.NameOverride = "";
        if (invalid == "name-space") function.NameOverride = " ";
        if (invalid == "description-empty") function.DescriptionOverride = "";
        if (invalid == "description-space") function.DescriptionOverride = " ";
        if (invalid == "reserved") function.NameOverride = TypeSafeToolSelector.NoMatch;
        using var handler = ToolFixture.Handler();
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        if (invalid == "duplicate")
        {
            var first = ToolFixture.FixedDefinition(function);
            var second = ToolFixture.FixedDefinition(ToolFixture.FixedFunction("preview", () => invocations++));
            Assert.Throws<ArgumentException>(() => new TypeSafeToolSelector(spy, [first, second], 0.8));
        }
        else Assert.Throws<ArgumentException>(() => ToolFixture.FixedDefinition(function));
        Assert.Equal(0, spy.Calls);
        Assert.Equal(0, handler.Sends);
        Assert.Equal(0, invocations);
    }

    [Theory]
    [InlineData("array")]
    [InlineData("inferred-number")]
    [InlineData("nonobject")]
    public async Task Configuration_CompilesEvenUnselectedCandidatesBeforeHttp(string invalid)
    {
        int invocations = 0;
        var valid = ToolFixture.FixedFunction("first", () => invocations++);
        var unselected = ToolFixture.FixedFunction("unselected", () => invocations++);
        unselected.SchemaOverride = invalid == "nonobject" ? NativeFixture.Json("""{"type":"boolean"}""") :
            SchemaFixture.ObjectSchema(invalid == "array" ? """{"type":"array"}""" : """{"type":"number"}""");
        using var handler = ToolFixture.Handler("first");
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        await Assert.ThrowsAsync<TypeSafeSchemaException>(() =>
        {
            IEnumerable<TypeSafeToolDefinition> Candidates()
            {
                yield return ToolFixture.FixedDefinition(valid);
                yield return new(unselected, invalid == "inferred-number"
                    ? new Dictionary<string, TypeSafeFieldBinding> { ["/value"] = SchemaFixture.Score() }
                    : null, invalid == "nonobject" ? new Dictionary<string, JsonElement> { [""] = NativeFixture.Json("true") } : null);
            }
            return new TypeSafeToolSelector(spy, Candidates(), 0.8).SelectAsync(SchemaFixture.State,
                cancellationToken: TestContext.Current.CancellationToken);
        });
        Assert.Equal(0, spy.Calls);
        Assert.Equal(0, handler.Sends);
        Assert.Equal(0, invocations);
    }

    [Theory]
    [InlineData("integer", "1", true)]
    [InlineData("integer", "3", true)]
    [InlineData("integer", "0", false)]
    [InlineData("integer", "4", false)]
    [InlineData("integer", "1.5", false)]
    [InlineData("integer", "\"2\"", false)]
    [InlineData("number", "1.25", true)]
    [InlineData("boolean", "true", true)]
    [InlineData("boolean", "\"true\"", false)]
    [InlineData("string", "\"Billing\"", true)]
    [InlineData("string", "\"Other\"", false)]
    public async Task FixedArguments_ValidateScalarTypeEnumAndInclusiveRange(string type, string value, bool accepted)
    {
        int invocations = 0;
        var function = ToolFixture.FixedFunction("fixed", () => invocations++);
        string leaf = type switch
        {
            "integer" => """{"type":"integer","minimum":1,"maximum":3}""",
            "number" => """{"type":"number","minimum":1,"maximum":3}""",
            "boolean" => """{"type":"boolean"}""",
            _ => """{"type":"string","enum":["Billing","Technical"]}"""
        };
        function.SchemaOverride = SchemaFixture.ObjectSchema(leaf);
        var fixedValues = new Dictionary<string, JsonElement> { ["/value"] = NativeFixture.Json(value) };
        using var handler = ToolFixture.Handler("fixed");
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        if (!accepted)
        {
            var error = Assert.Throws<TypeSafeSchemaException>(() => new TypeSafeToolDefinition(function, fixedValues: fixedValues));
            Assert.Equal("/properties/value", error.Pointer);
            Assert.Equal(0, handler.Sends);
        }
        else
        {
            var result = await new TypeSafeToolSelector(spy, [new(function, fixedValues: fixedValues)], 0.8)
                .SelectAsync(SchemaFixture.State, cancellationToken: TestContext.Current.CancellationToken);
            Assert.Equal(TypeSafeToolSelectionStatus.Accepted, result.Status);
            Assert.True(JsonElement.DeepEquals(NativeFixture.Json(value), Assert.IsType<JsonElement>(result.FunctionCall!.Arguments!["value"])));
            Assert.Single(result.Evidence);
            Assert.Equal(1, spy.Calls);
            Assert.Equal(1, handler.Sends);
        }
        Assert.Equal(0, invocations);
    }

    [Theory]
    [InlineData("pattern")]
    [InlineData("format")]
    public void FixedArguments_DoNotWaiveUnsupportedAssertions(string keyword)
    {
        int invocations = 0;
        var function = ToolFixture.FixedFunction("fixed", () => invocations++);
        var leaf = new JsonObject { ["type"] = "string", [keyword] = keyword == "pattern" ? "^[a-z]+$" : "email" };
        function.SchemaOverride = SchemaFixture.ObjectSchema(leaf.ToJsonString());
        var error = Assert.Throws<TypeSafeSchemaException>(() => new TypeSafeToolDefinition(function,
            fixedValues: new Dictionary<string, JsonElement> { ["/value"] = NativeFixture.Json("\"synthetic\"") }));
        Assert.Equal("/properties/value/" + keyword, error.Pointer);
        Assert.Equal(0, invocations);
    }

    [Theory]
    [InlineData("fixed")]
    [InlineData("omit")]
    [InlineData("none")]
    [InlineData("no-default")]
    [InlineData("conflict")]
    [InlineData("omit-required")]
    public async Task OptionalArguments_RequireExplicitFixedOrDeclaredDefaultOmission(string strategy)
    {
        int invocations = 0;
        var function = new ToolProbe(AIFunctionFactory.Create((int ticketId, string note = "declared-default") =>
        {
            invocations++;
            return $"{ticketId}:{note}";
        }, ToolFixture.Options("optional")));
        if (strategy == "no-default")
        {
            var schema = JsonNode.Parse(function.JsonSchema.GetRawText())!;
            schema["properties"]!["note"]!.AsObject().Remove("default");
            function.SchemaOverride = JsonSerializer.SerializeToElement(schema);
        }
        Dictionary<string, JsonElement> fixedValues = new() { ["/ticketId"] = NativeFixture.Json("17") };
        if (strategy is "fixed" or "conflict") fixedValues["/note"] = NativeFixture.Json("\"host-note\"");
        if (strategy == "omit-required") fixedValues.Remove("/ticketId");
        string[] omit = strategy switch
        {
            "omit" or "no-default" or "conflict" => ["/note"],
            "omit-required" => ["/ticketId", "/note"],
            _ => []
        };
        using var handler = ToolFixture.Handler("optional");
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        if (strategy is not ("fixed" or "omit"))
        {
            var error = Assert.Throws<TypeSafeSchemaException>(() => new TypeSafeToolDefinition(function, fixedValues: fixedValues, omitDefaultParameters: omit));
            if (strategy is "no-default" or "omit-required") Assert.Contains("Omission requires", error.Message);
            if (strategy == "conflict") Assert.Contains("Conflicting", error.Message);
            Assert.Equal(0, spy.Calls);
            Assert.Equal(0, handler.Sends);
        }
        else
        {
            var definition = new TypeSafeToolDefinition(function, fixedValues: fixedValues, omitDefaultParameters: omit);
            var result = await new TypeSafeToolSelector(spy, [definition], 0.8).SelectAsync(SchemaFixture.State,
                cancellationToken: TestContext.Current.CancellationToken);
            Assert.Equal(TypeSafeToolSelectionStatus.Accepted, result.Status);
            Assert.Equal(17, Assert.IsType<JsonElement>(result.FunctionCall!.Arguments!["ticketId"]).GetInt32());
            Assert.Equal(strategy == "fixed", result.FunctionCall.Arguments.ContainsKey("note"));
            if (strategy == "fixed") Assert.Equal("host-note", Assert.IsType<JsonElement>(result.FunctionCall.Arguments["note"]).GetString());
            Assert.Single(result.Evidence);
            Assert.Equal(1, handler.Sends);
        }
        Assert.Equal(0, invocations);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Select_NoMatchOrLowConfidenceStopsAfterSelectionWithoutExecutableCall(bool lowConfidence)
    {
        int invocations = 0;
        var function = ToolFixture.InferredFunction(() => invocations++);
        using var handler = ToolFixture.Handler(lowConfidence ? "preview" : TypeSafeToolSelector.NoMatch,
            confidence: lowConfidence ? 0.799999 : 0.9);
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        var result = await new TypeSafeToolSelector(spy, [ToolFixture.InferredDefinition(function)], 0.8)
            .SelectAsync(SchemaFixture.State, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(lowConfidence ? TypeSafeToolSelectionStatus.Abstained : TypeSafeToolSelectionStatus.NoMatch, result.Status);
        Assert.Null(result.FunctionCall);
        Assert.Null(result.Function);
        var evidence = Assert.Single(result.Evidence);
        Assert.Equal(lowConfidence ? 0.799999 : 0.9, evidence.GetAnswer<ChoiceAnswer>("tool").Confidence);
        Assert.Equal(lowConfidence ? 1 : 0, result.Rejections.Count);
        Assert.Equal(1, spy.Calls);
        Assert.Equal(1, handler.Sends);
        Assert.Equal(0, invocations);
    }

    [Fact]
    public async Task Select_InfersOnlySelectedArgumentsAndSnapshotsOptionsOnceAcrossBothBatches()
    {
        int selectedCalls = 0, otherCalls = 0;
        var selected = ToolFixture.InferredFunction(() => selectedCalls++);
        var other = new ToolProbe(AIFunctionFactory.Create((bool unrelated) => { otherCalls++; return unrelated; }, ToolFixture.Options("other")));
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var handler = new RecordingHandler(async (request, attempt, token) =>
        {
            if (attempt == 1) { entered.SetResult(); await release.Task.WaitAsync(token); }
            return NativeFixture.Response(ToolFixture.Envelope(NativeFixture.Json(await request.Content!.ReadAsStringAsync(token)), "preview"));
        });
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        var definitions = new List<TypeSafeToolDefinition>
        {
            ToolFixture.InferredDefinition(selected),
            new(other, new Dictionary<string, TypeSafeFieldBinding> { ["/unrelated"] = SchemaFixture.Boolean("Judge the unrelated candidate.") })
        };
        var selector = new TypeSafeToolSelector(spy, definitions, 0.8);
        definitions.Clear();
        var tags = new List<string> { "original" };
        var options = new ChatOptions { ModelId = "snapshot-model", AdditionalProperties = new() { ["synthetic-tags"] = tags } };
        var pending = selector.SelectAsync(SchemaFixture.State, options, TestContext.Current.CancellationToken);
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        options.ModelId = "caller-mutated";
        options.Temperature = 0.5f;
        tags[0] = "changed";
        release.SetResult();
        var result = await pending;

        Assert.Equal(TypeSafeToolSelectionStatus.Accepted, result.Status);
        Assert.Same(selected, result.Function);
        Assert.Equal("preview", result.FunctionCall!.Name);
        Assert.Equal(["destination", "followup"], result.FunctionCall.Arguments!.Keys.Order());
        Assert.Equal("Billing", Assert.IsType<JsonElement>(result.FunctionCall.Arguments["destination"]).GetString());
        Assert.True(Assert.IsType<JsonElement>(result.FunctionCall.Arguments["followup"]).GetBoolean());
        Assert.Equal(2, result.Evidence.Count);
        Assert.All(result.Evidence, e => Assert.Equal("snapshot-model", e.Model));
        Assert.Equal(2, spy.Calls);
        Assert.Equal(2, handler.Sends);
        Assert.All(spy.Options, o =>
        {
            Assert.Equal("snapshot-model", o!.ModelId);
            Assert.Null(o.Temperature);
            Assert.Equal("original", Assert.IsType<JsonElement>(o.AdditionalProperties!["synthetic-tags"])[0].GetString());
        });
        var requests = handler.Requests.ToArray();
        Assert.Equal(["tool"], NativeFixture.Json(requests[0].Body!).GetProperty("questions").EnumerateObject().Select(p => p.Name));
        var arguments = NativeFixture.Json(requests[1].Body!).GetProperty("questions");
        Assert.Equal(2, arguments.EnumerateObject().Count());
        Assert.Equal(["choice", "noul"], arguments.EnumerateObject().Select(p => p.Value.GetProperty("type").GetString()));
        Assert.DoesNotContain("unrelated", requests[1].Body!);
        Assert.Equal("caller-mutated", options.ModelId);
        Assert.Equal(0.5f, options.Temperature);
        Assert.Equal("changed", tags[0]);
        Assert.Equal(0, selectedCalls);
        Assert.Equal(0, otherCalls);
        Assert.Equal(0, selected.Calls);
        Assert.Equal(0, other.Calls);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Select_UncertainOrApplicationInvalidArgumentsHaveNoExecutableCall(bool application)
    {
        int invocations = 0, validations = 0;
        var function = ToolFixture.InferredFunction(() => invocations++);
        string? Validate(JsonElement arguments)
        {
            validations++;
            Assert.True(arguments.GetProperty("followup").GetBoolean());
            return "Synthetic application veto.";
        }
        using var handler = ToolFixture.Handler("preview", argumentNoul: application ? 0.9 : 0.5);
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        var result = await new TypeSafeToolSelector(spy, [ToolFixture.InferredDefinition(function, application ? Validate : null)], 0.8)
            .SelectAsync(SchemaFixture.State, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(TypeSafeToolSelectionStatus.Abstained, result.Status);
        Assert.Same(function, result.Function);
        Assert.Null(result.FunctionCall);
        Assert.Equal(application ? "" : "/followup", Assert.Single(result.Rejections).Pointer);
        if (application) Assert.Equal("Synthetic application veto.", result.Rejections[0].Reason);
        Assert.Equal(application ? 1 : 0, validations);
        Assert.Equal(2, result.Evidence.Count);
        Assert.Equal(2, spy.Calls);
        Assert.Equal(2, handler.Sends);
        Assert.Equal(0, invocations);
    }

    [Fact]
    public async Task Select_FixedConfigurationSnapshotsAndUniqueIdsNeedNoArgumentInference()
    {
        int invocations = 0;
        var function = ToolFixture.FixedFunction("fixed", () => invocations++);
        Dictionary<string, JsonElement> fixedValues;
        TypeSafeToolDefinition definition;
        using (var document = JsonDocument.Parse("17"))
        {
            fixedValues = new() { ["/ticketId"] = document.RootElement };
            definition = new(function, fixedValues: fixedValues);
        }
        fixedValues["/ticketId"] = NativeFixture.Json("99");
        using var handler = ToolFixture.Handler("fixed");
        using var services = NativeFixture.Services(handler, cache: true);
        using var spy = new SchemaSpy(services.Client());
        var selector = new TypeSafeToolSelector(spy, [definition], 0.8);
        var first = await selector.SelectAsync(SchemaFixture.State, cancellationToken: TestContext.Current.CancellationToken);
        var second = await selector.SelectAsync(SchemaFixture.State, cancellationToken: TestContext.Current.CancellationToken);
        foreach (var result in new[] { first, second })
        {
            Assert.Equal(TypeSafeToolSelectionStatus.Accepted, result.Status);
            Assert.Equal(17, Assert.IsType<JsonElement>(result.FunctionCall!.Arguments!["ticketId"]).GetInt32());
            Assert.True(Guid.TryParseExact(result.FunctionCall.CallId, "N", out var id));
            Assert.NotEqual(Guid.Empty, id);
            Assert.Single(result.Evidence);
        }
        Assert.NotEqual(first.FunctionCall!.CallId, second.FunctionCall!.CallId);
        Assert.Equal(99, fixedValues["/ticketId"].GetInt32());
        Assert.True(JsonElement.DeepEquals(first.Evidence[0].Json, second.Evidence[0].Json));
        Assert.Equal(2, spy.Calls);
        Assert.Equal(1, handler.Sends);
        Assert.Equal(0, invocations);
    }

    [Fact]
    public async Task Select_ZeroParameterFactoryFunctionUsesOneNativeBatch()
    {
        int invocations = 0;
        var function = new ToolProbe(AIFunctionFactory.Create(() => { invocations++; return "synthetic"; }, ToolFixture.Options("empty")));
        using var handler = ToolFixture.Handler("empty");
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        var result = await new TypeSafeToolSelector(spy, [new TypeSafeToolDefinition(function)], 0.8)
            .SelectAsync(SchemaFixture.State, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(TypeSafeToolSelectionStatus.Accepted, result.Status);
        Assert.Empty(result.FunctionCall!.Arguments!);
        Assert.Single(result.Evidence);
        Assert.Equal(1, spy.Calls);
        Assert.Equal(1, handler.Sends);
        Assert.Equal(0, invocations);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task OptionalArguments_ReferenceSiblingDefaultOverridesResolvedAnnotation(bool targetHasDefault)
    {
        int invocations = 0;
        var function = new ToolProbe(AIFunctionFactory.Create((int ticketId, string note = "declared-default") =>
        {
            invocations++;
            return $"{ticketId}:{note}";
        }, ToolFixture.Options("ref_default")));
        var schema = JsonNode.Parse(function.JsonSchema.GetRawText())!;
        var target = new JsonObject { ["type"] = "string", ["enum"] = new JsonArray("declared-default") };
        if (targetHasDefault) target["default"] = "stale-definition-default";
        schema["$defs"] = new JsonObject
        {
            ["noteType"] = target,
            ["alias"] = new JsonObject { ["$ref"] = "#/$defs/noteType" }
        };
        schema["properties"]!["note"] = new JsonObject { ["$ref"] = "#/$defs/alias", ["default"] = "declared-default" };
        function.SchemaOverride = JsonSerializer.SerializeToElement(schema);
        var definition = new TypeSafeToolDefinition(function,
            fixedValues: new Dictionary<string, JsonElement> { ["/ticketId"] = NativeFixture.Json("17") },
            omitDefaultParameters: ["/note"]);
        using var handler = ToolFixture.Handler("ref_default");
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));

        var selection = await new TypeSafeToolSelector(spy, [definition], 0.8).SelectAsync(SchemaFixture.State,
            cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(TypeSafeToolSelectionStatus.Accepted, selection.Status);
        Assert.Equal(["ticketId"], selection.FunctionCall!.Arguments!.Keys);
        Assert.Equal(17, Assert.IsType<JsonElement>(selection.FunctionCall.Arguments["ticketId"]).GetInt32());
        Assert.Equal(0, invocations);
        var result = await TypeSafeToolInvoker.InvokeAsync(selection, (_, _) => ValueTask.FromResult(true),
            cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal("17:declared-default", Assert.IsType<JsonElement>(result.Result).GetString());
        var originalTarget = function.JsonSchema.GetProperty("$defs").GetProperty("noteType");
        Assert.Equal(targetHasDefault, originalTarget.TryGetProperty("default", out var originalDefault));
        if (targetHasDefault) Assert.Equal("stale-definition-default", originalDefault.GetString());
        Assert.Single(selection.Evidence);
        Assert.Equal(1, spy.Calls);
        Assert.Equal(1, handler.Sends);
        Assert.Equal(1, function.Calls);
        Assert.Equal(1, invocations);
    }

    [Theory]
    [InlineData("1.0000000000000001", false)]
    [InlineData("1.00000000000000000000000000001", false)]
    [InlineData("9007199254740992.5", false)]
    [InlineData("-9007199254740992.5", false)]
    [InlineData("1e-400", false)]
    [InlineData("9007199254740993", true)]
    [InlineData("-9007199254740993", true)]
    public async Task FixedArguments_ValidateIntegerLexemesWithoutRounding(string number, bool accepted)
    {
        int invocations = 0;
        var function = new ToolProbe(AIFunctionFactory.Create((long value) => { invocations++; return value; },
            ToolFixture.Options("exact_integer")));
        var fixedValues = new Dictionary<string, JsonElement> { ["/value"] = NativeFixture.Json(number) };
        using var handler = ToolFixture.Handler("exact_integer");
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        if (!accepted)
        {
            var error = Assert.Throws<TypeSafeSchemaException>(() => new TypeSafeToolDefinition(function, fixedValues: fixedValues));
            Assert.Equal("/properties/value", error.Pointer);
            Assert.Equal(0, spy.Calls);
            Assert.Equal(0, handler.Sends);
            Assert.Equal(0, function.Calls);
            Assert.Equal(0, invocations);
        }
        else
        {
            var selection = await new TypeSafeToolSelector(spy, [new(function, fixedValues: fixedValues)], 0.8)
                .SelectAsync(SchemaFixture.State, cancellationToken: TestContext.Current.CancellationToken);
            Assert.Equal(TypeSafeToolSelectionStatus.Accepted, selection.Status);
            Assert.Equal(number, Assert.IsType<JsonElement>(selection.FunctionCall!.Arguments!["value"]).GetRawText());
            Assert.Equal(0, invocations);
            var result = await TypeSafeToolInvoker.InvokeAsync(selection, (_, _) => ValueTask.FromResult(true),
                cancellationToken: TestContext.Current.CancellationToken);
            Assert.Equal(long.Parse(number, System.Globalization.CultureInfo.InvariantCulture),
                Assert.IsType<JsonElement>(result.Result).GetInt64());
            Assert.Single(selection.Evidence);
            Assert.Equal(1, spy.Calls);
            Assert.Equal(1, handler.Sends);
            Assert.Equal(1, function.Calls);
            Assert.Equal(1, invocations);
        }
    }

    [Theory]
    [InlineData("9007199254740992", false)]
    [InlineData("9007199254740993", true)]
    public async Task FixedArguments_EnforceExactMinimumBeyondDoublePrecision(string number, bool accepted)
    {
        int invocations = 0;
        var function = new ToolProbe(AIFunctionFactory.Create((long value) => { invocations++; return value; },
            ToolFixture.Options("exact_minimum")))
        {
            SchemaOverride = SchemaFixture.ObjectSchema("""{"type":"integer","minimum":9007199254740993}""")
        };
        var fixedValues = new Dictionary<string, JsonElement> { ["/value"] = NativeFixture.Json(number) };
        using var handler = ToolFixture.Handler("exact_minimum");
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        if (!accepted)
        {
            var error = Assert.Throws<TypeSafeSchemaException>(() => new TypeSafeToolDefinition(function, fixedValues: fixedValues));
            Assert.Equal("/properties/value", error.Pointer);
            Assert.Contains("numeric constraint", error.Message);
            Assert.Equal(0, spy.Calls);
            Assert.Equal(0, handler.Sends);
            Assert.Equal(0, function.Calls);
            Assert.Equal(0, invocations);
        }
        else
        {
            var selection = await new TypeSafeToolSelector(spy, [new(function, fixedValues: fixedValues)], 0.8)
                .SelectAsync(SchemaFixture.State, cancellationToken: TestContext.Current.CancellationToken);
            Assert.Equal(TypeSafeToolSelectionStatus.Accepted, selection.Status);
            Assert.Equal(9007199254740993L, Assert.IsType<JsonElement>(selection.FunctionCall!.Arguments!["value"]).GetInt64());
            Assert.Equal(0, invocations);
            var result = await TypeSafeToolInvoker.InvokeAsync(selection, (_, _) => ValueTask.FromResult(true),
                cancellationToken: TestContext.Current.CancellationToken);
            Assert.Equal(9007199254740993L, Assert.IsType<JsonElement>(result.Result).GetInt64());
            Assert.Single(selection.Evidence);
            Assert.Equal(1, spy.Calls);
            Assert.Equal(1, handler.Sends);
            Assert.Equal(1, function.Calls);
            Assert.Equal(1, invocations);
        }
    }
}

internal static class ToolFixture
{
    internal static AIFunctionFactoryOptions Options(string name) => new()
    {
        Name = name,
        Description = "Preview a synthetic support operation without persistence.",
        JsonSchemaCreateOptions = new() { TransformOptions = new() { DisallowAdditionalProperties = true } }
    };
    internal static ToolProbe FixedFunction(string name, Action invoked) =>
        new(AIFunctionFactory.Create((int ticketId) => { invoked(); return ticketId; }, Options(name)));
    internal static TypeSafeToolDefinition FixedDefinition(AIFunction function) =>
        new(function, fixedValues: new Dictionary<string, JsonElement> { ["/ticketId"] = NativeFixture.Json("17") });
    internal static ToolProbe InferredFunction(Action invoked) =>
        new(AIFunctionFactory.Create((TestDestination destination, bool followup) =>
        {
            invoked();
            return new { destination, followup };
        }, Options("preview")));
    internal static TypeSafeToolDefinition InferredDefinition(AIFunction function, Func<JsonElement, string?>? validate = null) =>
        new(function, new Dictionary<string, TypeSafeFieldBinding>
        {
            ["/destination"] = SchemaFixture.Choice(),
            ["/followup"] = SchemaFixture.Boolean()
        }, validate: validate);
    internal static RecordingHandler Handler(string? selected = null, double confidence = 0.9, double argumentNoul = 0.9) =>
        new(async (request, _, token) => NativeFixture.Response(Envelope(
            NativeFixture.Json(await request.Content!.ReadAsStringAsync(token)), selected, confidence, argumentNoul)));
    internal static string Envelope(JsonElement request, string? selected = null, double confidence = 0.9, double argumentNoul = 0.9)
    {
        var response = JsonNode.Parse(SchemaFixture.Envelope(request, noul: argumentNoul, confidence: confidence))!;
        if (selected is not null && request.GetProperty("questions").TryGetProperty("tool", out _))
            response["answers"]!["tool"]!["choice"] = selected;
        return response.ToJsonString();
    }
}

internal sealed class ToolProbe(AIFunction inner) : DelegatingAIFunction(inner)
{
    internal string? NameOverride { get; set; }
    internal string? DescriptionOverride { get; set; }
    internal JsonElement? SchemaOverride { get; set; }
    internal int Calls { get; private set; }
    internal AIFunctionArguments? Arguments { get; private set; }
    internal CancellationToken Token { get; private set; }
    public override string Name => NameOverride ?? base.Name;
    public override string Description => DescriptionOverride ?? base.Description;
    public override JsonElement JsonSchema => SchemaOverride ?? base.JsonSchema;
    protected override ValueTask<object?> InvokeCoreAsync(AIFunctionArguments arguments, CancellationToken cancellationToken)
    {
        Calls++;
        Arguments = arguments;
        Token = cancellationToken;
        return base.InvokeCoreAsync(arguments, cancellationToken);
    }
}
