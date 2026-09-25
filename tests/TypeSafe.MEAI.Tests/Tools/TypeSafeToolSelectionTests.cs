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
    [InlineData("auto", 2)]
    [InlineData("required", 2)]
    [InlineData("specific", 1)]
    [InlineData("none", 0)]
    public async Task StandardToolModes_UseOrdinaryFactorySchemasAndBoundNativeBatches(string mode, int batches)
    {
        int effects = 0;
        var function = ToolFixture.InferredFunction(() => effects++);
        using var handler = ToolFixture.Handler("preview");
        using var http = new HttpClient(handler);
        using IChatClient client = new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key);
        var options = new ChatOptions
        {
            Tools = [function], AllowMultipleToolCalls = false,
            ToolMode = mode switch { "required" => ChatToolMode.RequireAny, "specific" => ChatToolMode.RequireSpecific("preview"),
                "none" => ChatToolMode.None, _ => ChatToolMode.Auto }
        };
        var response = await client.GetResponseAsync(NativeFixture.Messages(), options, TestContext.Current.CancellationToken);
        var evidence = response.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>().ToArray();
        Assert.Equal(batches, handler.Sends);
        Assert.Equal(batches, evidence.Length);
        Assert.Equal(0, effects);
        Assert.Equal(0, function.Calls);
        if (batches == 0)
        {
            Assert.Empty(Assert.Single(response.Messages).Contents);
            Assert.Null(response.Usage);
            Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
            return;
        }
        var call = Assert.Single(response.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>());
        Assert.Equal("preview", call.Name);
        Assert.True(Assert.IsType<JsonElement>(call.Arguments!["followup"]).GetBoolean());
        Assert.Equal("Billing", Assert.IsType<JsonElement>(call.Arguments["destination"]).GetString());
        Assert.Equal(11 * batches, response.Usage!.InputTokenCount);
        Assert.Equal(7 * batches, response.Usage.OutputTokenCount);
        Assert.Equal(18 * batches, response.Usage.TotalTokenCount);
        var requests = handler.Requests.ToArray();
        if (mode == "specific") Assert.False(NativeFixture.Json(requests[0].Body!).GetProperty("questions").TryGetProperty("tool", out _));
        else
        {
            var criteria = NativeFixture.Json(requests[0].Body!).GetProperty("questions").GetProperty("tool").GetProperty("criteria");
            Assert.Equal(mode == "auto", criteria.TryGetProperty("__typesafe_no_match__", out _));
            Assert.Equal("tool-selection", Assert.IsType<string>(evidence[0].AdditionalProperties!["typesafe.v1.batch-role"]));
        }
        Assert.Equal("tool-arguments", Assert.IsType<string>(evidence[^1].AdditionalProperties!["typesafe.v1.batch-role"]));
    }

    [Theory]
    [InlineData("multiple")]
    [InlineData("unknown-specific")]
    [InlineData("required-no-match")]
    public async Task StandardToolModes_RejectUnsupportedOrInvalidRequiredSelectionWithoutExecution(string invalid)
    {
        int effects = 0;
        var function = ToolFixture.InferredFunction(() => effects++);
        using var handler = ToolFixture.Handler("__typesafe_no_match__");
        using var http = new HttpClient(handler);
        using IChatClient client = new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key);
        var options = new ChatOptions { Tools = [function],
            AllowMultipleToolCalls = invalid == "multiple",
            ToolMode = invalid == "unknown-specific" ? ChatToolMode.RequireSpecific("not_registered") : ChatToolMode.RequireAny };
        Task<ChatResponse> Invoke() => client.GetResponseAsync(NativeFixture.Messages(), options, TestContext.Current.CancellationToken);
        if (invalid == "multiple") await Assert.ThrowsAsync<NotSupportedException>(Invoke);
        else if (invalid == "unknown-specific") await Assert.ThrowsAsync<ArgumentException>(Invoke);
        else await Assert.ThrowsAsync<TypeSafeProtocolException>(Invoke);
        Assert.Equal(invalid == "required-no-match" ? 1 : 0, handler.Sends);
        Assert.Equal(0, effects);
        Assert.Equal(0, function.Calls);
    }

    [Theory]
    [InlineData("1e999")]
    [InlineData("0e99999999999999")]
    [InlineData("1e-1025")]
    [InlineData("long-literal")]
    public async Task FixedArguments_RejectHugeUnsupportedNumericLexemesBeforeHttp(string input)
    {
        string number = input == "long-literal" ? "1." + new string('0', 1024) : input;
        var function = ToolFixture.FixedFunction("fixed", () => throw new InvalidOperationException("Must not execute."));
        using var handler = ToolFixture.Handler("fixed");
        using var http = new HttpClient(handler);
        using IChatClient client = new TypeSafeChatClient(http, NativeFixture.Settings with
        { Tools = new Dictionary<string, TypeSafeToolOptions> { ["fixed"] = new(fixedValues:
            new Dictionary<string, JsonElement> { ["/ticketId"] = NativeFixture.Json(number) }) } }, NativeFixture.Key);
        var error = await Assert.ThrowsAsync<TypeSafeSchemaException>(() => client.GetResponseAsync(NativeFixture.Messages(),
            new ChatOptions { Tools = [function] }, TestContext.Current.CancellationToken));
        Assert.Equal("/properties/ticketId", error.Pointer);
        Assert.Equal(0, handler.Sends);
        Assert.Equal(0, function.Calls);
    }

    [Theory]
    [InlineData(0, false)]
    [InlineData(254, true)]
    [InlineData(255, false)]
    public async Task Select_Enforces254ToolsPlusReservedSentinel(int count, bool accepted)
    {
        int invocations = 0;
        var functions = Enumerable.Range(0, count).Select(i =>
            ToolFixture.FixedFunction("tool_" + i, () => invocations++)).ToArray();
        using var handler = ToolFixture.Handler(confidence: 0.8);
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        using var client = new TypeSafeChatClient(spy, NativeFixture.Settings with
        { Tools = functions.ToDictionary(f => f.Name, _ => ToolFixture.FixedOptions()) });
        var options = new ChatOptions { Tools = functions.Cast<AITool>().ToList() };
        if (!accepted)
        {
            await Assert.ThrowsAsync<NotSupportedException>(() => client.GetResponseAsync(NativeFixture.Messages(), options, TestContext.Current.CancellationToken));
            Assert.Equal(0, handler.Sends);
            Assert.Equal(0, spy.Calls);
        }
        else
        {
            var result = await client.GetResponseAsync(NativeFixture.Messages(), options, TestContext.Current.CancellationToken);
            Assert.Equal(ChatFinishReason.ToolCalls, result.FinishReason);
            Assert.Equal("tool_0", Assert.Single(result.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>()).Name);
            var criteria = NativeFixture.Json(Assert.Single(handler.Requests).Body!).GetProperty("questions")
                .GetProperty("tool").GetProperty("criteria");
            Assert.Equal(255, criteria.EnumerateObject().Count());
            Assert.True(criteria.TryGetProperty("__typesafe_no_match__", out _));
            Assert.Equal(0.8, Assert.IsType<ChoiceAnswer>(Assert.Single(result.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>()).Answers["tool"]).Confidence);
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
    public async Task Configuration_RejectsEmptyReservedOrDuplicateDeclarations(string invalid)
    {
        int invocations = 0;
        var function = ToolFixture.FixedFunction("preview", () => invocations++);
        if (invalid == "name-empty") function.NameOverride = "";
        if (invalid == "name-space") function.NameOverride = " ";
        if (invalid == "description-empty") function.DescriptionOverride = "";
        if (invalid == "description-space") function.DescriptionOverride = " ";
        if (invalid == "reserved") function.NameOverride = "__typesafe_no_match__";
        using var handler = ToolFixture.Handler();
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        using var client = new TypeSafeChatClient(spy, NativeFixture.Settings with
        { Tools = new Dictionary<string, TypeSafeToolOptions> { [function.Name] = ToolFixture.FixedOptions() } });
        var tools = new List<AITool> { function };
        if (invalid == "duplicate") tools.Add(ToolFixture.FixedFunction("preview", () => invocations++));
        await Assert.ThrowsAsync<ArgumentException>(() => client.GetResponseAsync(NativeFixture.Messages(),
            new ChatOptions { Tools = tools }, TestContext.Current.CancellationToken));
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
        using var client = new TypeSafeChatClient(spy, NativeFixture.Settings with
        { Tools = new Dictionary<string, TypeSafeToolOptions>
            {
                [valid.Name] = ToolFixture.FixedOptions(),
                [unselected.Name] = new(invalid == "inferred-number"
                    ? new Dictionary<string, TypeSafeFieldOptions> { ["/value"] = SchemaFixture.Score() }
                    : null, invalid == "nonobject" ? new Dictionary<string, JsonElement> { [""] = NativeFixture.Json("true") } : null)
            }
        });
        await Assert.ThrowsAsync<TypeSafeSchemaException>(() => client.GetResponseAsync(NativeFixture.Messages(),
            new ChatOptions { Tools = [valid, unselected] }, TestContext.Current.CancellationToken));
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
        using var client = ToolFixture.Client(spy, function, new(fixedValues: fixedValues));
        Task<ChatResponse> Invoke() => client.GetResponseAsync(NativeFixture.Messages(), new ChatOptions { Tools = [function] }, TestContext.Current.CancellationToken);
        if (!accepted)
        {
            var error = await Assert.ThrowsAsync<TypeSafeSchemaException>(Invoke);
            Assert.Equal("/properties/value", error.Pointer);
            Assert.Equal(0, handler.Sends);
        }
        else
        {
            var result = await Invoke();
            Assert.Equal(ChatFinishReason.ToolCalls, result.FinishReason);
            Assert.True(JsonElement.DeepEquals(NativeFixture.Json(value), Assert.IsType<JsonElement>(Assert.Single(result.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>()).Arguments!["value"])));
            Assert.Single(result.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>());
            Assert.Equal(1, spy.Calls);
            Assert.Equal(1, handler.Sends);
        }
        Assert.Equal(0, invocations);
    }

    [Theory]
    [InlineData("pattern")]
    [InlineData("format")]
    public async Task FixedArguments_DoNotWaiveUnsupportedAssertions(string keyword)
    {
        int invocations = 0;
        var function = ToolFixture.FixedFunction("fixed", () => invocations++);
        var leaf = new JsonObject { ["type"] = "string", [keyword] = keyword == "pattern" ? "^[a-z]+$" : "email" };
        function.SchemaOverride = SchemaFixture.ObjectSchema(leaf.ToJsonString());
        using var handler = ToolFixture.Handler();
        using var http = new HttpClient(handler);
        using var client = new TypeSafeChatClient(http, NativeFixture.Settings with
        { Tools = new Dictionary<string, TypeSafeToolOptions> { [function.Name] = new(
            fixedValues: new Dictionary<string, JsonElement> { ["/value"] = NativeFixture.Json("\"synthetic\"") }) } }, NativeFixture.Key);
        var error = await Assert.ThrowsAsync<TypeSafeSchemaException>(() => client.GetResponseAsync(NativeFixture.Messages(),
            new ChatOptions { Tools = [function] }, TestContext.Current.CancellationToken));
        Assert.Equal("/properties/value/" + keyword, error.Pointer);
        Assert.Equal(0, invocations);
        Assert.Equal(0, handler.Sends);
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
        using var client = ToolFixture.Client(spy, function, new(fixedValues: fixedValues, omitDefaultParameters: omit));
        Task<ChatResponse> Invoke() => client.GetResponseAsync(NativeFixture.Messages(), new ChatOptions { Tools = [function] }, TestContext.Current.CancellationToken);
        if (strategy is not ("fixed" or "omit"))
        {
            var error = await Assert.ThrowsAsync<TypeSafeSchemaException>(Invoke);
            if (strategy is "no-default" or "omit-required") Assert.Contains("Omission requires", error.Message);
            if (strategy == "conflict") Assert.Contains("Conflicting", error.Message);
            Assert.Equal(0, spy.Calls);
            Assert.Equal(0, handler.Sends);
        }
        else
        {
            var result = await Invoke();
            var call = Assert.Single(result.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>());
            Assert.Equal(ChatFinishReason.ToolCalls, result.FinishReason);
            Assert.Equal(17, Assert.IsType<JsonElement>(call.Arguments!["ticketId"]).GetInt32());
            Assert.Equal(strategy == "fixed", call.Arguments.ContainsKey("note"));
            if (strategy == "fixed") Assert.Equal("host-note", Assert.IsType<JsonElement>(call.Arguments["note"]).GetString());
            Assert.Single(result.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>());
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
        using var handler = ToolFixture.Handler(lowConfidence ? "preview" : "__typesafe_no_match__",
            confidence: lowConfidence ? 0.799999 : 0.9);
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        JevResultContent evidence;
        if (lowConfidence)
        {
            var error = await Assert.ThrowsAsync<TypeSafeAbstentionException>(() => spy.GetResponseAsync(NativeFixture.Messages(),
                new ChatOptions { Tools = [function] }, TestContext.Current.CancellationToken));
            Assert.Single(error.Rejections);
            evidence = Assert.Single(error.Evidence);
        }
        else
        {
            var result = await spy.GetResponseAsync(NativeFixture.Messages(), new ChatOptions { Tools = [function] }, TestContext.Current.CancellationToken);
            Assert.Empty(result.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>());
            Assert.Equal(ChatFinishReason.Stop, result.FinishReason);
            evidence = Assert.Single(result.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>());
        }
        Assert.Equal(lowConfidence ? 0.799999 : 0.9, Assert.IsType<ChoiceAnswer>(evidence.Answers["tool"]).Confidence);
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
        using var client = new TypeSafeChatClient(spy, NativeFixture.Settings);
        var tags = new List<string> { "original" };
        var options = new ChatOptions { Tools = [selected, other], ModelId = "snapshot-model", AdditionalProperties = new() { ["synthetic-tags"] = tags } };
        var pending = client.GetResponseAsync(NativeFixture.Messages(), options, TestContext.Current.CancellationToken);
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        options.ModelId = "caller-mutated";
        options.Temperature = 0.5f;
        tags[0] = "changed";
        release.SetResult();
        var result = await pending;
        var call = Assert.Single(result.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>());
        var evidence = result.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>().ToArray();

        Assert.Equal(ChatFinishReason.ToolCalls, result.FinishReason);
        Assert.Equal("preview", call.Name);
        Assert.Equal(["destination", "followup"], call.Arguments!.Keys.Order());
        Assert.Equal("Billing", Assert.IsType<JsonElement>(call.Arguments["destination"]).GetString());
        Assert.True(Assert.IsType<JsonElement>(call.Arguments["followup"]).GetBoolean());
        Assert.Equal(2, evidence.Length);
        Assert.All(evidence, e => Assert.Equal("snapshot-model", e.Model));
        Assert.Equal(22, result.Usage!.InputTokenCount);
        Assert.Equal(14, result.Usage.OutputTokenCount);
        Assert.Equal(36, result.Usage.TotalTokenCount);
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
        using var client = ToolFixture.Client(spy, function, new(validate: application ? Validate : null));
        var result = await Assert.ThrowsAsync<TypeSafeAbstentionException>(() => client.GetResponseAsync(NativeFixture.Messages(),
            new ChatOptions { Tools = [function] }, TestContext.Current.CancellationToken));
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
        TypeSafeToolOptions config;
        using (var document = JsonDocument.Parse("17"))
        {
            fixedValues = new() { ["/ticketId"] = document.RootElement };
            config = new(fixedValues: fixedValues);
        }
        fixedValues["/ticketId"] = NativeFixture.Json("99");
        using var handler = ToolFixture.Handler("fixed");
        using var services = NativeFixture.Services(handler, cache: true);
        using var spy = new SchemaSpy(services.Client());
        using var client = ToolFixture.Client(spy, function, config);
        var options = new ChatOptions { Tools = [function] };
        var first = await client.GetResponseAsync(NativeFixture.Messages(), options, TestContext.Current.CancellationToken);
        var second = await client.GetResponseAsync(NativeFixture.Messages(), options, TestContext.Current.CancellationToken);
        foreach (var result in new[] { first, second })
        {
            var call = Assert.Single(result.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>());
            Assert.Equal(ChatFinishReason.ToolCalls, result.FinishReason);
            Assert.Equal(17, Assert.IsType<JsonElement>(call.Arguments!["ticketId"]).GetInt32());
            Assert.True(Guid.TryParseExact(call.CallId, "N", out var id));
            Assert.NotEqual(Guid.Empty, id);
            Assert.Single(result.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>());
        }
        Assert.NotEqual(Assert.Single(first.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>()).CallId,
            Assert.Single(second.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>()).CallId);
        Assert.Equal(99, fixedValues["/ticketId"].GetInt32());
        Assert.True(JsonElement.DeepEquals(Assert.Single(first.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>()).Json,
            Assert.Single(second.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>()).Json));
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
        var result = await spy.GetResponseAsync(NativeFixture.Messages(), new ChatOptions { Tools = [function] }, TestContext.Current.CancellationToken);
        Assert.Equal(ChatFinishReason.ToolCalls, result.FinishReason);
        Assert.Empty(Assert.Single(result.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>()).Arguments!);
        Assert.Single(result.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>());
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
        var config = new TypeSafeToolOptions(
            fixedValues: new Dictionary<string, JsonElement> { ["/ticketId"] = NativeFixture.Json("17") },
            omitDefaultParameters: ["/note"]);
        using var handler = ToolFixture.Handler("ref_default");
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));

        using var client = ToolFixture.Client(spy, function, config);
        var selection = await client.GetResponseAsync(NativeFixture.Messages(), new ChatOptions { Tools = [function] }, TestContext.Current.CancellationToken);
        var call = Assert.Single(selection.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>());
        Assert.Equal(ChatFinishReason.ToolCalls, selection.FinishReason);
        Assert.Equal(["ticketId"], call.Arguments!.Keys);
        Assert.Equal(17, Assert.IsType<JsonElement>(call.Arguments["ticketId"]).GetInt32());
        Assert.Equal(0, invocations);
        var result = await ToolFixture.ExecuteProposedAsync(selection, function, TestContext.Current.CancellationToken);

        Assert.Equal("17:declared-default", Assert.IsType<JsonElement>(result.Result).GetString());
        var originalTarget = function.JsonSchema.GetProperty("$defs").GetProperty("noteType");
        Assert.Equal(targetHasDefault, originalTarget.TryGetProperty("default", out var originalDefault));
        if (targetHasDefault) Assert.Equal("stale-definition-default", originalDefault.GetString());
        Assert.Single(selection.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>());
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
        using var client = ToolFixture.Client(spy, function, new(fixedValues: fixedValues));
        Task<ChatResponse> Select() => client.GetResponseAsync(NativeFixture.Messages(), new ChatOptions { Tools = [function] }, TestContext.Current.CancellationToken);
        if (!accepted)
        {
            var error = await Assert.ThrowsAsync<TypeSafeSchemaException>(Select);
            Assert.Equal("/properties/value", error.Pointer);
            Assert.Equal(0, spy.Calls);
            Assert.Equal(0, handler.Sends);
            Assert.Equal(0, function.Calls);
            Assert.Equal(0, invocations);
        }
        else
        {
            var selection = await Select();
            Assert.Equal(ChatFinishReason.ToolCalls, selection.FinishReason);
            Assert.Equal(number, Assert.IsType<JsonElement>(Assert.Single(selection.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>()).Arguments!["value"]).GetRawText());
            Assert.Equal(0, invocations);
            var result = await ToolFixture.ExecuteProposedAsync(selection, function, TestContext.Current.CancellationToken);
            Assert.Equal(long.Parse(number, System.Globalization.CultureInfo.InvariantCulture),
                Assert.IsType<JsonElement>(result.Result).GetInt64());
            Assert.Single(selection.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>());
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
        using var client = ToolFixture.Client(spy, function, new(fixedValues: fixedValues));
        Task<ChatResponse> Select() => client.GetResponseAsync(NativeFixture.Messages(), new ChatOptions { Tools = [function] }, TestContext.Current.CancellationToken);
        if (!accepted)
        {
            var error = await Assert.ThrowsAsync<TypeSafeSchemaException>(Select);
            Assert.Equal("/properties/value", error.Pointer);
            Assert.Contains("numeric constraint", error.Message);
            Assert.Equal(0, spy.Calls);
            Assert.Equal(0, handler.Sends);
            Assert.Equal(0, function.Calls);
            Assert.Equal(0, invocations);
        }
        else
        {
            var selection = await Select();
            Assert.Equal(ChatFinishReason.ToolCalls, selection.FinishReason);
            Assert.Equal(9007199254740993L, Assert.IsType<JsonElement>(Assert.Single(selection.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>()).Arguments!["value"]).GetInt64());
            Assert.Equal(0, invocations);
            var result = await ToolFixture.ExecuteProposedAsync(selection, function, TestContext.Current.CancellationToken);
            Assert.Equal(9007199254740993L, Assert.IsType<JsonElement>(result.Result).GetInt64());
            Assert.Single(selection.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>());
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
        Description = "Preview a synthetic support operation without persistence."
    };
    internal static ToolProbe FixedFunction(string name, Action invoked) =>
        new(AIFunctionFactory.Create((int ticketId) => { invoked(); return ticketId; }, Options(name)));
    internal static TypeSafeToolOptions FixedOptions() =>
        new(fixedValues: new Dictionary<string, JsonElement> { ["/ticketId"] = NativeFixture.Json("17") });
    internal static ToolProbe InferredFunction(Action invoked) =>
        new(AIFunctionFactory.Create((TestDestination destination, bool followup) =>
        {
            invoked();
            return new { destination, followup };
        }, Options("preview")));
    internal static TypeSafeChatClient Client(IChatClient native, AIFunction function, TypeSafeToolOptions config) =>
        new(native, NativeFixture.Settings with { Tools = new Dictionary<string, TypeSafeToolOptions> { [function.Name] = config } });

    // Test-only replay transport. Execution still goes through the real pinned middleware;
    // a second provider request is an error, not a scripted successful post-execution turn.
    internal static async Task<FunctionResultContent> ExecuteProposedAsync(ChatResponse proposal, AIFunction function, CancellationToken token)
    {
        using var transport = new ProposalClient(proposal);
        using var executing = transport.AsBuilder().UseFunctionInvocation(configure: invocation =>
        {
            invocation.MaximumConsecutiveErrorsPerRequest = 0;
            invocation.FunctionInvoker = async (context, cancellationToken) =>
            {
                Assert.Same(function, context.Function);
                cancellationToken.ThrowIfCancellationRequested();
                context.Terminate = true;
                return await context.Function.InvokeAsync(context.Arguments, cancellationToken);
            };
        }).Build();
        var result = await executing.GetResponseAsync("Execute the approved synthetic proposal.",
            new ChatOptions { Tools = [function] }, token);
        Assert.Equal(1, transport.Calls);
        return Assert.Single(result.Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>());
    }

    private sealed class ProposalClient(ChatResponse proposal) : IChatClient
    {
        internal int Calls;
        public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
        {
            if (++Calls != 1) throw new InvalidOperationException("Unexpected post-execution inference.");
            return Task.FromResult(proposal);
        }
        public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default) => throw new NotSupportedException();
        public object? GetService(Type serviceType, object? serviceKey = null) => null;
        public void Dispose() { }
    }
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
