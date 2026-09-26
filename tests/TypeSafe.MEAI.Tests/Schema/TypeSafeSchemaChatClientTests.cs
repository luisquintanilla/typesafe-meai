using System.Net;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using TypeSafe.MEAI.Tests.Fixtures;
using Xunit;

namespace TypeSafe.MEAI.Tests.Schema;

[Trait("Stage", "C")]
public sealed class TypeSafeSchemaChatClientTests
{
    [Theory]
    [InlineData(null)]
    [InlineData("Shared ticket policy.")]
    public async Task TypedApi_RootAndNestedDescriptionAttributesReachOnlyDescendantQuestions(string? shared)
    {
        using var handler = SchemaFixture.Handler();
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        IChatClient client = spy;
        var options = new ChatOptions { Instructions = shared };

        var response = await client.GetResponseAsync<DescribedDecision>(
            [new(ChatRole.User, "Synthetic ticket.")], SchemaFixture.Serializer, options,
            cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(response.Result.Approved);
        Assert.True(response.Result.Details.Followup);
        var message = Assert.Single(response.Messages);
        Assert.Equal(ChatRole.Assistant, message.Role);
        Assert.Equal(2, message.Contents.Count);
        var projected = NativeFixture.Json(Assert.Single(message.Contents.OfType<TextContent>()).Text!);
        Assert.True(projected.GetProperty("approved").GetBoolean());
        Assert.True(projected.GetProperty("case/~").GetProperty("follow/~").GetBoolean());
        var evidence = Assert.Single(message.Contents.OfType<JevResultContent>());
        Assert.Equal(2, evidence.Answers.Count);
        Assert.Equal(18, response.Usage!.TotalTokenCount);
        var questions = NativeFixture.Json(Assert.Single(handler.Requests).Body!).GetProperty("questions");
        Assert.Equal(2, questions.EnumerateObject().Count());
        AssertSchemaInstructions(questions.GetProperty("q0000"), options.Instructions, "Overall approval predicate.",
            ("responseFormat", "", "Root ticket context."),
            ("schema", "", "Root ticket context."));
        AssertSchemaInstructions(questions.GetProperty("q0001"), options.Instructions, "Escalation predicate.",
            ("responseFormat", "", "Root ticket context."),
            ("schema", "", "Root ticket context."),
            ("schema", "/case~1~0", "Nested escalation context."));
        var format = Assert.IsType<ChatResponseFormatJson>(Assert.Single(spy.Options)!.ResponseFormat);
        Assert.Equal("Root ticket context.", format.SchemaDescription);
        Assert.Equal("Root ticket context.", format.Schema!.Value.GetProperty("description").GetString());
        Assert.Equal("Nested escalation context.", format.Schema.Value.GetProperty("properties")
            .GetProperty("case/~").GetProperty("description").GetString());
        Assert.Null(options.ResponseFormat);
        Assert.Equal(shared, options.Instructions);
        Assert.Equal(1, spy.Calls);
        Assert.Equal(1, handler.Sends);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task SchemaDescriptions_ObjectReferenceOverlaysPreserveAncestorsAndLeafOverrides(bool overrideLeaf)
    {
        var schema = NativeFixture.Json("""
            {"type":"object","description":"Root schema context.",
             "$defs":{"node/~":{"type":"object","description":"Definition object context.",
                                "properties":{"flag/~":{"type":"boolean","description":"Definition leaf predicate."}}},
                      "alias":{"$ref":"#/$defs/node~1~0","description":"Alias object context."}},
             "properties":{"left/~":{"$ref":"#/$defs/alias","description":"Left use-site context."},
                           "right/~":{"$ref":"#/$defs/alias","description":"Right use-site context."},
                           "inherited/~":{"$ref":"#/$defs/alias"}}}
            """);
        string original = schema.GetRawText();
        var fields = new Dictionary<string, TypeSafeFieldOptions>();
        if (overrideLeaf) fields["/left~1~0/flag~1~0"] = new("Host leaf predicate.");
        using var handler = SchemaFixture.Handler();
        using var http = new HttpClient(handler);
        using IChatClient client = new TypeSafeChatClient(http, NativeFixture.Settings with { Fields = fields }, NativeFixture.Key);
        var format = ChatResponseFormat.ForJsonSchema(schema, schemaDescription: "Response format context.");
        var options = new ChatOptions { ResponseFormat = format, Instructions = "Shared host policy." };

        var response = await client.GetResponseAsync(NativeFixture.Messages(), options, TestContext.Current.CancellationToken);

        var questions = NativeFixture.Json(Assert.Single(handler.Requests).Body!).GetProperty("questions");
        Assert.Equal(3, questions.EnumerateObject().Count());
        AssertSchemaInstructions(questions.GetProperty("q0000"), options.Instructions,
            overrideLeaf ? "Host leaf predicate." : "Definition leaf predicate.",
            ("responseFormat", "", "Response format context."),
            ("schema", "", "Root schema context."),
            ("schema", "/left~1~0", "Left use-site context."));
        AssertSchemaInstructions(questions.GetProperty("q0001"), options.Instructions, "Definition leaf predicate.",
            ("responseFormat", "", "Response format context."),
            ("schema", "", "Root schema context."),
            ("schema", "/right~1~0", "Right use-site context."));
        AssertSchemaInstructions(questions.GetProperty("q0002"), options.Instructions, "Definition leaf predicate.",
            ("responseFormat", "", "Response format context."),
            ("schema", "", "Root schema context."),
            ("schema", "/inherited~1~0", "Alias object context."));
        var projected = NativeFixture.Json(response.Text);
        Assert.Equal(["left/~", "right/~", "inherited/~"], projected.EnumerateObject().Select(p => p.Name));
        Assert.All(projected.EnumerateObject(), p => Assert.True(p.Value.GetProperty("flag/~").GetBoolean()));
        Assert.Equal(3, Assert.Single(response.Messages[^1].Contents.OfType<JevResultContent>()).Answers.Count);
        Assert.Equal(original, schema.GetRawText());
        Assert.Equal(original, format.Schema!.Value.GetRawText());
        Assert.Same(format, options.ResponseFormat);
        Assert.Equal("Shared host policy.", options.Instructions);
        Assert.Equal("Response format context.", format.SchemaDescription);
        Assert.Equal(1, handler.Sends);
    }

    [Theory]
    [InlineData("root")]
    [InlineData("inline-object")]
    [InlineData("reference-object")]
    [InlineData("response-format")]
    public async Task SchemaDescriptions_OnlyDescriptionChangesMissNativeCacheWhileThresholdsReuse(string changed)
    {
        var schema = NativeFixture.Json("""
            {"type":"object","description":"Root context A.",
             "$defs":{"node":{"type":"object","description":"Definition context.",
                             "properties":{"value":{"type":"boolean","description":"Leaf predicate."}}}},
             "properties":{"inline/~":{"type":"object","description":"Inline context A.",
                                       "properties":{"value":{"type":"boolean","description":"Leaf predicate."}}},
                           "ref/~":{"$ref":"#/$defs/node","description":"Reference context A."}}}
            """);
        string original = schema.GetRawText();
        var revised = JsonNode.Parse(original)!.AsObject();
        string formatDescription = "Format context A.";
        switch (changed)
        {
            case "root": revised["description"] = "Root context B."; break;
            case "inline-object": revised["properties"]!["inline/~"]!["description"] = "Inline context B."; break;
            case "reference-object": revised["properties"]!["ref/~"]!["description"] = "Reference context B."; break;
            case "response-format": formatDescription = "Format context B."; break;
            default: throw new ArgumentOutOfRangeException(nameof(changed));
        }
        var originalOptions = new ChatOptions
        {
            Instructions = "Shared cache policy.",
            ResponseFormat = ChatResponseFormat.ForJsonSchema(schema, schemaDescription: "Format context A.")
        };
        var revisedOptions = new ChatOptions
        {
            Instructions = originalOptions.Instructions,
            ResponseFormat = ChatResponseFormat.ForJsonSchema(JsonSerializer.SerializeToElement(revised), schemaDescription: formatDescription)
        };
        using var handler = SchemaFixture.Handler(noul: 0.9);
        var registrations = new ServiceCollection();
        registrations.AddDistributedMemoryCache();
        registrations.AddTypeSafeChatClient("relaxed", NativeFixture.Key, NativeFixture.Settings, useCache: true)
            .ConfigurePrimaryHttpMessageHandler(() => handler);
        registrations.AddTypeSafeChatClient("strict", NativeFixture.Key, NativeFixture.Settings with
        { Policy = new(new TypeSafeBooleanPolicy(0.2, 0.95), 0.8) }, useCache: true)
            .ConfigurePrimaryHttpMessageHandler(() => handler);
        using var services = registrations.BuildServiceProvider();
        var relaxed = services.GetRequiredKeyedService<IChatClient>("relaxed");
        var strict = services.GetRequiredKeyedService<IChatClient>("strict");

        var first = await relaxed.GetResponseAsync(NativeFixture.Messages(), originalOptions, TestContext.Current.CancellationToken);
        var firstEvidence = Assert.Single(first.Messages[^1].Contents.OfType<JevResultContent>());
        var rejected = await Assert.ThrowsAsync<TypeSafeAbstentionException>(() =>
            strict.GetResponseAsync(NativeFixture.Messages(), originalOptions, TestContext.Current.CancellationToken));
        Assert.Equal(["/inline~1~0/value", "/ref~1~0/value"], rejected.Rejections.Select(r => r.Pointer));
        Assert.Equal(firstEvidence.Data.ToArray(), Assert.Single(rejected.Evidence).Data.ToArray());
        Assert.Equal(1, handler.Sends);

        var second = await relaxed.GetResponseAsync(NativeFixture.Messages(), revisedOptions, TestContext.Current.CancellationToken);
        Assert.Equal(2, handler.Sends);
        var requests = handler.Requests.ToArray();
        var before = NativeFixture.Json(requests[0].Body!).GetProperty("questions");
        var after = NativeFixture.Json(requests[1].Body!).GetProperty("questions");
        Assert.Equal(2, before.EnumerateObject().Count());
        Assert.Equal(2, after.EnumerateObject().Count());
        Assert.Equal(changed == "reference-object", JsonElement.DeepEquals(before.GetProperty("q0000"), after.GetProperty("q0000")));
        Assert.Equal(changed == "inline-object", JsonElement.DeepEquals(before.GetProperty("q0001"), after.GetProperty("q0001")));
        AssertSchemaInstructions(after.GetProperty("q0000"), originalOptions.Instructions, "Leaf predicate.",
            ("responseFormat", "", formatDescription),
            ("schema", "", changed == "root" ? "Root context B." : "Root context A."),
            ("schema", "/inline~1~0", changed == "inline-object" ? "Inline context B." : "Inline context A."));
        AssertSchemaInstructions(after.GetProperty("q0001"), originalOptions.Instructions, "Leaf predicate.",
            ("responseFormat", "", formatDescription),
            ("schema", "", changed == "root" ? "Root context B." : "Root context A."),
            ("schema", "/ref~1~0", changed == "reference-object" ? "Reference context B." : "Reference context A."));

        var revisedRejection = await Assert.ThrowsAsync<TypeSafeAbstentionException>(() =>
            strict.GetResponseAsync(NativeFixture.Messages(), revisedOptions, TestContext.Current.CancellationToken));
        Assert.Equal(["/inline~1~0/value", "/ref~1~0/value"], revisedRejection.Rejections.Select(r => r.Pointer));
        var secondEvidence = Assert.Single(second.Messages[^1].Contents.OfType<JevResultContent>());
        Assert.Equal(secondEvidence.Data.ToArray(), Assert.Single(revisedRejection.Evidence).Data.ToArray());
        var originalAgain = await relaxed.GetResponseAsync(NativeFixture.Messages(), originalOptions, TestContext.Current.CancellationToken);
        Assert.Equal(2, handler.Sends);
        foreach (var accepted in new[] { first, second, originalAgain })
        {
            var projected = NativeFixture.Json(accepted.Text);
            Assert.True(projected.GetProperty("inline/~").GetProperty("value").GetBoolean());
            Assert.True(projected.GetProperty("ref/~").GetProperty("value").GetBoolean());
            Assert.Equal(18, accepted.Usage!.TotalTokenCount);
            Assert.All(Assert.Single(accepted.Messages[^1].Contents.OfType<JevResultContent>()).Answers.Values,
                answer => Assert.Equal(0.9, Assert.IsType<NoulAnswer>(answer).Noul));
        }
        Assert.Equal(original, schema.GetRawText());
        Assert.Equal("Format context A.", Assert.IsType<ChatResponseFormatJson>(originalOptions.ResponseFormat).SchemaDescription);
    }

    [Fact]
    public async Task TypedApi_NormalOpenOptionalDtoUsesClientPolicyAndStandardDescriptions()
    {
        using var handler = SchemaFixture.Handler();
        using var http = new HttpClient(handler);
        using IChatClient client = new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key);
        var options = new ChatOptions { Instructions = "Classify this synthetic support ticket." };
        var response = await client.GetResponseAsync<OrdinaryDecision>(
            [new(ChatRole.User, "Payment failed; choose a team and decide whether to follow up.")],
            options: options, cancellationToken: TestContext.Current.CancellationToken);
        Assert.True(response.Result.Followup);
        Assert.Equal(TestDestination.Billing, response.Result.Destination);
        var message = Assert.Single(response.Messages);
        Assert.Equal(ChatRole.Assistant, message.Role);
        Assert.Equal(2, message.Contents.Count);
        var projected = NativeFixture.Json(Assert.Single(message.Contents.OfType<TextContent>()).Text!);
        Assert.True(projected.GetProperty("followup").GetBoolean());
        Assert.Equal("Billing", projected.GetProperty("destination").GetString());
        Assert.Equal(2, projected.EnumerateObject().Count());
        var evidence = Assert.Single(message.Contents.OfType<JevResultContent>());
        Assert.Equal(2, evidence.Answers.Count);
        Assert.Equal(18, response.Usage!.TotalTokenCount);
        var questions = NativeFixture.Json(Assert.Single(handler.Requests).Body!).GetProperty("questions");
        Assert.Equal("Does the payment issue need follow-up?", questions.GetProperty("q0000").GetProperty("instructions").GetProperty("fieldInstructions").GetString());
        Assert.Equal("Choose the support team.", questions.GetProperty("q0001").GetProperty("instructions").GetProperty("fieldInstructions").GetString());
        Assert.All(questions.EnumerateObject(), q => Assert.Equal(options.Instructions,
            q.Value.GetProperty("instructions").GetProperty("sharedInstructions").GetString()));
        Assert.All(questions.GetProperty("q0001").GetProperty("criteria").EnumerateObject(), p => Assert.Equal(JsonValueKind.Null, p.Value.ValueKind));
        Assert.Null(options.ResponseFormat);
        Assert.Empty(NativeFixture.Settings.Fields);
        Assert.Equal(1, handler.Sends);
    }

    [Fact]
    public async Task RegisteredNativeCache_ReappliesChangedClientPolicyAndOwnsPerCallMetadata()
    {
        using var handler = SchemaFixture.Handler(noul: 0.9);
        var registrations = new ServiceCollection();
        registrations.AddDistributedMemoryCache();
        registrations.AddTypeSafeChatClient("relaxed", NativeFixture.Key, NativeFixture.Settings, useCache: true)
            .ConfigurePrimaryHttpMessageHandler(() => handler);
        registrations.AddTypeSafeChatClient("strict", NativeFixture.Key, NativeFixture.Settings with
        { Policy = new(new TypeSafeBooleanPolicy(0.2, 0.95), 0.8) }, useCache: true)
            .ConfigurePrimaryHttpMessageHandler(() => handler);
        using var services = registrations.BuildServiceProvider();
        var relaxed = services.GetRequiredKeyedService<IChatClient>("relaxed");
        var strict = services.GetRequiredKeyedService<IChatClient>("strict");
        var first = await relaxed.GetResponseAsync<bool>("Synthetic flag", cancellationToken: TestContext.Current.CancellationToken);
        Assert.True(first.Result);
        var firstEvidence = Assert.Single(first.Messages[^1].Contents.OfType<JevResultContent>());
        firstEvidence.AdditionalProperties!["typesafe.v1.fields"] = "caller-mutated";
        firstEvidence.AdditionalProperties["caller"] = "must not leak";
        var rejected = await Assert.ThrowsAsync<TypeSafeAbstentionException>(() =>
            strict.GetResponseAsync<bool>("Synthetic flag", cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal("/data", Assert.Single(rejected.Rejections).Pointer);
        var strictEvidence = Assert.Single(rejected.Evidence);
        Assert.Equal(0.9, Assert.IsType<NoulAnswer>(strictEvidence.Answers["q0000"]).Noul);
        Assert.Equal("/data", Assert.IsType<JsonElement>(strictEvidence.AdditionalProperties!["typesafe.v1.fields"]).GetProperty("q0000").GetString());
        Assert.False(strictEvidence.AdditionalProperties.ContainsKey("caller"));
        Assert.Equal(firstEvidence.Data.ToArray(), strictEvidence.Data.ToArray());
        Assert.NotSame(firstEvidence, strictEvidence);
        var again = await relaxed.GetResponseAsync<bool>("Synthetic flag", cancellationToken: TestContext.Current.CancellationToken);
        Assert.True(again.Result);
        Assert.Equal(18, again.Usage!.TotalTokenCount);
        Assert.Equal(1, handler.Sends);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("synthetic-host-model")]
    public async Task TypedApi_ProjectsRequiredNestedDtoAndResolvesNativeDefaultWithoutMutatingCaller(string? model)
    {
        using var handler = SchemaFixture.Handler();
        using var services = NativeFixture.Services(handler, cache: true);
        using var spy = new SchemaSpy(services.Client());
        var bindings = SchemaFixture.Bindings();
        using var adapter = new TypeSafeChatClient(spy, NativeFixture.Settings with { Fields = bindings });
        bindings.Clear();
        var options = new ChatOptions { ModelId = model, Instructions = "Shared synthetic policy." };

        var response = await adapter.GetResponseAsync<DecisionDto>([SchemaFixture.State.ToMessage()], SchemaFixture.Serializer, options,
            cancellationToken: TestContext.Current.CancellationToken);
        var decision = await adapter.GetResponseAsync<DecisionDto>(NativeFixture.Messages(), SchemaFixture.Serializer, options,
            cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(response.Result.Followup);
        Assert.Equal(TestDestination.Billing, response.Result.Details.Destination);
        Assert.Equal(1.25, response.Result.Details.Quality);
        Assert.Equal(TestDestination.Billing, decision.Result.Details.Destination);
        Assert.Equal(1.25, decision.Result.Details.Quality);
        Assert.Equal(model ?? "jev-1.13.0", response.ModelId);
        Assert.Equal(11, response.Usage!.InputTokenCount);
        Assert.Equal(7, response.Usage.OutputTokenCount);
        Assert.Equal(18, response.Usage.TotalTokenCount);
        Assert.Equal(2, spy.Calls);
        Assert.Equal(1, handler.Sends);
        Assert.Equal(model, options.ModelId);
        Assert.Equal("Shared synthetic policy.", options.Instructions);
        Assert.Null(options.ResponseFormat);
        Assert.Null(options.AdditionalProperties);
        var nativeOptions = spy.Options[0]!;
        Assert.NotSame(options, nativeOptions);
        Assert.Equal(model ?? "jev-1.13.0", nativeOptions.ModelId);
        Assert.Null(nativeOptions.Instructions);
        Assert.Null(Assert.IsType<ChatResponseFormatJson>(nativeOptions.ResponseFormat).Schema);
        var body = NativeFixture.Json(Assert.Single(handler.Requests).Body!);
        Assert.Equal(model ?? "jev-1.13.0", body.GetProperty("model").GetString());
        Assert.True(JsonElement.DeepEquals(NativeFixture.Json(NativeFixture.State), body.GetProperty("state")));
        Assert.Equal(3, body.GetProperty("questions").EnumerateObject().Count());
        foreach (var question in body.GetProperty("questions").EnumerateObject())
        {
            var instructions = question.Value.GetProperty("instructions");
            Assert.Equal("Shared synthetic policy.", instructions.GetProperty("sharedInstructions").GetString());
            Assert.False(string.IsNullOrWhiteSpace(instructions.GetProperty("fieldInstructions").GetString()));
        }
        Assert.Equal("Billing", NativeFixture.Json(response.Text).GetProperty("details").GetProperty("destination").GetString());
    }

    [Fact]
    public async Task TypedApi_PrimitiveBooleanUsesRealMeaiDataWrapper()
    {
        using var handler = SchemaFixture.Handler();
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        IChatClient adapter = spy;
        var response = await adapter.GetResponseAsync<bool>([SchemaFixture.State.ToMessage()],
            cancellationToken: TestContext.Current.CancellationToken);
        Assert.True(response.Result);
        Assert.Equal(0.9, Assert.IsType<NoulAnswer>(Assert.Single(response.Messages[^1].Contents.OfType<JevResultContent>()).Answers["q0000"]).Noul);
        Assert.Equal("{\"data\":true}", Assert.Single(response.Messages[^1].Contents.OfType<TextContent>()).Text);
        Assert.Equal(2, response.Messages[^1].Contents.Count);
        Assert.Equal(1, spy.Calls);
        Assert.Equal(1, handler.Sends);
        Assert.Equal("noul", Assert.Single(NativeFixture.Json(Assert.Single(handler.Requests).Body!)
            .GetProperty("questions").EnumerateObject()).Value.GetProperty("type").GetString());
    }

    [Fact]
    public async Task NativeCache_ReusesThresholdChangesButMissesRubricAndBothInstructionKinds()
    {
        using var handler = SchemaFixture.Handler();
        using var services = NativeFixture.Services(handler, cache: true);
        using var spy = new SchemaSpy(services.Client());
        using var relaxed = new TypeSafeChatClient(spy, NativeFixture.Settings with { Fields = SchemaFixture.Bindings() });
        var initial = await relaxed.GetResponseAsync<DecisionDto>(NativeFixture.Messages(), cancellationToken: TestContext.Current.CancellationToken);
        Assert.True(initial.Result.Followup);
        var initialEvidence = Assert.Single(initial.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>());
        Assert.Equal(1, handler.Sends);

        var strictBindings = SchemaFixture.Bindings();
        strictBindings["/details/destination"] = SchemaFixture.Choice(0.95);
        using var strict = new TypeSafeChatClient(spy, NativeFixture.Settings with { Fields = strictBindings });
        var changed = await Assert.ThrowsAsync<TypeSafeAbstentionException>(() => strict.GetResponseAsync<DecisionDto>(NativeFixture.Messages(), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal("/details/destination", Assert.Single(changed.Rejections).Pointer);
        Assert.True(JsonElement.DeepEquals(initialEvidence.Json, Assert.Single(changed.Evidence).Json));
        Assert.Equal(1, handler.Sends);
        Assert.Null(spy.Responses[1].RawRepresentation);

        var original = SchemaFixture.Score().Rubric!;
        using var rubric = new TypeSafeChatClient(spy, NativeFixture.Settings with { Fields = SchemaFixture.Bindings(rubric: [original[1], original[0], original[2]]) });
        var reordered = await rubric.GetResponseAsync<DecisionDto>(NativeFixture.Messages(), cancellationToken: TestContext.Current.CancellationToken);
        Assert.True(reordered.Result.Followup);
        Assert.Equal(2, handler.Sends);
        Assert.Equal("poor", Assert.IsType<ScoreAnswer>(initialEvidence.Answers["q0002"]).Legend.GetProperty("0").GetString());
        Assert.Equal("partial", Assert.IsType<ScoreAnswer>(Assert.Single(reordered.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>()).Answers["q0002"]).Legend.GetProperty("0").GetProperty("level").GetString());

        var shared = await relaxed.GetResponseAsync<DecisionDto>(NativeFixture.Messages(),
            options: new ChatOptions { Instructions = "Revised shared synthetic instruction." }, cancellationToken: TestContext.Current.CancellationToken);
        Assert.True(shared.Result.Followup);
        Assert.Equal(3, handler.Sends);
        using var field = new TypeSafeChatClient(spy, NativeFixture.Settings with { Fields = SchemaFixture.Bindings(scoreInstructions: "A revised synthetic quality judgment.") });
        var instructed = await field.GetResponseAsync<DecisionDto>(NativeFixture.Messages(), cancellationToken: TestContext.Current.CancellationToken);
        Assert.True(instructed.Result.Followup);
        Assert.Equal(4, handler.Sends);
        Assert.Equal(5, spy.Calls);
        Assert.All(new[] { initial, reordered, shared, instructed }, d => Assert.Equal(1.25, d.Result.Details.Quality));
    }

    [Fact]
    public async Task ProjectedResponse_RoundtripsJevContentSeparatelyFromTypedJson()
    {
        using var handler = SchemaFixture.Handler();
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        using var adapter = new TypeSafeChatClient(spy, NativeFixture.Settings with { Fields = SchemaFixture.Bindings() });
        var response = await adapter.GetResponseAsync<DecisionDto>([SchemaFixture.State.ToMessage()],
            SchemaFixture.Serializer, cancellationToken: TestContext.Current.CancellationToken);
        var json = NativeFixture.ContentJson();
        var serialized = JsonSerializer.SerializeToElement<ChatResponse>(response, json);
        var restored = serialized.Deserialize<ChatResponse>(json)!;
        Assert.Null(restored.RawRepresentation);
        var evidence = Assert.Single(restored.Messages[^1].Contents.OfType<JevResultContent>());
        Assert.IsType<JsonElement>(evidence.AdditionalProperties!["typesafe.v1.fields"]);
        Assert.True(JsonElement.DeepEquals(Assert.Single(response.Messages[^1].Contents.OfType<JevResultContent>()).Json, evidence.Json));
        Assert.Equal("synthetic-preserved", evidence.Answers["q0001"].Json.GetProperty("vendor_answer").GetString());
        Assert.Equal("retained", evidence.Json.GetProperty("usage").GetProperty("vendor_usage").GetString());
        Assert.Equal(JsonValueKind.Null, Assert.IsType<ScoreAnswer>(evidence.Answers["q0002"]).Legend.GetProperty("2").ValueKind);
        Assert.Equal(17, evidence.Json.GetProperty("vendor_envelope").GetProperty("revision").GetInt32());
        var projection = NativeFixture.Json(restored.Text);
        Assert.False(projection.TryGetProperty("answers", out _));
        Assert.True(projection.GetProperty("followup").GetBoolean());
        Assert.Equal(1.25, projection.GetProperty("details").GetProperty("quality").GetDouble());
        Assert.Equal(response.ModelId, restored.ModelId);
        Assert.Equal(18, restored.Usage!.TotalTokenCount);
        Assert.Equal(1, handler.Sends);
    }

    [Fact]
    public async Task BufferedStreaming_ReconstructsProjectionEvidenceAndCountsUsageOnce()
    {
        using var handler = SchemaFixture.Handler();
        using var services = NativeFixture.Services(handler, cache: true);
        using var adapter = new TypeSafeChatClient(services.Client(), NativeFixture.Settings with { Fields = SchemaFixture.Bindings() });
        var options = SchemaFixture.Options(SchemaFixture.Schema);
        var normal = await adapter.GetResponseAsync([SchemaFixture.State.ToMessage()], options, TestContext.Current.CancellationToken);
        List<ChatResponseUpdate> updates = [];
        var reconstructed = await Observe(adapter.GetStreamingResponseAsync([SchemaFixture.State.ToMessage()], options,
            TestContext.Current.CancellationToken), updates, TestContext.Current.CancellationToken).ToChatResponseAsync(TestContext.Current.CancellationToken);
        Assert.Equal(normal.Text, reconstructed.Text);
        Assert.Equal(normal.ModelId, reconstructed.ModelId);
        Assert.Equal(ChatFinishReason.Stop, reconstructed.FinishReason);
        Assert.True(JsonElement.DeepEquals(Assert.Single(normal.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>()).Json,
            Assert.Single(reconstructed.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>()).Json));
        Assert.Equal(18, reconstructed.Usage!.TotalTokenCount);
        Assert.Equal(18, Assert.Single(updates.SelectMany(u => u.Contents).OfType<UsageContent>()).Details.TotalTokenCount);
        Assert.Equal(1, handler.Sends);
    }

    [Theory]
    [InlineData("before")]
    [InlineData("send")]
    [InlineData("enumeration")]
    public async Task BufferedStreaming_CancellationStopsWorkOrSubsequentUpdates(string phase)
    {
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var handler = new RecordingHandler(async (request, _, token) =>
        {
            if (phase == "send")
            {
                entered.SetResult();
                await Task.Delay(Timeout.InfiniteTimeSpan, token);
            }
            return NativeFixture.Response(SchemaFixture.Envelope(NativeFixture.Json(await request.Content!.ReadAsStringAsync(token))));
        });
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        using var adapter = new TypeSafeChatClient(spy, NativeFixture.Settings with { Fields = SchemaFixture.Bindings() });
        if (phase == "before") cancellation.Cancel();
        await using var iterator = adapter.GetStreamingResponseAsync([SchemaFixture.State.ToMessage()],
            SchemaFixture.Options(SchemaFixture.Schema), cancellation.Token).GetAsyncEnumerator(cancellation.Token);
        var next = iterator.MoveNextAsync().AsTask();
        if (phase == "send")
        {
            await entered.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
            cancellation.Cancel();
        }
        if (phase == "enumeration")
        {
            Assert.True(await next);
            Assert.Contains("Billing", string.Concat(iterator.Current.Contents.OfType<TextContent>().Select(t => t.Text)));
            cancellation.Cancel();
            next = iterator.MoveNextAsync().AsTask();
        }
        var error = await Assert.ThrowsAnyAsync<OperationCanceledException>(() => next);
        Assert.Equal(cancellation.Token, error.CancellationToken);
        Assert.Equal(phase == "before" ? 0 : 1, spy.Calls);
        Assert.Equal(phase == "before" ? 0 : 1, handler.Sends);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ResponseCancellation_AfterNativeCompletionPreventsEvidenceAndTypedProjection(bool typed)
    {
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        using var handler = typed ? SchemaFixture.Handler() : new RecordingHandler();
        using var services = NativeFixture.Services(handler, cache: true);
        using var spy = new SchemaSpy(services.Client());
        using var cancelAfterResponse = new CancelOnResponseClient(spy, cancellation);
        int validations = 0;
        using IChatClient client = new TypeSafeChatClient(cancelAfterResponse, NativeFixture.Settings with
        {
            ValidateResult = _ => { validations++; return null; }
        });

        var error = await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            if (typed)
                await client.GetResponseAsync<bool>("Synthetic follow-up decision", cancellationToken: cancellation.Token);
            else
                await client.GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), cancellation.Token);
        });

        Assert.Equal(cancellation.Token, error.CancellationToken);
        Assert.Equal(0, validations);
        Assert.Equal(1, spy.Calls);
        Assert.Equal(1, handler.Sends);
        var native = Assert.Single(spy.Responses);
        var evidence = Assert.IsType<JevResultContent>(Assert.Single(Assert.Single(native.Messages).Contents));
        Assert.Equal("", native.Text);
        Assert.Equal("jev-1.13.0", evidence.Model);
        Assert.Equal(18, evidence.TotalTokens);
        Assert.Equal(typed ? 0.9 : 0.82, Assert.Single(evidence.Answers.Values.OfType<NoulAnswer>()).Noul);
    }

    [Fact]
    public async Task TypedApi_RejectsPromptAppendingButAcceptsHistoryThroughDecoratedHttp()
    {
        using var handler = SchemaFixture.Handler();
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        using var adapter = new TypeSafeChatClient(spy, NativeFixture.Settings with { Fields = SchemaFixture.Bindings() });
        var error = await Assert.ThrowsAsync<NotSupportedException>(() => adapter.GetResponseAsync<DecisionDto>(
            [SchemaFixture.State.ToMessage()], SchemaFixture.Serializer, useJsonSchemaResponseFormat: false,
            cancellationToken: TestContext.Current.CancellationToken));
        Assert.Contains("Free-form", error.Message);
        var history = await adapter.GetResponseAsync(
            [new ChatMessage(ChatRole.System, "Synthetic history."), SchemaFixture.State.ToMessage()],
            SchemaFixture.Options(SchemaFixture.Schema), TestContext.Current.CancellationToken);
        Assert.True(NativeFixture.Json(history.Text).GetProperty("followup").GetBoolean());
        Assert.Equal(1, spy.Calls);
        Assert.Equal(1, handler.Sends);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task TypedApi_ServiceFailureOrMissingRequiredAnswerCannotBecomeDecision(bool missingAnswer)
    {
        using var handler = new RecordingHandler(async (request, _, token) =>
        {
            if (!missingAnswer) return NativeFixture.Response("Synthetic schema service failure.", 422);
            var envelope = JsonNode.Parse(SchemaFixture.Envelope(NativeFixture.Json(await request.Content!.ReadAsStringAsync(token))))!;
            envelope["answers"]!.AsObject().Remove("q0002");
            return NativeFixture.Response(envelope.ToJsonString());
        });
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        using var adapter = new TypeSafeChatClient(spy, NativeFixture.Settings with { Fields = SchemaFixture.Bindings() });
        Task<ChatResponse<DecisionDto>> Invoke() => adapter.GetResponseAsync<DecisionDto>(NativeFixture.Messages(),
            cancellationToken: TestContext.Current.CancellationToken);
        if (missingAnswer) await Assert.ThrowsAsync<TypeSafeProtocolException>(Invoke);
        else
        {
            var error = await Assert.ThrowsAsync<TypeSafeApiException>(Invoke);
            Assert.Equal(HttpStatusCode.UnprocessableEntity, error.StatusCode);
        }
        Assert.Equal(1, spy.Calls);
        Assert.Equal(1, handler.Sends);
    }

    [Fact]
    public async Task Adapter_SnapshotsExtraOptionsAndResolvesEffectiveModelBeforeDecoratedEntry()
    {
        var tags = new List<string> { "original" };
        var metadata = new Dictionary<string, object?> { ["tags"] = tags };
        var options = SchemaFixture.Options(SchemaFixture.Schema);
        options.AdditionalProperties = new() { ["synthetic-context"] = metadata };
        int inspections = 0;
        using var handler = SchemaFixture.Handler();
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http,
            NativeFixture.Settings with { ModelId = "synthetic-inner-default" }, NativeFixture.Key));
        using var inspect = new InspectingOptionsClient(spy, captured =>
        {
            inspections++;
            Assert.NotSame(options, captured);
            Assert.NotSame(options.AdditionalProperties, captured.AdditionalProperties);
            Assert.Equal("jev-1.13.0", captured.ModelId);
            var payload = Assert.IsType<JsonElement>(captured.AdditionalProperties!["synthetic-context"]);
            tags[0] = "caller-mutated";
            metadata["extra"] = "new caller metadata";
            options.ModelId = "caller-mutated-model";
            options.Temperature = 0.5f;
            Assert.Equal("original", payload.GetProperty("tags")[0].GetString());
            Assert.False(payload.TryGetProperty("extra", out _));
            Assert.Null(captured.Temperature);
        });
        using var adapter = new TypeSafeChatClient(inspect, NativeFixture.Settings with { Fields = SchemaFixture.Bindings() });

        var response = await adapter.GetResponseAsync([SchemaFixture.State.ToMessage()], options, TestContext.Current.CancellationToken);

        Assert.Equal(1, inspections);
        Assert.Equal(1, spy.Calls);
        Assert.Equal(1, handler.Sends);
        Assert.Equal("jev-1.13.0", response.ModelId);
        Assert.Equal("jev-1.13.0", NativeFixture.Json(Assert.Single(handler.Requests).Body!).GetProperty("model").GetString());
        Assert.True(NativeFixture.Json(response.Text).GetProperty("followup").GetBoolean());
        Assert.Equal(1.25, NativeFixture.Json(response.Text).GetProperty("details").GetProperty("quality").GetDouble());
        Assert.Same(metadata, options.AdditionalProperties["synthetic-context"]);
        Assert.Equal("caller-mutated", tags[0]);
        Assert.Equal("caller-mutated-model", options.ModelId);
        Assert.Equal(0.5f, options.Temperature);
    }

    private static async IAsyncEnumerable<ChatResponseUpdate> Observe(IAsyncEnumerable<ChatResponseUpdate> source,
        List<ChatResponseUpdate> captured, [EnumeratorCancellation] CancellationToken token)
    {
        await foreach (var update in source.WithCancellation(token))
        {
            captured.Add(update);
            yield return update;
        }
    }

    private sealed class InspectingOptionsClient(IChatClient inner, Action<ChatOptions> inspect) : DelegatingChatClient(inner)
    {
        public override Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            inspect(Assert.IsType<ChatOptions>(options));
            return InnerClient.GetResponseAsync(messages, options, cancellationToken);
        }

    }

    private sealed class CancelOnResponseClient(IChatClient inner, CancellationTokenSource cancellation) : DelegatingChatClient(inner)
    {
        public override async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            var response = await InnerClient.GetResponseAsync(messages, options, cancellationToken);
            cancellation.Cancel();
            return response;
        }
    }

    private static void AssertSchemaInstructions(JsonElement question, string? shared, string field,
        params (string Source, string Pointer, string Description)[] expected)
    {
        var instructions = question.GetProperty("instructions");
        Assert.Equal(["fieldInstructions", "schemaDescriptions", "sharedInstructions"],
            instructions.EnumerateObject().Select(p => p.Name).Order());
        Assert.Equal(shared, instructions.GetProperty("sharedInstructions").GetString());
        Assert.Equal(field, instructions.GetProperty("fieldInstructions").GetString());
        var descriptions = instructions.GetProperty("schemaDescriptions").EnumerateArray().ToArray();
        Assert.Equal(expected.Length, descriptions.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(["description", "pointer", "source"], descriptions[i].EnumerateObject().Select(p => p.Name).Order());
            Assert.Equal(expected[i].Source, descriptions[i].GetProperty("source").GetString());
            Assert.Equal(expected[i].Pointer, descriptions[i].GetProperty("pointer").GetString());
            Assert.Equal(expected[i].Description, descriptions[i].GetProperty("description").GetString());
        }
    }

    [Description("Root ticket context.")]
    public sealed class DescribedDecision
    {
        [Description("Overall approval predicate.")]
        public bool Approved { get; set; }
        [JsonPropertyName("case/~")]
        public DescribedDetails Details { get; set; } = new();
    }

    [Description("Nested escalation context.")]
    public sealed class DescribedDetails
    {
        [Description("Escalation predicate.")]
        [JsonPropertyName("follow/~")]
        public bool Followup { get; set; }
    }

    public sealed class OrdinaryDecision
    {
        [Description("Does the payment issue need follow-up?")]
        public bool Followup { get; set; }
        [Description("Choose the support team.")]
        public TestDestination Destination { get; set; }
    }
}
