using System.Net;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.AI;
using TypeSafe.MEAI.Tests.Fixtures;
using Xunit;

namespace TypeSafe.MEAI.Tests.Schema;

[Trait("Stage", "C")]
public sealed class TypeSafeSchemaChatClientTests
{
    [Theory]
    [InlineData(null)]
    [InlineData("synthetic-host-model")]
    public async Task TypedApi_ProjectsRequiredNestedDtoWithoutMutatingCallerOrOwningNativeDefault(string? model)
    {
        using var handler = SchemaFixture.Handler();
        using var services = NativeFixture.Services(handler, cache: true);
        using var spy = new SchemaSpy(services.Client());
        var bindings = SchemaFixture.Bindings();
        using var adapter = new TypeSafeSchemaChatClient(spy, bindings);
        bindings.Clear();
        var options = new ChatOptions { ModelId = model, Instructions = "Shared synthetic policy." };

        var response = await adapter.GetResponseAsync<DecisionDto>([SchemaFixture.State.ToMessage()], SchemaFixture.Serializer, options,
            cancellationToken: TestContext.Current.CancellationToken);
        var decision = await adapter.GetTypeSafeDecisionAsync<DecisionDto>(SchemaFixture.State, options, SchemaFixture.Serializer,
            TestContext.Current.CancellationToken);

        Assert.True(response.Result.Followup);
        Assert.Equal(TestDestination.Billing, response.Result.Details.Destination);
        Assert.Equal(1.25, response.Result.Details.Quality);
        Assert.True(decision.Accepted);
        Assert.Empty(decision.Rejections);
        Assert.Equal(TestDestination.Billing, decision.Value.Details.Destination);
        Assert.Equal(1.25, decision.Value.Details.Quality);
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
        Assert.Equal(model, nativeOptions.ModelId);
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
        using var adapter = new TypeSafeSchemaChatClient(spy, new Dictionary<string, TypeSafeFieldBinding> { ["/data"] = SchemaFixture.Boolean() });
        var response = await adapter.GetResponseAsync<bool>([SchemaFixture.State.ToMessage()],
            cancellationToken: TestContext.Current.CancellationToken);
        Assert.True(response.Result);
        Assert.Equal(0.9, response.GetTypeSafeEvidence().GetAnswer<NoulAnswer>("q0000").Noul);
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
        using var relaxed = new TypeSafeSchemaChatClient(spy, SchemaFixture.Bindings());
        var initial = await relaxed.GetTypeSafeDecisionAsync<DecisionDto>(SchemaFixture.State, cancellationToken: TestContext.Current.CancellationToken);
        Assert.True(initial.Accepted);
        Assert.Equal(1, handler.Sends);

        var strictBindings = SchemaFixture.Bindings();
        strictBindings["/details/destination"] = SchemaFixture.Choice(0.95);
        using var strict = new TypeSafeSchemaChatClient(spy, strictBindings);
        var changed = await strict.GetTypeSafeDecisionAsync<DecisionDto>(SchemaFixture.State, cancellationToken: TestContext.Current.CancellationToken);
        Assert.False(changed.Accepted);
        Assert.Equal("/details/destination", Assert.Single(changed.Rejections).Pointer);
        Assert.True(JsonElement.DeepEquals(initial.Evidence.Json, changed.Evidence.Json));
        Assert.Equal(1, handler.Sends);
        Assert.Null(spy.Responses[1].RawRepresentation);

        var original = SchemaFixture.Score().Rubric!;
        using var rubric = new TypeSafeSchemaChatClient(spy, SchemaFixture.Bindings(rubric: [original[1], original[0], original[2]]));
        var reordered = await rubric.GetTypeSafeDecisionAsync<DecisionDto>(SchemaFixture.State, cancellationToken: TestContext.Current.CancellationToken);
        Assert.True(reordered.Accepted);
        Assert.Equal(2, handler.Sends);
        Assert.Equal("poor", initial.Evidence.GetAnswer<ScoreAnswer>("q0002").Legend.GetProperty("0").GetString());
        Assert.Equal("partial", reordered.Evidence.GetAnswer<ScoreAnswer>("q0002").Legend.GetProperty("0").GetProperty("level").GetString());

        var shared = await relaxed.GetTypeSafeDecisionAsync<DecisionDto>(SchemaFixture.State,
            new ChatOptions { Instructions = "Revised shared synthetic instruction." }, cancellationToken: TestContext.Current.CancellationToken);
        Assert.True(shared.Accepted);
        Assert.Equal(3, handler.Sends);
        using var field = new TypeSafeSchemaChatClient(spy, SchemaFixture.Bindings(scoreInstructions: "A revised synthetic quality judgment."));
        var instructed = await field.GetTypeSafeDecisionAsync<DecisionDto>(SchemaFixture.State, cancellationToken: TestContext.Current.CancellationToken);
        Assert.True(instructed.Accepted);
        Assert.Equal(4, handler.Sends);
        Assert.Equal(5, spy.Calls);
        Assert.All(new[] { initial, reordered, shared, instructed }, d => Assert.Equal(1.25, d.Value.Details.Quality));
    }

    [Fact]
    public async Task ProjectedResponse_RoundtripsJsonElementEvidenceSeparatelyFromTypedJson()
    {
        using var handler = SchemaFixture.Handler();
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        using var adapter = new TypeSafeSchemaChatClient(spy, SchemaFixture.Bindings());
        var response = await adapter.GetResponseAsync<DecisionDto>([SchemaFixture.State.ToMessage()],
            SchemaFixture.Serializer, cancellationToken: TestContext.Current.CancellationToken);
        var serialized = JsonSerializer.SerializeToElement<ChatResponse>(response, AIJsonUtilities.DefaultOptions);
        var restored = serialized.Deserialize<ChatResponse>(AIJsonUtilities.DefaultOptions)!;
        Assert.Null(restored.RawRepresentation);
        Assert.IsType<JsonElement>(restored.AdditionalProperties!["typesafe.v1.evidence"]);
        Assert.IsType<JsonElement>(restored.AdditionalProperties["typesafe.v1.evidence.questions"]);
        var evidence = restored.GetTypeSafeEvidence();
        Assert.True(JsonElement.DeepEquals(response.GetTypeSafeEvidence().Json, evidence.Json));
        Assert.Equal("synthetic-preserved", evidence.Answers["q0001"].Json.GetProperty("vendor_answer").GetString());
        Assert.Equal("retained", evidence.Json.GetProperty("usage").GetProperty("vendor_usage").GetString());
        Assert.Equal(JsonValueKind.Null, evidence.GetAnswer<ScoreAnswer>("q0002").Legend.GetProperty("2").ValueKind);
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
        using var adapter = new TypeSafeSchemaChatClient(services.Client(), SchemaFixture.Bindings());
        var options = SchemaFixture.Options(SchemaFixture.Schema);
        var normal = await adapter.GetResponseAsync([SchemaFixture.State.ToMessage()], options, TestContext.Current.CancellationToken);
        List<ChatResponseUpdate> updates = [];
        var reconstructed = await Observe(adapter.GetStreamingResponseAsync([SchemaFixture.State.ToMessage()], options,
            TestContext.Current.CancellationToken), updates, TestContext.Current.CancellationToken).ToChatResponseAsync(TestContext.Current.CancellationToken);
        Assert.Equal(normal.Text, reconstructed.Text);
        Assert.Equal(normal.ModelId, reconstructed.ModelId);
        Assert.Equal(ChatFinishReason.Stop, reconstructed.FinishReason);
        Assert.True(JsonElement.DeepEquals(normal.GetTypeSafeEvidence().Json, reconstructed.GetTypeSafeEvidence().Json));
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
        using var adapter = new TypeSafeSchemaChatClient(spy, SchemaFixture.Bindings());
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

    [Fact]
    public async Task TypedApi_RejectsPromptAppendingAndHistoryBeforeDecoratedHttp()
    {
        using var handler = SchemaFixture.Handler();
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        using var adapter = new TypeSafeSchemaChatClient(spy, SchemaFixture.Bindings());
        var error = await Assert.ThrowsAsync<NotSupportedException>(() => adapter.GetResponseAsync<DecisionDto>(
            [SchemaFixture.State.ToMessage()], SchemaFixture.Serializer, useJsonSchemaResponseFormat: false,
            cancellationToken: TestContext.Current.CancellationToken));
        Assert.Contains("prompt-appending", error.Message);
        var historyError = await Assert.ThrowsAsync<NotSupportedException>(() => adapter.GetResponseAsync(
            [new ChatMessage(ChatRole.System, "Synthetic history."), SchemaFixture.State.ToMessage()],
            SchemaFixture.Options(SchemaFixture.Schema), TestContext.Current.CancellationToken));
        Assert.Contains("exactly one user message", historyError.Message);
        Assert.Equal(0, spy.Calls);
        Assert.Equal(0, handler.Sends);
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
        using var adapter = new TypeSafeSchemaChatClient(spy, SchemaFixture.Bindings());
        Task<TypeSafeDecision<DecisionDto>> Invoke() => adapter.GetTypeSafeDecisionAsync<DecisionDto>(SchemaFixture.State,
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
    public async Task Adapter_SnapshotsExtraOptionsBeforeDecoratedEntryAndLeavesNativeModelDefaultUnresolved()
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
            Assert.Null(captured.ModelId);
            var payload = Assert.IsType<JsonElement>(captured.AdditionalProperties!["synthetic-context"]);
            tags[0] = "caller-mutated";
            metadata["extra"] = "new caller metadata";
            options.ModelId = "caller-mutated-model";
            options.Temperature = 0.5f;
            Assert.Equal("original", payload.GetProperty("tags")[0].GetString());
            Assert.False(payload.TryGetProperty("extra", out _));
            Assert.Null(captured.Temperature);
        });
        using var adapter = new TypeSafeSchemaChatClient(inspect, SchemaFixture.Bindings());

        var response = await adapter.GetResponseAsync([SchemaFixture.State.ToMessage()], options, TestContext.Current.CancellationToken);

        Assert.Equal(1, inspections);
        Assert.Equal(1, spy.Calls);
        Assert.Equal(1, handler.Sends);
        Assert.Equal("synthetic-inner-default", response.ModelId);
        Assert.Equal("synthetic-inner-default", NativeFixture.Json(Assert.Single(handler.Requests).Body!).GetProperty("model").GetString());
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
}
