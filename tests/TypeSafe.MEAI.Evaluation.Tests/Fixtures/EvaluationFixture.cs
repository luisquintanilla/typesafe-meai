using System.Collections.Concurrent;
using System.Globalization;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.AI.Evaluation;
using TypeSafe.MEAI.Tests.Fixtures;
using Xunit;

namespace TypeSafe.MEAI.Evaluation.Tests.Fixtures;

internal static class EvaluationFixture
{
    internal static readonly string[] MetricNames = ["follow-up-probability", "destination-label", "quality-score"];
    internal static TypeSafeMetricDefinition[] Definitions()
    {
        var questions = NativeFixture.Questions();
        return questions.Select((question, index) => new TypeSafeMetricDefinition(MetricNames[index], question)).ToArray();
    }

    internal static EvaluationInput Input(int ticket = 17)
    {
        byte[] data = Encoding.UTF8.GetBytes($$"""{"ticket":{{ticket}},"tags":["billing",null]}""");
        ChatMessage[] messages =
        [
            new(ChatRole.System, "Synthetic policy: preserve evidence.") { AuthorName = "policy" },
            new(ChatRole.User, [new TextContent($"Ticket {ticket}: payment issue."), new DataContent(data, "application/json")]) { AuthorName = "customer" },
            new(ChatRole.Assistant, "Earlier clarification.")
        ];
        var response = new ChatResponse(new ChatMessage(ChatRole.Assistant, "Check the payment details.") { AuthorName = "primary" })
        {
            ModelId = "scripted-primary-v1",
            Usage = new() { InputTokenCount = 100, OutputTokenCount = 20, TotalTokenCount = 120 }
        };
        using var contextDocument = JsonDocument.Parse($$"""{"ticket":{{ticket}},"expected":"Check the payment details.","levels":["low",null,{"high":"complete"}]}""");
        EvaluationContext[] contexts =
        [
            new TypeSafeEvaluationContext("reference", contextDocument.RootElement),
            new PlainContext("guidance",
            [
                new TextContent("Use the approved policy."),
                new DataContent("""{"policy":"synthetic-only","order":[1,2]}"""u8.ToArray(), "application/json")
            ])
        ];
        return new(messages, response, contexts, data);
    }

    internal static void AssertMetrics(EvaluationResult result, int ticket = 17, double noul = 0.82)
    {
        Assert.Equal(noul, Assert.IsType<NumericMetric>(result.Metrics[MetricNames[0]]).Value);
        Assert.Equal("billing", Assert.IsType<StringMetric>(result.Metrics[MetricNames[1]]).Value);
        Assert.Equal(1.25, Assert.IsType<NumericMetric>(result.Metrics[MetricNames[2]]).Value);
        var definitions = Definitions();
        var envelope = NativeFixture.Json(NativeFixture.Envelope.Replace("0.82",
            noul.ToString(CultureInfo.InvariantCulture), StringComparison.Ordinal));
        foreach (var definition in definitions)
        {
            var metric = result.Metrics[definition.Name];
            Assert.Equal(definition.Name, metric.Name);
            Assert.Null(metric.Reason);
            Assert.NotNull(metric.Metadata);
            Assert.True(JsonElement.DeepEquals(envelope.GetProperty("answers").GetProperty(definition.Question.Id),
                NativeFixture.Json(metric.Metadata["typesafe.v1.answer"])));
            Assert.True(JsonElement.DeepEquals(definition.Question.ToJson(), NativeFixture.Json(metric.Metadata["typesafe.v1.rubric"])));
            Assert.True(JsonElement.DeepEquals(envelope, NativeFixture.Json(metric.Metadata["typesafe.v1.envelope"])));
            Assert.Equal("jev-1.13.0", metric.Metadata["eval-model"]);
            Assert.Equal("11", metric.Metadata["eval-input-tokens"]);
            Assert.Equal("7", metric.Metadata["eval-output-tokens"]);
            Assert.Equal("18", metric.Metadata["eval-total-tokens"]);
            Assert.True(double.TryParse(metric.Metadata["eval-duration-ms"], NumberStyles.Float, CultureInfo.InvariantCulture, out var duration));
            Assert.True(double.IsFinite(duration) && duration >= 0);
            Assert.True(Guid.TryParseExact(metric.Metadata["typesafe.v1.batch"], "N", out var batch));
            Assert.NotEqual(Guid.Empty, batch);
            AssertContexts(metric, ticket);
        }
        Assert.Single(definitions.Select(d => result.Metrics[d.Name].Metadata!["typesafe.v1.batch"]).Distinct(StringComparer.Ordinal));
        Assert.Single(definitions.Select(d => result.Metrics[d.Name].Metadata!["eval-duration-ms"]).Distinct(StringComparer.Ordinal));
    }

    internal static void AssertContexts(EvaluationMetric metric, int ticket = 17)
    {
        Assert.NotNull(metric.Context);
        Assert.Equal(["guidance", "reference"], metric.Context.Keys.Order());
        var reference = metric.Context["reference"];
        Assert.Equal("reference", reference.Name);
        var data = Assert.IsType<DataContent>(Assert.Single(reference.Contents));
        Assert.Equal("application/json", data.MediaType);
        using var json = JsonDocument.Parse(data.Data);
        Assert.Equal(ticket, json.RootElement.GetProperty("ticket").GetInt32());
        Assert.Equal("Check the payment details.", json.RootElement.GetProperty("expected").GetString());
        Assert.Equal(JsonValueKind.Null, json.RootElement.GetProperty("levels")[1].ValueKind);
        Assert.Equal("complete", json.RootElement.GetProperty("levels")[2].GetProperty("high").GetString());
        var guidance = metric.Context["guidance"];
        Assert.Equal("guidance", guidance.Name);
        Assert.Equal(2, guidance.Contents.Count);
        Assert.Equal("Use the approved policy.", Assert.IsType<TextContent>(guidance.Contents[0]).Text);
        using var policy = JsonDocument.Parse(Assert.IsType<DataContent>(guidance.Contents[1]).Data);
        Assert.Equal("synthetic-only", policy.RootElement.GetProperty("policy").GetString());
        Assert.Equal([1, 2], policy.RootElement.GetProperty("order").EnumerateArray().Select(v => v.GetInt32()));
    }

    internal static void AssertConversation(IReadOnlyList<ChatMessage> messages, int ticket = 17)
    {
        Assert.Equal(3, messages.Count);
        Assert.Equal([ChatRole.System, ChatRole.User, ChatRole.Assistant], messages.Select(m => m.Role));
        Assert.Equal("policy", messages[0].AuthorName);
        Assert.Equal("Synthetic policy: preserve evidence.", messages[0].Text);
        Assert.Equal("customer", messages[1].AuthorName);
        Assert.Equal($"Ticket {ticket}: payment issue.", Assert.IsType<TextContent>(messages[1].Contents[0]).Text);
        using var json = JsonDocument.Parse(Assert.IsType<DataContent>(messages[1].Contents[1]).Data);
        Assert.Equal(ticket, json.RootElement.GetProperty("ticket").GetInt32());
        Assert.Equal("billing", json.RootElement.GetProperty("tags")[0].GetString());
        Assert.Equal(JsonValueKind.Null, json.RootElement.GetProperty("tags")[1].ValueKind);
        Assert.Equal("Earlier clarification.", messages[2].Text);
    }

    internal static void AssertWireState(CapturedRequest request, int ticket = 17)
    {
        Assert.Equal(HttpMethod.Post, request.Method);
        Assert.Equal("/v1/systemone", request.Uri.AbsolutePath);
        var root = NativeFixture.Json(request.Body!);
        var state = root.GetProperty("state");
        Assert.Equal(JsonValueKind.Object, state.ValueKind);
        var conversation = state.GetProperty("conversation");
        Assert.Equal(3, conversation.GetArrayLength());
        Assert.Equal(["system", "user", "assistant"], conversation.EnumerateArray().Select(m => m.GetProperty("role").GetString()));
        Assert.Equal("policy", conversation[0].GetProperty("author").GetString());
        Assert.Equal("Synthetic policy: preserve evidence.", AssertText(conversation[0].GetProperty("contents")[0]));
        Assert.Equal("customer", conversation[1].GetProperty("author").GetString());
        var userContents = conversation[1].GetProperty("contents");
        Assert.Equal(2, userContents.GetArrayLength());
        Assert.Equal($"Ticket {ticket}: payment issue.", AssertText(userContents[0]));
        Assert.Equal("json", userContents[1].GetProperty("type").GetString());
        var json = userContents[1].GetProperty("value");
        Assert.Equal(ticket, json.GetProperty("ticket").GetInt32());
        Assert.Equal("billing", json.GetProperty("tags")[0].GetString());
        Assert.Equal(JsonValueKind.Null, json.GetProperty("tags")[1].ValueKind);
        Assert.Equal("Earlier clarification.", AssertText(conversation[2].GetProperty("contents")[0]));
        var response = Assert.Single(state.GetProperty("response").EnumerateArray());
        Assert.Equal("assistant", response.GetProperty("role").GetString());
        Assert.Equal("primary", response.GetProperty("author").GetString());
        Assert.Equal("Check the payment details.", AssertText(response.GetProperty("contents")[0]));
        Assert.Equal("scripted-primary-v1", state.GetProperty("responseModel").GetString());
        var contexts = state.GetProperty("context");
        Assert.Equal(2, contexts.GetArrayLength());
        Assert.Equal("reference", contexts[0].GetProperty("name").GetString());
        Assert.Equal("guidance", contexts[1].GetProperty("name").GetString());
        var reference = Assert.Single(contexts[0].GetProperty("contents").EnumerateArray());
        Assert.Equal("json", reference.GetProperty("type").GetString());
        var context = reference.GetProperty("value");
        Assert.Equal(ticket, context.GetProperty("ticket").GetInt32());
        Assert.Equal("Check the payment details.", context.GetProperty("expected").GetString());
        Assert.Equal(JsonValueKind.Null, context.GetProperty("levels")[1].ValueKind);
        Assert.Equal("complete", context.GetProperty("levels")[2].GetProperty("high").GetString());
        var guidance = contexts[1].GetProperty("contents");
        Assert.Equal(2, guidance.GetArrayLength());
        Assert.Equal("Use the approved policy.", AssertText(guidance[0]));
        Assert.Equal("json", guidance[1].GetProperty("type").GetString());
        Assert.Equal("synthetic-only", guidance[1].GetProperty("value").GetProperty("policy").GetString());
        Assert.Equal([1, 2], guidance[1].GetProperty("value").GetProperty("order").EnumerateArray().Select(v => v.GetInt32()));
        Assert.Equal(3, root.GetProperty("questions").EnumerateObject().Count());
        foreach (var definition in Definitions())
            Assert.True(JsonElement.DeepEquals(definition.Question.ToJson(), root.GetProperty("questions").GetProperty(definition.Question.Id)));
    }

    private static string? AssertText(JsonElement content)
    {
        Assert.Equal("text", content.GetProperty("type").GetString());
        return content.GetProperty("value").GetString();
    }
}

internal sealed record EvaluationInput(ChatMessage[] Messages, ChatResponse Response, EvaluationContext[] Contexts, byte[] StateBytes);

internal sealed class PlainContext(string name, IEnumerable<AIContent> contents) : EvaluationContext(name, contents);

internal sealed class EvaluationSpy(IChatClient inner) : DelegatingChatClient(inner)
{
    private int _calls;
    private int _lookups;
    internal int Calls => Volatile.Read(ref _calls);
    internal int ServiceLookups => Volatile.Read(ref _lookups);
    internal ConcurrentQueue<ChatOptions?> Options { get; } = new();
    public override Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        Interlocked.Increment(ref _calls);
        Options.Enqueue(options);
        return InnerClient.GetResponseAsync(messages, options, cancellationToken);
    }
    public override object? GetService(Type serviceType, object? serviceKey = null)
    {
        Interlocked.Increment(ref _lookups);
        return base.GetService(serviceType, serviceKey);
    }
}
