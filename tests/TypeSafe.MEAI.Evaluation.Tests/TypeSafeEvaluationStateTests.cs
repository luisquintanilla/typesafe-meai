using System.Text;
using System.Text.Json;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.AI.Evaluation;
using TypeSafe.MEAI.Evaluation.Tests.Fixtures;
using TypeSafe.MEAI.Tests.Fixtures;
using Xunit;

namespace TypeSafe.MEAI.Evaluation.Tests;

[Trait("Stage", "B")]
public sealed class TypeSafeEvaluationStateTests
{
    private const string UnicodeText = "Synthetic literal \u00e9 \u732b \U0001f9ea";

    [Fact]
    public async Task Evaluate_ProjectsStructuredValuesWithoutEncodingAndKeepsReportingContents()
    {
        ChatMessage[] messages =
        [
            new(ChatRole.System, "Preserve structured values.") { AuthorName = "policy" },
            new(ChatRole.User, MixedContents("conversation")) { AuthorName = "customer" }
        ];
        var response = new ChatResponse(new ChatMessage(ChatRole.Assistant, MixedContents("response")) { AuthorName = "primary" })
        {
            ModelId = "scripted-projection-primary"
        };
        var context = new PlainContext("reference", MixedContents("context"));
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler);
        using var spy = new EvaluationSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        IEvaluator evaluator = new TypeSafeEvaluator(EvaluationFixture.Definitions());

        var result = await evaluator.EvaluateAsync(messages, response, new(spy), [context], TestContext.Current.CancellationToken);

        Assert.Equal(1, spy.Calls);
        Assert.Equal(0, spy.ServiceLookups);
        Assert.Equal(1, handler.Sends);
        var request = Assert.Single(handler.Requests);
        Assert.Equal(HttpMethod.Post, request.Method);
        Assert.Equal("/v1/systemone", request.Uri.AbsolutePath);
        var state = NativeFixture.Json(request.Body!).GetProperty("state");
        Assert.Equal(JsonValueKind.Object, state.ValueKind);
        Assert.Equal(["context", "conversation", "response", "responseModel"], state.EnumerateObject().Select(p => p.Name).Order());
        Assert.Equal("scripted-projection-primary", state.GetProperty("responseModel").GetString());
        var conversation = state.GetProperty("conversation");
        Assert.Equal(2, conversation.GetArrayLength());
        AssertMessage(conversation[0], "system", "policy");
        var policy = Assert.Single(conversation[0].GetProperty("contents").EnumerateArray());
        Assert.Equal("text", policy.GetProperty("type").GetString());
        Assert.Equal("Preserve structured values.", policy.GetProperty("value").GetString());
        AssertMessage(conversation[1], "user", "customer");
        AssertProjectedContents(conversation[1].GetProperty("contents"), "conversation");
        var projectedResponse = Assert.Single(state.GetProperty("response").EnumerateArray());
        AssertMessage(projectedResponse, "assistant", "primary");
        AssertProjectedContents(projectedResponse.GetProperty("contents"), "response");
        var projectedContext = Assert.Single(state.GetProperty("context").EnumerateArray());
        Assert.Equal(["contents", "name"], projectedContext.EnumerateObject().Select(p => p.Name).Order());
        Assert.Equal("reference", projectedContext.GetProperty("name").GetString());
        AssertProjectedContents(projectedContext.GetProperty("contents"), "context");

        Assert.Equal(EvaluationFixture.MetricNames.Order(), result.Metrics.Keys.Order());
        Assert.Equal(0.82, Assert.IsType<NumericMetric>(result.Metrics["follow-up-probability"]).Value);
        Assert.Equal("billing", Assert.IsType<StringMetric>(result.Metrics["destination-label"]).Value);
        Assert.Equal(1.25, Assert.IsType<NumericMetric>(result.Metrics["quality-score"]).Value);
        foreach (var metric in result.Metrics.Values)
        {
            var snapshot = Assert.Single(metric.Context!).Value;
            Assert.Equal("reference", snapshot.Name);
            Assert.NotSame(context, snapshot);
            Assert.Equal(6, snapshot.Contents.Count);
            Assert.Equal(ObjectJson("context"), Assert.IsType<TextContent>(snapshot.Contents[0]).Text);
            Assert.Equal(UnicodeText, Assert.IsType<TextContent>(snapshot.Contents[4]).Text);
            foreach (int index in new[] { 1, 2, 3, 5 })
            {
                var data = Assert.IsType<DataContent>(snapshot.Contents[index]);
                Assert.NotSame(context.Contents[index], data);
                Assert.Equal("application/json", data.MediaType);
                using var document = JsonDocument.Parse(data.Data);
                AssertJsonValue(document.RootElement, index, "context");
            }
        }
    }

    [Theory]
    [InlineData("conversation")]
    [InlineData("response")]
    [InlineData("context")]
    public async Task Evaluate_RejectsInvalidUtf8BeforeCallingJudge(string surface)
    {
        var input = EvaluationFixture.Input();
        byte[] bytes = [0x22, 0xFF, 0x22];
        var invalid = new DataContent(bytes, "application/json");
        var contents = surface switch
        {
            "conversation" => input.Messages[1].Contents,
            "response" => input.Response.Messages[0].Contents,
            "context" => input.Contexts[0].Contents,
            _ => throw new ArgumentOutOfRangeException(nameof(surface))
        };
        contents.Add(invalid);
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler);
        using var spy = new EvaluationSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        var evaluator = new TypeSafeEvaluator(EvaluationFixture.Definitions());

        await Assert.ThrowsAsync<DecoderFallbackException>(() => evaluator.EvaluateAsync(input.Messages, input.Response,
            new(spy), input.Contexts, TestContext.Current.CancellationToken).AsTask());

        Assert.Equal(0, spy.Calls);
        Assert.Equal(0, handler.Sends);
        Assert.Empty(handler.Requests);
        Assert.Equal(new byte[] { 0x22, 0xFF, 0x22 }, bytes);
        Assert.Same(invalid, contents[^1]);
    }

    private static AIContent[] MixedContents(string surface) =>
    [
        new TextContent(ObjectJson(surface)),
        new DataContent(Encoding.UTF8.GetBytes(ObjectJson(surface)), "application/json"),
        new DataContent(Encoding.UTF8.GetBytes(ArrayJson(surface)), "application/json"),
        new DataContent("null"u8.ToArray(), "application/json"),
        new TextContent(UnicodeText),
        new DataContent(Encoding.UTF8.GetBytes($"\"{UnicodeText}\""), "application/json")
    ];

    private static string ObjectJson(string surface) =>
        $$$"""{"surface":"{{{surface}}}","nested":{"items":[{"label":"billing"},null,["inner",null]]}}""";

    private static string ArrayJson(string surface) =>
        $$$"""[{"surface":"{{{surface}}}","nested":{"active":true,"score":1.25}},["inner",null],null]""";

    private static void AssertMessage(JsonElement message, string role, string author)
    {
        Assert.Equal(["author", "contents", "role"], message.EnumerateObject().Select(p => p.Name).Order());
        Assert.Equal(role, message.GetProperty("role").GetString());
        Assert.Equal(author, message.GetProperty("author").GetString());
    }

    private static void AssertProjectedContents(JsonElement contents, string surface)
    {
        Assert.Equal(6, contents.GetArrayLength());
        foreach (var content in contents.EnumerateArray())
            Assert.Equal(["type", "value"], content.EnumerateObject().Select(p => p.Name).Order());
        Assert.Equal(["text", "json", "json", "json", "text", "json"],
            contents.EnumerateArray().Select(c => c.GetProperty("type").GetString()));
        Assert.Equal(JsonValueKind.String, contents[0].GetProperty("value").ValueKind);
        Assert.Equal(ObjectJson(surface), contents[0].GetProperty("value").GetString());
        Assert.Equal(UnicodeText, contents[4].GetProperty("value").GetString());
        foreach (int index in new[] { 1, 2, 3, 5 })
            AssertJsonValue(contents[index].GetProperty("value"), index, surface);
    }

    private static void AssertJsonValue(JsonElement value, int index, string surface)
    {
        switch (index)
        {
            case 1:
                Assert.Equal(JsonValueKind.Object, value.ValueKind);
                Assert.True(JsonElement.DeepEquals(NativeFixture.Json(ObjectJson(surface)), value));
                Assert.Equal(surface, value.GetProperty("surface").GetString());
                var items = value.GetProperty("nested").GetProperty("items");
                Assert.Equal(3, items.GetArrayLength());
                Assert.Equal("billing", items[0].GetProperty("label").GetString());
                Assert.Equal(JsonValueKind.Null, items[1].ValueKind);
                Assert.Equal("inner", items[2][0].GetString());
                Assert.Equal(JsonValueKind.Null, items[2][1].ValueKind);
                break;
            case 2:
                Assert.Equal(JsonValueKind.Array, value.ValueKind);
                Assert.True(JsonElement.DeepEquals(NativeFixture.Json(ArrayJson(surface)), value));
                Assert.Equal(3, value.GetArrayLength());
                Assert.Equal(surface, value[0].GetProperty("surface").GetString());
                Assert.True(value[0].GetProperty("nested").GetProperty("active").GetBoolean());
                Assert.Equal(1.25, value[0].GetProperty("nested").GetProperty("score").GetDouble());
                Assert.Equal("inner", value[1][0].GetString());
                Assert.Equal(JsonValueKind.Null, value[1][1].ValueKind);
                Assert.Equal(JsonValueKind.Null, value[2].ValueKind);
                break;
            case 3:
                Assert.Equal(JsonValueKind.Null, value.ValueKind);
                break;
            case 5:
                Assert.Equal(JsonValueKind.String, value.ValueKind);
                Assert.Equal(UnicodeText, value.GetString());
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(index));
        }
    }
}
