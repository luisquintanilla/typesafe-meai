using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.AI;
using TypeSafe.MEAI.Tests.Fixtures;
using Xunit;

namespace TypeSafe.MEAI.Tests.Provider;

[Trait("Stage", "A")]
public sealed class NativeInputRegressionTests
{
    [Theory]
    [InlineData("capability")]
    [InlineData("reserved")]
    [InlineData("questions")]
    [InlineData("state")]
    public async Task RawNativeEntry_ValidatesUntrustedRequestsWithoutPublicNormalization(string invalid)
    {
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler);
        using var raw = new NativeChatClient(http, NativeFixture.Settings, NativeFixture.Key);
        var options = NativeFixture.Options();
        var messages = NativeFixture.Messages();
        if (invalid == "capability") options.RawRepresentationFactory = _ => throw new InvalidOperationException("must not invoke");
        if (invalid == "reserved") options.AdditionalProperties!["typesafe.unrecognized"] = true;
        if (invalid == "questions") options.AdditionalProperties![TypeSafeChatClient.QuestionsKey] =
            NativeFixture.Json("""{"q":{"type":"noul"},"q":{"type":"noul"}}""");
        if (invalid == "state") messages = NativeFixture.Messages("false");

        if (invalid == "questions") await Assert.ThrowsAnyAsync<JsonException>(() =>
            raw.GetResponseAsync(messages, options, TestContext.Current.CancellationToken));
        else if (invalid == "state") await Assert.ThrowsAsync<ArgumentException>(() =>
            raw.GetResponseAsync(messages, options, TestContext.Current.CancellationToken));
        else await Assert.ThrowsAsync<NotSupportedException>(() =>
            raw.GetResponseAsync(messages, options, TestContext.Current.CancellationToken));
        Assert.Equal(0, handler.Sends);
        Assert.Empty(handler.Requests);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task GetResponse_RejectsInvalidUtf8BeforeSend(bool factory)
    {
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler, disposeHandler: false);
        using var services = NativeFixture.Services(handler);
        using IChatClient client = factory ? services.Client() : new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key);
        byte[] bytes = [0x22, 0xFF, 0x22];
        ChatMessage[] messages = [new(ChatRole.User, [new DataContent(bytes, "application/json")])];

        await Assert.ThrowsAsync<DecoderFallbackException>(() =>
            client.GetResponseAsync(messages, NativeFixture.Options(), TestContext.Current.CancellationToken));

        Assert.Equal(0, handler.Sends);
        Assert.Empty(handler.Requests);
        Assert.Equal(new byte[] { 0x22, 0xFF, 0x22 }, bytes);
    }

    [Theory]
    [InlineData("object", false)]
    [InlineData("object", true)]
    [InlineData("array", false)]
    [InlineData("array", true)]
    [InlineData("string", false)]
    [InlineData("string", true)]
    public async Task GetResponse_PreservesValidUnicodeJsonState(string shape, bool factory)
    {
        const string text = "caf\u00E9 \u6771\u4EAC \U0001F680 e\u0301";
        string json = shape switch
        {
            "object" => $"{{\"label\":\"{text}\"}}",
            "array" => $"[\"{text}\",null]",
            "string" => $"\"{text}\"",
            _ => throw new ArgumentOutOfRangeException(nameof(shape))
        };
        byte[] bytes = new UTF8Encoding(false, true).GetBytes(json);
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler, disposeHandler: false);
        using var services = NativeFixture.Services(handler);
        using IChatClient client = factory ? services.Client() : new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key);
        ChatMessage[] messages = [new(ChatRole.User, [new DataContent(bytes, "application/json")])];

        var response = await client.GetResponseAsync(messages, NativeFixture.Options(), TestContext.Current.CancellationToken);

        NativeFixture.AssertChat(response);
        Assert.Contains((byte)0xC3, bytes);
        var sent = NativeFixture.Json(Assert.Single(handler.Requests).Body!).GetProperty("state");
        Assert.True(JsonElement.DeepEquals(NativeFixture.Json(json), sent));
        var sentText = shape switch
        {
            "object" => sent.GetProperty("label").GetString(),
            "array" => sent[0].GetString(),
            _ => sent.GetString()
        };
        Assert.Equal(text, sentText);
        Assert.DoesNotContain("\uFFFD", sentText!);
        Assert.Equal(1, handler.Sends);
    }

    public static TheoryData<string, string, bool> UnknownFields
    {
        get
        {
            TheoryData<string, string, bool> data = new();
            foreach (string id in new[] { "follow-up", "destination", "quality" })
            foreach (string fieldName in new[] { "future", "instrutions" })
            foreach (bool factory in new[] { false, true })
                data.Add(id, fieldName, factory);
            return data;
        }
    }

    [Theory]
    [MemberData(nameof(UnknownFields))]
    public async Task GetResponse_RejectsUnknownRawQuestionFieldsBeforeSend(string id, string field, bool factory)
    {
        var raw = RawQuestions();
        raw[id]![field] = JsonNode.Parse("""{"context":"must-not-be-dropped"}""");
        var options = new ChatOptions { AdditionalProperties = new() { [NativeFixture.QuestionsKey] = NativeFixture.Json(raw.ToJsonString()) } };
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler, disposeHandler: false);
        using var services = NativeFixture.Services(handler);
        using IChatClient client = factory ? services.Client() : new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key);

        var error = await Assert.ThrowsAsync<ArgumentException>(() =>
            client.GetResponseAsync(NativeFixture.Messages(), options, TestContext.Current.CancellationToken));

        Assert.Contains("top-level native question field", error.Message);
        Assert.Equal(0, handler.Sends);
        Assert.Empty(handler.Requests);
        var unchanged = Assert.IsType<JsonElement>(options.AdditionalProperties[NativeFixture.QuestionsKey]);
        Assert.Equal("must-not-be-dropped", unchanged.GetProperty(id).GetProperty(field).GetProperty("context").GetString());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task GetResponse_PreservesNestedStructuredQuestionFields(bool factory)
    {
        var raw = RawQuestions();
        foreach (var question in raw)
            question.Value!["instructions"] = JsonNode.Parse("""{"future":{"version":17},"instrutions":["keep",null],"type":"application-context"}""");
        raw["follow-up"]!["criteria"]!["true"] = JsonNode.Parse("""{"future":"noul-context"}""");
        raw["destination"]!["criteria"]!["billing"] = JsonNode.Parse("""{"instrutions":["choice-context"]}""");
        raw["quality"]!["criteria"]![1] = JsonNode.Parse("""{"future":{"criteria":"score-context"}}""");
        var payload = NativeFixture.Json(raw.ToJsonString());
        var options = new ChatOptions { AdditionalProperties = new() { [NativeFixture.QuestionsKey] = payload } };
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler, disposeHandler: false);
        using var services = NativeFixture.Services(handler);
        using IChatClient client = factory ? services.Client() : new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key);

        var response = await client.GetResponseAsync(NativeFixture.Messages(), options, TestContext.Current.CancellationToken);

        NativeFixture.AssertChat(response);
        var sent = NativeFixture.Json(Assert.Single(handler.Requests).Body!).GetProperty("questions");
        Assert.True(JsonElement.DeepEquals(payload, sent));
        Assert.Equal(17, sent.GetProperty("quality").GetProperty("instructions").GetProperty("future").GetProperty("version").GetInt32());
        Assert.Equal("noul-context", sent.GetProperty("follow-up").GetProperty("criteria").GetProperty("true").GetProperty("future").GetString());
        Assert.Equal("choice-context", sent.GetProperty("destination").GetProperty("criteria").GetProperty("billing").GetProperty("instrutions")[0].GetString());
        Assert.Equal("score-context", sent.GetProperty("quality").GetProperty("criteria")[1].GetProperty("future").GetProperty("criteria").GetString());
        Assert.Equal(1, handler.Sends);
    }

    private static JsonObject RawQuestions() => new(NativeFixture.Questions()
        .Select(question => KeyValuePair.Create(question.Id, JsonNode.Parse(question.ToJson().GetRawText()))));

    [Theory]
    [InlineData("system")]
    [InlineData("assistant")]
    [InlineData("tool")]
    [InlineData("user")]
    public async Task StandardMessages_PreserveRolesOrderContentBoundariesAndLiteralVersusJson(string role)
    {
        using var handler = new RecordingHandler();
        using var services = NativeFixture.Services(handler, cache: true);
        var evidence = new JevResultContent(Encoding.UTF8.GetBytes(NativeFixture.Envelope));
        ChatMessage[] messages =
        [
            new(new ChatRole(role), [new TextContent("""{"ticket":17}"""), new DataContent("""{"ticket":17}"""u8.ToArray(), "application/json")]) { AuthorName = "first" },
            new(ChatRole.User, [new TextContent("café 猫 🧪"), evidence]) { AuthorName = "second" }
        ];
        NativeFixture.AssertChat(await services.Client().GetResponseAsync(messages, NativeFixture.Options(), TestContext.Current.CancellationToken));
        var conversation = NativeFixture.Json(Assert.Single(handler.Requests).Body!).GetProperty("state").GetProperty("conversation");
        Assert.Equal([role, "user"], conversation.EnumerateArray().Select(m => m.GetProperty("role").GetString()));
        Assert.Equal(["first", "second"], conversation.EnumerateArray().Select(m => m.GetProperty("author").GetString()));
        Assert.Equal(["text", "json"], conversation[0].GetProperty("contents").EnumerateArray().Select(c => c.GetProperty("type").GetString()));
        Assert.Equal("""{"ticket":17}""", conversation[0].GetProperty("contents")[0].GetProperty("value").GetString());
        Assert.Equal(17, conversation[0].GetProperty("contents")[1].GetProperty("value").GetProperty("ticket").GetInt32());
        Assert.Equal("café 猫 🧪", conversation[1].GetProperty("contents")[0].GetProperty("value").GetString());
        Assert.True(JsonElement.DeepEquals(evidence.Json, conversation[1].GetProperty("contents")[1].GetProperty("value")));
        Assert.DoesNotContain("data:", conversation.GetRawText());
        Assert.Equal(1, handler.Sends);
    }

    [Fact]
    public async Task NativeInstructions_ParticipateInCacheIdentityWithoutMutatingQuestions()
    {
        using var handler = new RecordingHandler();
        using var services = NativeFixture.Services(handler, cache: true);
        var options = NativeFixture.Options();
        var original = Assert.IsType<JsonElement>(options.AdditionalProperties![TypeSafeChatClient.QuestionsKey]).GetRawText();
        foreach (string? instructions in new[] { null, "Task A", "Task A", "Task B" })
        {
            options.Instructions = instructions;
            NativeFixture.AssertChat(await services.Client().GetResponseAsync(NativeFixture.Messages(), options, TestContext.Current.CancellationToken));
        }
        Assert.Equal(3, handler.Sends);
        var captured = handler.Requests.ToArray();
        var combined = NativeFixture.Json(captured[1].Body!).GetProperty("questions").GetProperty("follow-up").GetProperty("instructions");
        Assert.Equal("Task A", combined.GetProperty("taskInstructions").GetString());
        Assert.Equal("Follow up?", combined.GetProperty("questionInstructions").GetString());
        Assert.Equal("Task B", NativeFixture.Json(captured[2].Body!).GetProperty("questions").GetProperty("follow-up").GetProperty("instructions").GetProperty("taskInstructions").GetString());
        Assert.Equal(original, Assert.IsType<JsonElement>(options.AdditionalProperties[TypeSafeChatClient.QuestionsKey]).GetRawText());
    }
}
