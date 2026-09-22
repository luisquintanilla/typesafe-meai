using System.Text;
using System.Text.Json;
using Microsoft.Extensions.AI;
using TypeSafe.MEAI.Tests.Fixtures;
using Xunit;

namespace TypeSafe.MEAI.Tests.Provider;

[Trait("Stage", "A")]
public sealed class NativeChatClientTests
{
    [Theory]
    [InlineData("https://api.typesafe.ai")]
    [InlineData("https://api.typesafe.ai/")]
    public async Task GetResponse_MixedBatchPreservesCanonicalEnvelope(string origin)
    {
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler);
        using var client = new TypeSafeChatClient(http, NativeFixture.Settings with { Endpoint = new(origin) }, NativeFixture.Key);
        var options = NativeFixture.Options();
        options.ResponseFormat = ChatResponseFormat.Json;
        options.ModelId = "jev";
        options.AdditionalProperties!["application-metadata"] = new { correlation = 17 };
        var response = await client.GetResponseAsync(NativeFixture.Messages(), options, TestContext.Current.CancellationToken);
        NativeFixture.AssertChat(response);
        Assert.True(JsonElement.DeepEquals(NativeFixture.Json(NativeFixture.Envelope), NativeFixture.Json(response.Text)));
        var request = Assert.Single(handler.Requests);
        Assert.Equal(HttpMethod.Post, request.Method);
        Assert.Equal("https://api.typesafe.ai/v1/systemone", request.Uri.AbsoluteUri);
        Assert.Equal("Bearer", request.Scheme);
        Assert.Equal(NativeFixture.Key, request.Key);
        Assert.Equal("application/json", request.MediaType);
        var body = NativeFixture.Json(request.Body!);
        Assert.Equal(["state", "model", "questions"], body.EnumerateObject().Select(p => p.Name));
        Assert.Equal(17, body.GetProperty("state").GetProperty("ticket").GetInt32());
        Assert.Equal("jev", body.GetProperty("model").GetString());
        Assert.Equal(3, body.GetProperty("questions").EnumerateObject().Count());
        Assert.Equal("noul", body.GetProperty("questions").GetProperty("follow-up").GetProperty("type").GetString());
        Assert.Equal("choice", body.GetProperty("questions").GetProperty("destination").GetProperty("type").GetString());
        Assert.Equal("score", body.GetProperty("questions").GetProperty("quality").GetProperty("type").GetString());
        Assert.Equal(JsonValueKind.Null, body.GetProperty("questions").GetProperty("quality").GetProperty("criteria")[2].ValueKind);
        Assert.False(body.TryGetProperty("application-metadata", out _));
        Assert.Equal(1, handler.Sends);
        Assert.Equal("jev", options.ModelId);
        Assert.True(options.AdditionalProperties.ContainsKey("application-metadata"));
    }

    [Theory]
    [InlineData("""{"ticket":17}""", false, JsonValueKind.Object)]
    [InlineData("""[17,"billing"]""", false, JsonValueKind.Array)]
    [InlineData("null", false, JsonValueKind.Null)]
    [InlineData("\"literal\"", false, JsonValueKind.String)]
    [InlineData("""{"ticket":17}""", true, JsonValueKind.String)]
    [InlineData("[17]", true, JsonValueKind.String)]
    [InlineData("null", true, JsonValueKind.String)]
    public async Task GetResponse_PreservesJsonStateVersusLiteralText(string value, bool literal, JsonValueKind kind)
    {
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler);
        using var client = new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key);
        var state = literal ? TypeSafeState.FromText(value) : TypeSafeState.Parse(value);
        var response = await client.GetTypeSafeResponseAsync(state, NativeFixture.Questions(), cancellationToken: TestContext.Current.CancellationToken);
        var sent = NativeFixture.Json(Assert.Single(handler.Requests).Body!).GetProperty("state");
        Assert.Equal(kind, sent.ValueKind);
        if (literal) Assert.Equal(value, sent.GetString());
        else Assert.True(JsonElement.DeepEquals(NativeFixture.Json(value), sent));
        NativeFixture.AssertNative(response);
    }

    [Theory]
    [InlineData("empty")]
    [InlineData("multiple")]
    [InlineData("system")]
    [InlineData("assistant")]
    [InlineData("tool")]
    [InlineData("no-content")]
    [InlineData("multiple-content")]
    [InlineData("mixed")]
    [InlineData("image")]
    [InlineData("audio")]
    [InlineData("video")]
    [InlineData("mime")]
    [InlineData("function-result")]
    public async Task GetResponse_RejectsInvalidMessageShapesBeforeSend(string shape)
    {
        ChatMessage[] messages = shape switch
        {
            "empty" => [],
            "multiple" => [new(ChatRole.User, "one"), new(ChatRole.User, "two")],
            "system" => [new(ChatRole.System, "state")],
            "assistant" => [new(ChatRole.Assistant, "state")],
            "tool" => [new(ChatRole.Tool, "state")],
            "no-content" => [new(ChatRole.User, new List<AIContent>())],
            "multiple-content" => [new(ChatRole.User, [new TextContent("one"), new TextContent("two")])],
            "mixed" => [new(ChatRole.User, [new TextContent("one"), new DataContent("{}"u8.ToArray(), "application/json")])],
            "image" => [new(ChatRole.User, [new DataContent(new byte[] { 1 }, "image/png")])],
            "audio" => [new(ChatRole.User, [new DataContent(new byte[] { 1 }, "audio/wav")])],
            "video" => [new(ChatRole.User, [new DataContent(new byte[] { 1 }, "video/mp4")])],
            "mime" => [new(ChatRole.User, [new DataContent("{}"u8.ToArray(), "text/plain")])],
            "function-result" => [new(ChatRole.User, [new FunctionResultContent("call-17", "result")])],
            _ => throw new ArgumentOutOfRangeException(nameof(shape))
        };
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler);
        using var client = new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key);
        await Assert.ThrowsAsync<NotSupportedException>(() =>
            client.GetResponseAsync(messages, NativeFixture.Options(), TestContext.Current.CancellationToken));
        Assert.Equal(0, handler.Sends);
        Assert.Empty(handler.Requests);
    }

    [Theory]
    [InlineData("{", true)]
    [InlineData("17", false)]
    [InlineData("true", false)]
    public async Task GetResponse_RejectsInvalidJsonStateBeforeSend(string json, bool malformed)
    {
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler);
        using var client = new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key);
        ChatMessage[] messages = [new(ChatRole.User, [new DataContent(Encoding.UTF8.GetBytes(json), "application/json")])];
        if (malformed)
            await Assert.ThrowsAnyAsync<JsonException>(() => client.GetResponseAsync(messages, NativeFixture.Options(), TestContext.Current.CancellationToken));
        else
            await Assert.ThrowsAsync<ArgumentException>(() => client.GetResponseAsync(messages, NativeFixture.Options(), TestContext.Current.CancellationToken));
        Assert.Equal(0, handler.Sends);
    }

    [Theory]
    [InlineData("""{"q":{"type":"noul"},"q":{"type":"noul"}}""")]
    [InlineData("""{"q":{"type":"noul","type":"choice"}}""")]
    [InlineData("""{"q":{"type":"choice","criteria":{"a":"first","a":"second"}}}""")]
    public async Task GetResponse_RejectsDuplicateQuestionPropertiesBeforeSend(string payload)
    {
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler);
        using var client = new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key);
        var options = new ChatOptions { AdditionalProperties = new() { [NativeFixture.QuestionsKey] = NativeFixture.Json(payload) } };
        await Assert.ThrowsAsync<JsonException>(() => client.GetResponseAsync(NativeFixture.Messages(), options, TestContext.Current.CancellationToken));
        Assert.Equal(0, handler.Sends);
    }

    [Fact]
    public async Task GetTypeSafeResponse_UsesDecoratedClientAndStandardContent()
    {
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler);
        using var spy = new NativeSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        var result = await spy.GetTypeSafeResponseAsync(TypeSafeState.Parse(NativeFixture.State),
            NativeFixture.Questions(), cancellationToken: TestContext.Current.CancellationToken);
        NativeFixture.AssertNative(result);
        Assert.Equal(1, spy.Calls);
        Assert.Equal(0, spy.ServiceLookups);
        Assert.Equal(1, handler.Sends);
        Assert.NotNull(spy.SeenOptions);
        Assert.False(spy.SeenOptions.ResponseFormat is ChatResponseFormatJson { Schema: not null });
    }

    [Fact]
    public void WithTypeSafeQuestions_ClonesOptionsAndOwnsSerializablePayload()
    {
        var original = new ChatOptions { ModelId = "jev-1.13.0", AdditionalProperties = new() { ["metadata"] = "original" } };
        var questions = NativeFixture.Questions().ToList();
        var result = original.WithTypeSafeQuestions(questions);
        questions.Clear();
        original.AdditionalProperties["metadata"] = "changed";
        Assert.NotSame(original, result);
        Assert.NotSame(original.AdditionalProperties, result.AdditionalProperties);
        Assert.False(original.AdditionalProperties.ContainsKey(NativeFixture.QuestionsKey));
        Assert.Equal("original", result.AdditionalProperties!["metadata"]);
        var restored = JsonSerializer.Deserialize<ChatOptions>(JsonSerializer.Serialize(result))!;
        var json = Assert.IsType<JsonElement>(restored.AdditionalProperties![NativeFixture.QuestionsKey]);
        Assert.Equal(3, json.EnumerateObject().Count());
        Assert.Equal("Assess quality", json.GetProperty("quality").GetProperty("instructions").GetString());
        Assert.Equal("jev-1.13.0", restored.ModelId);
    }

    [Fact]
    public void GetService_ExposesMetadataWithoutInventedCapabilities()
    {
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler);
        using var client = new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key);
        var metadata = Assert.IsType<ChatClientMetadata>(client.GetService(typeof(ChatClientMetadata)));
        Assert.Equal("typesafe", metadata.ProviderName);
        Assert.Equal(new Uri("https://api.typesafe.ai"), metadata.ProviderUri);
        Assert.Equal("jev-1.13.0", metadata.DefaultModelId);
        Assert.Same(client, client.GetService(typeof(IChatClient)));
        Assert.Null(client.GetService(typeof(IChatClient), "key"));
        Assert.Null(client.GetService(typeof(IDistributedCacheMarker)));
        Assert.Throws<ArgumentNullException>(() => client.GetService(null!));
        Assert.Equal(0, handler.Sends);
    }

    [Theory]
    [InlineData("no-options")]
    [InlineData("no-questions")]
    [InlineData("empty-questions")]
    [InlineData("empty-model")]
    public async Task GetResponse_RejectsMissingQuestionsAndEmptyEffectiveModel(string invalid)
    {
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler);
        using var client = new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key);
        ChatOptions? options = invalid switch
        {
            "no-options" => null,
            "no-questions" => new ChatOptions(),
            _ => NativeFixture.Options()
        };
        if (invalid == "empty-questions") options!.AdditionalProperties![NativeFixture.QuestionsKey] = NativeFixture.Json("{}");
        if (invalid == "empty-model") options!.ModelId = "";
        await Assert.ThrowsAsync<ArgumentException>(() => client.GetResponseAsync(NativeFixture.Messages(), options, TestContext.Current.CancellationToken));
        Assert.Equal(0, handler.Sends);
        Assert.Empty(handler.Requests);
    }

    private interface IDistributedCacheMarker;

    private sealed class NativeSpy(IChatClient inner) : DelegatingChatClient(inner)
    {
        internal int Calls { get; private set; }
        internal int ServiceLookups { get; private set; }
        internal ChatOptions? SeenOptions { get; private set; }
        public override async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            Calls++;
            SeenOptions = options;
            var response = await InnerClient.GetResponseAsync(messages, options, cancellationToken);
            response.RawRepresentation = null;
            return response;
        }
        public override object? GetService(Type serviceType, object? serviceKey = null)
        {
            ServiceLookups++;
            return base.GetService(serviceType, serviceKey);
        }
    }
}
