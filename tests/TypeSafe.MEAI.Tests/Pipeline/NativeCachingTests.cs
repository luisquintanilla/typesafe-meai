using System.Text.Json;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using TypeSafe.MEAI.Tests.Fixtures;
using Xunit;

namespace TypeSafe.MEAI.Tests.Pipeline;

[Trait("Stage", "A")]
public sealed class NativeCachingTests
{
    [Fact]
    public async Task CacheHit_RestoresEveryNativeFieldThroughRealSerialization()
    {
        using var handler = new RecordingHandler();
        using var services = NativeFixture.Services(handler, cache: true);
        var client = services.Client();
        var options = JsonSerializer.Deserialize<ChatOptions>(JsonSerializer.Serialize(NativeFixture.Options()))!;
        Assert.IsType<JsonElement>(options.AdditionalProperties![NativeFixture.QuestionsKey]);
        var first = await client.GetResponseAsync(NativeFixture.Messages(), options, TestContext.Current.CancellationToken);
        var restored = await client.GetResponseAsync(NativeFixture.Messages(), options, TestContext.Current.CancellationToken);
        Assert.NotSame(first, restored);
        Assert.NotSame(first.Messages[0], restored.Messages[0]);
        Assert.Null(restored.RawRepresentation);
        NativeFixture.AssertChat(first);
        NativeFixture.AssertChat(restored);
        Assert.True(JsonElement.DeepEquals(NativeFixture.Json(NativeFixture.Envelope), NativeFixture.Json(restored.Text)));
        Assert.Equal(1, handler.Sends);
        Assert.Null(options.ModelId);
        Assert.False(options.AdditionalProperties.ContainsKey("typesafe.v1.cache-scope"));
    }

    [Theory]
    [InlineData("state")]
    [InlineData("question-id")]
    [InlineData("instructions")]
    [InlineData("noul-criteria")]
    [InlineData("choice-description")]
    [InlineData("score-entry")]
    [InlineData("score-order")]
    [InlineData("explicit-model")]
    [InlineData("default-model")]
    [InlineData("endpoint")]
    [InlineData("tenant")]
    public async Task CacheKey_ChangesForEachInferenceMutationAcrossSharedKeyedClients(string mutation)
    {
        using var handler = new RecordingHandler((_, attempt, _) =>
        {
            var body = attempt == 1 ? NativeFixture.Envelope : NativeFixture.Envelope.Replace("0.82", "0.83", StringComparison.Ordinal);
            if (attempt > 1 && mutation == "question-id") body = body.Replace("follow-up", "follow-up-2", StringComparison.Ordinal);
            return Task.FromResult(NativeFixture.Response(body));
        });
        var settings = NativeFixture.Settings;
        var changedSettings = mutation switch
        {
            "default-model" => settings with { ModelId = "jev-1.14.0" },
            "endpoint" => settings with { Endpoint = new("https://other.example") },
            "tenant" => settings with { CacheNamespace = "synthetic-tenant-b" },
            _ => settings
        };
        var services = new ServiceCollection();
        services.AddDistributedMemoryCache();
        services.AddTypeSafeChatClient("first", NativeFixture.Key, settings, useCache: true).ConfigurePrimaryHttpMessageHandler(() => handler);
        services.AddTypeSafeChatClient("second", NativeFixture.Key, changedSettings, useCache: true).ConfigurePrimaryHttpMessageHandler(() => handler);
        using var provider = services.BuildServiceProvider();
        var firstClient = provider.GetRequiredKeyedService<IChatClient>("first");
        var secondClient = provider.GetRequiredKeyedService<IChatClient>("second");
        var first = await firstClient.GetTypeSafeResponseAsync(TypeSafeState.Parse(NativeFixture.State),
            NativeFixture.Questions(), cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(0.82, first.GetAnswer<NoulAnswer>("follow-up").Noul);
        var questions = NativeFixture.Questions();
        switch (mutation)
        {
            case "question-id": questions[0] = new NoulQuestion("follow-up-2", questions[0].Instructions, ((NoulQuestion)questions[0]).Criteria); break;
            case "instructions": questions[0] = new NoulQuestion("follow-up", TypeSafeJson.Entry("Different instructions"), ((NoulQuestion)questions[0]).Criteria); break;
            case "noul-criteria": questions[0] = new NoulQuestion("follow-up", questions[0].Instructions, NativeFixture.Json("""{"true":"different","false":null}""")); break;
            case "choice-description":
                var entries = ((ChoiceQuestion)questions[1]).Criteria.ToDictionary();
                entries["billing"] = TypeSafeJson.Entry("different description");
                questions[1] = new ChoiceQuestion("destination", questions[1].Instructions, entries);
                break;
            case "score-entry":
                var levels = ((ScoreQuestion)questions[2]).Criteria.ToArray();
                levels[1] = TypeSafeJson.Entry("different level");
                questions[2] = new ScoreQuestion("quality", questions[2].Instructions, levels);
                break;
            case "score-order": questions[2] = new ScoreQuestion("quality", questions[2].Instructions, ((ScoreQuestion)questions[2]).Criteria.Reverse()); break;
        }
        var state = TypeSafeState.Parse(mutation == "state" ? """{"ticket":18,"category":"billing"}""" : NativeFixture.State);
        var options = mutation == "explicit-model" ? new ChatOptions { ModelId = "jev-1.14.0" } : null;
        var second = await secondClient.GetTypeSafeResponseAsync(state, questions, options, TestContext.Current.CancellationToken);
        Assert.Equal(0.83, second.GetAnswer<NoulAnswer>(mutation == "question-id" ? "follow-up-2" : "follow-up").Noul);
        Assert.Equal(2, handler.Sends);
        var again = await secondClient.GetTypeSafeResponseAsync(state, questions, options, TestContext.Current.CancellationToken);
        Assert.Equal(0.83, again.GetAnswer<NoulAnswer>(mutation == "question-id" ? "follow-up-2" : "follow-up").Noul);
        Assert.Equal(2, handler.Sends);
        var requests = handler.Requests.ToArray();
        Assert.Equal(mutation == "endpoint" ? "other.example" : "api.typesafe.ai", requests[1].Uri.Host);
        Assert.Equal(mutation is "default-model" or "explicit-model" ? "jev-1.14.0" : "jev-1.13.0",
            NativeFixture.Json(requests[1].Body!).GetProperty("model").GetString());
    }

    [Theory]
    [InlineData(false, 2)]
    [InlineData(true, 1)]
    public async Task CacheScope_IsOptInAndKeyRotationWithinTenantReusesEvidence(bool enabled, int expectedSends)
    {
        using var handler = new RecordingHandler();
        var services = new ServiceCollection();
        services.AddDistributedMemoryCache();
        services.AddTypeSafeChatClient("first", NativeFixture.Key, NativeFixture.Settings, enabled).ConfigurePrimaryHttpMessageHandler(() => handler);
        services.AddTypeSafeChatClient("second", "SYNTHETIC-ROTATED-KEY", NativeFixture.Settings, enabled).ConfigurePrimaryHttpMessageHandler(() => handler);
        using var provider = services.BuildServiceProvider();
        var first = await provider.GetRequiredKeyedService<IChatClient>("first").GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), TestContext.Current.CancellationToken);
        var second = await provider.GetRequiredKeyedService<IChatClient>("second").GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), TestContext.Current.CancellationToken);
        NativeFixture.AssertChat(first);
        NativeFixture.AssertChat(second);
        Assert.Equal(expectedSends, handler.Sends);
        Assert.All(handler.Requests, request => Assert.DoesNotContain("SYNTHETIC", request.Body!));
    }

    [Fact]
    public async Task CacheScope_NormalizationContainsNonsecretVersionedProviderIdentity()
    {
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler);
        using var capture = new ScopeSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        using var client = capture.AsBuilder().UseTypeSafeRequestNormalization(NativeFixture.Settings).Build();
        var response = await client.GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), TestContext.Current.CancellationToken);
        NativeFixture.AssertChat(response);
        var normalized = Assert.IsType<ChatOptions>(capture.Options);
        Assert.Equal("jev-1.13.0", normalized.ModelId);
        var scope = Assert.IsType<JsonElement>(normalized.AdditionalProperties!["typesafe.v1.cache-scope"]);
        Assert.Equal(["provider", "protocol", "endpoint", "account"], scope.EnumerateObject().Select(p => p.Name));
        Assert.Equal("typesafe", scope.GetProperty("provider").GetString());
        Assert.Equal(1, scope.GetProperty("protocol").GetInt32());
        Assert.Equal("https://api.typesafe.ai/", scope.GetProperty("endpoint").GetString());
        Assert.Equal("synthetic-tenant-a", scope.GetProperty("account").GetString());
        Assert.DoesNotContain(NativeFixture.Key, scope.GetRawText());
        Assert.Equal(1, handler.Sends);
    }

    [Fact]
    public async Task KeyedRegistration_PreservesOrdinaryClientAndPeerLifetime()
    {
        using var handler = new RecordingHandler();
        using var ordinary = new OrdinaryClient();
        var services = new ServiceCollection();
        services.AddSingleton<IChatClient>(ordinary);
        services.AddTypeSafeChatClient("first", NativeFixture.Key, NativeFixture.Settings).ConfigurePrimaryHttpMessageHandler(() => handler);
        services.AddTypeSafeChatClient("second", NativeFixture.Key, NativeFixture.Settings with { Endpoint = new("https://second.example") })
            .ConfigurePrimaryHttpMessageHandler(() => handler);
        using var provider = services.BuildServiceProvider();
        var first = provider.GetRequiredKeyedService<IChatClient>("first");
        var second = provider.GetRequiredKeyedService<IChatClient>("second");
        Assert.Same(ordinary, provider.GetRequiredService<IChatClient>());
        Assert.NotSame(first, second);
        NativeFixture.AssertChat(await first.GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), TestContext.Current.CancellationToken));
        first.Dispose();
        NativeFixture.AssertChat(await second.GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), TestContext.Current.CancellationToken));
        Assert.Equal(0, handler.Disposals);
        Assert.Equal(["api.typesafe.ai", "second.example"], handler.Requests.Select(r => r.Uri.Host));
        Assert.Equal(0, ordinary.Calls);
        var result = await provider.GetRequiredService<IChatClient>().GetResponseAsync(
            [new(ChatRole.User, "ordinary question")], cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal("ordinary answer", result.Text);
        Assert.Equal(1, ordinary.Calls);
        Assert.Equal(2, handler.Sends);
    }

    private sealed class OrdinaryClient : IChatClient
    {
        internal int Calls { get; private set; }
        public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            Calls++;
            return Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, "ordinary answer")));
        }
        public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages,
            ChatOptions? options = null, CancellationToken cancellationToken = default) => throw new NotSupportedException();
        public object? GetService(Type serviceType, object? serviceKey = null) =>
            serviceKey is null && serviceType.IsInstanceOfType(this) ? this : null;
        public void Dispose() { }
    }

    private sealed class ScopeSpy(IChatClient inner) : DelegatingChatClient(inner)
    {
        internal ChatOptions? Options { get; private set; }
        public override Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            Options = options;
            return InnerClient.GetResponseAsync(messages, options, cancellationToken);
        }
    }
}
