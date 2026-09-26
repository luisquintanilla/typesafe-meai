using System.Net;
using System.Text;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using TypeSafe.MEAI.Tests.Fixtures;
using Xunit;

namespace TypeSafe.MEAI.Tests.Transport;

[Trait("Stage", "A")]
public sealed class TypeSafeModelAndOwnershipTests
{
    [Theory]
    [InlineData("https://models.example")]
    [InlineData("https://models.example/")]
    public async Task GetModels_DecodesWrapperUsingSharedGetBearerAndOrigin(string endpoint)
    {
        using var handler = new RecordingHandler((request, _, _) => Task.FromResult(request.Method == HttpMethod.Get
            ? NativeFixture.Response("""{"models":["jev","jev-latest"],"vendor":{"revision":17}}""")
            : NativeFixture.Response()));
        using var http = new HttpClient(handler);
        var settings = NativeFixture.Settings with { Endpoint = new(endpoint) };
        var models = new TypeSafeModelClient(http, settings, NativeFixture.Key);
        var result = await models.GetModelsAsync(TestContext.Current.CancellationToken);
        Assert.Equal(["jev", "jev-latest"], result.GetProperty("models").EnumerateArray().Select(m => m.GetString()));
        Assert.Equal(17, result.GetProperty("vendor").GetProperty("revision").GetInt32());
        var request = Assert.Single(handler.Requests);
        Assert.Equal(HttpMethod.Get, request.Method);
        Assert.Equal("https://models.example/v1/models", request.Uri.AbsoluteUri);
        Assert.Equal("Bearer", request.Scheme);
        Assert.Equal(NativeFixture.Key, request.Key);
        Assert.Null(request.Body);
        Assert.Null(request.MediaType);
        using var provider = new TypeSafeChatClient(http, settings, NativeFixture.Key);
        var inference = await provider.GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), TestContext.Current.CancellationToken);
        NativeFixture.AssertChat(inference);
        Assert.Equal("jev-1.13.0", NativeFixture.Json(handler.Requests.Last().Body!).GetProperty("model").GetString());
        Assert.Equal(2, handler.Sends);
    }

    [Theory]
    [InlineData("{}")]
    [InlineData("""{"models":{}}""")]
    [InlineData("""{"models":null}""")]
    [InlineData("{")]
    [InlineData("""{"models":[],"models":[]}""")]
    public async Task GetModels_RejectsMalformedWrapper(string body)
    {
        using var handler = new RecordingHandler((_, _, _) => Task.FromResult(NativeFixture.Response(body)));
        using var http = new HttpClient(handler);
        var models = new TypeSafeModelClient(http, NativeFixture.Settings, NativeFixture.Key);
        await Assert.ThrowsAsync<TypeSafeProtocolException>(() => models.GetModelsAsync(TestContext.Current.CancellationToken));
        Assert.Equal(1, handler.Sends);
        Assert.Equal(HttpMethod.Get, Assert.Single(handler.Requests).Method);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task GetModels_InvalidUtf8RejectsWithoutCharsetFallbackOrRetry(bool factory)
    {
        const string secret = "SYNTHETIC-PRIVATE-UTF8-MODELS";
        byte[] bytes = Encoding.UTF8.GetBytes($$"""{"models":["jev"],"vendor":"{{secret}}-@"}""");
        bytes[Array.IndexOf(bytes, (byte)'@')] = 0xff;
        var clock = new TestClock();
        var stream = new TrackedStream(bytes);
        var content = new TrackedContent(stream);
        content.Headers.ContentType = new("application/json") { CharSet = "iso-8859-1" };
        var response = new TrackedResponse(HttpStatusCode.OK) { Content = content };
        using var handler = new RecordingHandler((_, _, _) => Task.FromResult<HttpResponseMessage>(response));
        using var services = factory ? NativeFixture.Services(handler, clock: clock) : null;
        using var http = factory
            ? services!.GetRequiredService<IHttpClientFactory>().CreateClient("typesafe:native")
            : new HttpClient(handler);
        var models = new TypeSafeModelClient(http, NativeFixture.Settings, NativeFixture.Key);

        var error = await Assert.ThrowsAsync<TypeSafeProtocolException>(() =>
            models.GetModelsAsync(TestContext.Current.CancellationToken));

        Assert.Equal("Malformed model listing wrapper.", error.Message);
        Assert.Null(error.InnerException);
        Assert.DoesNotContain(secret, error.ToString());
        Assert.DoesNotContain(NativeFixture.Key, error.ToString());
        Assert.True(stream.Disposed);
        Assert.True(content.Disposed);
        Assert.True(response.Disposed);
        clock.Advance(TimeSpan.FromMinutes(1));
        Assert.Equal(1, handler.Sends);
        var request = Assert.Single(handler.Requests);
        Assert.Equal(HttpMethod.Get, request.Method);
        Assert.Equal("/v1/models", request.Uri.AbsolutePath);
        Assert.Null(request.Body);
    }

    [Theory]
    [InlineData(429, 2)]
    [InlineData(401, 1)]
    public async Task GetModels_UsesActualRegisteredResilienceAndPropagatesErrors(int status, int attempts)
    {
        var clock = new TestClock();
        using var handler = new RecordingHandler((_, attempt, _) => Task.FromResult(NativeFixture.Response(
            attempt == 1 ? "synthetic-error" : """{"models":["jev"]}""", attempt == 1 ? status : 200)));
        using var services = NativeFixture.Services(handler, clock: clock);
        using var http = services.GetRequiredService<IHttpClientFactory>().CreateClient("typesafe:native");
        var models = new TypeSafeModelClient(http, NativeFixture.Settings, NativeFixture.Key);
        var pending = models.GetModelsAsync(TestContext.Current.CancellationToken);
        if (status == 429)
        {
            await clock.AdvanceRetryAsync();
            Assert.Equal("jev", (await pending).GetProperty("models")[0].GetString());
        }
        else
        {
            var error = await Assert.ThrowsAsync<TypeSafeApiException>(() => pending);
            Assert.Equal(HttpStatusCode.Unauthorized, error.StatusCode);
        }
        Assert.Equal(attempts, handler.Sends);
        Assert.All(handler.Requests, r => Assert.Equal("/v1/models", r.Uri.AbsolutePath));
    }

    [Fact]
    public async Task GetModels_PreCanceledDoesNotSend()
    {
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler);
        var models = new TypeSafeModelClient(http, NativeFixture.Settings, NativeFixture.Key);
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        var error = await Assert.ThrowsAnyAsync<OperationCanceledException>(() => models.GetModelsAsync(cancellation.Token));
        Assert.Equal(cancellation.Token, error.CancellationToken);
        Assert.Equal(0, handler.Sends);
    }

    [Fact]
    public async Task Dispose_BorrowedHttpClientRemainsUsableByPeerProviderAndModels()
    {
        using var handler = new RecordingHandler((request, _, _) => Task.FromResult(request.Method == HttpMethod.Get
            ? NativeFixture.Response("""{"models":["jev"]}""") : NativeFixture.Response()));
        using var http = new HttpClient(handler);
        var first = new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key);
        using var second = new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key);
        var models = new TypeSafeModelClient(http, NativeFixture.Settings, NativeFixture.Key);
        using var decorated = first.AsBuilder().Use(inner =>
            new LoggingChatClient(inner, Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance)).Build();
        decorated.Dispose();
        first.Dispose();
        Assert.Equal(0, handler.Disposals);
        NativeFixture.AssertChat(await second.GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), TestContext.Current.CancellationToken));
        Assert.Equal("jev", (await models.GetModelsAsync(TestContext.Current.CancellationToken)).GetProperty("models")[0].GetString());
        await Assert.ThrowsAsync<ObjectDisposedException>(() => first.GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), TestContext.Current.CancellationToken));
        Assert.Equal(2, handler.Sends);
        Assert.Equal(0, handler.Disposals);
    }

    [Fact]
    public async Task Dispose_ExplicitlyOwnedHttpClientAndHandlerAreReleasedExactlyOnce()
    {
        var handler = new RecordingHandler();
        var http = new HttpClient(handler);
        var client = new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key, ownsHttpClient: true);
        NativeFixture.AssertChat(await client.GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), TestContext.Current.CancellationToken));
        client.Dispose();
        client.Dispose();
        Assert.Equal(1, handler.Disposals);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => http.GetAsync("https://never-sent.example", TestContext.Current.CancellationToken));
        Assert.Equal(1, handler.Sends);
    }

    [Theory]
    [InlineData("http://api.typesafe.ai")]
    [InlineData("https://api.typesafe.ai/v1")]
    [InlineData("https://api.typesafe.ai?query=1")]
    [InlineData("https://api.typesafe.ai#fragment")]
    [InlineData("https://user:password@api.typesafe.ai")]
    public void Configuration_RejectsNonOriginEndpointsBeforeSend(string endpoint)
    {
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler);
        Assert.Throws<ArgumentException>(() => new TypeSafeChatClient(http, NativeFixture.Settings with { Endpoint = new(endpoint) }, NativeFixture.Key));
        Assert.Throws<ArgumentException>(() => new TypeSafeModelClient(http, NativeFixture.Settings with { Endpoint = new(endpoint) }, NativeFixture.Key));
        Assert.Equal(0, handler.Sends);
    }

    [Fact]
    public void Configuration_RejectsEmptyModelKeyNamespaceAndNonpositiveLimit()
    {
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler);
        Assert.Throws<ArgumentException>(() => new TypeSafeChatClient(http, NativeFixture.Settings with { ModelId = "" }, NativeFixture.Key));
        Assert.Throws<ArgumentException>(() => new TypeSafeChatClient(http, NativeFixture.Settings, ""));
        Assert.Throws<ArgumentException>(() => new TypeSafeChatClient(http, NativeFixture.Settings, "key with whitespace"));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TypeSafeChatClient(http, NativeFixture.Settings with { MaximumResponseBytes = 0 }, NativeFixture.Key));
        Assert.Throws<ArgumentException>(() => new ServiceCollection().AddTypeSafeChatClient("native", NativeFixture.Key,
            NativeFixture.Settings with { CacheNamespace = "" }, useCache: true));
        Assert.Equal(4 * 1024 * 1024, new TypeSafeClientOptions().MaximumResponseBytes);
        Assert.Equal(0, handler.Sends);
    }
}
