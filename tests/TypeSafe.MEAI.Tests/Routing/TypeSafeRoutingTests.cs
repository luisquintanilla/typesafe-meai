using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.AI;
using TypeSafe.MEAI.Tests.Fixtures;
using Xunit;

namespace TypeSafe.MEAI.Tests.Routing;

[Trait("Stage", "C")]
public sealed class TypeSafeRoutingTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Route_UsesDecoratedClassifierOnceAndDelegatesUnchangedRequestAndUsage(bool streaming)
    {
        using var handler = new RecordingHandler((_, _, _) => Task.FromResult(Choice("billing", 0.8)));
        using var services = NativeFixture.Services(handler);
        var classifier = new Spy(services.Client());
        var billing = new Target();
        var other = new Target();
        var observed = new List<TypeSafeRouteDecision>();
        using var router = new TypeSafeRoutingChatClient(classifier, Routes(billing, other), 0.8, "jev-classifier",
            observe: (d, _) => { observed.Add(d); return ValueTask.CompletedTask; });
        ChatMessage[] original = [new(ChatRole.System, "System context"), new(ChatRole.User, "Billing question") { AuthorName = "synthetic" }];
        int enumerations = 0;
        IEnumerable<ChatMessage> Once()
        {
            if (++enumerations != 1) throw new InvalidOperationException("One-shot enumerable was enumerated twice.");
            foreach (var message in original) yield return message;
        }
        var options = new ChatOptions { ModelId = "downstream-model", Temperature = 0.25f, Instructions = "Downstream only", AdditionalProperties = new() { ["custom"] = 17 } };
        ChatResponse result = streaming
            ? await router.GetStreamingResponseAsync(Once(), options, TestContext.Current.CancellationToken).ToChatResponseAsync(cancellationToken: TestContext.Current.CancellationToken)
            : await router.GetResponseAsync(Once(), options, TestContext.Current.CancellationToken);
        Assert.Equal(1, enumerations);
        Assert.Equal(1, classifier.Calls);
        Assert.Equal(1, handler.Sends);
        Assert.Equal(1, billing.Calls);
        Assert.Equal(0, other.Calls);
        Assert.Equal(streaming, billing.Streamed);
        Assert.Equal(original, billing.Messages);
        Assert.NotSame(options, billing.Options);
        Assert.Equal("downstream-model", billing.Options!.ModelId);
        Assert.Equal(0.25f, billing.Options.Temperature);
        Assert.Equal("Downstream only", billing.Options.Instructions);
        Assert.Equal(17, billing.Options.AdditionalProperties!["custom"]);
        Assert.Equal("downstream-model", options.ModelId);
        Assert.Equal("actual-target-model", result.ModelId);
        Assert.Equal("target result", result.Text);
        Assert.Equal(7, result.Usage!.InputTokenCount);
        Assert.Equal(2, result.Usage.OutputTokenCount);
        Assert.Equal(9, result.Usage.TotalTokenCount);
        var decision = Assert.Single(observed);
        Assert.Equal("billing", decision.RouteId);
        Assert.False(decision.UsedFallback);
        Assert.Equal(0.8, decision.Evidence.GetAnswer<ChoiceAnswer>("route").Confidence);
        Assert.Equal(0.95, decision.Evidence.GetAnswer<ChoiceAnswer>("route").Probabilities.GetProperty("billing").GetDouble());
        var body = NativeFixture.Json(Assert.Single(handler.Requests).Body!);
        Assert.Equal("jev-classifier", body.GetProperty("model").GetString());
        Assert.Equal("system", body.GetProperty("state")[0].GetProperty("role").GetString());
        Assert.Equal("System context", body.GetProperty("state")[0].GetProperty("contents")[0].GetString());
        Assert.Equal("synthetic", body.GetProperty("state")[1].GetProperty("author").GetString());
        Assert.Equal("Billing question", body.GetProperty("state")[1].GetProperty("contents")[0].GetString());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Route_LowServerConfidenceAbstainsOrUsesOnlyExplicitFallback(bool hasFallback)
    {
        using var handler = new RecordingHandler((_, _, _) => Task.FromResult(Choice("billing", 0.799999)));
        using var services = NativeFixture.Services(handler);
        var billing = new Target();
        var other = new Target();
        var fallback = new Target();
        TypeSafeRouteDecision? observed = null;
        using var router = new TypeSafeRoutingChatClient(services.Client(), Routes(billing, other), 0.8, fallback: hasFallback ? fallback : null,
            observe: (d, _) => { observed = d; return ValueTask.CompletedTask; });
        if (hasFallback) await router.GetResponseAsync("question", cancellationToken: TestContext.Current.CancellationToken);
        else
        {
            var error = await Assert.ThrowsAsync<TypeSafeRoutingAbstentionException>(() => router.GetResponseAsync("question", cancellationToken: TestContext.Current.CancellationToken));
            Assert.Same(observed, error.Decision);
        }
        Assert.NotNull(observed);
        Assert.Null(observed.RouteId);
        Assert.Equal(hasFallback, observed.UsedFallback);
        Assert.Equal(0, billing.Calls);
        Assert.Equal(0, other.Calls);
        Assert.Equal(hasFallback ? 1 : 0, fallback.Calls);
        Assert.Equal(1, handler.Sends);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Route_HttpAndProtocolErrorsNeverBecomeConfidenceFallback(bool malformed)
    {
        using var handler = new RecordingHandler((_, _, _) => Task.FromResult(malformed ? NativeFixture.Response("{}") : NativeFixture.Response("denied", 401)));
        using var services = NativeFixture.Services(handler);
        var target = new Target();
        var fallback = new Target();
        using var router = new TypeSafeRoutingChatClient(services.Client(), Routes(target, target), 0, fallback: fallback);
        if (malformed) await Assert.ThrowsAsync<TypeSafeProtocolException>(() => router.GetResponseAsync("question", cancellationToken: TestContext.Current.CancellationToken));
        else await Assert.ThrowsAsync<TypeSafeApiException>(() => router.GetResponseAsync("question", cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal(0, target.Calls);
        Assert.Equal(0, fallback.Calls);
        Assert.Equal(1, handler.Sends);
    }

    [Fact]
    public async Task Route_HandlesAndUnsupportedDefaultModalitiesRejectBeforeClassifier_CustomProjectionIsExplicit()
    {
        using var handler = new RecordingHandler((_, _, _) => Task.FromResult(Choice("billing", 1)));
        using var services = NativeFixture.Services(handler);
        var target = new Target();
        using var router = new TypeSafeRoutingChatClient(services.Client(), Routes(target, target), 0.8);
        await Assert.ThrowsAsync<NotSupportedException>(() => router.GetResponseAsync("text", new() { ConversationId = "handle" }, TestContext.Current.CancellationToken));
        await Assert.ThrowsAsync<NotSupportedException>(() => router.GetResponseAsync("text", new() { ContinuationToken = ResponseContinuationToken.FromBytes(new byte[] { 17 }) }, TestContext.Current.CancellationToken));
        ChatMessage[] json = [TypeSafeState.Parse("""{"ticket":17}""").ToMessage()];
        await Assert.ThrowsAsync<NotSupportedException>(() => router.GetResponseAsync(json, cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal(0, handler.Sends);
        using var custom = new TypeSafeRoutingChatClient(services.Client(), Routes(target, target), 0.8,
            stateProjector: messages => TypeSafeState.Parse(System.Text.Encoding.UTF8.GetString(((DataContent)messages[0].Contents[0]).Data.Span)));
        await custom.GetResponseAsync(json, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(17, NativeFixture.Json(Assert.Single(handler.Requests).Body!).GetProperty("state").GetProperty("ticket").GetInt32());
        Assert.Equal(1, target.Calls);
    }

    [Fact]
    public async Task Route_CancellationAfterObservationPreventsTarget_AndDisposalBorrowsAllClients()
    {
        using var handler = new RecordingHandler((_, _, _) => Task.FromResult(Choice("billing", 1)));
        using var services = NativeFixture.Services(handler);
        var classifier = new Spy(services.Client());
        var target = new Target();
        var fallback = new Target();
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        var router = new TypeSafeRoutingChatClient(classifier, Routes(target, target), 0.8, fallback: fallback,
            observe: (_, _) => { cancellation.Cancel(); return ValueTask.CompletedTask; });
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => router.GetResponseAsync("text", cancellationToken: cancellation.Token));
        Assert.Equal(0, target.Calls);
        Assert.Equal(0, fallback.Calls);
        Assert.Equal(1, handler.Sends);
        router.Dispose();
        router.Dispose();
        Assert.Equal(0, classifier.Disposals);
        Assert.Equal(0, target.Disposals);
        Assert.Equal(0, fallback.Disposals);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => router.GetResponseAsync("text", cancellationToken: TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task Route_ThirtyTwoControlledRequestsKeepDecisionsOptionsAndTargetsIsolated()
    {
        var allEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var handler = new RecordingHandler(async (request, attempt, token) =>
        {
            var state = NativeFixture.Json(await request.Content!.ReadAsStringAsync(token)).GetProperty("state");
            int index = int.Parse(state[0].GetProperty("contents")[0].GetString()!, System.Globalization.CultureInfo.InvariantCulture);
            if (attempt == 32) allEntered.SetResult();
            await release.Task.WaitAsync(token);
            return Choice(index % 2 == 0 ? "billing" : "other", 1, index);
        });
        using var services = NativeFixture.Services(handler);
        var billing = new Target();
        var other = new Target();
        var observations = new ConcurrentBag<TypeSafeRouteDecision>();
        using var router = new TypeSafeRoutingChatClient(services.Client(), Routes(billing, other), 0.8,
            observe: (d, _) => { observations.Add(d); return ValueTask.CompletedTask; });
        var shared = new ChatOptions { ModelId = "downstream" };
        var tasks = Enumerable.Range(0, 32).Select(i => router.GetResponseAsync(i.ToString(System.Globalization.CultureInfo.InvariantCulture),
            shared, TestContext.Current.CancellationToken)).ToArray();
        await allEntered.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        Assert.All(tasks, t => Assert.False(t.IsCompleted));
        Assert.Equal("downstream", shared.ModelId);
        release.SetResult();
        await Task.WhenAll(tasks);
        Assert.Equal(32, handler.Sends);
        Assert.Equal(16, billing.Calls);
        Assert.Equal(16, other.Calls);
        Assert.Equal(32, observations.Count);
        Assert.Equal(32, observations.Select(d => d.Evidence.Json.GetProperty("echo").GetInt32()).Distinct().Count());
        Assert.All(observations, d => Assert.Equal(d.Evidence.Json.GetProperty("echo").GetInt32() % 2 == 0 ? "billing" : "other", d.RouteId));
        Assert.Equal("downstream", shared.ModelId);
    }

    [Fact]
    public void Route_RejectsInvalidConfigurationWithoutCalls()
    {
        var classifier = new Target();
        var target = new Target();
        Assert.Throws<ArgumentException>(() => new TypeSafeRoutingChatClient(classifier, [], 0.8));
        Assert.Throws<ArgumentException>(() => new TypeSafeRoutingChatClient(classifier, [new("same", "First", target), new("same", "Second", target)], 0.8));
        Assert.Throws<ArgumentException>(() => new TypeSafeRoutingChatClient(classifier, [new("a", "", target)], 0.8));
        Assert.Throws<ArgumentException>(() => new TypeSafeRoutingChatClient(classifier, [new("a", "Self", classifier)], 0.8));
        Assert.Throws<ArgumentException>(() => new TypeSafeRoutingChatClient(classifier, Routes(target, target), 0.8, fallback: classifier));
        Assert.Throws<ArgumentException>(() => new TypeSafeRoutingChatClient(classifier, Enumerable.Range(0, 256).Select(i => new TypeSafeRoute($"r{i}", "route", target)), 0.8));
        foreach (double threshold in new[] { -0.1, 1.1, double.NaN, double.PositiveInfinity })
            Assert.Throws<ArgumentOutOfRangeException>(() => new TypeSafeRoutingChatClient(classifier, Routes(target, target), threshold));
        Assert.Equal(0, classifier.Calls);
        Assert.Equal(0, target.Calls);
    }

    private static TypeSafeRoute[] Routes(IChatClient first, IChatClient second) => [new("billing", "Payment issues", first), new("other", "Other issues", second)];
    private static HttpResponseMessage Choice(string selected, double confidence, int echo = 0) => NativeFixture.Response(JsonSerializer.Serialize(new
    {
        model = "jev-actual", usage = new { input_tokens = 50, output_tokens = 10 }, echo,
        answers = new { route = new { type = "choice", choice = selected, confidence, probabilities = new Dictionary<string, double> { ["billing"] = selected == "billing" ? 0.95 : 0.05, ["other"] = selected == "other" ? 0.95 : 0.05 } } }
    }));

    private sealed class Spy(IChatClient inner) : DelegatingChatClient(inner)
    {
        internal int Calls;
        internal int Disposals;
        public override Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
        {
            Interlocked.Increment(ref Calls);
            return base.GetResponseAsync(messages, options, cancellationToken);
        }
        public override object? GetService(Type serviceType, object? serviceKey = null) => throw new InvalidOperationException("Do not unwrap the classifier.");
        protected override void Dispose(bool disposing) { if (disposing) Disposals++; }
    }

    private sealed class Target : IChatClient
    {
        internal int Calls;
        internal int Disposals;
        internal bool Streamed;
        internal ChatMessage[]? Messages;
        internal ChatOptions? Options;
        public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Interlocked.Increment(ref Calls);
            Messages = messages.ToArray();
            Options = options;
            return Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, "target result")) { ModelId = "actual-target-model", Usage = new() { InputTokenCount = 7, OutputTokenCount = 2, TotalTokenCount = 9 } });
        }
        public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            Streamed = true;
            foreach (var update in (await GetResponseAsync(messages, options, cancellationToken)).ToChatResponseUpdates()) yield return update;
        }
        public object? GetService(Type serviceType, object? serviceKey = null) => null;
        public void Dispose() => Disposals++;
    }
}
