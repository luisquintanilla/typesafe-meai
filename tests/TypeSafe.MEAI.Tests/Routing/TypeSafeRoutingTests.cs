using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.AI;
using TypeSafe.MEAI.Tests.Fixtures;
using Xunit;

#pragma warning disable MEAI001 // Explicitly exercise pinned experimental RoutingChatClient.
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
        var observed = new List<HostDecision>();
        using var router = Compose(classifier, Routes(billing, other), 0.8, "jev-classifier",
            observe: (d, _) => { observed.Add(d); return ValueTask.CompletedTask; });
        ChatMessage[] original = [new(ChatRole.System, "System context"), new(ChatRole.User, "Billing question") { AuthorName = "synthetic" }];
        int enumerations = 0;
        IEnumerable<ChatMessage> Once()
        {
            if (++enumerations != 1) throw new InvalidOperationException("One-shot enumerable was enumerated twice.");
            foreach (var message in original) yield return message;
        }
        var options = new ChatOptions { ModelId = "downstream-model", Temperature = 0.25f, Instructions = "Downstream only", AdditionalProperties = new() { ["custom"] = 17 } };
        // RoutingContext deliberately requires a repeatable sequence. The host owns
        // enumeration, so snapshot a one-shot source before selection and downstream.
        var snapshot = Once().ToArray();
        ChatResponse result = streaming
            ? await router.GetStreamingResponseAsync(snapshot, options, TestContext.Current.CancellationToken).ToChatResponseAsync(cancellationToken: TestContext.Current.CancellationToken)
            : await router.GetResponseAsync(snapshot, options, TestContext.Current.CancellationToken);
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
        Assert.Equal(0.8, Assert.IsType<ChoiceAnswer>(decision.Evidence.Answers["route"]).Confidence);
        Assert.Equal(0.95, Assert.IsType<ChoiceAnswer>(decision.Evidence.Answers["route"]).Probabilities.GetProperty("billing").GetDouble());
        var body = NativeFixture.Json(Assert.Single(handler.Requests).Body!);
        Assert.Equal("jev-classifier", body.GetProperty("model").GetString());
        var conversation = body.GetProperty("state").GetProperty("conversation");
        Assert.Equal("system", conversation[0].GetProperty("role").GetString());
        Assert.Equal("System context", conversation[0].GetProperty("contents")[0].GetProperty("value").GetString());
        Assert.Equal("synthetic", conversation[1].GetProperty("author").GetString());
        Assert.Equal("Billing question", conversation[1].GetProperty("contents")[0].GetProperty("value").GetString());
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
        HostDecision? observed = null;
        using var router = Compose(services.Client(), Routes(billing, other), 0.8, fallback: hasFallback ? fallback : null,
            observe: (d, _) => { observed = d; return ValueTask.CompletedTask; });
        if (hasFallback) await router.GetResponseAsync("question", cancellationToken: TestContext.Current.CancellationToken);
        else
        {
            var error = await Assert.ThrowsAsync<TypeSafeAbstentionException>(() => router.GetResponseAsync("question", cancellationToken: TestContext.Current.CancellationToken));
            Assert.Same(observed!.Evidence, Assert.Single(error.Evidence));
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
        using var router = Compose(services.Client(), Routes(target, target), 0, fallback: fallback);
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
        using var router = Compose(services.Client(), Routes(target, target), 0.8);
        await Assert.ThrowsAsync<NotSupportedException>(() => router.GetResponseAsync("text", new() { ConversationId = "handle" }, TestContext.Current.CancellationToken));
        await Assert.ThrowsAsync<NotSupportedException>(() => router.GetResponseAsync("text", new() { ContinuationToken = ResponseContinuationToken.FromBytes(new byte[] { 17 }) }, TestContext.Current.CancellationToken));
        ChatMessage[] unsupported = [new(ChatRole.User, [new DataContent(new byte[] { 1 }, "image/png")])];
        await Assert.ThrowsAsync<NotSupportedException>(() => router.GetResponseAsync(unsupported, cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal(0, handler.Sends);
        await router.GetResponseAsync(NativeFixture.Messages("""{"ticket":17}"""), cancellationToken: TestContext.Current.CancellationToken);
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
        var router = Compose(classifier, Routes(target, target), 0.8, fallback: fallback,
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
        Assert.Equal(0, target.Calls);
    }

    [Fact]
    public async Task Route_ThirtyTwoControlledRequestsKeepDecisionsOptionsAndTargetsIsolated()
    {
        var allEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var handler = new RecordingHandler(async (request, attempt, token) =>
        {
            var state = NativeFixture.Json(await request.Content!.ReadAsStringAsync(token)).GetProperty("state");
            int index = int.Parse(state.GetString()!, System.Globalization.CultureInfo.InvariantCulture);
            if (attempt == 32) allEntered.SetResult();
            await release.Task.WaitAsync(token);
            return Choice(index % 2 == 0 ? "billing" : "other", 1, index);
        });
        using var services = NativeFixture.Services(handler);
        var billing = new Target();
        var other = new Target();
        var observations = new ConcurrentBag<HostDecision>();
        using var router = Compose(services.Client(), Routes(billing, other), 0.8,
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
        Assert.Throws<ArgumentException>(() => Compose(classifier, [], 0.8));
        Assert.Throws<ArgumentException>(() => Compose(classifier, [new("same", "First", target), new("same", "Second", target)], 0.8));
        Assert.Throws<ArgumentException>(() => Compose(classifier, [new("a", "", target)], 0.8));
        Assert.Throws<ArgumentException>(() => Compose(classifier, [new("a", "Self", classifier)], 0.8));
        Assert.Throws<ArgumentException>(() => Compose(classifier, Routes(target, target), 0.8, fallback: classifier));
        Assert.Throws<ArgumentException>(() => Compose(classifier, Enumerable.Range(0, 256).Select(i => new HostRoute($"r{i}", "route", target)), 0.8));
        foreach (double threshold in new[] { -0.1, 1.1, double.NaN, double.PositiveInfinity })
            Assert.Throws<ArgumentOutOfRangeException>(() => Compose(classifier, Routes(target, target), threshold));
        Assert.Equal(0, classifier.Calls);
        Assert.Equal(0, target.Calls);
    }

    private sealed record HostRoute(string Id, string Description, IChatClient Client);
    private sealed record HostDecision(string? RouteId, bool UsedFallback, JevResultContent Evidence);
    private static HostRoute[] Routes(IChatClient first, IChatClient second) => [new("billing", "Payment issues", first), new("other", "Other issues", second)];
    private static RoutingChatClient Compose(IChatClient classifier, IEnumerable<HostRoute> routes, double threshold,
        string? model = null, IChatClient? fallback = null, Func<HostDecision, CancellationToken, ValueTask>? observe = null)
    {
        var candidates = routes.ToArray();
        if (!double.IsFinite(threshold) || threshold is < 0 or > 1) throw new ArgumentOutOfRangeException(nameof(threshold));
        if (candidates.Length is < 1 or > 255 || candidates.Any(r => string.IsNullOrWhiteSpace(r.Id) ||
            string.IsNullOrWhiteSpace(r.Description) || ReferenceEquals(r.Client, classifier)) ||
            candidates.Select(r => r.Id).Distinct(StringComparer.Ordinal).Count() != candidates.Length ||
            ReferenceEquals(fallback, classifier)) throw new ArgumentException("Invalid host routes.");
        return RoutingChatClient.Create(async (context, token) =>
        {
            if (context.ChatOptions?.ConversationId is not null || context.ChatOptions?.ContinuationToken is not null)
                throw new NotSupportedException("Host routing does not transfer provider handles.");
            var options = NativeFixture.Options([new ChoiceQuestion("route", TypeSafeJson.Entry("Select a support route."),
                candidates.Select(r => KeyValuePair.Create(r.Id, TypeSafeJson.Entry(r.Description))))]);
            options.ModelId = model;
            var response = await classifier.GetResponseAsync(context.Messages, options, token);
            var evidence = Assert.Single(response.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>());
            var answer = Assert.IsType<ChoiceAnswer>(evidence.Answers["route"]);
            bool accepted = answer.Confidence >= threshold;
            var decision = new HostDecision(accepted ? answer.Choice : null, !accepted && fallback is not null, evidence);
            if (observe is not null) await observe(decision, token);
            token.ThrowIfCancellationRequested();
            return accepted ? candidates.Single(r => r.Id == answer.Choice).Client :
                fallback ?? throw new TypeSafeAbstentionException([evidence], [new("", "Host route confidence below threshold.")]);
        });
    }
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
