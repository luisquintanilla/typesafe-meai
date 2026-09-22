using System.Net;
using System.Net.Http.Headers;
using System.Text;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Polly.Timeout;
using TypeSafe.MEAI.Tests.Fixtures;
using Xunit;

namespace TypeSafe.MEAI.Tests.Transport;

[Trait("Stage", "A")]
public sealed class TypeSafeHttpTransportTests
{
    [Theory]
    [InlineData(429)]
    [InlineData(529)]
    [InlineData(408)]
    [InlineData(500)]
    [InlineData(502)]
    [InlineData(503)]
    [InlineData(504)]
    public async Task RetryPolicy_RetryableStatusThenSuccessUsesOneHttpRetryOwner(int status)
    {
        var clock = new TestClock();
        var failedStream = new TrackedStream("private-error"u8.ToArray());
        var failedContent = new TrackedContent(failedStream);
        var failedResponse = new TrackedResponse((HttpStatusCode)status) { Content = failedContent };
        using var handler = new RecordingHandler((_, attempt, _) => Task.FromResult(attempt == 1
            ? failedResponse
            : NativeFixture.Response()));
        using var services = NativeFixture.Services(handler, clock: clock);
        var pending = Send(services.Client());
        Assert.Equal(1, handler.Sends);
        await clock.AdvanceRetryAsync();
        NativeFixture.AssertChat(await pending);
        Assert.Equal(2, handler.Sends);
        Assert.True(failedStream.Disposed);
        Assert.True(failedContent.Disposed);
        Assert.True(failedResponse.Disposed);
        Assert.All(handler.Requests, request =>
        {
            Assert.Equal(HttpMethod.Post, request.Method);
            Assert.Equal("Bearer", request.Scheme);
            Assert.Equal("/v1/systemone", request.Uri.AbsolutePath);
        });
    }

    [Theory]
    [InlineData(429, false)]
    [InlineData(529, false)]
    [InlineData(408, false)]
    [InlineData(500, false)]
    [InlineData(429, true)]
    public async Task RetryPolicy_BoundsTwoRetriesAndPreservesTerminalStatus(int status, bool succeeds)
    {
        var clock = new TestClock();
        using var handler = new RecordingHandler((_, attempt, _) =>
        {
            if (succeeds && attempt == 3) return Task.FromResult(NativeFixture.Response());
            var response = NativeFixture.Response("SYNTHETIC-PRIVATE-ERROR", attempt == 2 && succeeds ? 529 : status);
            response.Headers.Add("x-request-id", "request-17");
            return Task.FromResult(response);
        });
        using var services = NativeFixture.Services(handler, clock: clock);
        var task = Send(services.Client());
        await clock.AdvanceRetryAsync();
        await clock.AdvanceRetryAsync();
        if (succeeds) NativeFixture.AssertChat(await task);
        else
        {
            var error = await Assert.ThrowsAsync<TypeSafeApiException>(() => task);
            Assert.Equal((HttpStatusCode)status, error.StatusCode);
            Assert.Equal("request-17", error.RequestId);
            Assert.Equal("SYNTHETIC-PRIVATE-ERROR", error.DiagnosticBody);
            Assert.DoesNotContain("SYNTHETIC-PRIVATE-ERROR", error.ToString());
        }
        Assert.Equal(3, handler.Sends);
        clock.Advance(TimeSpan.FromMinutes(1));
        Assert.Equal(3, handler.Sends);
    }

    [Theory]
    [InlineData(401)]
    [InlineData(422)]
    public async Task Errors_PermanentStatusesNeverRetryAndDoNotLeakDiagnosticBodies(int status)
    {
        var clock = new TestClock();
        var stream = new TrackedStream("SYNTHETIC-PRIVATE-ERROR"u8.ToArray());
        var content = new TrackedContent(stream);
        using var handler = new RecordingHandler((_, _, _) =>
        {
            var response = new HttpResponseMessage((HttpStatusCode)status) { Content = content };
            response.Headers.Add("x-request-id", "request-17");
            return Task.FromResult(response);
        });
        using var services = NativeFixture.Services(handler, clock: clock);
        var error = await Assert.ThrowsAsync<TypeSafeApiException>(() => Send(services.Client()));
        Assert.Equal((HttpStatusCode)status, error.StatusCode);
        Assert.Equal("request-17", error.RequestId);
        Assert.Equal("SYNTHETIC-PRIVATE-ERROR", error.DiagnosticBody);
        Assert.DoesNotContain("SYNTHETIC-PRIVATE-ERROR", error.ToString());
        Assert.DoesNotContain(NativeFixture.Key, error.ToString());
        clock.Advance(TimeSpan.FromMinutes(1));
        Assert.Equal(1, handler.Sends);
        Assert.True(stream.Disposed);
        Assert.True(content.Disposed);
    }

    [Fact]
    public async Task RetryPolicy_TransientConnectionFailureThenSuccess()
    {
        var clock = new TestClock();
        using var handler = new RecordingHandler((_, attempt, _) => attempt == 1
            ? Task.FromException<HttpResponseMessage>(new HttpRequestException(HttpRequestError.ConnectionError, "synthetic connection failure"))
            : Task.FromResult(NativeFixture.Response()));
        using var services = NativeFixture.Services(handler, clock: clock);
        var task = Send(services.Client());
        await clock.AdvanceRetryAsync();
        NativeFixture.AssertChat(await task);
        Assert.Equal(2, handler.Sends);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task RetryAfter_RespectsDeltaOrDateBeforeStartingNextAttempt(bool date)
    {
        var clock = new TestClock();
        using var handler = new RecordingHandler((_, attempt, _) =>
        {
            var response = NativeFixture.Response(attempt == 1 ? "retry" : NativeFixture.Envelope, attempt == 1 ? 429 : 200);
            if (attempt == 1) response.Headers.RetryAfter = date
                ? new RetryConditionHeaderValue(clock.GetUtcNow() + TimeSpan.FromSeconds(2))
                : new RetryConditionHeaderValue(TimeSpan.FromSeconds(2));
            return Task.FromResult(response);
        });
        using var services = NativeFixture.Services(handler, clock: clock);
        var task = Send(services.Client());
        Assert.Equal(1, handler.Sends);
        Assert.False(task.IsCompleted);
        var retry = await clock.NextTimerAsync(delay => delay < TimeSpan.FromSeconds(10));
        Assert.Equal(TimeSpan.FromSeconds(2), retry.Delay);
        clock.Advance(TimeSpan.FromMilliseconds(1999));
        Assert.Equal(1, handler.Sends);
        Assert.False(task.IsCompleted);
        clock.Advance(TimeSpan.FromMilliseconds(1));
        NativeFixture.AssertChat(await task);
        Assert.Equal(2, handler.Sends);
    }

    [Fact]
    public async Task RetryAfter_LongerThanTotalBudgetNeverStartsLateAttempt()
    {
        var clock = new TestClock();
        using var handler = new RecordingHandler((_, _, _) =>
        {
            var response = NativeFixture.Response("retry", 429);
            response.Headers.RetryAfter = new(TimeSpan.FromSeconds(60));
            return Task.FromResult(response);
        });
        using var services = NativeFixture.Services(handler, clock: clock);
        var task = Send(services.Client());
        var retry = await clock.NextTimerAsync(delay => delay == TimeSpan.FromSeconds(60));
        Assert.Equal(TimeSpan.FromSeconds(60), retry.Delay);
        clock.Advance(TimeSpan.FromSeconds(30));
        await Assert.ThrowsAsync<TimeoutRejectedException>(() => task);
        clock.Advance(TimeSpan.FromMinutes(1));
        Assert.Equal(1, handler.Sends);
    }

    [Theory]
    [InlineData("before", 0)]
    [InlineData("send", 1)]
    [InlineData("backoff", 1)]
    [InlineData("body", 1)]
    public async Task Cancellation_DuringSendBackoffOrBodyStopsFurtherAttempts(string point, int sends)
    {
        var clock = new TestClock();
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var stream = new TrackedStream(Encoding.UTF8.GetBytes(NativeFixture.Envelope), stall: true);
        var content = new TrackedContent(stream);
        using var handler = new RecordingHandler(async (_, _, token) =>
        {
            if (point == "send")
            {
                entered.SetResult();
                await Task.Delay(Timeout.InfiniteTimeSpan, token);
            }
            return point == "body" ? new(HttpStatusCode.OK) { Content = content } : NativeFixture.Response("retry", 429);
        });
        using var services = NativeFixture.Services(handler, clock: clock);
        if (point == "before") cancellation.Cancel();
        async Task<ChatResponse> InvokeAsync() => await services.Client().GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), cancellation.Token);
        var task = InvokeAsync();
        if (point == "send") await entered.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        if (point == "body") await stream.Entered.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        if (point == "backoff") await clock.NextTimerAsync(delay => delay < TimeSpan.FromSeconds(5));
        cancellation.Cancel();
        var error = await Assert.ThrowsAnyAsync<OperationCanceledException>(() => task);
        Assert.Equal(cancellation.Token, error.CancellationToken);
        clock.Advance(TimeSpan.FromMinutes(1));
        Assert.Equal(sends, handler.Sends);
        if (point == "body")
        {
            Assert.True(stream.Canceled.Task.IsCompletedSuccessfully);
            Assert.True(stream.Disposed);
            Assert.True(content.Disposed);
        }
    }

    [Theory]
    [InlineData(10, 30)]
    [InlineData(2, 5)]
    public async Task BodyRead_StallIsInsideAttemptBudgetAndRetryDisposesFirstResponse(int attemptSeconds, int totalSeconds)
    {
        var clock = new TestClock();
        var stream = new TrackedStream(Encoding.UTF8.GetBytes(NativeFixture.Envelope), stall: true);
        var content = new TrackedContent(stream);
        using var handler = new RecordingHandler((_, attempt, _) => Task.FromResult(attempt == 1
            ? new HttpResponseMessage(HttpStatusCode.OK) { Content = content } : NativeFixture.Response()));
        using var services = NativeFixture.Services(handler, clock: clock,
            attempt: attemptSeconds == 10 ? null : TimeSpan.FromSeconds(attemptSeconds),
            total: totalSeconds == 30 ? null : TimeSpan.FromSeconds(totalSeconds));
        using var configuredHttp = services.GetRequiredService<IHttpClientFactory>().CreateClient("typesafe:native");
        Assert.Equal(Timeout.InfiniteTimeSpan, configuredHttp.Timeout);
        var task = Send(services.Client());
        await stream.Entered.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        clock.Advance(TimeSpan.FromSeconds(attemptSeconds) - TimeSpan.FromTicks(1));
        Assert.False(stream.Canceled.Task.IsCompleted);
        Assert.False(task.IsCompleted);
        clock.Advance(TimeSpan.FromTicks(1));
        await stream.Canceled.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        // Ignore queued deadline registrations; only a newly scheduled subsecond-to-5s retry is advanced.
        var retry = await clock.NextTimerAsync(delay => delay > TimeSpan.Zero && delay < TimeSpan.FromSeconds(attemptSeconds));
        Assert.True(stream.Disposed);
        Assert.True(content.Disposed);
        Assert.Equal(1, handler.Sends);
        clock.Advance(retry.DueAt - clock.GetUtcNow());
        NativeFixture.AssertChat(await task);
        Assert.Equal(2, handler.Sends);
    }

    [Theory]
    [InlineData(10, 30)]
    [InlineData(2, 5)]
    public async Task BodyRead_RepeatedStallsCannotEscapeTotalOperationBudget(int attemptSeconds, int totalSeconds)
    {
        var clock = new TestClock();
        var streams = new List<TrackedStream>();
        var contents = new List<TrackedContent>();
        var secondEntered = new TaskCompletionSource<TrackedStream>(TaskCreationOptions.RunContinuationsAsynchronously);
        using var handler = new RecordingHandler((_, attempt, _) =>
        {
            var stream = new TrackedStream(Encoding.UTF8.GetBytes(NativeFixture.Envelope), stall: true);
            var content = new TrackedContent(stream);
            streams.Add(stream);
            contents.Add(content);
            if (attempt == 2) secondEntered.SetResult(stream);
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = content });
        });
        using var services = NativeFixture.Services(handler, clock: clock,
            attempt: attemptSeconds == 10 ? null : TimeSpan.FromSeconds(attemptSeconds),
            total: totalSeconds == 30 ? null : TimeSpan.FromSeconds(totalSeconds));
        var started = clock.GetUtcNow();
        var task = Send(services.Client());
        await streams[0].Entered.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        clock.Advance(TimeSpan.FromSeconds(attemptSeconds));
        var retry = await clock.NextTimerAsync(delay => delay > TimeSpan.Zero && delay < TimeSpan.FromSeconds(attemptSeconds));
        clock.Advance(retry.DueAt - clock.GetUtcNow());
        var second = await secondEntered.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        await second.Entered.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        // The total timer must cancel whichever attempt/backoff is active, regardless of its remaining budget.
        clock.Advance(started + TimeSpan.FromSeconds(totalSeconds) - clock.GetUtcNow());
        var error = await Assert.ThrowsAsync<TimeoutRejectedException>(() => task);
        Assert.Equal(TimeSpan.FromSeconds(totalSeconds), error.Timeout);
        Assert.Equal(2, handler.Sends);
        var before = handler.Sends;
        clock.Advance(TimeSpan.FromMinutes(1));
        Assert.Equal(before, handler.Sends);
        Assert.All(streams, stream => Assert.True(stream.Disposed));
        Assert.All(contents, content => Assert.True(content.Disposed));
        Assert.True(streams[0].Canceled.Task.IsCompletedSuccessfully);
    }

    public static TheoryData<int, bool, int, bool> BodyLimits
    {
        get
        {
            TheoryData<int, bool, int, bool> data = new();
            foreach (int limit in new[] { 1024, 4 * 1024 * 1024 })
            foreach (bool known in new[] { false, true })
            foreach (int status in new[] { 200, 422 })
            foreach (bool excess in new[] { false, true })
                data.Add(limit, known, status, excess);
            return data;
        }
    }

    [Theory]
    [MemberData(nameof(BodyLimits))]
    public async Task BodyLimit_EnforcesExactSuccessAndErrorBoundariesWithAndWithoutContentLength(int limit, bool known, int status, bool excess)
    {
        string prefix = status == 200 ? NativeFixture.Envelope : "SYNTHETIC-PRIVATE-ERROR";
        var bytes = Encoding.UTF8.GetBytes(prefix.PadRight(limit + (excess ? 1 : 0), ' '));
        var stream = new TrackedStream(bytes, knownLength: known);
        var content = new TrackedContent(stream);
        var response = new TrackedResponse((HttpStatusCode)status) { Content = content };
        using var handler = new RecordingHandler((_, _, _) => Task.FromResult<HttpResponseMessage>(response));
        using var services = NativeFixture.Services(handler, settings: NativeFixture.Settings with { MaximumResponseBytes = limit });
        if (excess) await Assert.ThrowsAsync<TypeSafeBodyLimitException>(() => Send(services.Client()));
        else if (status == 200) NativeFixture.AssertChat(await Send(services.Client()));
        else
        {
            var error = await Assert.ThrowsAsync<TypeSafeApiException>(() => Send(services.Client()));
            Assert.Equal(limit, Encoding.UTF8.GetByteCount(error.DiagnosticBody));
            Assert.Equal(HttpStatusCode.UnprocessableEntity, error.StatusCode);
        }
        Assert.Equal(1, handler.Sends);
        Assert.True(content.Disposed);
        Assert.True(stream.Disposed);
        Assert.True(response.Disposed);
        if (known && excess) Assert.Equal(0, stream.ReadCount);
        else Assert.True(stream.ReadCount > 0);
    }

    [Fact]
    public async Task BodyRead_FailureRetriesWithinPipelineAndDisposesFailedAndSuccessfulStreams()
    {
        var clock = new TestClock();
        var failed = new TrackedStream("ignored"u8.ToArray(), failRead: true);
        var successful = new TrackedStream(Encoding.UTF8.GetBytes(NativeFixture.Envelope));
        var failedContent = new TrackedContent(failed);
        var successfulContent = new TrackedContent(successful);
        var failedResponse = new TrackedResponse(HttpStatusCode.OK) { Content = failedContent };
        var successfulResponse = new TrackedResponse(HttpStatusCode.OK) { Content = successfulContent };
        using var handler = new RecordingHandler((_, attempt, _) => Task.FromResult<HttpResponseMessage>(
            attempt == 1 ? failedResponse : successfulResponse));
        using var services = NativeFixture.Services(handler, clock: clock);
        var task = Send(services.Client());
        await clock.AdvanceRetryAsync();
        NativeFixture.AssertChat(await task);
        Assert.Equal(2, handler.Sends);
        Assert.True(failed.Disposed);
        Assert.True(failedContent.Disposed);
        Assert.True(successful.Disposed);
        Assert.True(successfulContent.Disposed);
        Assert.True(failedResponse.Disposed);
        Assert.True(successfulResponse.Disposed);
    }

    [Fact]
    public async Task BodyRead_ExhaustedFailuresRemainHttpErrorsAndDisposeEveryResponse()
    {
        var clock = new TestClock();
        var streams = Enumerable.Range(0, 3).Select(_ => new TrackedStream("ignored"u8.ToArray(), failRead: true)).ToArray();
        var responses = streams.Select(stream => new TrackedResponse(HttpStatusCode.OK) { Content = new TrackedContent(stream) }).ToArray();
        using var handler = new RecordingHandler((_, attempt, _) => Task.FromResult<HttpResponseMessage>(responses[attempt - 1]));
        using var services = NativeFixture.Services(handler, clock: clock);
        var pending = Send(services.Client());
        await clock.AdvanceRetryAsync();
        await clock.AdvanceRetryAsync();
        var error = await Assert.ThrowsAsync<HttpRequestException>(() => pending);
        Assert.Equal(HttpRequestError.ResponseEnded, error.HttpRequestError);
        Assert.DoesNotContain("SYNTHETIC-PRIVATE-READ-ERROR", error.ToString());
        Assert.Equal(3, handler.Sends);
        Assert.All(streams, stream => Assert.True(stream.Disposed));
        Assert.All(responses, response => Assert.True(response.Disposed));
    }

    [Theory]
    [InlineData(301)]
    [InlineData(302)]
    [InlineData(307)]
    [InlineData(308)]
    public async Task Redirects_FailExplicitlyWithoutForeignOriginSend(int status)
    {
        using var handler = new RecordingHandler((_, _, _) =>
        {
            var response = NativeFixture.Response("redirect", status);
            response.Headers.Location = new("https://foreign.example/steal");
            return Task.FromResult(response);
        });
        using var services = NativeFixture.Services(handler);
        var error = await Assert.ThrowsAsync<TypeSafeApiException>(() => Send(services.Client()));
        Assert.Equal((HttpStatusCode)status, error.StatusCode);
        Assert.Equal("api.typesafe.ai", Assert.Single(handler.Requests).Uri.Host);
        Assert.DoesNotContain(handler.Requests, r => r.Uri.Host == "foreign.example");
        Assert.Equal(1, handler.Sends);
    }

    private static Task<ChatResponse> Send(IChatClient client) =>
        client.GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), TestContext.Current.CancellationToken);
}
