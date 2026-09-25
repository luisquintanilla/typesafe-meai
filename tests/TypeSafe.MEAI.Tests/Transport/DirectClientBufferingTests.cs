using System.Net;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.AI;
using TypeSafe.MEAI.Tests.Fixtures;
using Xunit;

namespace TypeSafe.MEAI.Tests.Transport;

[Trait("Stage", "A")]
public sealed class DirectClientBufferingTests
{
    private const string Models = """{"models":["jev"],"vendor":{"revision":17}}""";

    [Theory]
    [InlineData(false, 200, false)]
    [InlineData(false, 200, true)]
    [InlineData(false, 422, false)]
    [InlineData(false, 422, true)]
    [InlineData(true, 200, false)]
    [InlineData(true, 200, true)]
    [InlineData(true, 422, false)]
    [InlineData(true, 422, true)]
    public async Task DirectClient_EnforcesUnknownLengthBodyBoundary(bool models, int status, bool excess)
    {
        const int limit = 1024;
        var prefix = status == 200 ? models ? Models : NativeFixture.Envelope : "SYNTHETIC-PRIVATE-DIRECT-ERROR";
        var stream = new TrackedStream(Encoding.UTF8.GetBytes(prefix.PadRight(limit + (excess ? 1 : 0), ' ')));
        var content = new TrackedContent(stream);
        var response = new TrackedResponse((HttpStatusCode)status) { Content = content };
        using var handler = new RecordingHandler((_, _, _) => Task.FromResult<HttpResponseMessage>(response));
        using var http = new HttpClient(handler) { Timeout = Timeout.InfiniteTimeSpan };
        var settings = NativeFixture.Settings with { MaximumResponseBytes = limit };
        Assert.Null(content.Headers.ContentLength);

        if (excess)
        {
            var error = await Assert.ThrowsAsync<TypeSafeBodyLimitException>(() =>
                InvokeAsync(http, models, settings, TestContext.Current.CancellationToken));
            Assert.DoesNotContain(prefix, error.ToString());
        }
        else if (status != 200)
        {
            var error = await Assert.ThrowsAsync<TypeSafeApiException>(() =>
                InvokeAsync(http, models, settings, TestContext.Current.CancellationToken));
            Assert.Equal(HttpStatusCode.UnprocessableEntity, error.StatusCode);
            Assert.Equal(limit, Encoding.UTF8.GetByteCount(error.DiagnosticBody));
            Assert.StartsWith("SYNTHETIC-PRIVATE-DIRECT-ERROR", error.DiagnosticBody);
            Assert.DoesNotContain("SYNTHETIC-PRIVATE-DIRECT-ERROR", error.ToString());
        }
        else
        {
            var result = await InvokeAsync(http, models, settings, TestContext.Current.CancellationToken);
            AssertPayload(result, models);
        }

        Assert.Equal(1, handler.Sends);
        Assert.True(stream.ReadCount > 0);
        Assert.True(stream.Disposed);
        Assert.True(content.Disposed);
        Assert.True(response.Disposed);
        Assert.Equal(0, handler.Disposals);
        Assert.Equal(models ? "/v1/models" : "/v1/systemone", Assert.Single(handler.Requests).Uri.AbsolutePath);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public async Task DirectClient_CancellationOrCallerDeadlineDisposesStalledBodyWithoutRetry(bool models, bool deadline)
    {
        var clock = new TestClock();
        using var cancellation = deadline
            ? new CancellationTokenSource(TimeSpan.FromSeconds(5), clock)
            : CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        var stream = new TrackedStream(Encoding.UTF8.GetBytes(models ? Models : NativeFixture.Envelope), stall: true);
        var content = new TrackedContent(stream);
        var response = new TrackedResponse(HttpStatusCode.OK) { Content = content };
        using var handler = new RecordingHandler((_, attempt, _) => Task.FromResult<HttpResponseMessage>(
            attempt == 1 ? response : NativeFixture.Response(models ? Models : NativeFixture.Envelope)));
        using var http = new HttpClient(handler) { Timeout = Timeout.InfiniteTimeSpan };
        Assert.Null(content.Headers.ContentLength);

        var pending = InvokeAsync(http, models, NativeFixture.Settings, cancellation.Token);
        await stream.Entered.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        Assert.False(pending.IsCompleted);
        if (deadline)
        {
            clock.Advance(TimeSpan.FromSeconds(5) - TimeSpan.FromTicks(1));
            Assert.False(cancellation.IsCancellationRequested);
            Assert.False(stream.Canceled.Task.IsCompleted);
            clock.Advance(TimeSpan.FromTicks(1));
        }
        else cancellation.Cancel();

        var error = await Assert.ThrowsAnyAsync<OperationCanceledException>(() => pending);
        Assert.Equal(cancellation.Token, error.CancellationToken);
        Assert.True(stream.Canceled.Task.IsCompletedSuccessfully);
        Assert.True(stream.Disposed);
        Assert.True(content.Disposed);
        Assert.True(response.Disposed);
        Assert.Equal(1, handler.Sends);
        clock.Advance(TimeSpan.FromMinutes(1));
        Assert.Equal(1, handler.Sends);
        Assert.Equal(0, handler.Disposals);

        var subsequent = await InvokeAsync(http, models, NativeFixture.Settings, TestContext.Current.CancellationToken);
        AssertPayload(subsequent, models);
        Assert.Equal(2, handler.Sends);
        Assert.Equal(0, handler.Disposals);
    }

    private static async Task<JsonElement> InvokeAsync(HttpClient http, bool models, TypeSafeClientOptions settings,
        CancellationToken cancellationToken)
    {
        if (models)
            return await new TypeSafeModelClient(http, settings, NativeFixture.Key).GetModelsAsync(cancellationToken);
        using var client = new TypeSafeChatClient(http, settings, NativeFixture.Key);
        var response = await client.GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), cancellationToken);
        NativeFixture.AssertChat(response);
        return Assert.Single(response.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>()).Json;
    }

    private static void AssertPayload(JsonElement result, bool models)
    {
        if (models)
        {
            Assert.Equal("jev", result.GetProperty("models")[0].GetString());
            Assert.Equal(17, result.GetProperty("vendor").GetProperty("revision").GetInt32());
        }
        else
        {
            Assert.Equal("jev-1.13.0", result.GetProperty("model").GetString());
            Assert.Equal(0.82, result.GetProperty("answers").GetProperty("follow-up").GetProperty("noul").GetDouble());
        }
    }
}
