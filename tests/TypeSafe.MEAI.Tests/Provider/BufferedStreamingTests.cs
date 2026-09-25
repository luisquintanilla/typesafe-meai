using System.Net;
using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.Extensions.AI;
using TypeSafe.MEAI.Tests.Fixtures;
using Xunit;

namespace TypeSafe.MEAI.Tests.Provider;

[Trait("Stage", "A")]
public sealed class BufferedStreamingTests
{
    [Fact]
    public async Task GetStreamingResponse_ReconstructsNormalSemanticsAndUsageOnceWithMeaiAggregator()
    {
        using var handler = new RecordingHandler();
        using var services = NativeFixture.Services(handler);
        var client = services.Client();
        var normal = await client.GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), TestContext.Current.CancellationToken);
        var updates = new List<ChatResponseUpdate>();
        var reconstructed = await Observe(client.GetStreamingResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), TestContext.Current.CancellationToken),
            updates, TestContext.Current.CancellationToken).ToChatResponseAsync(TestContext.Current.CancellationToken);
        NativeFixture.AssertChat(normal);
        NativeFixture.AssertChat(reconstructed);
        Assert.Equal(normal.Text, reconstructed.Text);
        Assert.Equal(normal.ModelId, reconstructed.ModelId);
        Assert.Equal(normal.FinishReason, reconstructed.FinishReason);
        var usage = Assert.Single(updates.SelectMany(u => u.Contents).OfType<UsageContent>());
        Assert.Equal(11, usage.Details.InputTokenCount);
        Assert.Equal(7, usage.Details.OutputTokenCount);
        Assert.Equal(18, usage.Details.TotalTokenCount);
        Assert.Equal(2, handler.Sends);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task GetStreamingResponse_CancellationDuringEnumerationStopsUpdates(bool before)
    {
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        using var handler = new RecordingHandler();
        using var services = NativeFixture.Services(handler);
        if (before) cancellation.Cancel();
        await using var iterator = services.Client().GetStreamingResponseAsync(NativeFixture.Messages(), NativeFixture.Options(),
            cancellation.Token).GetAsyncEnumerator(cancellation.Token);
        if (!before)
        {
            Assert.True(await iterator.MoveNextAsync());
            Assert.Empty(iterator.Current.Contents.OfType<TextContent>());
            Assert.Equal(17, Assert.Single(iterator.Current.Contents.OfType<JevResultContent>()).Json
                .GetProperty("vendor_envelope").GetProperty("revision").GetInt32());
            cancellation.Cancel();
        }
        var error = await Assert.ThrowsAnyAsync<OperationCanceledException>(() => iterator.MoveNextAsync().AsTask());
        Assert.Equal(cancellation.Token, error.CancellationToken);
        Assert.Equal(before ? 0 : 1, handler.Sends);
    }

    [Theory]
    [InlineData("valid")]
    [InlineData("malformed")]
    [InlineData("oversize")]
    public async Task GetStreamingResponse_ValidatesFullBodyBeforePublishingUpdates(string kind)
    {
        var payload = kind switch { "malformed" => "{", "oversize" => new string('x', 1025), _ => NativeFixture.Envelope };
        var stream = new TrackedStream(Encoding.UTF8.GetBytes(payload), stall: true);
        var content = new TrackedContent(stream);
        using var handler = new RecordingHandler((_, _, _) => Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = content }));
        using var services = NativeFixture.Services(handler, settings: NativeFixture.Settings with { MaximumResponseBytes = 1024 });
        await using var iterator = services.Client().GetStreamingResponseAsync(NativeFixture.Messages(), NativeFixture.Options(),
            TestContext.Current.CancellationToken).GetAsyncEnumerator(TestContext.Current.CancellationToken);
        var first = iterator.MoveNextAsync().AsTask();
        await stream.Entered.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        Assert.False(first.IsCompleted);
        stream.Release.SetResult();
        if (kind == "valid")
        {
            Assert.True(await first);
            var all = new List<ChatResponseUpdate> { iterator.Current };
            while (await iterator.MoveNextAsync()) all.Add(iterator.Current);
            NativeFixture.AssertChat(all.ToChatResponse());
        }
        else if (kind == "malformed") await Assert.ThrowsAsync<TypeSafeProtocolException>(() => first);
        else await Assert.ThrowsAsync<TypeSafeBodyLimitException>(() => first);
        Assert.Equal(1, handler.Sends);
        Assert.True(content.Disposed);
        Assert.True(stream.Disposed);
    }

    private static async IAsyncEnumerable<ChatResponseUpdate> Observe(IAsyncEnumerable<ChatResponseUpdate> source,
        List<ChatResponseUpdate> captured, [EnumeratorCancellation] CancellationToken token)
    {
        await foreach (var update in source.WithCancellation(token))
        {
            captured.Add(update);
            yield return update;
        }
    }
}
