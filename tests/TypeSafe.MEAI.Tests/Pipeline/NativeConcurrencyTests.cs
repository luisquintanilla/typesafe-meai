using System.Text.Json;
using Microsoft.Extensions.AI;
using TypeSafe.MEAI.Tests.Fixtures;
using Xunit;

namespace TypeSafe.MEAI.Tests.Pipeline;

[Trait("Stage", "A")]
public sealed class NativeConcurrencyTests
{
    [Fact]
    public async Task GetTypeSafeResponse_ThirtyTwoControlledCallsRemainIsolatedAndOwnCallerSnapshots()
    {
        const int count = 32;
        var entered = Enumerable.Range(0, count).Select(_ => new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously)).ToArray();
        var release = Enumerable.Range(0, count).Select(_ => new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously)).ToArray();
        var criteria = Enumerable.Range(0, count).ToDictionary(i => $"ticket-{i}", i => TypeSafeJson.Entry($"description-{i}"));
        var questions = new List<TypeSafeQuestion> { new ChoiceQuestion("route", TypeSafeJson.Entry("Select matching ticket"), criteria) };
        var sharedOptions = new ChatOptions { ModelId = "jev-1.13.0", AdditionalProperties = new() { ["metadata"] = "before" } };
        using var handler = new RecordingHandler(async (request, _, token) =>
        {
            var body = NativeFixture.Json(await request.Content!.ReadAsStringAsync(token));
            var ticket = body.GetProperty("state").GetProperty("ticket").GetInt32();
            entered[ticket].SetResult();
            await release[ticket].Task.WaitAsync(token);
            Assert.Equal("jev-1.13.0", body.GetProperty("model").GetString());
            Assert.Equal("description-0", body.GetProperty("questions").GetProperty("route").GetProperty("criteria").GetProperty("ticket-0").GetString());
            var probabilities = Enumerable.Range(0, count).ToDictionary(i => $"ticket-{i}", i => i == ticket ? 0.8 : 0.2 / 31);
            return NativeFixture.Response(JsonSerializer.Serialize(new
            {
                model = "jev-1.13.0",
                usage = new { input_tokens = 11, output_tokens = 7 },
                answers = new { route = new { type = "choice", choice = $"ticket-{ticket}", confidence = 0.91, probabilities } }
            }));
        });
        using var services = NativeFixture.Services(handler);
        var client = services.Client();
        var tasks = Enumerable.Range(0, count).Select(i => client.GetTypeSafeResponseAsync(
            TypeSafeState.Parse($$"""{"ticket":{{i}}}"""), questions, sharedOptions, TestContext.Current.CancellationToken)).ToArray();
        await Task.WhenAll(entered.Select(e => e.Task)).WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        Assert.Equal(count, handler.Sends);
        Assert.All(tasks, task => Assert.False(task.IsCompleted));
        Assert.Equal("jev-1.13.0", sharedOptions.ModelId);
        Assert.Equal("before", sharedOptions.AdditionalProperties["metadata"]);
        Assert.False(sharedOptions.AdditionalProperties.ContainsKey(NativeFixture.QuestionsKey));
        sharedOptions.ModelId = "caller-mutated-model";
        sharedOptions.AdditionalProperties["metadata"] = "caller-mutated-metadata";
        criteria["ticket-0"] = TypeSafeJson.Entry("caller-mutated-description");
        questions.Clear();
        for (int i = count - 1; i >= 0; i--)
        {
            release[i].SetResult();
            var answer = (await tasks[i]).GetAnswer<ChoiceAnswer>("route");
            Assert.Equal($"ticket-{i}", answer.Choice);
            Assert.Equal(0.91, answer.Confidence);
            Assert.Equal(0.8, answer.Probabilities.GetProperty($"ticket-{i}").GetDouble());
            Assert.Equal(0.2 / 31, answer.Probabilities.GetProperty($"ticket-{(i + 1) % count}").GetDouble());
            Assert.Equal(count, answer.Probabilities.EnumerateObject().Count());
        }
        Assert.Equal(count, handler.Sends);
        Assert.Equal(count, handler.Requests.Select(r => NativeFixture.Json(r.Body!).GetProperty("state").GetProperty("ticket").GetInt32()).Distinct().Count());
        Assert.Equal("caller-mutated-model", sharedOptions.ModelId);
        Assert.Equal("caller-mutated-metadata", sharedOptions.AdditionalProperties["metadata"]);
        Assert.Empty(questions);
    }

    [Fact]
    public async Task GetResponse_SnapshotsMutableMessagesAndJsonBytesBeforeInFlightMutation()
    {
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var handler = new RecordingHandler(async (request, _, token) =>
        {
            entered.SetResult();
            await release.Task.WaitAsync(token);
            var body = NativeFixture.Json(await request.Content!.ReadAsStringAsync(token));
            Assert.Equal(17, body.GetProperty("state").GetProperty("ticket").GetInt32());
            Assert.Equal("jev-1.13.0", body.GetProperty("model").GetString());
            Assert.Equal("Follow up?", body.GetProperty("questions").GetProperty("follow-up").GetProperty("instructions").GetString());
            return NativeFixture.Response();
        });
        using var services = NativeFixture.Services(handler, cache: true);
        byte[] data = """{"ticket":17}"""u8.ToArray();
        var message = new ChatMessage(ChatRole.User, [new DataContent(data, "application/json")]);
        var messages = new List<ChatMessage> { message };
        var options = NativeFixture.Options();
        var task = services.Client().GetResponseAsync(messages, options, TestContext.Current.CancellationToken);
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        Array.Fill(data, (byte)' ');
        message.Role = ChatRole.System;
        message.Contents.Clear();
        messages.Clear();
        options.ModelId = "changed";
        options.AdditionalProperties!.Clear();
        release.SetResult();
        NativeFixture.AssertChat(await task);
        var originalRequestAgain = await services.Client().GetResponseAsync(
            [TypeSafeState.Parse("""{"ticket":17}""").ToMessage()], NativeFixture.Options(), TestContext.Current.CancellationToken);
        NativeFixture.AssertChat(originalRequestAgain);
        Assert.Equal(1, handler.Sends);
        Assert.Empty(messages);
        Assert.Empty(options.AdditionalProperties);
    }

    [Fact]
    public async Task ConcurrentCacheMisses_PreserveAnswersWithoutSingleFlightClaim()
    {
        var both = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var handler = new RecordingHandler(async (_, attempt, token) =>
        {
            if (attempt == 2) both.SetResult();
            await release.Task.WaitAsync(token);
            return NativeFixture.Response();
        });
        using var services = NativeFixture.Services(handler, cache: true);
        var first = services.Client().GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), TestContext.Current.CancellationToken);
        var second = services.Client().GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), TestContext.Current.CancellationToken);
        await both.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        Assert.Equal(2, handler.Sends);
        release.SetResult();
        NativeFixture.AssertChat(await first);
        NativeFixture.AssertChat(await second);
        NativeFixture.AssertChat(await services.Client().GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), TestContext.Current.CancellationToken));
        Assert.Equal(2, handler.Sends);
    }
}
