using System.Net;
using System.Text.Json;
using Microsoft.Extensions.AI;
using TypeSafe.MEAI.Tests.Fixtures;
using Xunit;

namespace TypeSafe.MEAI.Tests.Tools;

[Trait("Stage", "B")]
public sealed class TypeSafeFunctionFactoryTests
{
    [Fact]
    public async Task Create_ExposesStateOnlySchemaAndRetainsHostConfigurationAndQuestionSnapshot()
    {
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler);
        using var spy = new NativeSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        var questions = NativeFixture.Questions().ToList();
        var function = TypeSafeFunctionFactory.Create(spy, questions, "judge_ticket", "Apply the host-approved synthetic rubric.");
        questions.Clear();

        Assert.Equal("judge_ticket", function.Name);
        Assert.Equal("Apply the host-approved synthetic rubric.", function.Description);
        var schema = function.JsonSchema;
        Assert.Equal("object", schema.GetProperty("type").GetString());
        Assert.Equal(["state"], schema.GetProperty("properties").EnumerateObject().Select(p => p.Name));
        Assert.Equal("string", schema.GetProperty("properties").GetProperty("state").GetProperty("type").GetString());
        Assert.Equal(["state"], schema.GetProperty("required").EnumerateArray().Select(v => v.GetString()));
        foreach (var forbidden in new[] { "apiKey", "credentials", "endpoint", "model", "questions", "rubrics", "cancellationToken" })
            Assert.False(schema.GetProperty("properties").TryGetProperty(forbidden, out _));

        var result = Assert.IsType<JsonElement>(await function.InvokeAsync(new AIFunctionArguments
        {
            ["state"] = """{"ticket":17}"""
        }, TestContext.Current.CancellationToken));

        Assert.True(JsonElement.DeepEquals(NativeFixture.Json(NativeFixture.Envelope), result));
        NativeFixture.AssertNative(TypeSafeResponse.Parse(result.GetRawText(), NativeFixture.Questions()));
        var request = Assert.Single(handler.Requests);
        Assert.Equal("Bearer", request.Scheme);
        Assert.Equal(NativeFixture.Key, request.Key);
        Assert.Equal("https://api.typesafe.ai/v1/systemone", request.Uri.AbsoluteUri);
        var body = NativeFixture.Json(request.Body!);
        Assert.Equal(JsonValueKind.String, body.GetProperty("state").ValueKind);
        Assert.Equal("""{"ticket":17}""", body.GetProperty("state").GetString());
        Assert.Equal("jev-1.13.0", body.GetProperty("model").GetString());
        Assert.Equal(3, body.GetProperty("questions").EnumerateObject().Count());
        foreach (var question in NativeFixture.Questions())
            Assert.True(JsonElement.DeepEquals(question.ToJson(), body.GetProperty("questions").GetProperty(question.Id)));
        Assert.Equal(1, handler.Sends);
        Assert.Equal(1, spy.Calls);
        Assert.Equal(0, spy.ServiceLookups);
        Assert.Empty(questions);
    }

    [Fact]
    public async Task FunctionInvocation_TwoTurnsReturnFullNativeEnvelopeThroughDecoratedClient()
    {
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler);
        using var spy = new NativeSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        var function = TypeSafeFunctionFactory.Create(spy, NativeFixture.Questions());
        using var script = new ScriptedConversation(function.Name);
        using var conversation = script.AsBuilder().UseFunctionInvocation().Build();
        ChatMessage[] messages = [new(ChatRole.System, "Use the approved judge."), new(ChatRole.User, "Evaluate this synthetic ticket.")];

        var response = await conversation.GetResponseAsync(messages, new ChatOptions { Tools = [function] }, TestContext.Current.CancellationToken);

        Assert.Equal("evaluation complete", response.Text);
        Assert.Equal("scripted-conversation", response.ModelId);
        Assert.Equal(2, script.Calls);
        Assert.Equal(1, handler.Sends);
        Assert.Equal(1, spy.Calls);
        Assert.Equal(0, spy.ServiceLookups);
        var result = Assert.IsType<FunctionResultContent>(script.Result);
        Assert.Equal("call-17", result.CallId);
        Assert.Null(result.Exception);
        var native = Assert.IsType<JsonElement>(result.Result);
        Assert.Equal(JsonValueKind.Object, native.ValueKind);
        Assert.True(JsonElement.DeepEquals(NativeFixture.Json(NativeFixture.Envelope), native));
        NativeFixture.AssertNative(TypeSafeResponse.Parse(native.GetRawText(), NativeFixture.Questions()));
        Assert.Equal([ChatRole.System, ChatRole.User], script.Turns[0].Select(m => m.Role));
        Assert.Equal("Evaluate this synthetic ticket.", script.Turns[0][1].Text);
        var second = script.Turns[1];
        Assert.Equal([ChatRole.System, ChatRole.User, ChatRole.Assistant, ChatRole.Tool], second.Select(m => m.Role));
        Assert.Equal(function.Name, Assert.Single(second.SelectMany(m => m.Contents).OfType<FunctionCallContent>()).Name);
        Assert.Same(result, Assert.Single(second.SelectMany(m => m.Contents).OfType<FunctionResultContent>()));
        var request = Assert.Single(handler.Requests);
        Assert.Equal(HttpMethod.Post, request.Method);
        Assert.Equal("Synthetic support: payment failed.", NativeFixture.Json(request.Body!).GetProperty("state").GetString());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task FunctionInvocation_CancellationDoesNotInvokeNativeJudge(bool before)
    {
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler);
        using var spy = new NativeSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        var function = TypeSafeFunctionFactory.Create(spy, NativeFixture.Questions());
        using var script = new ScriptedConversation(function.Name, before ? null : cancellation.Cancel);
        using var conversation = script.AsBuilder().UseFunctionInvocation().Build();
        if (before) cancellation.Cancel();

        var error = await Assert.ThrowsAnyAsync<OperationCanceledException>(() => conversation.GetResponseAsync(
            [new(ChatRole.User, "Evaluate this synthetic ticket.")], new ChatOptions { Tools = [function] }, cancellation.Token));

        Assert.Equal(cancellation.Token, error.CancellationToken);
        Assert.Equal(before ? 0 : 1, script.Calls);
        Assert.Equal(0, handler.Sends);
        Assert.Equal(0, spy.Calls);
        Assert.Null(script.Result);
    }

    [Fact]
    public async Task FunctionInvocation_ServiceFailureRemainsVisibleInsteadOfSuccess()
    {
        using var handler = new RecordingHandler((_, _, _) => Task.FromResult(NativeFixture.Response("SYNTHETIC-NATIVE-ERROR", 422)));
        using var http = new HttpClient(handler);
        using var spy = new NativeSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        var function = TypeSafeFunctionFactory.Create(spy, NativeFixture.Questions());
        using var script = new ScriptedConversation(function.Name);
        using var conversation = script.AsBuilder().UseFunctionInvocation().Build();

        var response = await conversation.GetResponseAsync([new(ChatRole.User, "Evaluate this synthetic ticket.")],
            new ChatOptions { Tools = [function] }, TestContext.Current.CancellationToken);

        Assert.Equal("evaluation failed", response.Text);
        var result = Assert.IsType<FunctionResultContent>(script.Result);
        var error = Assert.IsType<TypeSafeApiException>(result.Exception);
        Assert.Equal(HttpStatusCode.UnprocessableEntity, error.StatusCode);
        Assert.Equal("call-17", result.CallId);
        var payload = JsonSerializer.SerializeToElement(result.Result);
        Assert.False(payload.ValueKind == JsonValueKind.Object && payload.TryGetProperty("answers", out _));
        Assert.Equal(2, script.Calls);
        Assert.Equal(1, spy.Calls);
        Assert.Equal(1, handler.Sends);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Create_RejectsUnapprovedEmptyOrDuplicateQuestions(bool duplicate)
    {
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler);
        using var spy = new NativeSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        TypeSafeQuestion[] questions = duplicate ? [new NoulQuestion("same"), new NoulQuestion("same")] : [];
        Assert.Throws<ArgumentException>(() => TypeSafeFunctionFactory.Create(spy, questions));
        Assert.Equal(0, spy.Calls);
        Assert.Equal(0, handler.Sends);
    }

    private sealed class NativeSpy(IChatClient inner) : DelegatingChatClient(inner)
    {
        internal int Calls { get; private set; }
        internal int ServiceLookups { get; private set; }
        public override Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            Calls++;
            return InnerClient.GetResponseAsync(messages, options, cancellationToken);
        }
        public override object? GetService(Type serviceType, object? serviceKey = null)
        {
            ServiceLookups++;
            return base.GetService(serviceType, serviceKey);
        }
    }

    private sealed class ScriptedConversation(string functionName, Action? afterFirstTurn = null) : IChatClient
    {
        internal int Calls { get; private set; }
        internal List<ChatMessage[]> Turns { get; } = [];
        internal FunctionResultContent? Result { get; private set; }
        public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Calls++;
            var snapshot = messages.ToArray();
            Turns.Add(snapshot);
            if (Calls == 1)
            {
                var response = new ChatResponse(new ChatMessage(ChatRole.Assistant,
                [
                    new FunctionCallContent("call-17", functionName,
                        new Dictionary<string, object?> { ["state"] = "Synthetic support: payment failed." })
                ]));
                afterFirstTurn?.Invoke();
                return Task.FromResult(response);
            }
            if (Calls != 2) throw new InvalidOperationException("Unexpected additional scripted model turn.");
            Result = snapshot.SelectMany(m => m.Contents).OfType<FunctionResultContent>().Single();
            return Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant,
                Result.Exception is null ? "evaluation complete" : "evaluation failed")) { ModelId = "scripted-conversation" });
        }
        public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages,
            ChatOptions? options = null, CancellationToken cancellationToken = default) => throw new NotSupportedException();
        public object? GetService(Type serviceType, object? serviceKey = null) =>
            serviceKey is null && serviceType.IsInstanceOfType(this) ? this : null;
        public void Dispose() { }
    }
}
