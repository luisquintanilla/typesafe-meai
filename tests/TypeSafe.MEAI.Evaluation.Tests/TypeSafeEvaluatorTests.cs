using System.Net;
using System.Text.Json;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.AI.Evaluation;
using TypeSafe.MEAI.Evaluation.Tests.Fixtures;
using TypeSafe.MEAI.Tests.Fixtures;
using Xunit;

namespace TypeSafe.MEAI.Evaluation.Tests;

[Trait("Stage", "B")]
public sealed class TypeSafeEvaluatorTests
{
    [Fact]
    public async Task Evaluate_BatchesMetricsThroughDecoratedClientAndPreservesStateAndEvidence()
    {
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler);
        using var spy = new EvaluationSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        IEvaluator evaluator = new TypeSafeEvaluator(EvaluationFixture.Definitions());
        var input = EvaluationFixture.Input();

        var result = await evaluator.EvaluateAsync(input.Messages, input.Response, new ChatConfiguration(spy), input.Contexts,
            TestContext.Current.CancellationToken);

        Assert.Equal(EvaluationFixture.MetricNames, evaluator.EvaluationMetricNames);
        Assert.Equal(EvaluationFixture.MetricNames.Order(), result.Metrics.Keys.Order());
        Assert.Equal(3, result.Metrics.Count);
        EvaluationFixture.AssertMetrics(result);
        Assert.All(result.Metrics.Values, metric => Assert.Null(metric.Interpretation));
        Assert.Equal(1, spy.Calls);
        Assert.Equal(0, spy.ServiceLookups);
        Assert.Equal(1, handler.Sends);
        Assert.Null(Assert.Single(spy.Options)!.ResponseFormat);
        EvaluationFixture.AssertWireState(Assert.Single(handler.Requests));
        Assert.Equal("scripted-primary-v1", input.Response.ModelId);
        Assert.Equal(120, input.Response.Usage!.TotalTokenCount);
        Assert.Equal("11", result.Metrics[EvaluationFixture.MetricNames[0]].Metadata!["eval-input-tokens"]);
    }

    [Theory]
    [InlineData(0.82, false)]
    [InlineData(0.820001, true)]
    public async Task Evaluate_UsesOnlyExplicitHostInterpretation(double threshold, bool failed)
    {
        var definitions = EvaluationFixture.Definitions();
        int interpretations = 0;
        definitions[0] = new(definitions[0].Name, definitions[0].Question, answer =>
        {
            interpretations++;
            var noul = Assert.IsType<NoulAnswer>(answer);
            Assert.Equal(0.82, noul.Noul);
            bool rejected = noul.Noul < threshold;
            return new(rejected ? EvaluationRating.Poor : EvaluationRating.Good, rejected, "Explicit synthetic host policy.");
        });
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler);
        using var spy = new EvaluationSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        var evaluator = new TypeSafeEvaluator(definitions);
        var input = EvaluationFixture.Input();

        var result = await evaluator.EvaluateAsync(input.Messages, input.Response, new(spy), input.Contexts, TestContext.Current.CancellationToken);

        EvaluationFixture.AssertMetrics(result);
        var interpretation = Assert.IsType<EvaluationMetricInterpretation>(result.Metrics[EvaluationFixture.MetricNames[0]].Interpretation);
        Assert.Equal(failed, interpretation.Failed);
        Assert.Equal(failed ? EvaluationRating.Poor : EvaluationRating.Good, interpretation.Rating);
        Assert.Equal("Explicit synthetic host policy.", interpretation.Reason);
        Assert.Null(result.Metrics[EvaluationFixture.MetricNames[1]].Interpretation);
        Assert.Null(result.Metrics[EvaluationFixture.MetricNames[2]].Interpretation);
        Assert.Equal(1, interpretations);
        Assert.Equal(1, spy.Calls);
        Assert.Equal(1, handler.Sends);
    }

    [Theory]
    [InlineData("empty")]
    [InlineData("duplicate-name")]
    [InlineData("duplicate-id")]
    public void Configuration_RejectsEmptyAndDuplicateDefinitions(string invalid)
    {
        var definitions = EvaluationFixture.Definitions();
        TypeSafeMetricDefinition[] configured = invalid switch
        {
            "empty" => [],
            "duplicate-name" => [definitions[0], new(definitions[0].Name, definitions[1].Question)],
            "duplicate-id" => [definitions[0], new("different-name", definitions[0].Question)],
            _ => throw new ArgumentOutOfRangeException(nameof(invalid))
        };
        var error = Assert.Throws<ArgumentException>(() => new TypeSafeEvaluator(configured));
        Assert.Contains("distinct", error.Message);
    }

    [Theory]
    [InlineData("")]
    [InlineData(" ")]
    [InlineData(null)]
    public void MetricDefinition_RejectsInvalidNameOrQuestion(string? invalidName)
    {
        var nameError = Assert.ThrowsAny<ArgumentException>(() => new TypeSafeMetricDefinition(invalidName!, NativeFixture.Questions()[0]));
        var questionError = Assert.Throws<ArgumentNullException>(() => new TypeSafeMetricDefinition("valid", null!));
        Assert.Equal("name", nameError.ParamName);
        Assert.Equal("question", questionError.ParamName);
    }

    [Fact]
    public async Task Evaluate_RequiresConfiguredJudgeBeforeSending()
    {
        var evaluator = new TypeSafeEvaluator(EvaluationFixture.Definitions());
        var input = EvaluationFixture.Input();
        var error = await Assert.ThrowsAsync<ArgumentException>(() =>
            evaluator.EvaluateAsync(input.Messages, input.Response, additionalContext: input.Contexts,
                cancellationToken: TestContext.Current.CancellationToken).AsTask());
        Assert.Equal("chatConfiguration", error.ParamName);
        Assert.Contains("decorated judge", error.Message);
    }

    [Theory]
    [InlineData("conversation")]
    [InlineData("response")]
    [InlineData("context")]
    [InlineData("context-scalar")]
    public async Task Evaluate_RejectsUnsupportedContentAndContextBeforeSending(string location)
    {
        var input = EvaluationFixture.Input();
        var unsupported = new DataContent(new byte[] { 17 }, "image/png");
        if (location == "conversation") input.Messages[1].Contents.Add(unsupported);
        if (location == "response") input.Response.Messages[0].Contents.Add(unsupported);
        if (location == "context") input.Contexts[0].Contents.Add(unsupported);
        if (location == "context-scalar") input.Contexts[0].Contents = [new DataContent("true"u8.ToArray(), "application/json")];
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler);
        using var spy = new EvaluationSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        var evaluator = new TypeSafeEvaluator(EvaluationFixture.Definitions());
        Task<EvaluationResult> Invoke() => evaluator.EvaluateAsync(input.Messages, input.Response, new(spy), input.Contexts,
            TestContext.Current.CancellationToken).AsTask();

        if (location == "context-scalar") await Assert.ThrowsAsync<ArgumentException>(Invoke);
        else await Assert.ThrowsAsync<NotSupportedException>(Invoke);
        Assert.Equal(0, spy.Calls);
        Assert.Equal(0, handler.Sends);
    }

    [Theory]
    [InlineData("http")]
    [InlineData("protocol")]
    public async Task Evaluate_PropagatesServiceAndProtocolErrors(string failure)
    {
        using var handler = new RecordingHandler((_, _, _) => Task.FromResult(failure == "http"
            ? NativeFixture.Response("synthetic-evaluation-error", 422) : NativeFixture.Response("{}")));
        using var http = new HttpClient(handler);
        using var spy = new EvaluationSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        var input = EvaluationFixture.Input();
        var evaluator = new TypeSafeEvaluator(EvaluationFixture.Definitions());
        Task<EvaluationResult> Invoke() => evaluator.EvaluateAsync(input.Messages, input.Response, new(spy), input.Contexts,
            TestContext.Current.CancellationToken).AsTask();
        if (failure == "http")
        {
            var error = await Assert.ThrowsAsync<TypeSafeApiException>(Invoke);
            Assert.Equal(HttpStatusCode.UnprocessableEntity, error.StatusCode);
        }
        else await Assert.ThrowsAsync<TypeSafeProtocolException>(Invoke);
        Assert.Equal(1, spy.Calls);
        Assert.Equal(1, handler.Sends);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Evaluate_PropagatesCancellationBeforeOrDuringSend(bool before)
    {
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var handler = new RecordingHandler(async (_, _, token) =>
        {
            entered.SetResult();
            await Task.Delay(Timeout.InfiniteTimeSpan, token);
            return NativeFixture.Response();
        });
        using var http = new HttpClient(handler);
        using var spy = new EvaluationSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        var input = EvaluationFixture.Input();
        var evaluator = new TypeSafeEvaluator(EvaluationFixture.Definitions());
        if (before) cancellation.Cancel();
        var pending = evaluator.EvaluateAsync(input.Messages, input.Response, new(spy), input.Contexts, cancellation.Token).AsTask();
        if (!before)
        {
            await entered.Task.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
            cancellation.Cancel();
        }
        var error = await Assert.ThrowsAnyAsync<OperationCanceledException>(() => pending);
        Assert.Equal(cancellation.Token, error.CancellationToken);
        Assert.Equal(before ? 0 : 1, spy.Calls);
        Assert.Equal(before ? 0 : 1, handler.Sends);
    }

    [Fact]
    public async Task Evaluate_ConcurrentCallsOwnInputSnapshotsAndSeparateBatchIds()
    {
        var definitions = EvaluationFixture.Definitions().ToList();
        var evaluator = new TypeSafeEvaluator(definitions);
        definitions.Clear();
        var inputs = new[] { EvaluationFixture.Input(17), EvaluationFixture.Input(18) };
        var release = new[] { new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously), new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously) };
        var entered = new[] { new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously), new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously) };
        using var handler = new RecordingHandler(async (request, _, token) =>
        {
            var root = NativeFixture.Json(await request.Content!.ReadAsStringAsync(token));
            int ticket = root.GetProperty("state").GetProperty("context")[0].GetProperty("contents")[0]
                .GetProperty("value").GetProperty("ticket").GetInt32();
            entered[ticket - 17].SetResult();
            await release[ticket - 17].Task.WaitAsync(token);
            return NativeFixture.Response(ticket == 17 ? NativeFixture.Envelope : NativeFixture.Envelope.Replace("0.82", "0.27", StringComparison.Ordinal));
        });
        using var http = new HttpClient(handler);
        using var spy = new EvaluationSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        var tasks = inputs.Select(input => evaluator.EvaluateAsync(input.Messages, input.Response, new(spy), input.Contexts,
            TestContext.Current.CancellationToken).AsTask()).ToArray();
        await Task.WhenAll(entered.Select(e => e.Task)).WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        foreach (var input in inputs)
        {
            input.Messages[1].Contents.Clear();
            Array.Fill(input.StateBytes, (byte)' ');
            input.Response.Messages[0].Contents.Clear();
            input.Response.ModelId = "caller-changed";
            input.Contexts[0].Name = "caller-changed";
            input.Contexts[0].Contents.Clear();
        }
        release[1].SetResult();
        var second = await tasks[1];
        release[0].SetResult();
        var first = await tasks[0];
        EvaluationFixture.AssertMetrics(first);
        EvaluationFixture.AssertMetrics(second, ticket: 18, noul: 0.27);
        Assert.NotEqual(first.Metrics[EvaluationFixture.MetricNames[0]].Metadata!["typesafe.v1.batch"],
            second.Metrics[EvaluationFixture.MetricNames[0]].Metadata!["typesafe.v1.batch"]);
        Assert.Equal(EvaluationFixture.MetricNames, evaluator.EvaluationMetricNames);
        Assert.Equal(2, spy.Calls);
        Assert.Equal(2, handler.Sends);
        var requests = handler.Requests.ToArray();
        EvaluationFixture.AssertWireState(requests[0], 17);
        EvaluationFixture.AssertWireState(requests[1], 18);
        Assert.All(inputs, input => Assert.Equal("caller-changed", input.Contexts[0].Name));
    }

    [Fact]
    public async Task Evaluate_RejectsDuplicateContextNamesBeforeSending()
    {
        var input = EvaluationFixture.Input();
        input.Contexts[1].Name = input.Contexts[0].Name;
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler);
        using var spy = new EvaluationSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        var evaluator = new TypeSafeEvaluator(EvaluationFixture.Definitions());

        var error = await Assert.ThrowsAsync<ArgumentException>(() => evaluator.EvaluateAsync(input.Messages, input.Response,
            new(spy), input.Contexts, TestContext.Current.CancellationToken).AsTask());

        Assert.Contains("context names must be distinct", error.Message);
        Assert.Equal(0, spy.Calls);
        Assert.Equal(0, handler.Sends);
    }
}
