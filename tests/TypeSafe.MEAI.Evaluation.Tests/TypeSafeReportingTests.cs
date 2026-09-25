using Microsoft.Extensions.AI;
using System.Text.Json;
using Microsoft.Extensions.AI.Evaluation;
using Microsoft.Extensions.AI.Evaluation.Reporting;
using Microsoft.Extensions.AI.Evaluation.Reporting.Storage;
using TypeSafe.MEAI.Evaluation.Tests.Fixtures;
using TypeSafe.MEAI.Tests.Fixtures;
using Xunit;

namespace TypeSafe.MEAI.Evaluation.Tests;

[Trait("Stage", "B")]
public sealed class TypeSafeReportingTests
{
    [Fact]
    public async Task Reporting_JevPrimaryResponseUsesLosslessPortableHostSnapshotWithoutMutatingLiveContent()
    {
        var directory = Directory.CreateTempSubdirectory("TypeSafe.MEAI.Revision-");
        try
        {
            using var handler = new RecordingHandler();
            using var services = NativeFixture.Services(handler);
            var live = await services.Client().GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), TestContext.Current.CancellationToken);
            var original = Assert.IsType<JevResultContent>(Assert.Single(Assert.Single(live.Messages).Contents));
            original.AdditionalProperties!["host-report"] = NativeFixture.Json("""{"sequence":17,"scope":"synthetic"}""");
            original.Annotations = [new CitationAnnotation { Title = "Synthetic evidence", FileId = "fixture-17", Snippet = "lossless" }];
            byte[] expected = original.Data.ToArray();
            Assert.Throws<NotSupportedException>(() => JsonSerializer.Serialize(live, AIJsonUtilities.DefaultOptions));
            var snapshot = PortableReportSnapshot(live);
            var plain = Assert.IsType<DataContent>(Assert.Single(Assert.Single(snapshot.Messages).Contents));
            Assert.Equal(expected, plain.Data.ToArray());
            Assert.NotSame(live, snapshot);
            Assert.Same(original, Assert.Single(live.Messages[0].Contents));
            var configuration = DiskBasedReportingConfiguration.Create(directory.FullName, [new DraftPresentEvaluator()],
                enableResponseCaching: false, executionName: "revision-jev-primary");
            await using (var scenario = await configuration.CreateScenarioRunAsync("portable-primary",
                iterationName: "snapshot-17", cancellationToken: TestContext.Current.CancellationToken))
            {
                var metrics = await scenario.EvaluateAsync(NativeFixture.Messages(), snapshot, cancellationToken: TestContext.Current.CancellationToken);
                Assert.False(Assert.IsType<BooleanMetric>(metrics.Metrics["draft-present"]).Value);
            }
            var saved = Assert.Single(await ReadAsync(new DiskBasedResultStore(directory.FullName), "revision-jev-primary", "portable-primary"));
            var restored = Assert.IsType<DataContent>(Assert.Single(Assert.Single(saved.ModelResponse.Messages).Contents));
            Assert.Equal(JevResultContent.ContentMediaType, restored.MediaType);
            Assert.Equal(expected, restored.Data.ToArray());
            Assert.Equal(17, Assert.IsType<JsonElement>(restored.AdditionalProperties!["host-report"]).GetProperty("sequence").GetInt32());
            Assert.Equal("native", Assert.IsType<JsonElement>(restored.AdditionalProperties["typesafe.v1.batch-role"]).GetString());
            var annotation = Assert.IsType<CitationAnnotation>(Assert.Single(restored.Annotations!));
            Assert.Equal("Synthetic evidence", annotation.Title);
            Assert.Equal("fixture-17", annotation.FileId);
            Assert.Equal("lossless", annotation.Snippet);
            NativeFixture.AssertNative(new JevResultContent(restored.Data));
            Assert.Equal("", saved.ModelResponse.Text);
            Assert.Equal("jev-1.13.0", saved.ModelResponse.ModelId);
            Assert.Equal(ChatFinishReason.Stop, saved.ModelResponse.FinishReason);
            Assert.Equal(11, saved.ModelResponse.Usage!.InputTokenCount);
            Assert.Equal(7, saved.ModelResponse.Usage.OutputTokenCount);
            Assert.Equal(18, saved.ModelResponse.Usage.TotalTokenCount);
            Assert.Equal("snapshot-17", saved.IterationName);
            Assert.Same(original, live.Messages[0].Contents[0]);
            Assert.Equal(expected, original.Data.ToArray());
            Assert.Equal("Synthetic evidence", Assert.IsType<CitationAnnotation>(Assert.Single(original.Annotations!)).Title);
            Assert.Equal(1, handler.Sends);
        }
        finally { directory.Delete(recursive: true); }
    }

    // Local reporting-boundary adaptation; no new public writer, converter or API.
    private static ChatResponse PortableReportSnapshot(ChatResponse response)
    {
        var portable = new ChatResponse(response.Messages.Select(message =>
            new ChatMessage(message.Role, message.Contents.Select(content => content is JevResultContent evidence ?
                new DataContent(evidence.Data.ToArray(), evidence.MediaType)
                { AdditionalProperties = evidence.AdditionalProperties, Annotations = evidence.Annotations } : content).ToArray())
            { AuthorName = message.AuthorName, AdditionalProperties = message.AdditionalProperties }).ToArray())
        {
            ModelId = response.ModelId, ResponseId = response.ResponseId, CreatedAt = response.CreatedAt,
            FinishReason = response.FinishReason, Usage = response.Usage, AdditionalProperties = response.AdditionalProperties
        };
        return JsonSerializer.Deserialize<ChatResponse>(JsonSerializer.Serialize(portable, AIJsonUtilities.DefaultOptions), AIJsonUtilities.DefaultOptions)!;
    }

    [Fact]
    public async Task Reporting_DisposalPersistsAndReloadsCompleteMetricsContextAndPrimaryResponse()
    {
        var directory = Directory.CreateTempSubdirectory("TypeSafe.MEAI.StageB-");
        try
        {
            using var handler = new RecordingHandler();
            using var http = new HttpClient(handler);
            using var spy = new EvaluationSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
            var deterministic = new DraftPresentEvaluator();
            var evaluator = new TypeSafeEvaluator(EvaluationFixture.Definitions());
            var configuration = DiskBasedReportingConfiguration.Create(directory.FullName,
                [new CompositeEvaluator(evaluator, deterministic)], new ChatConfiguration(spy),
                enableResponseCaching: false, executionName: "stage-b-synthetic", tags: ["offline"]);
            Assert.Null(configuration.ResponseCacheProvider);
            var input = EvaluationFixture.Input();
            EvaluationResult live;

            await using (var scenario = await configuration.CreateScenarioRunAsync("fixed-draft", iterationName: "iteration-17",
                cancellationToken: TestContext.Current.CancellationToken))
            {
                live = await scenario.EvaluateAsync(input.Messages, input.Response, input.Contexts, TestContext.Current.CancellationToken);
                Assert.Equal(4, live.Metrics.Count);
                EvaluationFixture.AssertMetrics(live);
                Assert.True(Assert.IsType<BooleanMetric>(live.Metrics["draft-present"]).Value);
                Assert.Empty(await ReadAsync(new DiskBasedResultStore(directory.FullName), "stage-b-synthetic", "fixed-draft"));
            }

            var restored = Assert.Single(await ReadAsync(new DiskBasedResultStore(directory.FullName), "stage-b-synthetic", "fixed-draft"));
            Assert.Equal("stage-b-synthetic", restored.ExecutionName);
            Assert.Equal("fixed-draft", restored.ScenarioName);
            Assert.Equal("iteration-17", restored.IterationName);
            Assert.Contains("offline", restored.Tags!);
            Assert.NotSame(live, restored.EvaluationResult);
            Assert.Equal(4, restored.EvaluationResult.Metrics.Count);
            EvaluationFixture.AssertMetrics(restored.EvaluationResult);
            Assert.True(Assert.IsType<BooleanMetric>(restored.EvaluationResult.Metrics["draft-present"]).Value);
            foreach (var name in EvaluationFixture.MetricNames)
            {
                var metric = restored.EvaluationResult.Metrics[name];
                Assert.Null(metric.Interpretation);
                Assert.Equal(live.Metrics[name].Metadata!.OrderBy(p => p.Key), metric.Metadata!.OrderBy(p => p.Key));
            }
            EvaluationFixture.AssertConversation(restored.Messages.ToArray());
            Assert.Equal("scripted-primary-v1", restored.ModelResponse.ModelId);
            Assert.Equal("Check the payment details.", restored.ModelResponse.Text);
            Assert.Equal(ChatRole.Assistant, Assert.Single(restored.ModelResponse.Messages).Role);
            Assert.Equal("primary", restored.ModelResponse.Messages[0].AuthorName);
            Assert.Equal(120, restored.ModelResponse.Usage!.TotalTokenCount);
            Assert.Equal(1, handler.Sends);
            Assert.Equal(1, spy.Calls);
            Assert.Equal(1, deterministic.Calls);
            EvaluationFixture.AssertWireState(Assert.Single(handler.Requests));
        }
        finally
        {
            directory.Delete(recursive: true);
        }
    }

    [Fact]
    public async Task Composite_FailingChildPersistsErrorDiagnosticAndCannotPass()
    {
        var directory = Directory.CreateTempSubdirectory("TypeSafe.MEAI.StageB-");
        try
        {
            var deterministic = new DraftPresentEvaluator();
            var broken = new FailingEvaluator();
            var composite = new CompositeEvaluator(deterministic, broken);
            Assert.Equal(["draft-present", "failed-child"], composite.EvaluationMetricNames.Order());
            var configuration = DiskBasedReportingConfiguration.Create(directory.FullName, [composite],
                enableResponseCaching: false, executionName: "stage-b-failure");
            Assert.Null(configuration.ResponseCacheProvider);
            var input = EvaluationFixture.Input();

            await using (var scenario = await configuration.CreateScenarioRunAsync("visible-error",
                cancellationToken: TestContext.Current.CancellationToken))
            {
                var result = await scenario.EvaluateAsync(input.Messages, input.Response, cancellationToken: TestContext.Current.CancellationToken);
                AssertFailure(result);
            }

            var restored = Assert.Single(await ReadAsync(new DiskBasedResultStore(directory.FullName), "stage-b-failure", "visible-error"));
            AssertFailure(restored.EvaluationResult);
            Assert.Equal("visible-error", restored.ScenarioName);
            Assert.Equal("scripted-primary-v1", restored.ModelResponse.ModelId);
            Assert.Equal(1, deterministic.Calls);
            Assert.Equal(1, broken.Calls);
        }
        finally
        {
            directory.Delete(recursive: true);
        }
    }

    private static void AssertFailure(EvaluationResult result)
    {
        Assert.Equal(2, result.Metrics.Count);
        Assert.True(Assert.IsType<BooleanMetric>(result.Metrics["draft-present"]).Value);
        var failed = result.Metrics["failed-child"];
        Assert.Contains(failed.Diagnostics!, diagnostic => diagnostic.Severity == EvaluationDiagnosticSeverity.Error
            && diagnostic.Message.Contains("intentional synthetic child failure", StringComparison.Ordinal));
        Assert.False(failed.Interpretation is { Failed: false });
        Assert.NotEqual(EvaluationRating.Good, failed.Interpretation?.Rating);
        Assert.NotEqual(EvaluationRating.Exceptional, failed.Interpretation?.Rating);
    }

    private static async Task<List<ScenarioRunResult>> ReadAsync(IEvaluationResultStore store, string execution, string scenario)
    {
        List<ScenarioRunResult> results = [];
        await foreach (var result in store.ReadResultsAsync(execution, scenario, cancellationToken: TestContext.Current.CancellationToken))
            results.Add(result);
        return results;
    }

    private sealed class DraftPresentEvaluator : IEvaluator
    {
        private int _calls;
        internal int Calls => Volatile.Read(ref _calls);
        public IReadOnlyCollection<string> EvaluationMetricNames => ["draft-present"];
        public ValueTask<EvaluationResult> EvaluateAsync(IEnumerable<ChatMessage> messages, ChatResponse modelResponse,
            ChatConfiguration? chatConfiguration = null, IEnumerable<EvaluationContext>? additionalContext = null,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Interlocked.Increment(ref _calls);
            return ValueTask.FromResult(new EvaluationResult(new BooleanMetric("draft-present",
                modelResponse.Text.Contains("payment", StringComparison.Ordinal))));
        }
    }

    private sealed class FailingEvaluator : IEvaluator
    {
        private int _calls;
        internal int Calls => Volatile.Read(ref _calls);
        public IReadOnlyCollection<string> EvaluationMetricNames => ["failed-child"];
        public ValueTask<EvaluationResult> EvaluateAsync(IEnumerable<ChatMessage> messages, ChatResponse modelResponse,
            ChatConfiguration? chatConfiguration = null, IEnumerable<EvaluationContext>? additionalContext = null,
            CancellationToken cancellationToken = default)
        {
            Interlocked.Increment(ref _calls);
            throw new InvalidOperationException("intentional synthetic child failure");
        }
    }
}
