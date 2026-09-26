using System.Text.Json;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.AI.Evaluation;
using Microsoft.Extensions.AI.Evaluation.Reporting.Storage;
using TypeSafe.MEAI.Evaluation;

namespace TypeSafe.MEAI.Sample;

internal static class IntegrationScenarios
{
    internal static async Task EvaluateAsync(SampleEnvironment environment, string directory)
    {
        var evaluator = new TypeSafeEvaluator(SampleEnvironment.Questions.Select(q => new TypeSafeMetricDefinition(q.Id, q)));
        var configuration = DiskBasedReportingConfiguration.Create(directory,
            [new CompositeEvaluator(evaluator, new DraftPresentEvaluator())],
            new ChatConfiguration(environment.Client), enableResponseCaching: false, executionName: "synthetic-support");
        await using (var scenario = await configuration.CreateScenarioRunAsync("fixed-draft"))
        {
            var result = await scenario.EvaluateAsync([new ChatMessage(ChatRole.User, "Synthetic: payment failed.")],
                new ChatResponse(new ChatMessage(ChatRole.Assistant, "Please check the payment details.")) { ModelId = "scripted-primary" },
                [new TypeSafeEvaluationContext("reference", TypeSafeJson.Entry(new { approved = "Check payment details" }))]);
            if (environment.Handler.Sends != 1 || result.Metrics.Count != 4 ||
                result.Metrics["quality"] is not NumericMetric { Value: 1.05 } ||
                result.Metrics["destination"] is not StringMetric { Value: "billing" })
                throw new InvalidOperationException("Batched evaluator proof failed.");
            Console.WriteLine(JsonSerializer.Serialize(result, AIJsonUtilities.DefaultOptions));
        }
        var primary = await environment.RootClient.GetResponseAsync<bool>("Does this synthetic ticket need follow-up?");
        string proofRun = Guid.NewGuid().ToString("N");
        primary.AdditionalProperties = new() { ["proof-run"] = proofRun };
        var portable = ReportSnapshot(primary);
        await using (var scenario = await configuration.CreateScenarioRunAsync("jev-primary"))
        {
            await scenario.EvaluateAsync([new ChatMessage(ChatRole.User, "Synthetic primary decision")], portable);
        }
        if (primary.Messages.Last().Contents.OfType<JevResultContent>().Count() != 1 ||
            portable.Messages.Last().Contents.Any(c => c is JevResultContent))
            throw new InvalidOperationException("Report ingress changed the live response or retained an unsupported subtype.");
        int readCount = 0;
        await foreach (var saved in configuration.ResultStore.ReadResultsAsync("synthetic-support", "jev-primary"))
        {
            readCount++;
            var data = saved.ModelResponse!.Messages.Last().Contents.OfType<DataContent>().Single();
            var restored = new JevResultContent(data.Data) { AdditionalProperties = data.AdditionalProperties, Annotations = data.Annotations };
            if (saved.ModelResponse.AdditionalProperties?["proof-run"]?.ToString() != proofRun || data is JevResultContent ||
                !JsonElement.DeepEquals(restored.Json, primary.Messages.Last().Contents.OfType<JevResultContent>().Single().Json))
                throw new InvalidOperationException("Persisted primary evidence did not round-trip as portable DataContent.");
            Console.WriteLine($"Read stored primary evidence without a question snapshot: {restored.Model}.");
        }
        if (readCount != 1) throw new InvalidOperationException("Expected exactly one newly persisted Jev-primary report.");
        var errorConfiguration = DiskBasedReportingConfiguration.Create(directory,
            [new CompositeEvaluator(new BrokenEvaluator(), new DraftPresentEvaluator())],
            enableResponseCaching: false, executionName: "synthetic-support");
        await using (var scenario = await errorConfiguration.CreateScenarioRunAsync("visible-child-failure"))
        {
            var result = await scenario.EvaluateAsync([new ChatMessage(ChatRole.User, "Synthetic prompt")],
                new ChatResponse(new ChatMessage(ChatRole.Assistant, "Fixed response")));
            if (!result.Metrics.Values.Any(m => m.Diagnostics?.Any(d => d.Severity == EvaluationDiagnosticSeverity.Error) == true))
                throw new InvalidOperationException("Composite evaluator failure was hidden.");
            Console.WriteLine("Composite child failure is persisted as error diagnostics, not a passed evaluation.");
        }
        Console.WriteLine($"Persisted existing MEAI report data at {directory}; reporting response cache is disabled.");
    }

    internal static async Task ToolAsync(SampleEnvironment environment)
    {
        var function = AIFunctionFactory.Create(async (string state, CancellationToken cancellationToken) =>
        {
            var native = await environment.Client.GetResponseAsync(state, SampleEnvironment.NativeOptions(), cancellationToken);
            return native.Messages.Last().Contents.OfType<JevResultContent>().Single().Json;
        }, "judge_ticket", "Judge a synthetic support ticket in one native batch.");
        var script = new ScriptedConversation(function.Name);
        using var conversation = script.AsBuilder().UseFunctionInvocation().Build();
        var response = await conversation.GetResponseAsync("Evaluate the synthetic support ticket.",
            new ChatOptions { Tools = [function] });
        if (script.Calls != 2 || environment.Handler.Sends != 1 || response.Text != "Scripted conversational response: evaluation received.")
            throw new InvalidOperationException("Function invocation middleware proof failed.");
        Console.WriteLine(response.Text);
        Console.WriteLine("Two scripted conversational turns, one native batch, real UseFunctionInvocation middleware.");
    }

    internal static ChatResponse ReportSnapshot(ChatResponse response)
    {
        AIContent Portable(AIContent content) => content is JevResultContent jev ? new DataContent(jev.Data.ToArray(), jev.MediaType)
        {
            AdditionalProperties = jev.AdditionalProperties, Annotations = jev.Annotations
        } : content;
        var snapshot = new ChatResponse(response.Messages.Select(message => new ChatMessage(message.Role, message.Contents.Select(Portable).ToArray())
        {
            AuthorName = message.AuthorName, MessageId = message.MessageId, CreatedAt = message.CreatedAt, AdditionalProperties = message.AdditionalProperties
        }).ToArray())
        {
            ResponseId = response.ResponseId, ConversationId = response.ConversationId, CreatedAt = response.CreatedAt,
            FinishReason = response.FinishReason, ModelId = response.ModelId, Usage = response.Usage,
            AdditionalProperties = response.AdditionalProperties, ContinuationToken = response.ContinuationToken
        };
        return JsonSerializer.Deserialize<ChatResponse>(JsonSerializer.Serialize(snapshot, AIJsonUtilities.DefaultOptions), AIJsonUtilities.DefaultOptions)!;
    }

    private sealed class ScriptedConversation(string functionName) : IChatClient
    {
        internal int Calls { get; private set; }
        public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Calls++;
            var result = messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>().LastOrDefault();
            if (result is null)
                return Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant,
                    [new FunctionCallContent("synthetic-call", functionName, new Dictionary<string, object?> { ["state"] = "Synthetic support: payouts failed." })])));
            var json = JsonSerializer.SerializeToElement(result.Result);
            if (result.Exception is not null || json.GetProperty("model").GetString() != "jev-1.13.0" || json.GetProperty("answers").EnumerateObject().Count() != 3)
                throw new InvalidOperationException("Full native tool result did not reach the conversational client.");
            return Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, "Scripted conversational response: evaluation received.")));
        }
        public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default) =>
            throw new NotSupportedException("This offline scripted interaction uses the nonstreaming middleware entry point.");
        public object? GetService(Type serviceType, object? serviceKey = null) => serviceKey is null && serviceType.IsInstanceOfType(this) ? this : null;
        public void Dispose() { }
    }

    private sealed class DraftPresentEvaluator : IEvaluator
    {
        public IReadOnlyCollection<string> EvaluationMetricNames => ["draft-present"];
        public ValueTask<EvaluationResult> EvaluateAsync(IEnumerable<ChatMessage> messages, ChatResponse modelResponse, ChatConfiguration? chatConfiguration = null,
            IEnumerable<EvaluationContext>? additionalContext = null, CancellationToken cancellationToken = default) =>
            ValueTask.FromResult(new EvaluationResult(new BooleanMetric("draft-present", !string.IsNullOrWhiteSpace(modelResponse.Text))));
    }

    private sealed class BrokenEvaluator : IEvaluator
    {
        public IReadOnlyCollection<string> EvaluationMetricNames => ["intentional-failure"];
        public ValueTask<EvaluationResult> EvaluateAsync(IEnumerable<ChatMessage> messages, ChatResponse modelResponse, ChatConfiguration? chatConfiguration = null,
            IEnumerable<EvaluationContext>? additionalContext = null, CancellationToken cancellationToken = default) =>
            throw new InvalidOperationException("Intentional synthetic evaluator failure.");
    }
}
