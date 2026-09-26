using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.AI;

#pragma warning disable MEAI001 // The pinned RoutingChatClient experiment is explicitly approved for this proof.

namespace TypeSafe.MEAI.Sample;

internal static class CompositionScenarios
{
    internal static async Task RouteAsync(SampleEnvironment environment)
    {
        var billing = new DestinationClient("billing");
        var technical = new DestinationClient("technical");
        var general = new DestinationClient("general");
        var routes = new Dictionary<string, IChatClient> { ["billing"] = billing, ["technical"] = technical, ["general"] = general };
        var question = new ChoiceQuestion("route", TypeSafeJson.Entry("Choose the support destination."),
            new Dictionary<string, JsonElement>
            {
                ["billing"] = TypeSafeJson.Entry("Payment problems"), ["technical"] = TypeSafeJson.Entry("Technical problems"),
                ["general"] = TypeSafeJson.Entry("Other questions")
            });
        RoutingChatClient Router(double threshold) => RoutingChatClient.Create(async (context, token) =>
        {
            var classified = await environment.Client.GetResponseAsync(context.Messages, new ChatOptions
            {
                ModelId = "jev-1.13.0",
                AdditionalProperties = new() { [TypeSafeChatClient.QuestionsKey] = JsonSerializer.SerializeToElement(new Dictionary<string, JsonElement> { ["route"] = question.ToJson() }) }
            }, token);
            var evidence = classified.Messages.Last().Contents.OfType<JevResultContent>().Single();
            var answer = (ChoiceAnswer)evidence.Answers["route"];
            bool fallback = answer.Confidence < threshold;
            Console.WriteLine($"Route={answer.Choice}, explicit confidence fallback={fallback}; {evidence.Json}");
            return fallback ? general : routes[answer.Choice];
        });
        using var router = Router(0.8);
        var options = new ChatOptions { ModelId = "scripted-downstream", Temperature = 0.25f };
        // Materialize once before the router: both classification and downstream consume the same history.
        ChatMessage[] history = [new(ChatRole.User, "Synthetic support: payment failed.")];
        var response = await router.GetResponseAsync(history, options);
        if (billing.Calls != 1 || technical.Calls != 0 || general.Calls != 0 || response.ModelId != "scripted-billing" ||
            response.Usage?.TotalTokenCount != 7 || options.ModelId != "scripted-downstream")
            throw new InvalidOperationException("Routing delegation proof failed.");
        Console.WriteLine(response.Text);
        using var uncertain = Router(0.95);
        await uncertain.GetResponseAsync([new(ChatRole.User, "Synthetic support: ambiguous question.")], options);
        if (general.Calls != 1 || environment.Handler.Sends != 2) throw new InvalidOperationException("Explicit confidence fallback proof failed.");
    }

    internal static async Task SelectToolAsync(SampleEnvironment environment, bool dispatch)
    {
        int invocations = 0;
        var function = AIFunctionFactory.Create((Destination destination, bool followup) =>
        {
            invocations++;
            return new { previewOnly = true, destination, followup };
        }, new AIFunctionFactoryOptions
        {
            Name = "preview_queue_assignment",
            Description = "Preview (never persist) the support queue destination and follow-up flag."
        });
        var approvedFunction = new ApprovalRequiredAIFunction(function);
        var options = new ChatOptions { Tools = [approvedFunction], AllowMultipleToolCalls = false };
        if (!dispatch)
        {
            var selection = await environment.Client.GetResponseAsync(SampleEnvironment.State, options);
            var calls = selection.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>().ToArray();
            if (calls.Length != 1 || calls[0].Name != "preview_queue_assignment" || invocations != 0 || environment.Handler.Sends != 2 ||
                selection.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>().Count() != 2)
                throw new InvalidOperationException("Two-step selection-only proof failed.");
            Console.WriteLine(JsonSerializer.Serialize(calls[0], AIJsonUtilities.DefaultOptions));
            Console.WriteLine("Selection complete: two native batches, zero tool invocations. Thresholds are illustrative, not calibrated.");
            return;
        }
        string? trustedCallId = null;
        JsonElement trustedArguments = default;
        bool approved = false, consumed = false;
        using var executing = environment.Client.AsBuilder().UseFunctionInvocation(configure: invocation =>
        {
            invocation.MaximumConsecutiveErrorsPerRequest = 0;
            invocation.TerminateOnUnknownCalls = true;
            invocation.FunctionInvoker = async (context, token) =>
            {
                if (!approved || consumed || !ReferenceEquals(context.Function, approvedFunction) ||
                    context.CallContent.CallId != trustedCallId || context.CallContent.Name != approvedFunction.Name ||
                    !JsonElement.DeepEquals(JsonSerializer.SerializeToElement(context.Arguments), trustedArguments))
                    throw new UnauthorizedAccessException("The trusted one-shot approval does not match this invocation.");
                token.ThrowIfCancellationRequested();
                consumed = true;
                context.Terminate = true;
                return await context.Function.InvokeAsync(context.Arguments, token);
            };
        }).Build();
        var pending = await executing.GetResponseAsync(SampleEnvironment.State, options);
        var request = pending.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>().Single();
        var call = (FunctionCallContent)request.ToolCall;
        trustedCallId = call.CallId;
        trustedArguments = JsonSerializer.SerializeToElement(call.Arguments);
        if (invocations != 0 || environment.Handler.Sends != 2) throw new InvalidOperationException("Approval did not pause execution.");
        // Synthetic host approval for this harmless preview only. A real host persists/authenticates this grant.
        approved = true;
        // Denial or cancellation ends the host operation here: never send a denial back into a model loop.
        if (!approved) return;
        var result = await executing.GetResponseAsync([new(ChatRole.User, [request.CreateResponse(approved)])], options);
        if (invocations != 1 || !consumed ||
            result.Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>().Single().CallId != trustedCallId || environment.Handler.Sends != 2)
            throw new InvalidOperationException("Separate approved one-shot invocation proof failed.");
        Console.WriteLine("Standard ToolApprovalRequestContent/ToolApprovalResponseContent resumption completed.");
        Console.WriteLine("Host separately authorized and approved this harmless preview; invoked once, no further model turn.");
    }

    private sealed class DestinationClient(string name) : IChatClient
    {
        internal int Calls { get; private set; }
        public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (options?.ModelId != "scripted-downstream" || options.Temperature != 0.25f)
                throw new InvalidOperationException("Downstream options were changed.");
            Calls++;
            return Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, $"Scripted {name} reply to: {messages.Last().Text}"))
            {
                ModelId = $"scripted-{name}", Usage = new() { InputTokenCount = 4, OutputTokenCount = 3, TotalTokenCount = 7 }
            });
        }
        public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            foreach (var update in (await GetResponseAsync(messages, options, cancellationToken)).ToChatResponseUpdates()) yield return update;
        }
        public object? GetService(Type serviceType, object? serviceKey = null) => serviceKey is null && serviceType.IsInstanceOfType(this) ? this : null;
        public void Dispose() { }
    }
}
