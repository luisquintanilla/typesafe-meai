using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.AI;

namespace TypeSafe.MEAI.Sample;

internal static class CompositionScenarios
{
    internal static async Task RouteAsync(SampleEnvironment environment)
    {
        var billing = new DestinationClient("billing");
        var technical = new DestinationClient("technical");
        var general = new DestinationClient("general");
        TypeSafeRoute[] routes = [new("billing", "Payment problems", billing), new("technical", "Technical problems", technical), new("general", "Other questions", general)];
        ValueTask Observe(TypeSafeRouteDecision decision, CancellationToken _)
        {
            Console.WriteLine($"Route={decision.RouteId ?? "abstained"}, explicit fallback={decision.UsedFallback}; {decision.Evidence.Json}");
            return ValueTask.CompletedTask;
        }
        using var router = new TypeSafeRoutingChatClient(environment.Client, routes, 0.8, "jev-1.13.0", observe: Observe);
        var options = new ChatOptions { ModelId = "scripted-downstream", Temperature = 0.25f };
        var response = await router.GetResponseAsync("Synthetic support: payment failed.", options);
        if (billing.Calls != 1 || technical.Calls != 0 || general.Calls != 0 || response.ModelId != "scripted-billing" ||
            response.Usage?.TotalTokenCount != 7 || options.ModelId != "scripted-downstream")
            throw new InvalidOperationException("Routing delegation proof failed.");
        Console.WriteLine(response.Text);
        using var uncertain = new TypeSafeRoutingChatClient(environment.Client, routes, 0.95, "jev-1.13.0", fallback: general, observe: Observe);
        await uncertain.GetResponseAsync("Synthetic support: ambiguous question.", options);
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
            Description = "Preview (never persist) the support queue destination and follow-up flag.",
            JsonSchemaCreateOptions = new() { TransformOptions = new() { DisallowAdditionalProperties = true } }
        });
        var approvedFunction = new ApprovalRequiredAIFunction(function);
        var bindings = SchemaScenario.Bindings().Where(p => p.Key != "/quality").ToDictionary();
        var selector = new TypeSafeToolSelector(environment.Client, [new TypeSafeToolDefinition(approvedFunction, bindings)], 0.8);
        var selection = await selector.SelectAsync(SampleEnvironment.State);
        if (selection.Status != TypeSafeToolSelectionStatus.Accepted || invocations != 0 || environment.Handler.Sends != 2 ||
            selection.FunctionCall?.Name != "preview_queue_assignment" || selection.Evidence.Count != 2)
            throw new InvalidOperationException("Two-step selection-only proof failed.");
        foreach (var evidence in selection.Evidence) Console.WriteLine(evidence.Json);
        Console.WriteLine(JsonSerializer.Serialize(selection.FunctionCall, AIJsonUtilities.DefaultOptions));
        Console.WriteLine("Selection complete: two native batches, zero tool invocations. Thresholds are illustrative, not calibrated.");
        if (!dispatch) return;
        var result = await TypeSafeToolInvoker.InvokeAsync(selection,
            authorize: (_, _) => ValueTask.FromResult(true),
            approve: (_, _) => ValueTask.FromResult(true));
        if (invocations != 1 || result.CallId != selection.FunctionCall.CallId || environment.Handler.Sends != 2)
            throw new InvalidOperationException("Separate approved one-shot invocation proof failed.");
        Console.WriteLine(JsonSerializer.Serialize(result, AIJsonUtilities.DefaultOptions));
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
