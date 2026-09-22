using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using TypeSafe.MEAI.Tests.Fixtures;
using TypeSafe.MEAI.Tests.Schema;
using Xunit;

namespace TypeSafe.MEAI.Tests.Tools;

[Trait("Stage", "C")]
public sealed class TypeSafeToolInvokerTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Invoke_AuthorizedSelectionPreservesOriginalOuterWrappersApprovalAndHostContext(bool approvalRequired)
    {
        int invocations = 0, authorizations = 0, approvals = 0;
        List<string> sequence = [];
        var marker = new HostMarker("host-owned-service");
        using var services = new ServiceCollection().AddSingleton(marker).BuildServiceProvider();
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        var core = new ToolProbe(AIFunctionFactory.Create(
            (TestDestination destination, bool followup, IServiceProvider provider, CancellationToken token) =>
            {
                invocations++;
                sequence.Add("invoke");
                Assert.Equal(TestDestination.Billing, destination);
                Assert.True(followup);
                Assert.Same(services, provider);
                Assert.Same(marker, provider.GetRequiredService<HostMarker>());
                Assert.Equal(cancellation.Token, token);
                return new { destination = destination.ToString(), followup, host = marker.Value };
            }, ToolFixture.Options("preview")));
        AIFunction registered = core;
        ApprovalRequiredAIFunction? approval = null;
        if (approvalRequired) registered = approval = new ApprovalRequiredAIFunction(core);
        var middle = new ToolProbe(registered);
        var outer = new ToolProbe(middle);
        Assert.Same(approval, outer.GetService(typeof(ApprovalRequiredAIFunction)));
        using var handler = ToolFixture.Handler("preview");
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        var selection = await new TypeSafeToolSelector(spy, [ToolFixture.InferredDefinition(outer)], 0.8)
            .SelectAsync(SchemaFixture.State, cancellationToken: cancellation.Token);
        Assert.Equal(TypeSafeToolSelectionStatus.Accepted, selection.Status);
        Assert.Same(outer, selection.Function);
        Assert.Equal(0, invocations);
        Assert.Equal(0, outer.Calls);
        Assert.Equal(2, selection.Evidence.Count);
        Assert.Equal("preview", selection.Evidence[0].GetAnswer<ChoiceAnswer>("tool").Choice);
        Assert.Equal(0.9, selection.Evidence[0].GetAnswer<ChoiceAnswer>("tool").Confidence);
        Assert.Equal("Billing", selection.Evidence[1].GetAnswer<ChoiceAnswer>("q0000").Choice);
        Assert.Equal(0.9, selection.Evidence[1].GetAnswer<NoulAnswer>("q0001").Noul);
        Assert.All(selection.Evidence, evidence =>
        {
            Assert.Equal(18, evidence.TotalTokens);
            Assert.Equal(17, evidence.Json.GetProperty("vendor_envelope").GetProperty("revision").GetInt32());
        });
        Assert.Equal(["destination", "followup"], selection.FunctionCall!.Arguments!.Keys.Order());
        Dictionary<object, object?> context = new() { ["invocation-id"] = "synthetic-context-17" };

        var result = await TypeSafeToolInvoker.InvokeAsync(selection,
            authorize: (observed, token) =>
            {
                Assert.Same(selection, observed);
                Assert.Equal(cancellation.Token, token);
                Assert.Equal(0, outer.Calls);
                authorizations++;
                sequence.Add("authorize");
                return ValueTask.FromResult(true);
            },
            approve: (observed, token) =>
            {
                Assert.Same(selection, observed);
                Assert.Equal(cancellation.Token, token);
                Assert.Equal(1, authorizations);
                Assert.Equal(0, outer.Calls);
                approvals++;
                sequence.Add("approve");
                return ValueTask.FromResult(true);
            }, context, services, cancellation.Token);

        Assert.Equal(1, authorizations);
        Assert.Equal(approvalRequired ? 1 : 0, approvals);
        Assert.Equal(approvalRequired ? ["authorize", "approve", "invoke"] : ["authorize", "invoke"], sequence);
        Assert.Equal(1, invocations);
        Assert.Equal(1, outer.Calls);
        Assert.Equal(1, middle.Calls);
        Assert.Equal(1, core.Calls);
        foreach (var probe in new[] { outer, middle, core })
        {
            Assert.Same(context, probe.Arguments!.Context);
            Assert.Same(services, probe.Arguments.Services);
            Assert.Equal(cancellation.Token, probe.Token);
        }
        Assert.Equal(selection.FunctionCall.CallId, result.CallId);
        Assert.Null(result.Exception);
        var payload = Assert.IsType<JsonElement>(result.Result);
        Assert.Equal("Billing", payload.GetProperty("destination").GetString());
        Assert.True(payload.GetProperty("followup").GetBoolean());
        Assert.Equal("host-owned-service", payload.GetProperty("host").GetString());
        Assert.Equal(2, spy.Calls);
        Assert.Equal(2, handler.Sends);
    }

    [Theory]
    [InlineData("authorization")]
    [InlineData("missing-approval")]
    [InlineData("approval")]
    public async Task Invoke_DeniedAuthorizationOrMissingSeparateApprovalCannotBypassNestedMarker(string denied)
    {
        int invocations = 0, authorizations = 0, approvals = 0;
        var core = ToolFixture.InferredFunction(() => invocations++);
        var approval = new ApprovalRequiredAIFunction(core);
        var outer = new ToolProbe(new ToolProbe(approval));
        Assert.Same(approval, outer.GetService(typeof(ApprovalRequiredAIFunction)));
        using var handler = ToolFixture.Handler("preview");
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        var selection = await new TypeSafeToolSelector(spy, [ToolFixture.InferredDefinition(outer)], 0.8)
            .SelectAsync(SchemaFixture.State, cancellationToken: TestContext.Current.CancellationToken);
        Func<TypeSafeToolSelection, CancellationToken, ValueTask<bool>>? approve = denied == "missing-approval" ? null :
            (_, _) => { approvals++; return ValueTask.FromResult(false); };

        await Assert.ThrowsAsync<UnauthorizedAccessException>(() => TypeSafeToolInvoker.InvokeAsync(selection,
            (_, _) => { authorizations++; return ValueTask.FromResult(denied != "authorization"); },
            approve, cancellationToken: TestContext.Current.CancellationToken).AsTask());

        Assert.Equal(1, authorizations);
        Assert.Equal(denied == "approval" ? 1 : 0, approvals);
        Assert.Equal(0, invocations);
        Assert.Equal(0, outer.Calls);
        Assert.Equal(0, core.Calls);
        Assert.Equal(2, handler.Sends);
    }

    [Theory]
    [InlineData("no-match")]
    [InlineData("selection-abstain")]
    [InlineData("arguments-abstain")]
    public async Task Invoke_NonacceptedSelectionCannotReachAuthorizationOrFunction(string outcome)
    {
        int invocations = 0, authorizations = 0;
        var function = ToolFixture.InferredFunction(() => invocations++);
        using var handler = ToolFixture.Handler(outcome == "no-match" ? TypeSafeToolSelector.NoMatch : "preview",
            confidence: outcome == "selection-abstain" ? 0.7 : 0.9,
            argumentNoul: outcome == "arguments-abstain" ? 0.5 : 0.9);
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        var selection = await new TypeSafeToolSelector(spy, [ToolFixture.InferredDefinition(function)], 0.8)
            .SelectAsync(SchemaFixture.State, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(outcome == "no-match" ? TypeSafeToolSelectionStatus.NoMatch : TypeSafeToolSelectionStatus.Abstained, selection.Status);
        Assert.Null(selection.FunctionCall);

        await Assert.ThrowsAsync<InvalidOperationException>(() => TypeSafeToolInvoker.InvokeAsync(selection,
            (_, _) => { authorizations++; return ValueTask.FromResult(true); },
            cancellationToken: TestContext.Current.CancellationToken).AsTask());

        Assert.Equal(0, authorizations);
        Assert.Equal(0, invocations);
        Assert.Equal(0, function.Calls);
        Assert.Equal(outcome == "arguments-abstain" ? 2 : 1, handler.Sends);
    }

    [Theory]
    [InlineData("name")]
    [InlineData("description")]
    [InlineData("schema")]
    [InlineData("arguments")]
    [InlineData("informational")]
    [InlineData("validation")]
    public async Task Invoke_RechecksDeclarationArgumentsAndApplicationPolicyAfterAuthorization(string mutation)
    {
        int invocations = 0, authorizations = 0;
        bool allowed = true;
        var function = ToolFixture.InferredFunction(() => invocations++);
        using var handler = ToolFixture.Handler("preview");
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        var definition = ToolFixture.InferredDefinition(function, _ => allowed ? null : "Host policy changed.");
        var selection = await new TypeSafeToolSelector(spy, [definition], 0.8).SelectAsync(SchemaFixture.State,
            cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(TypeSafeToolSelectionStatus.Accepted, selection.Status);

        var error = await Assert.ThrowsAsync<InvalidOperationException>(() => TypeSafeToolInvoker.InvokeAsync(selection,
            (_, _) =>
            {
                authorizations++;
                switch (mutation)
                {
                    case "name": function.NameOverride = "unregistered"; break;
                    case "description": function.DescriptionOverride = "Changed task semantics."; break;
                    case "schema":
                        var schema = JsonNode.Parse(function.JsonSchema.GetRawText())!;
                        schema["title"] = "Changed declaration.";
                        function.SchemaOverride = JsonSerializer.SerializeToElement(schema);
                        break;
                    case "arguments": selection.FunctionCall!.Arguments!["followup"] = false; break;
                    case "informational": selection.FunctionCall!.InformationalOnly = true; break;
                    case "validation": allowed = false; break;
                }
                return ValueTask.FromResult(true);
            }, cancellationToken: TestContext.Current.CancellationToken).AsTask());

        Assert.Contains(mutation == "validation" ? "Host policy changed." : "changed", error.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(1, authorizations);
        Assert.Equal(0, function.Calls);
        Assert.Equal(0, invocations);
        Assert.Equal(2, handler.Sends);
    }

    [Theory]
    [InlineData("before")]
    [InlineData("authorization")]
    [InlineData("approval")]
    public async Task Invoke_CancellationAtAnyHostGatePreventsSideEffects(string phase)
    {
        int invocations = 0, authorizations = 0, approvals = 0;
        var function = ToolFixture.InferredFunction(() => invocations++);
        var outer = new ToolProbe(new ApprovalRequiredAIFunction(function));
        using var handler = ToolFixture.Handler("preview");
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        var selection = await new TypeSafeToolSelector(spy, [ToolFixture.InferredDefinition(outer)], 0.8)
            .SelectAsync(SchemaFixture.State, cancellationToken: TestContext.Current.CancellationToken);
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        if (phase == "before") cancellation.Cancel();

        var error = await Assert.ThrowsAnyAsync<OperationCanceledException>(() => TypeSafeToolInvoker.InvokeAsync(selection,
            (_, _) =>
            {
                authorizations++;
                if (phase == "authorization") cancellation.Cancel();
                return ValueTask.FromResult(true);
            },
            (_, _) =>
            {
                approvals++;
                if (phase == "approval") cancellation.Cancel();
                return ValueTask.FromResult(true);
            }, cancellationToken: cancellation.Token).AsTask());

        Assert.Equal(cancellation.Token, error.CancellationToken);
        Assert.Equal(phase == "before" ? 0 : 1, authorizations);
        Assert.Equal(phase == "approval" ? 1 : 0, approvals);
        Assert.Equal(0, invocations);
        Assert.Equal(0, outer.Calls);
        Assert.Equal(0, function.Calls);
        Assert.Equal(2, handler.Sends);
    }

    [Fact]
    public async Task Invoke_FunctionExceptionPropagatesOnceWithoutAnotherModelOrToolLoop()
    {
        int invocations = 0, authorizations = 0;
        var failure = new InvalidOperationException("Synthetic function execution failed.");
        object Fail(TestDestination destination, bool followup)
        {
            invocations++;
            Assert.Equal(TestDestination.Billing, destination);
            Assert.True(followup);
            throw failure;
        }
        var function = new ToolProbe(AIFunctionFactory.Create(Fail, ToolFixture.Options("preview")));
        var outer = new ToolProbe(function);
        using var handler = ToolFixture.Handler("preview");
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        var selection = await new TypeSafeToolSelector(spy, [ToolFixture.InferredDefinition(outer)], 0.8)
            .SelectAsync(SchemaFixture.State, cancellationToken: TestContext.Current.CancellationToken);

        var error = await Assert.ThrowsAsync<InvalidOperationException>(() => TypeSafeToolInvoker.InvokeAsync(selection,
            (_, _) => { authorizations++; return ValueTask.FromResult(true); },
            cancellationToken: TestContext.Current.CancellationToken).AsTask());

        Assert.Same(failure, error);
        Assert.Equal(1, authorizations);
        Assert.Equal(1, invocations);
        Assert.Equal(1, outer.Calls);
        Assert.Equal(1, function.Calls);
        Assert.Equal(2, spy.Calls);
        Assert.Equal(2, handler.Sends);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Invoke_UsesFixedAndDeclaredDefaultArgumentsWithoutAnotherInference(bool omit)
    {
        int invocations = 0;
        var function = new ToolProbe(AIFunctionFactory.Create((int ticketId, string note = "declared-default") =>
        {
            invocations++;
            return $"{ticketId}:{note}";
        }, ToolFixture.Options("optional")));
        Dictionary<string, JsonElement> fixedValues = new() { ["/ticketId"] = NativeFixture.Json("17") };
        if (!omit) fixedValues["/note"] = NativeFixture.Json("\"host-note\"");
        var definition = new TypeSafeToolDefinition(function, fixedValues: fixedValues, omitDefaultParameters: omit ? ["/note"] : []);
        using var handler = ToolFixture.Handler("optional");
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        var selection = await new TypeSafeToolSelector(spy, [definition], 0.8).SelectAsync(SchemaFixture.State,
            cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(0, invocations);
        Assert.Single(selection.Evidence);
        var result = await TypeSafeToolInvoker.InvokeAsync(selection, (_, _) => ValueTask.FromResult(true),
            cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal(omit ? "17:declared-default" : "17:host-note", Assert.IsType<JsonElement>(result.Result).GetString());
        Assert.Equal(selection.FunctionCall!.CallId, result.CallId);
        Assert.Equal(1, invocations);
        Assert.Equal(1, function.Calls);
        Assert.Equal(1, spy.Calls);
        Assert.Equal(1, handler.Sends);
    }

    private sealed record HostMarker(string Value);
}
