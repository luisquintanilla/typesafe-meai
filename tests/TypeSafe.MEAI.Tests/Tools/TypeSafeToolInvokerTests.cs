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
    [InlineData("arguments")]
    [InlineData("call-id")]
    [InlineData("request-id")]
    [InlineData("name")]
    [InlineData("replay")]
    public async Task ApprovalResume_HostRejectsTamperUnknownCorrelationAndReplayBeforeMiddleware(string mutation)
    {
        int effects = 0;
        var core = ToolFixture.InferredFunction(() => effects++);
        var outer = new ToolProbe(new ApprovalRequiredAIFunction(core));
        using var handler = ToolFixture.Handler("preview");
        using var http = new HttpClient(handler);
        using var provider = new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key);
        using var host = new OneShotHost(provider, outer);
        var first = await host.StartAsync(TestContext.Current.CancellationToken);
        var request = Assert.Single(first.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>());
        var original = Assert.IsType<FunctionCallContent>(request.ToolCall);
        Assert.Equal(0, effects);
        Assert.Equal(0, host.Authorizations);
        Assert.Equal(2, handler.Sends);
        var arguments = original.Arguments!.ToDictionary();
        if (mutation == "arguments") arguments["followup"] = false;
        var changed = new FunctionCallContent(mutation == "call-id" ? "unknown-call" : original.CallId,
            mutation == "name" ? "unregistered" : original.Name, arguments);
        var approval = new ToolApprovalResponseContent(mutation == "request-id" ? "unknown-request" : request.RequestId, true, changed);
        if (mutation == "replay")
        {
            var completed = await host.ResumeAsync(approval, TestContext.Current.CancellationToken);
            var result = Assert.Single(completed.Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>());
            Assert.Equal(original.CallId, result.CallId);
            Assert.Equal("Billing", Assert.IsType<JsonElement>(result.Result).GetProperty("destination").GetString());
            Assert.True(Assert.IsType<JsonElement>(result.Result).GetProperty("followup").GetBoolean());
        }
        await Assert.ThrowsAsync<InvalidOperationException>(() => host.ResumeAsync(approval, TestContext.Current.CancellationToken));
        Assert.Equal(mutation == "replay" ? 1 : 0, effects);
        Assert.Equal(mutation == "replay" ? 1 : 0, core.Calls);
        Assert.Equal(mutation == "replay" ? 1 : 0, outer.Calls);
        Assert.Equal(mutation == "replay" ? 1 : 0, host.Authorizations);
        Assert.Equal(1, host.ProviderCalls);
        Assert.Equal(2, handler.Sends);
    }

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
        Dictionary<object, object?> context = new() { ["invocation-id"] = "synthetic-context-17" };
        using var host = new OneShotHost(spy, outer) { Services = services, Context = context };
        host.Authorize = (observed, token) =>
            {
                Assert.Same(outer, observed.Function);
                Assert.Equal(["destination", "followup"], observed.Arguments.Keys.Order());
                Assert.Equal(cancellation.Token, token);
                Assert.Equal(0, outer.Calls);
                authorizations++;
                sequence.Add("authorize");
                return ValueTask.FromResult(true);
            };
        var first = await host.StartAsync(cancellation.Token);
        var evidence = first.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>().ToArray();
        Assert.Equal(2, evidence.Length);
        Assert.Equal("preview", Assert.IsType<ChoiceAnswer>(evidence[0].Answers["tool"]).Choice);
        Assert.Equal(0.9, Assert.IsType<ChoiceAnswer>(evidence[0].Answers["tool"]).Confidence);
        Assert.Equal("Billing", Assert.IsType<ChoiceAnswer>(evidence[1].Answers["q0000"]).Choice);
        Assert.Equal(0.9, Assert.IsType<NoulAnswer>(evidence[1].Answers["q0001"]).Noul);
        Assert.All(evidence, e =>
        {
            Assert.Equal(18, e.TotalTokens);
            Assert.Equal(17, e.Json.GetProperty("vendor_envelope").GetProperty("revision").GetInt32());
        });
        var completed = first;
        string callId;
        if (approvalRequired)
        {
            Assert.Equal(0, invocations);
            Assert.Equal(0, outer.Calls);
            var request = Assert.Single(first.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>());
            callId = Assert.IsType<FunctionCallContent>(request.ToolCall).CallId;
            approvals++;
            sequence.Add("approve");
            completed = await host.ResumeAsync(request.CreateResponse(true), cancellation.Token);
        }
        else callId = Assert.Single(first.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>()).CallId;
        var result = Assert.Single(completed.Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>());

        Assert.Equal(1, authorizations);
        Assert.Equal(approvalRequired ? 1 : 0, approvals);
        Assert.Equal(approvalRequired ? ["approve", "authorize", "invoke"] : ["authorize", "invoke"], sequence);
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
        Assert.Equal(callId, result.CallId);
        Assert.Null(result.Exception);
        var payload = Assert.IsType<JsonElement>(result.Result);
        Assert.Equal("Billing", payload.GetProperty("destination").GetString());
        Assert.True(payload.GetProperty("followup").GetBoolean());
        Assert.Equal("host-owned-service", payload.GetProperty("host").GetString());
        Assert.Equal(1, spy.Calls);
        Assert.Equal(1, host.ProviderCalls);
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
        using var host = new OneShotHost(spy, outer);
        host.Authorize = (_, _) => { authorizations++; return ValueTask.FromResult(denied != "authorization"); };
        var selection = await host.StartAsync(TestContext.Current.CancellationToken);
        var request = Assert.Single(selection.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>());
        if (denied != "missing-approval")
        {
            if (denied == "approval") approvals++;
            await Assert.ThrowsAsync<UnauthorizedAccessException>(() =>
                host.ResumeAsync(request.CreateResponse(denied == "authorization"), TestContext.Current.CancellationToken));
        }
        Assert.Equal(denied == "authorization" ? 1 : 0, authorizations);
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
        using var handler = ToolFixture.Handler(outcome == "no-match" ? "__typesafe_no_match__" : "preview",
            confidence: outcome == "selection-abstain" ? 0.7 : 0.9,
            argumentNoul: outcome == "arguments-abstain" ? 0.5 : 0.9);
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        using var host = new OneShotHost(spy, function);
        host.Authorize = (_, _) => { authorizations++; return ValueTask.FromResult(true); };
        if (outcome == "no-match")
        {
            var selection = await host.StartAsync(TestContext.Current.CancellationToken);
            Assert.Empty(selection.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>());
            Assert.Empty(selection.Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>());
        }
        else
        {
            var error = await Assert.ThrowsAsync<TypeSafeAbstentionException>(() => host.StartAsync(TestContext.Current.CancellationToken));
            Assert.Equal(outcome == "arguments-abstain" ? 2 : 1, error.Evidence.Count);
            Assert.Single(error.Rejections);
        }

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
        using var host = new OneShotHost(spy, function) { PolicyStillAllows = () => allowed };
        host.Authorize = (context, _) =>
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
                    case "arguments": context.Arguments["followup"] = false; break;
                    case "informational": context.CallContent.InformationalOnly = true; break;
                    case "validation": allowed = false; break;
                }
                return ValueTask.FromResult(true);
            };
        var error = await Assert.ThrowsAsync<InvalidOperationException>(() => host.StartAsync(TestContext.Current.CancellationToken));

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
        using var host = new OneShotHost(spy, outer);
        var selection = await host.StartAsync(TestContext.Current.CancellationToken);
        var request = Assert.Single(selection.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>());
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        if (phase == "before") cancellation.Cancel();

        host.Authorize = (_, _) =>
            {
                authorizations++;
                if (phase == "authorization") cancellation.Cancel();
                return ValueTask.FromResult(true);
            };
        if (phase == "approval") { approvals++; cancellation.Cancel(); }
        var error = await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            host.ResumeAsync(request.CreateResponse(true), cancellation.Token));

        Assert.Equal(cancellation.Token, error.CancellationToken);
        Assert.Equal(phase == "authorization" ? 1 : 0, authorizations);
        Assert.Equal(phase == "approval" ? 1 : 0, approvals);
        Assert.Equal(0, invocations);
        Assert.Equal(0, outer.Calls);
        Assert.Equal(0, function.Calls);
        Assert.Equal(2, handler.Sends);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Invoke_FunctionExceptionPropagatesOnceWithoutAnotherModelOrToolLoop(bool approvalRequired)
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
        var outer = new ToolProbe(approvalRequired ? new ApprovalRequiredAIFunction(function) : function);
        using var handler = ToolFixture.Handler("preview");
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        using var host = new OneShotHost(spy, outer);
        host.Authorize = (_, _) => { authorizations++; return ValueTask.FromResult(true); };
        InvalidOperationException error;
        if (approvalRequired)
        {
            var pending = await host.StartAsync(TestContext.Current.CancellationToken);
            Assert.Equal(0, invocations);
            Assert.Equal(0, authorizations);
            var request = Assert.Single(pending.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>());
            error = await Assert.ThrowsAsync<InvalidOperationException>(() => host.ResumeAsync(request.CreateResponse(true), TestContext.Current.CancellationToken));
        }
        else error = await Assert.ThrowsAsync<InvalidOperationException>(() => host.StartAsync(TestContext.Current.CancellationToken));

        Assert.Same(failure, error);
        Assert.Equal(1, authorizations);
        Assert.Equal(1, invocations);
        Assert.Equal(1, outer.Calls);
        Assert.Equal(1, function.Calls);
        Assert.Equal(1, spy.Calls);
        Assert.Equal(1, host.ProviderCalls);
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
        var config = new TypeSafeToolOptions(fixedValues: fixedValues, omitDefaultParameters: omit ? ["/note"] : []);
        using var handler = ToolFixture.Handler("optional");
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        using var client = ToolFixture.Client(spy, function, config);
        using var host = new OneShotHost(client, function);
        var selection = await host.StartAsync(TestContext.Current.CancellationToken);
        Assert.Single(selection.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>());
        var call = Assert.Single(selection.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>());
        var result = Assert.Single(selection.Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>());

        Assert.Equal(omit ? "17:declared-default" : "17:host-note", Assert.IsType<JsonElement>(result.Result).GetString());
        Assert.Equal(call.CallId, result.CallId);
        Assert.Equal(1, invocations);
        Assert.Equal(1, function.Calls);
        Assert.Equal(1, spy.Calls);
        Assert.Equal(1, handler.Sends);
    }

    private sealed record HostMarker(string Value);
}
