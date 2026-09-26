using System.Text.Json;
using Microsoft.Extensions.AI;
using TypeSafe.MEAI.Tests.Fixtures;
using Xunit;

namespace TypeSafe.MEAI.Tests.Tools;

[Trait("Stage", "C")]
public sealed class TypeSafeFixedIntegerInvocationTests
{
    [Theory]
    [InlineData("1.0", 1L)]
    [InlineData("1e2", 100L)]
    [InlineData("9007199254740993", 9007199254740993L)]
    [InlineData("9007199254740993.0", 9007199254740993L)]
    [InlineData("9223372036854775807.0", long.MaxValue)]
    [InlineData("-9223372036854775808.0", long.MinValue)]
    public async Task FixedIntegralJson_InvokesRealApprovedLongFunctionWithoutRounding(string json, long expected)
    {
        int invocations = 0, authorizations = 0, approvals = 0;
        long observed = 0;
        var function = new ApprovalRequiredAIFunction(AIFunctionFactory.Create((long value) =>
        {
            invocations++;
            observed = value;
            return value;
        }, new AIFunctionFactoryOptions
        {
            Name = "fixed_integer", Description = "Read-only synthetic integer echo."
        }));
        var config = new TypeSafeToolOptions(fixedValues:
            new Dictionary<string, JsonElement> { ["/value"] = NativeFixture.Json(json) });
        using var handler = new RecordingHandler((_, _, _) => Task.FromResult(NativeFixture.Response("""
            {"model":"jev-1.13.0","usage":{"input_tokens":3,"output_tokens":2},
             "answers":{"tool":{"type":"choice","choice":"fixed_integer","confidence":1,
             "probabilities":{"fixed_integer":1,"__typesafe_no_match__":0}}}}
            """)));
        using var services = NativeFixture.Services(handler, settings: NativeFixture.Settings with
        { Tools = new Dictionary<string, TypeSafeToolOptions> { [function.Name] = config } });
        using var host = new OneShotHost(services.Client(), function);
        host.Authorize = (context, _) =>
        {
            Assert.Same(function, context.Function);
            Assert.Equal(expected, Assert.IsType<JsonElement>(context.Arguments["value"]).GetInt64());
            authorizations++;
            return ValueTask.FromResult(true);
        };
        var selection = await host.StartAsync(TestContext.Current.CancellationToken);
        var approval = Assert.Single(selection.Messages.SelectMany(m => m.Contents).OfType<ToolApprovalRequestContent>());
        var call = Assert.IsType<FunctionCallContent>(approval.ToolCall);
        Assert.Equal(expected.ToString(System.Globalization.CultureInfo.InvariantCulture),
            Assert.IsType<JsonElement>(call.Arguments!["value"]).GetRawText());
        Assert.Equal(0, invocations);
        approvals++;
        var completed = await host.ResumeAsync(approval.CreateResponse(true), TestContext.Current.CancellationToken);
        var result = Assert.Single(completed.Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>());
        Assert.Equal(expected, observed);
        Assert.Equal(expected, JsonSerializer.SerializeToElement(result.Result).GetInt64());
        Assert.Equal(call.CallId, result.CallId);
        Assert.Equal(1, invocations);
        Assert.Equal(1, authorizations);
        Assert.Equal(1, approvals);
        Assert.Equal(1, handler.Sends);
    }
}
