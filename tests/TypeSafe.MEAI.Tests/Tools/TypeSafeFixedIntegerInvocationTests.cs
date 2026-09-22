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
            Name = "fixed_integer", Description = "Read-only synthetic integer echo.",
            JsonSchemaCreateOptions = new() { TransformOptions = new() { DisallowAdditionalProperties = true } }
        }));
        var definition = new TypeSafeToolDefinition(function, fixedValues:
            new Dictionary<string, JsonElement> { ["/value"] = NativeFixture.Json(json) });
        using var handler = new RecordingHandler((_, _, _) => Task.FromResult(NativeFixture.Response("""
            {"model":"jev-1.13.0","usage":{"input_tokens":3,"output_tokens":2},
             "answers":{"tool":{"type":"choice","choice":"fixed_integer","confidence":1,
             "probabilities":{"fixed_integer":1,"__typesafe_no_match__":0}}}}
            """)));
        using var services = NativeFixture.Services(handler);
        var selector = new TypeSafeToolSelector(services.Client(), [definition], 0.8);
        var selection = await selector.SelectAsync(TypeSafeState.FromText("Synthetic integer proof."),
            cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(TypeSafeToolSelectionStatus.Accepted, selection.Status);
        Assert.Equal(0, invocations);
        var result = await TypeSafeToolInvoker.InvokeAsync(selection,
            authorize: (_, _) => { authorizations++; return ValueTask.FromResult(true); },
            approve: (_, _) => { approvals++; return ValueTask.FromResult(true); },
            cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(expected, observed);
        Assert.Equal(expected, JsonSerializer.SerializeToElement(result.Result).GetInt64());
        Assert.Equal(selection.FunctionCall!.CallId, result.CallId);
        Assert.Equal(1, invocations);
        Assert.Equal(1, authorizations);
        Assert.Equal(1, approvals);
        Assert.Equal(1, handler.Sends);
    }
}
