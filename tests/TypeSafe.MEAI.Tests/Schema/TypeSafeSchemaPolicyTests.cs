using System.Text.Json;
using Microsoft.Extensions.AI;
using TypeSafe.MEAI.Tests.Fixtures;
using Xunit;

namespace TypeSafe.MEAI.Tests.Schema;

[Trait("Stage", "C")]
public sealed class TypeSafeSchemaPolicyTests
{
    [Theory]
    [InlineData(0.199999, false)]
    [InlineData(0.2, false)]
    [InlineData(0.5, null)]
    [InlineData(0.8, true)]
    [InlineData(0.800001, true)]
    public async Task BooleanPolicy_InclusiveBoundsAndOpenIntervalNeverUseSchemaDefault(double probability, bool? expected)
    {
        using var handler = SchemaFixture.Handler(noul: probability);
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        using var adapter = new TypeSafeSchemaChatClient(spy, new Dictionary<string, TypeSafeFieldBinding> { [""] = SchemaFixture.Boolean() });
        var options = SchemaFixture.Options(NativeFixture.Json("""{"type":"boolean","default":true}"""));
        Task<ChatResponse> Invoke() => adapter.GetResponseAsync([SchemaFixture.State.ToMessage()], options, TestContext.Current.CancellationToken);
        TypeSafeResponse evidence;
        if (expected is { } value)
        {
            var response = await Invoke();
            Assert.Equal(value, NativeFixture.Json(response.Text).GetBoolean());
            evidence = response.GetTypeSafeEvidence();
        }
        else
        {
            var error = await Assert.ThrowsAsync<TypeSafeAbstentionException>(Invoke);
            Assert.Equal("", Assert.Single(error.Rejections).Pointer);
            Assert.Contains("abstention", error.Rejections[0].Reason);
            evidence = error.Evidence;
        }
        Assert.Equal(probability, evidence.GetAnswer<NoulAnswer>("q0000").Noul);
        Assert.Equal(18, evidence.TotalTokens);
        Assert.Equal(1, spy.Calls);
        Assert.Equal(1, handler.Sends);
    }

    [Theory]
    [InlineData("choice", 0.799999, false)]
    [InlineData("choice", 0.8, true)]
    [InlineData("score", 0.799999, false)]
    [InlineData("score", 0.8, true)]
    public async Task ConfidencePolicy_UsesInclusiveServerConfidenceNotDistribution(string type, double confidence, bool accepted)
    {
        double probability = accepted ? 0.4 : 0.99;
        using var handler = SchemaFixture.Handler(confidence: confidence, probability: probability);
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        var binding = type == "choice" ? SchemaFixture.Choice() : SchemaFixture.Score();
        using var adapter = new TypeSafeSchemaChatClient(spy, new Dictionary<string, TypeSafeFieldBinding> { ["/value"] = binding });
        var schema = SchemaFixture.ObjectSchema(type == "choice" ? """{"type":"string","enum":["Billing","Technical"]}""" : """{"type":"number"}""");
        Task<ChatResponse> Invoke() => adapter.GetResponseAsync([SchemaFixture.State.ToMessage()], SchemaFixture.Options(schema), TestContext.Current.CancellationToken);
        TypeSafeResponse evidence;
        if (accepted)
        {
            var response = await Invoke();
            var value = NativeFixture.Json(response.Text).GetProperty("value");
            if (type == "choice") Assert.Equal("Billing", value.GetString());
            else Assert.Equal(1.25, value.GetDouble());
            evidence = response.GetTypeSafeEvidence();
        }
        else
        {
            var error = await Assert.ThrowsAsync<TypeSafeAbstentionException>(Invoke);
            Assert.Equal("/value", Assert.Single(error.Rejections).Pointer);
            evidence = error.Evidence;
        }
        var native = evidence.Answers["q0000"].Json;
        Assert.Equal(confidence, native.GetProperty("confidence").GetDouble());
        Assert.Equal(probability, native.GetProperty("probabilities").EnumerateObject().First().Value.GetDouble());
        Assert.NotEqual(confidence, native.GetProperty("probabilities").EnumerateObject().Max(p => p.Value.GetDouble()));
        if (type == "score") Assert.Equal(1.25, evidence.GetAnswer<ScoreAnswer>("q0000").Score);
        Assert.Equal(1, spy.Calls);
        Assert.Equal(1, handler.Sends);
    }

    [Theory]
    [InlineData(-0.01, 0.8)]
    [InlineData(0.2, 1.01)]
    [InlineData(0.5, 0.5)]
    [InlineData(0.8, 0.2)]
    [InlineData(double.NaN, 0.8)]
    [InlineData(0.2, double.NaN)]
    [InlineData(0.2, double.PositiveInfinity)]
    public void BooleanPolicy_RejectsNonfiniteOutOfRangeOrOverlappingBounds(double falseMaximum, double trueMinimum)
    {
        var error = Assert.Throws<ArgumentException>(() => new TypeSafeBooleanPolicy(falseMaximum, trueMinimum));
        Assert.Contains("falseMaximum < trueMinimum", error.Message);
    }

    [Theory]
    [InlineData(-0.01, false)]
    [InlineData(1.01, false)]
    [InlineData(double.NaN, false)]
    [InlineData(double.PositiveInfinity, false)]
    [InlineData(0, true)]
    [InlineData(1, true)]
    public void FieldBinding_RequiresFiniteUnitConfidenceWithExplicitEndpointPolicies(double confidence, bool valid)
    {
        if (!valid)
        {
            var error = Assert.Throws<ArgumentException>(() => SchemaFixture.Score(confidence));
            Assert.Contains("finite in [0,1]", error.Message);
        }
        else
        {
            var binding = SchemaFixture.Score(confidence);
            Assert.Equal(confidence, binding.MinimumConfidence);
            Assert.Equal(3, binding.Rubric!.Count);
            var boolean = new TypeSafeBooleanPolicy(0, 1);
            Assert.Equal(0, boolean.FalseMaximum);
            Assert.Equal(1, boolean.TrueMinimum);
        }
    }

    [Fact]
    public async Task Decision_AbstainsWithoutDefaultTypedValueAndExceptionRetainsSameEvidence()
    {
        using var handler = SchemaFixture.Handler();
        using var services = NativeFixture.Services(handler, cache: true);
        using var spy = new SchemaSpy(services.Client());
        var bindings = SchemaFixture.Bindings();
        bindings["/details/destination"] = SchemaFixture.Choice(0.95);
        using var adapter = new TypeSafeSchemaChatClient(spy, bindings);

        var decision = await adapter.GetTypeSafeDecisionAsync<DecisionDto>(SchemaFixture.State, serializerOptions: SchemaFixture.Serializer,
            cancellationToken: TestContext.Current.CancellationToken);
        var error = await Assert.ThrowsAsync<TypeSafeAbstentionException>(() => adapter.GetResponseAsync<DecisionDto>(
            [SchemaFixture.State.ToMessage()], SchemaFixture.Serializer, cancellationToken: TestContext.Current.CancellationToken));

        Assert.False(decision.Accepted);
        Assert.Throws<InvalidOperationException>(() => decision.Value);
        Assert.Equal("/details/destination", Assert.Single(decision.Rejections).Pointer);
        Assert.Equal(decision.Rejections, error.Rejections);
        Assert.True(JsonElement.DeepEquals(decision.Evidence.Json, error.Evidence.Json));
        Assert.Equal(3, decision.Evidence.Answers.Count);
        Assert.Equal(0.9, decision.Evidence.GetAnswer<ChoiceAnswer>("q0001").Confidence);
        Assert.Equal(1.25, decision.Evidence.GetAnswer<ScoreAnswer>("q0002").Score);
        Assert.Equal("jev-1.13.0", decision.Evidence.Model);
        Assert.Equal(18, decision.Evidence.TotalTokens);
        Assert.Equal(2, spy.Calls);
        Assert.Equal(1, handler.Sends);
        Assert.Null(spy.Responses[1].RawRepresentation);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Decision_LeafOrApplicationValidationAbstainsWithExplicitReason(bool application)
    {
        int validations = 0;
        string? Validate(JsonElement value)
        {
            validations++;
            if (application)
            {
                Assert.True(value.GetProperty("followup").GetBoolean());
                Assert.Equal("Billing", value.GetProperty("details").GetProperty("destination").GetString());
                Assert.Equal(1.25, value.GetProperty("details").GetProperty("quality").GetDouble());
            }
            else Assert.Equal(1.25, value.GetDouble());
            return "Synthetic host validation rejected this decision.";
        }
        var bindings = SchemaFixture.Bindings();
        if (!application)
            bindings["/details/quality"] = new("Assess the synthetic draft.", minimumConfidence: 0.8,
                rubric: SchemaFixture.Score().Rubric, validate: Validate);
        using var handler = SchemaFixture.Handler();
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        using var adapter = new TypeSafeSchemaChatClient(spy, bindings, application ? Validate : null);
        var decision = await adapter.GetTypeSafeDecisionAsync<DecisionDto>(SchemaFixture.State,
            serializerOptions: SchemaFixture.Serializer, cancellationToken: TestContext.Current.CancellationToken);

        Assert.False(decision.Accepted);
        Assert.Throws<InvalidOperationException>(() => decision.Value);
        var rejection = Assert.Single(decision.Rejections);
        Assert.Equal(application ? "" : "/details/quality", rejection.Pointer);
        Assert.Equal("Synthetic host validation rejected this decision.", rejection.Reason);
        Assert.Equal(1, validations);
        Assert.Equal(1.25, decision.Evidence.GetAnswer<ScoreAnswer>("q0002").Score);
        Assert.Equal(1, spy.Calls);
        Assert.Equal(1, handler.Sends);
    }

    [Fact]
    public async Task Decision_ExplicitZeroConfidencePolicyAcceptsWithoutInventingConfidence()
    {
        using var handler = SchemaFixture.Handler(confidence: 0);
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        using var adapter = new TypeSafeSchemaChatClient(spy, SchemaFixture.Bindings(confidence: 0));
        var decision = await adapter.GetTypeSafeDecisionAsync<DecisionDto>(SchemaFixture.State,
            cancellationToken: TestContext.Current.CancellationToken);

        Assert.True(decision.Accepted);
        Assert.Empty(decision.Rejections);
        Assert.Equal(TestDestination.Billing, decision.Value.Details.Destination);
        Assert.Equal(1.25, decision.Value.Details.Quality);
        Assert.Equal(0, decision.Evidence.GetAnswer<ChoiceAnswer>("q0001").Confidence);
        Assert.Equal(0, decision.Evidence.GetAnswer<ScoreAnswer>("q0002").Confidence);
        Assert.Equal(1, spy.Calls);
        Assert.Equal(1, handler.Sends);
    }
}
