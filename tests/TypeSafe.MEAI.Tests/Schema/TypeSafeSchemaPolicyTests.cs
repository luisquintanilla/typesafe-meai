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
        using var adapter = new TypeSafeChatClient(spy, NativeFixture.Settings with
        { Fields = new Dictionary<string, TypeSafeFieldOptions> { [""] = SchemaFixture.Boolean() } });
        var options = SchemaFixture.Options(NativeFixture.Json("""{"type":"boolean","default":true}"""));
        Task<ChatResponse> Invoke() => adapter.GetResponseAsync([SchemaFixture.State.ToMessage()], options, TestContext.Current.CancellationToken);
        JevResultContent evidence;
        if (expected is { } value)
        {
            var response = await Invoke();
            Assert.Equal(value, NativeFixture.Json(response.Text).GetBoolean());
            evidence = Assert.Single(response.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>());
        }
        else
        {
            var error = await Assert.ThrowsAsync<TypeSafeAbstentionException>(Invoke);
            Assert.Equal("", Assert.Single(error.Rejections).Pointer);
            Assert.Contains("abstention", error.Rejections[0].Reason);
            evidence = Assert.Single(error.Evidence);
        }
        Assert.Equal(probability, Assert.IsType<NoulAnswer>(evidence.Answers["q0000"]).Noul);
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
        using var adapter = new TypeSafeChatClient(spy, NativeFixture.Settings with
        { Fields = new Dictionary<string, TypeSafeFieldOptions> { ["/value"] = binding } });
        var schema = SchemaFixture.ObjectSchema(type == "choice" ? """{"type":"string","enum":["Billing","Technical"]}""" : """{"type":"number"}""");
        Task<ChatResponse> Invoke() => adapter.GetResponseAsync([SchemaFixture.State.ToMessage()], SchemaFixture.Options(schema), TestContext.Current.CancellationToken);
        JevResultContent evidence;
        if (accepted)
        {
            var response = await Invoke();
            var value = NativeFixture.Json(response.Text).GetProperty("value");
            if (type == "choice") Assert.Equal("Billing", value.GetString());
            else Assert.Equal(1.25, value.GetDouble());
            evidence = Assert.Single(response.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>());
        }
        else
        {
            var error = await Assert.ThrowsAsync<TypeSafeAbstentionException>(Invoke);
            Assert.Equal("/value", Assert.Single(error.Rejections).Pointer);
            evidence = Assert.Single(error.Evidence);
        }
        var native = evidence.Answers["q0000"].Json;
        Assert.Equal(confidence, native.GetProperty("confidence").GetDouble());
        Assert.Equal(probability, native.GetProperty("probabilities").EnumerateObject().First().Value.GetDouble());
        Assert.NotEqual(confidence, native.GetProperty("probabilities").EnumerateObject().Max(p => p.Value.GetDouble()));
        if (type == "score") Assert.Equal(1.25, Assert.IsType<ScoreAnswer>(evidence.Answers["q0000"]).Score);
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
    public void FieldOptions_RequiresFiniteUnitConfidenceWithExplicitEndpointPolicies(double confidence, bool valid)
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
        using var adapter = new TypeSafeChatClient(spy, NativeFixture.Settings with { Fields = bindings });

        var first = await Assert.ThrowsAsync<TypeSafeAbstentionException>(() => adapter.GetResponseAsync<DecisionDto>(
            NativeFixture.Messages(), SchemaFixture.Serializer, cancellationToken: TestContext.Current.CancellationToken));
        var error = await Assert.ThrowsAsync<TypeSafeAbstentionException>(() => adapter.GetResponseAsync<DecisionDto>(
            [SchemaFixture.State.ToMessage()], SchemaFixture.Serializer, cancellationToken: TestContext.Current.CancellationToken));

        Assert.Equal("/details/destination", Assert.Single(first.Rejections).Pointer);
        Assert.Equal(first.Rejections, error.Rejections);
        var evidence = Assert.Single(error.Evidence);
        Assert.True(JsonElement.DeepEquals(Assert.Single(first.Evidence).Json, evidence.Json));
        Assert.Equal(3, evidence.Answers.Count);
        Assert.Equal(0.9, Assert.IsType<ChoiceAnswer>(evidence.Answers["q0001"]).Confidence);
        Assert.Equal(1.25, Assert.IsType<ScoreAnswer>(evidence.Answers["q0002"]).Score);
        Assert.Equal("jev-1.13.0", evidence.Model);
        Assert.Equal(18, evidence.TotalTokens);
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
        using var adapter = new TypeSafeChatClient(spy, NativeFixture.Settings with { Fields = bindings, ValidateResult = application ? Validate : null });
        var decision = await Assert.ThrowsAsync<TypeSafeAbstentionException>(() => adapter.GetResponseAsync<DecisionDto>(NativeFixture.Messages(),
            SchemaFixture.Serializer, cancellationToken: TestContext.Current.CancellationToken));
        var rejection = Assert.Single(decision.Rejections);
        Assert.Equal(application ? "" : "/details/quality", rejection.Pointer);
        Assert.Equal("Synthetic host validation rejected this decision.", rejection.Reason);
        Assert.Equal(1, validations);
        Assert.Equal(1.25, Assert.IsType<ScoreAnswer>(Assert.Single(decision.Evidence).Answers["q0002"]).Score);
        Assert.Equal(1, spy.Calls);
        Assert.Equal(1, handler.Sends);
    }

    [Fact]
    public async Task Decision_ExplicitZeroConfidencePolicyAcceptsWithoutInventingConfidence()
    {
        using var handler = SchemaFixture.Handler(confidence: 0);
        using var http = new HttpClient(handler);
        using var spy = new SchemaSpy(new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key));
        using var adapter = new TypeSafeChatClient(spy, NativeFixture.Settings with { Fields = SchemaFixture.Bindings(confidence: 0) });
        var decision = await adapter.GetResponseAsync<DecisionDto>(NativeFixture.Messages(),
            cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal(TestDestination.Billing, decision.Result.Details.Destination);
        Assert.Equal(1.25, decision.Result.Details.Quality);
        var evidence = Assert.Single(decision.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>());
        Assert.Equal(0, Assert.IsType<ChoiceAnswer>(evidence.Answers["q0001"]).Confidence);
        Assert.Equal(0, Assert.IsType<ScoreAnswer>(evidence.Answers["q0002"]).Confidence);
        Assert.Equal(1, spy.Calls);
        Assert.Equal(1, handler.Sends);
    }
}
