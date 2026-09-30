using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using DecisionInference.Ollama;
using OllamaSharp;

namespace DecisionInference.OllamaSharp.Tests;

public sealed class ComparisonTests
{
    private static CancellationToken Token => TestContext.Current.CancellationToken;

    [Fact]
    public async Task CommonDomainSameTicketAliasesAllKindsExactDoublesUsageAndPrompt()
    {
        var input = Fixtures.Input();
        string source = Fixtures.Native().ToJsonString();
        using var sdkHandler = new ScriptedHandler(_ => Fixtures.Response(source));
        using var customHandler = new ScriptedHandler(_ => Fixtures.Response(source));
        using var sdkHttp = Fixtures.Http(sdkHandler);
        using var customHttp = Fixtures.Http(customHandler);
        using var sdk = new OllamaApiClient(sdkHttp, Fixtures.Model);
        using var sdkGenerator = new OllamaSharpDecisionGenerator(sdk);
        using var customGenerator = new OllamaDecisionGenerator(customHttp, new() { ModelId = Fixtures.Model });
        var first = await sdkGenerator.GenerateAsync(input, new OllamaSharpGenerationOptions { KeepAliveSeconds = 60 }, Token);
        var second = await customGenerator.GenerateAsync(input, new OllamaGenerationOptions { KeepAliveSeconds = 60 }, Token);
        Assert.Equal(second.ModelId, first.ModelId);
        Assert.Equal(second.Answers.Select(a => a.QuestionId), first.Answers.Select(a => a.QuestionId));
        Assert.Equal(second.Answers.Select(a => a.Kind), first.Answers.Select(a => a.Kind));
        Assert.Equal(second.GetAnswer<BinaryDecisionAnswer>("account-access").TrueProbability,
            first.GetAnswer<BinaryDecisionAnswer>("account-access").TrueProbability);
        var firstChoice = first.GetAnswer<ChoiceDecisionAnswer>("support-team");
        var secondChoice = second.GetAnswer<ChoiceDecisionAnswer>("support-team");
        Assert.Equal(secondChoice.SelectedCandidateId, firstChoice.SelectedCandidateId);
        Assert.Equal(secondChoice.Probabilities, firstChoice.Probabilities);
        var firstScore = first.GetAnswer<ScoreDecisionAnswer>("urgency");
        var secondScore = second.GetAnswer<ScoreDecisionAnswer>("urgency");
        Assert.Equal(secondScore.Probabilities, firstScore.Probabilities);
        Assert.Equal(secondScore.NativeScore, firstScore.NativeScore);
        Assert.Equal(secondScore.ExpectedIndex, firstScore.ExpectedIndex);
        Assert.Equal(second.Usage!.InputTokenCount, first.Usage!.InputTokenCount);
        Assert.Equal(second.Usage.OutputTokenCount, first.Usage.OutputTokenCount);
        Assert.Equal(second.Usage.TotalTokenCount, first.Usage.TotalTokenCount);
        var sdkBody = Assert.Single(sdkHandler.Requests).Body;
        var customBody = Assert.Single(customHandler.Requests).Body;
        Assert.Equal(customBody.GetProperty("state").GetString(), sdkBody.GetProperty("state").GetString());
        Assert.True(JsonElement.DeepEquals(customBody.GetProperty("questions"), sdkBody.GetProperty("questions")));
        Assert.Equal(60, customBody.GetProperty("keep_alive").GetInt32());
        Assert.Equal("60s", sdkBody.GetProperty("keep_alive").GetString());
        Assert.True(sdkBody.TryGetProperty("CustomHeaders", out _));
        Assert.False(customBody.TryGetProperty("CustomHeaders", out _));
        Assert.False(second.AdditionalProperties!.ContainsKey("ollamasharp.rawRepresentationKind"));
        Assert.Equal("sdk-dto-snapshot", first.AdditionalProperties!["ollamasharp.rawRepresentationKind"]);
    }

    [Theory]
    [InlineData("{\"text\":\"café <tag>\"}")]
    [InlineData("[\"café <tag>\"]")]
    public async Task ObjectArraySemanticEqualityIsNotPinnedServerPromptEquality(string state)
    {
        var input = new DecisionInput(JsonSerializer.Deserialize<JsonElement>(state), Fixtures.Input().Questions);
        using var sdkHandler = new ScriptedHandler(_ => Fixtures.Response(Fixtures.Native(input).ToJsonString()));
        using var customHandler = new ScriptedHandler(_ => Fixtures.Response(Fixtures.Native(input).ToJsonString()));
        using var sdkHttp = Fixtures.Http(sdkHandler);
        using var customHttp = Fixtures.Http(customHandler);
        using var sdk = new OllamaApiClient(sdkHttp, Fixtures.Model);
        var encoder = sdk.OutgoingJsonSerializerOptions.Encoder;
        using var sdkGenerator = new OllamaSharpDecisionGenerator(sdk);
        using var customGenerator = new OllamaDecisionGenerator(customHttp, new() { ModelId = Fixtures.Model });
        await sdkGenerator.GenerateAsync(input, cancellationToken: Token);
        await customGenerator.GenerateAsync(input, cancellationToken: Token);
        var sdkSent = Assert.Single(sdkHandler.Requests);
        var customSent = Assert.Single(customHandler.Requests);
        var sdkState = sdkSent.Body.GetProperty("state");
        var customState = customSent.Body.GetProperty("state");
        Assert.True(JsonElement.DeepEquals(customState, sdkState));
        Assert.NotEqual(customState.GetRawText(), sdkState.GetRawText());
        Assert.Contains("café <tag>", sdkState.GetRawText());
        Assert.Contains("\\u", customState.GetRawText());
        Assert.Same(encoder, sdk.OutgoingJsonSerializerOptions.Encoder);
        Assert.Equal(Encoding.UTF8.GetByteCount(sdkSent.Text), sdkSent.Bytes.Length);
        Assert.Equal(Encoding.UTF8.GetByteCount(customSent.Text), customSent.Bytes.Length);
        // Pinned content(json.RawMessage) compacts object/array whitespace without unescaping strings.
        TestContext.Current.TestOutputHelper!.WriteLine(
            $"sdkState={sdkState.GetRawText()}; customState={customState.GetRawText()}; sdkBytes={sdkSent.Bytes.Length}; customBytes={customSent.Bytes.Length}");
    }

    [Theory]
    [InlineData("missing-noul")]
    [InlineData("missing-confidence")]
    [InlineData("missing-score")]
    [InlineData("missing-token")]
    [InlineData("duplicate")]
    [InlineData("inconsistent-total")]
    public async Task LostSourceInformationIsAnExplicitDifferenceNotTransportParity(string difference)
    {
        var root = Fixtures.Native();
        switch (difference)
        {
            case "missing-noul": root["answers"]!["q0"]!.AsObject().Remove("noul"); break;
            case "missing-confidence": root["answers"]!["q1"]!.AsObject().Remove("confidence"); break;
            case "missing-score":
                root["answers"]!["q2"]!.AsObject().Remove("score");
                root["answers"]!["q2"]!["probabilities"] = new JsonObject { ["0"] = 1, ["1"] = 0, ["2"] = 0 }; break;
            case "missing-token": root["usage"]!.AsObject().Remove("output_tokens"); break;
            case "inconsistent-total": root["usage"]!["total_tokens"] = 999; break;
            case "duplicate": break;
            default: throw new InvalidOperationException("Unknown test difference.");
        }
        string source = root.ToJsonString();
        if (difference == "duplicate") source = source.Replace("\"input_tokens\":1836", "\"input_tokens\":1836,\"input_tokens\":1836", StringComparison.Ordinal);
        using var sdkHandler = new ScriptedHandler(_ => Fixtures.Response(source));
        using var customHandler = new ScriptedHandler(_ => Fixtures.Response(source));
        using var sdkHttp = Fixtures.Http(sdkHandler);
        using var customHttp = Fixtures.Http(customHandler);
        using var sdk = new OllamaApiClient(sdkHttp, Fixtures.Model);
        using var sdkGenerator = new OllamaSharpDecisionGenerator(sdk);
        using var customGenerator = new OllamaDecisionGenerator(customHttp, new() { ModelId = Fixtures.Model });
        var result = await sdkGenerator.GenerateAsync(Fixtures.Input(), cancellationToken: Token);
        await Assert.ThrowsAsync<DecisionProtocolException>(() => customGenerator.GenerateAsync(Fixtures.Input(), cancellationToken: Token));
        var snapshot = Assert.IsType<JsonElement>(result.RawRepresentation);
        Assert.Equal("sdk-dto-snapshot", result.AdditionalProperties!["ollamasharp.rawRepresentationKind"]);
        if (difference == "missing-noul") Assert.Equal(0, result.GetAnswer<BinaryDecisionAnswer>("account-access").TrueProbability);
        if (difference == "missing-confidence") Assert.Equal(0, snapshot.GetProperty("answers").GetProperty("q1").GetProperty("confidence").GetDouble());
        if (difference == "missing-score") Assert.Equal(0, result.GetAnswer<ScoreDecisionAnswer>("urgency").NativeScore);
        if (difference == "missing-token") Assert.Equal(0, result.Usage!.OutputTokenCount);
        Assert.False(snapshot.GetProperty("usage").TryGetProperty("total_tokens", out _));
        Assert.Single(sdkHandler.Requests);
        Assert.Single(customHandler.Requests);
    }

    [Fact]
    public async Task UnknownOriginalFieldsSurviveOnlyCustomRawEvidence()
    {
        var root = Fixtures.Native();
        root["future"] = "original-only";
        root["answers"]!["q0"]!["future"] = "original-answer-only";
        using var sdkHandler = new ScriptedHandler(_ => Fixtures.Response(root.ToJsonString()));
        using var customHandler = new ScriptedHandler(_ => Fixtures.Response(root.ToJsonString()));
        using var sdkHttp = Fixtures.Http(sdkHandler);
        using var customHttp = Fixtures.Http(customHandler);
        using var sdk = new OllamaApiClient(sdkHttp, Fixtures.Model);
        using var sdkGenerator = new OllamaSharpDecisionGenerator(sdk);
        using var customGenerator = new OllamaDecisionGenerator(customHttp, new() { ModelId = Fixtures.Model });
        var first = await sdkGenerator.GenerateAsync(Fixtures.Input(), cancellationToken: Token);
        var second = await customGenerator.GenerateAsync(Fixtures.Input(), cancellationToken: Token);
        Assert.False(((JsonElement)first.RawRepresentation!).TryGetProperty("future", out _));
        Assert.Equal("original-only", ((JsonElement)second.RawRepresentation!).GetProperty("future").GetString());
        Assert.False(((JsonElement)first.Answers[0].RawRepresentation!).TryGetProperty("future", out _));
        Assert.Equal("original-answer-only", ((JsonElement)second.Answers[0].RawRepresentation!).GetProperty("future").GetString());
    }

    [Fact]
    public async Task Int64WireDomainIsWiderThanSdkInt32Counters()
    {
        var root = Fixtures.Native();
        root["usage"]!["input_tokens"] = 2147483648L;
        using var sdkHandler = new ScriptedHandler(_ => Fixtures.Response(root.ToJsonString()));
        using var customHandler = new ScriptedHandler(_ => Fixtures.Response(root.ToJsonString()));
        using var sdkHttp = Fixtures.Http(sdkHandler);
        using var customHttp = Fixtures.Http(customHandler);
        using var sdk = new OllamaApiClient(sdkHttp, Fixtures.Model);
        using var sdkGenerator = new OllamaSharpDecisionGenerator(sdk);
        using var customGenerator = new OllamaDecisionGenerator(customHttp, new() { ModelId = Fixtures.Model });
        await Assert.ThrowsAsync<DecisionProtocolException>(() => sdkGenerator.GenerateAsync(Fixtures.Input(), cancellationToken: Token));
        var result = await customGenerator.GenerateAsync(Fixtures.Input(), cancellationToken: Token);
        Assert.Equal(2147483648L, result.Usage!.InputTokenCount);
        Assert.Equal(2147483655L, result.Usage.TotalTokenCount);
    }

    [Fact]
    public async Task SdkFinalBodyHasNoAdapterByteCapWhileCustomProviderRetainsItsCap()
    {
        var input = new DecisionInput(JsonSerializer.SerializeToElement(new string('x', 65536)), Fixtures.Input().Questions);
        using var sdkHandler = new ScriptedHandler(_ => Fixtures.Response(Fixtures.Native(input).ToJsonString()));
        using var customHandler = new ScriptedHandler(_ => throw new InvalidOperationException("Custom provider must reject locally."));
        using var sdkHttp = Fixtures.Http(sdkHandler);
        using var customHttp = Fixtures.Http(customHandler);
        using var sdk = new OllamaApiClient(sdkHttp, Fixtures.Model);
        using var sdkGenerator = new OllamaSharpDecisionGenerator(sdk);
        using var customGenerator = new OllamaDecisionGenerator(customHttp, new() { ModelId = Fixtures.Model });
        await sdkGenerator.GenerateAsync(input, cancellationToken: Token);
        await Assert.ThrowsAsync<ArgumentException>(() => customGenerator.GenerateAsync(input, cancellationToken: Token));
        Assert.True(Assert.Single(sdkHandler.Requests).Bytes.Length > 65536);
        Assert.Empty(customHandler.Requests);
    }
}
