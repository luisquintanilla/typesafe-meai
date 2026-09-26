using System.Text.Json.Nodes;
using Microsoft.Extensions.AI;
using TypeSafe.MEAI.Tests.Fixtures;
using Xunit;

namespace TypeSafe.MEAI.Tests.Protocol;

[Trait("Stage", "A")]
public sealed class NativeResponseContractTests
{
    [Fact]
    public void Decode_PreservesEnvelopeAndTypedAnswers()
    {
        var response = TypeSafeResponse.Parse(NativeFixture.Envelope, NativeFixture.Questions());
        NativeFixture.AssertNative(new JevResultContent(System.Text.Encoding.UTF8.GetBytes(NativeFixture.Envelope)));
        Assert.True(System.Text.Json.JsonElement.DeepEquals(NativeFixture.Json(NativeFixture.Envelope), response.Json));
        Assert.Throws<TypeSafeProtocolException>(() => response.GetAnswer<ChoiceAnswer>("follow-up"));
        Assert.Throws<TypeSafeProtocolException>(() => response.GetAnswer<NoulAnswer>("FOLLOW-UP"));
        Assert.Throws<TypeSafeProtocolException>(() => response.GetAnswer<NoulAnswer>("missing"));
    }

    [Theory]
    [InlineData("0.6000000", true)]
    [InlineData("0.6000005", true)]
    [InlineData("0.5999995", true)]
    [InlineData("0.6000010", true)]
    [InlineData("0.5999990", true)]
    [InlineData("0.6000011", false)]
    [InlineData("0.5999989", false)]
    public async Task Decode_EnforcesAbsoluteSumToleranceWithoutRepair(string billing, bool accepted)
    {
        var json = NativeFixture.Envelope.Replace("\"billing\":0.60", $"\"billing\":{billing}", StringComparison.Ordinal);
        using var handler = new RecordingHandler((_, _, _) => Task.FromResult(NativeFixture.Response(json)));
        using var services = NativeFixture.Services(handler, cache: true);
        var client = services.Client();
        if (!accepted)
        {
            var error = await Assert.ThrowsAsync<TypeSafeProtocolException>(() =>
                client.GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), TestContext.Current.CancellationToken));
            Assert.Contains("tolerance", error.Message);
            Assert.Equal(1, handler.Sends);
            return;
        }
        for (int call = 0; call < 2; call++)
        {
            var response = await client.GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), TestContext.Current.CancellationToken);
            Assert.Null(response.RawRepresentation);
            var choice = Assert.IsType<ChoiceAnswer>(Assert.Single(response.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>()).Answers["destination"]);
            Assert.Equal(double.Parse(billing, System.Globalization.CultureInfo.InvariantCulture), choice.Probabilities.GetProperty("billing").GetDouble());
            Assert.Equal(0.30, choice.Probabilities.GetProperty("technical").GetDouble());
            Assert.Equal(0.10, choice.Probabilities.GetProperty("general").GetDouble());
            Assert.Equal(0.91, choice.Confidence);
        }
        Assert.Equal(1, handler.Sends);
    }

    [Theory]
    [InlineData(0, 0)]
    [InlineData(1, 2)]
    public void Decode_AcceptsUnitAndScoreEndpoints(double probability, double score)
    {
        var root = JsonNode.Parse(NativeFixture.Envelope)!;
        root["answers"]!["follow-up"]!["noul"] = probability;
        root["answers"]!["quality"]!["score"] = score;
        root["answers"]!["quality"]!["confidence"] = probability;
        root["answers"]!["destination"]!["confidence"] = probability;
        root["answers"]!["destination"]!["probabilities"] = JsonNode.Parse("""{"billing":1,"technical":0,"general":0}""");
        var response = TypeSafeResponse.Parse(root.ToJsonString(), NativeFixture.Questions());
        Assert.Equal(probability, response.GetAnswer<NoulAnswer>("follow-up").Noul);
        Assert.Equal(score, response.GetAnswer<ScoreAnswer>("quality").Score);
        Assert.Equal(probability, response.GetAnswer<ChoiceAnswer>("destination").Confidence);
        Assert.Equal(0, response.GetAnswer<ChoiceAnswer>("destination").Probabilities.GetProperty("general").GetDouble());
    }

    public static TheoryData<string> InvalidResponses => new(
        "malformed", "missing-model", "empty-model", "missing-usage", "missing-answers", "missing-answer", "extra-answer",
        "unknown-id", "case-id", "duplicate-answer", "duplicate-envelope", "duplicate-probability", "wrong-type", "unknown-type",
        "noul-low", "noul-high", "noul-infinite", "noul-nan", "choice-confidence-low", "choice-confidence-high",
        "score-confidence-low", "score-confidence-high", "choice-confidence-infinite", "score-confidence-infinite",
        "probability-low", "probability-high", "probability-infinite", "score-probability-low", "score-probability-high", "score-probability-infinite",
        "invalid-label", "missing-label", "extra-label", "score-low", "score-high", "score-infinite",
        "invalid-index", "invalid-legend-index", "invalid-legend-entry", "negative-input", "negative-output",
        "fractional-tokens", "overflow-tokens", "mismatched-total", "negative-total", "null-answer", "string-noul");

    internal static string Invalid(string variant)
    {
        var root = JsonNode.Parse(NativeFixture.Envelope)!;
        var answers = root["answers"]!.AsObject();
        var choice = answers["destination"]!;
        var score = answers["quality"]!;
        switch (variant)
        {
            case "malformed": return "{";
            case "missing-model": root.AsObject().Remove("model"); break;
            case "empty-model": root["model"] = ""; break;
            case "missing-usage": root.AsObject().Remove("usage"); break;
            case "missing-answers": root.AsObject().Remove("answers"); break;
            case "missing-answer": answers.Remove("follow-up"); break;
            case "extra-answer": answers["extra"] = answers["follow-up"]!.DeepClone(); break;
            case "unknown-id": answers["unknown"] = answers["follow-up"]!.DeepClone(); answers.Remove("follow-up"); break;
            case "case-id": answers["Follow-up"] = answers["follow-up"]!.DeepClone(); answers.Remove("follow-up"); break;
            case "duplicate-answer": return NativeFixture.Envelope.Replace("\"follow-up\":", "\"follow-up\":{\"type\":\"noul\",\"noul\":0.2},\"follow-up\":", StringComparison.Ordinal);
            case "duplicate-envelope": return NativeFixture.Envelope.Replace("\"model\":", "\"model\":\"other\",\"model\":", StringComparison.Ordinal);
            case "duplicate-probability": return NativeFixture.Envelope.Replace("\"billing\":0.60", "\"billing\":0.60,\"billing\":0.60", StringComparison.Ordinal);
            case "wrong-type": answers["follow-up"]!["type"] = "choice"; break;
            case "unknown-type": answers["follow-up"]!["type"] = "future"; break;
            case "noul-low": answers["follow-up"]!["noul"] = -0.001; break;
            case "noul-high": answers["follow-up"]!["noul"] = 1.001; break;
            case "noul-infinite": return NativeFixture.Envelope.Replace("\"noul\":0.82", "\"noul\":1e400", StringComparison.Ordinal);
            case "noul-nan": return NativeFixture.Envelope.Replace("\"noul\":0.82", "\"noul\":NaN", StringComparison.Ordinal);
            case "choice-confidence-low": choice["confidence"] = -0.001; break;
            case "choice-confidence-high": choice["confidence"] = 1.001; break;
            case "score-confidence-low": score["confidence"] = -0.001; break;
            case "score-confidence-high": score["confidence"] = 1.001; break;
            case "choice-confidence-infinite": return NativeFixture.Envelope.Replace("\"confidence\":0.91", "\"confidence\":1e400", StringComparison.Ordinal);
            case "score-confidence-infinite": return NativeFixture.Envelope.Replace("\"confidence\":0.76", "\"confidence\":1e400", StringComparison.Ordinal);
            case "probability-low": choice["probabilities"]!["billing"] = -0.1; break;
            case "probability-high": choice["probabilities"]!["billing"] = 1.1; break;
            case "probability-infinite": return NativeFixture.Envelope.Replace("\"billing\":0.60", "\"billing\":1e400", StringComparison.Ordinal);
            case "score-probability-low": score["probabilities"]!["0"] = -0.1; break;
            case "score-probability-high": score["probabilities"]!["0"] = 1.1; break;
            case "score-probability-infinite": return NativeFixture.Envelope.Replace("\"0\":0.20", "\"0\":1e400", StringComparison.Ordinal);
            case "invalid-label": choice["choice"] = "unknown"; break;
            case "missing-label": choice["probabilities"]!.AsObject().Remove("general"); break;
            case "extra-label": choice["probabilities"]!["unknown"] = 0; break;
            case "score-low": score["score"] = -0.001; break;
            case "score-high": score["score"] = 2.001; break;
            case "score-infinite": return NativeFixture.Envelope.Replace("\"score\":1.25", "\"score\":1e400", StringComparison.Ordinal);
            case "invalid-index": score["probabilities"]!["3"] = 0.3; score["probabilities"]!.AsObject().Remove("2"); break;
            case "invalid-legend-index": score["legend"]!["3"] = "unknown"; score["legend"]!.AsObject().Remove("2"); break;
            case "invalid-legend-entry": score["legend"]!["2"] = true; break;
            case "negative-input": root["usage"]!["input_tokens"] = -1; break;
            case "negative-output": root["usage"]!["output_tokens"] = -1; break;
            case "fractional-tokens": root["usage"]!["input_tokens"] = 1.5; break;
            case "overflow-tokens": root["usage"]!["input_tokens"] = long.MaxValue; break;
            case "mismatched-total": root["usage"]!["total_tokens"] = 19; break;
            case "negative-total": root["usage"]!["total_tokens"] = -1; break;
            case "null-answer": answers["follow-up"] = null; break;
            case "string-noul": answers["follow-up"]!["noul"] = "0.82"; break;
            default: throw new ArgumentOutOfRangeException(nameof(variant));
        }
        return root.ToJsonString();
    }

    [Theory]
    [MemberData(nameof(InvalidResponses))]
    public async Task GetResponse_RejectsCorrelationAndProtocolViolations(string variant)
    {
        using var handler = new RecordingHandler((_, _, _) => Task.FromResult(NativeFixture.Response(Invalid(variant))));
        using var http = new HttpClient(handler);
        using var client = new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key);
        var error = await Assert.ThrowsAsync<TypeSafeProtocolException>(() =>
            client.GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), TestContext.Current.CancellationToken));
        Assert.Equal(1, handler.Sends);
        Assert.DoesNotContain(NativeFixture.Key, error.ToString());
        Assert.DoesNotContain("noul-extra", error.ToString());
    }

    [Theory]
    [InlineData(11, 7, true, 18)]
    [InlineData(0, 0, true, 0)]
    [InlineData(0, 0, false, 0)]
    public async Task GetResponse_PreservesExplicitAndZeroUsageWithoutInventedTokens(long input, long output, bool totalPresent, long total)
    {
        var root = JsonNode.Parse(NativeFixture.Envelope)!;
        root["usage"]!["input_tokens"] = input;
        root["usage"]!["output_tokens"] = output;
        if (totalPresent) root["usage"]!["total_tokens"] = total;
        using var handler = new RecordingHandler((_, _, _) => Task.FromResult(NativeFixture.Response(root.ToJsonString())));
        using var services = NativeFixture.Services(handler, cache: true);
        for (int call = 0; call < 2; call++)
        {
            var response = await services.Client().GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), TestContext.Current.CancellationToken);
            var native = Assert.Single(response.Messages.SelectMany(m => m.Contents).OfType<JevResultContent>());
            Assert.Equal(input, response.Usage!.InputTokenCount);
            Assert.Equal(output, response.Usage.OutputTokenCount);
            Assert.Equal(total, response.Usage.TotalTokenCount);
            Assert.Equal(total, native.TotalTokens);
            Assert.Equal(totalPresent, native.Json.GetProperty("usage").TryGetProperty("total_tokens", out _));
            Assert.Equal(1.25, Assert.IsType<ScoreAnswer>(native.Answers["quality"]).Score);
        }
        Assert.Equal(1, handler.Sends);
    }
}
