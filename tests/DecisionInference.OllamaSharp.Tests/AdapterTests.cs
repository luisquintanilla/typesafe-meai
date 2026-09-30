using System.Globalization;
using System.Net;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using Microsoft.Extensions.AI;
using OllamaSharp;
using OllamaSharp.Models;
using OllamaSharp.Models.Exceptions;
using NativeRequest = OllamaSharp.Models.SystemOneRequest;

namespace DecisionInference.OllamaSharp.Tests;

public sealed class AdapterTests
{
    private static CancellationToken Token => TestContext.Current.CancellationToken;

    [Fact]
    public async Task AllKinds_OrdinaryBorrowedSdkOneNativePostAndOwnedHighPrecisionResults()
    {
        using var handler = new ScriptedHandler(_ => Fixtures.Response(Fixtures.Native().ToJsonString()));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http, Fixtures.Model);
        using IDecisionGenerator generator = new OllamaSharpDecisionGenerator(sdk);
        var result = await generator.GenerateAsync(Fixtures.Input(), cancellationToken: Token);
        var sent = Assert.Single(handler.Requests);
        Assert.Equal(HttpMethod.Post, sent.Method);
        Assert.Equal("/v1/systemone", sent.Uri.AbsolutePath);
        var questions = sent.Body.GetProperty("questions");
        Assert.Equal(["q0", "q1", "q2"], questions.EnumerateObject().Select(p => p.Name));
        Assert.Equal(["Account access?", "Choose team", "Rate urgency"], questions.EnumerateObject()
            .Select(p => p.Value.GetProperty("instructions").GetString()));
        Assert.Equal("Access problem", questions.GetProperty("q0").GetProperty("criteria").GetProperty("true").GetString());
        Assert.Equal("No access problem", questions.GetProperty("q0").GetProperty("criteria").GetProperty("false").GetString());
        Assert.Equal(["0", "1"], questions.GetProperty("q1").GetProperty("criteria").EnumerateObject().Select(p => p.Name));
        Assert.Equal(["Account", "Billing"], questions.GetProperty("q1").GetProperty("criteria").EnumerateObject().Select(p => p.Value.GetString()));
        Assert.Equal(["Routine", "Blocked", "Critical"], questions.GetProperty("q2").GetProperty("criteria").EnumerateArray().Select(p => p.GetString()));
        Assert.Equal(Fixtures.Ticket, sent.Body.GetProperty("state").GetString());
        Assert.Equal(Fixtures.Model, sent.Body.GetProperty("model").GetString());
        Assert.Empty(sent.Body.GetProperty("CustomHeaders").EnumerateObject());
        Assert.Equal(["account-access", "support-team", "urgency"], result.Answers.Select(a => a.QuestionId));
        Assert.All(result.Answers, a => Assert.Equal(DecisionPrecision.HighPrecision, a.Precision));
        Assert.Equal(.8807970779778823, result.GetAnswer<BinaryDecisionAnswer>("account-access").TrueProbability);
        var choice = result.GetAnswer<ChoiceDecisionAnswer>("support-team");
        Assert.Equal("billing", choice.SelectedCandidateId);
        Assert.Equal(["account", "billing"], choice.Probabilities.Keys);
        Assert.Equal(.11920292202211755, choice.Probabilities["account"]);
        Assert.Equal(.8807970779778823, choice.Probabilities["billing"]);
        var score = result.GetAnswer<ScoreDecisionAnswer>("urgency");
        Assert.Equal(1.5752103826044412, score.NativeScore!.Value, 14);
        Assert.Equal(score.NativeScore.Value, score.ExpectedIndex);
        Assert.Equal(1836, result.Usage!.InputTokenCount);
        Assert.Equal(7, result.Usage.OutputTokenCount);
        Assert.Equal(1843, result.Usage.TotalTokenCount);
        Assert.Null(result.Usage.ReasoningTokenCount);
        Assert.Equal("sdk-dto-snapshot", result.AdditionalProperties!["ollamasharp.rawRepresentationKind"]);
        Assert.All(result.Answers, a => Assert.IsType<JsonElement>(a.RawRepresentation));
        Assert.IsType<JsonElement>(result.RawRepresentation);
        Assert.Equal(Fixtures.Model, sdk.SelectedModel);
    }

    [Fact]
    public async Task OpaqueIdsCandidateOrderAndBinaryDescriptionOmissionRemainExact()
    {
        var input = new DecisionInput(JsonSerializer.SerializeToElement("state"),
        [
            new BinaryDecisionQuestion("é/ID", "assess", falseDescription: "negative"),
            new BinaryDecisionQuestion("É/id", "assess"),
            new ChoiceDecisionQuestion("Case", "choose", [new("A/é", "first"), new("a/É", "second")])
        ]);
        using var handler = new ScriptedHandler(_ => Fixtures.Response(Fixtures.Native(input, uniform: true).ToJsonString()));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http, Fixtures.Model);
        using var generator = new OllamaSharpDecisionGenerator(sdk);
        var result = await generator.GenerateAsync(input, cancellationToken: Token);
        Assert.Equal(["é/ID", "É/id", "Case"], result.Answers.Select(a => a.QuestionId));
        var choice = result.GetAnswer<ChoiceDecisionAnswer>("Case");
        Assert.Equal("A/é", choice.SelectedCandidateId);
        Assert.Equal(["A/é", "a/É"], choice.Probabilities.Keys);
        var body = Assert.Single(handler.Requests).Body.GetProperty("questions");
        Assert.False(body.GetProperty("q0").GetProperty("criteria").TryGetProperty("true", out _));
        Assert.Equal("negative", body.GetProperty("q0").GetProperty("criteria").GetProperty("false").GetString());
        Assert.False(body.GetProperty("q1").TryGetProperty("criteria", out _));
        var aliases = Assert.IsAssignableFrom<IReadOnlyDictionary<string, string>>(result.AdditionalProperties!["systemone.questionIds"]);
        Assert.Equal("é/ID", aliases["q0"]);
        Assert.Equal("É/id", aliases["q1"]);
    }

    [Fact]
    public async Task InteractingMaximum64QuestionsChoice26AndScore26Accepted65Rejected()
    {
        var input = new DecisionInput(JsonSerializer.SerializeToElement("state"),
        [
            .. Enumerable.Range(0, 62).Select(i => (DecisionQuestion)new BinaryDecisionQuestion($"b{i}", "assess")),
            new ChoiceDecisionQuestion("choice", "choose", Enumerable.Range(0, 26).Select(i => new DecisionCandidate($"c{i}", $"candidate{i}"))),
            new ScoreDecisionQuestion("score", "rate", Enumerable.Range(0, 26).Select(i => $"level{i}"))
        ]);
        using var handler = new ScriptedHandler(_ => Fixtures.Response(Fixtures.Native(input, uniform: true).ToJsonString()));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http, Fixtures.Model);
        using var generator = new OllamaSharpDecisionGenerator(sdk);
        var result = await generator.GenerateAsync(input, cancellationToken: Token);
        Assert.Equal(64, result.Answers.Count);
        Assert.Equal("c0", result.GetAnswer<ChoiceDecisionAnswer>("choice").SelectedCandidateId);
        Assert.Equal(26, result.GetAnswer<ScoreDecisionAnswer>("score").Probabilities.Count);
        var sent = Assert.Single(handler.Requests).Body.GetProperty("questions");
        Assert.Equal(64, sent.EnumerateObject().Count());
        Assert.Equal(26, sent.GetProperty("q62").GetProperty("criteria").EnumerateObject().Count());
        Assert.Equal(26, sent.GetProperty("q63").GetProperty("criteria").GetArrayLength());
        var tooMany = new DecisionInput(input.State, input.Questions.Append(new BinaryDecisionQuestion("extra", "assess")));
        await Assert.ThrowsAsync<ArgumentException>(() => generator.GenerateAsync(tooMany, cancellationToken: Token));
        Assert.Single(handler.Requests);
    }

    [Fact]
    public async Task ChoiceAndScoreEveryCount2Through26PreserveDescriptionsAndUniformTies()
    {
        for (int count = 2; count <= 26; count++)
        {
            var input = new DecisionInput(JsonSerializer.SerializeToElement("state"),
            [
                new ChoiceDecisionQuestion("c", "choose", Enumerable.Range(0, count).Select(i => new DecisionCandidate($"id{i}", $"description{i}"))),
                new ScoreDecisionQuestion("s", "rate", Enumerable.Range(0, count).Select(i => $"level{i}"))
            ]);
            using var handler = new ScriptedHandler(_ => Fixtures.Response(Fixtures.Native(input, uniform: true).ToJsonString()));
            using var http = Fixtures.Http(handler);
            using var sdk = new OllamaApiClient(http, Fixtures.Model);
            using var generator = new OllamaSharpDecisionGenerator(sdk);
            var result = await generator.GenerateAsync(input, cancellationToken: Token);
            Assert.Equal("id0", result.GetAnswer<ChoiceDecisionAnswer>("c").SelectedCandidateId);
            var score = result.GetAnswer<ScoreDecisionAnswer>("s");
            Assert.Equal(count, score.Probabilities.Count);
            Assert.All(score.Probabilities, p => Assert.Equal(1d / count, p));
            Assert.Equal((count - 1) / 2d, score.NativeScore!.Value, 12);
            var questions = Assert.Single(handler.Requests).Body.GetProperty("questions");
            Assert.Equal(Enumerable.Range(0, count).Select(i => $"description{i}"),
                questions.GetProperty("q0").GetProperty("criteria").EnumerateObject().Select(p => p.Value.GetString()));
            Assert.Equal(Enumerable.Range(0, count).Select(i => $"level{i}"),
                questions.GetProperty("q1").GetProperty("criteria").EnumerateArray().Select(p => p.GetString()));
        }
    }

    [Theory]
    [InlineData("choice", 1)]
    [InlineData("choice", 27)]
    [InlineData("score", 27)]
    public async Task UnsupportedCountsFailBeforeSdk(string kind, int count)
    {
        DecisionQuestion question = kind == "choice"
            ? new ChoiceDecisionQuestion("q", "choose", Enumerable.Range(0, count).Select(i => new DecisionCandidate($"c{i}", $"candidate{i}")))
            : new ScoreDecisionQuestion("q", "rate", Enumerable.Range(0, count).Select(i => $"level{i}"));
        using var handler = new ScriptedHandler(_ => throw new InvalidOperationException("Must not invoke."));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http, Fixtures.Model);
        using var generator = new OllamaSharpDecisionGenerator(sdk);
        await Assert.ThrowsAsync<ArgumentException>(() => generator.GenerateAsync(new(Fixtures.Input().State, [question]), cancellationToken: Token));
        Assert.Empty(handler.Requests);
    }

    [Theory]
    [InlineData("null")]
    [InlineData("true")]
    [InlineData("42")]
    [InlineData("\"\"")]
    [InlineData("\" \\t\\n\"")]
    public async Task InvalidStateFailsBeforeSdk(string state)
    {
        using var handler = new ScriptedHandler(_ => throw new InvalidOperationException("Must not invoke."));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http, Fixtures.Model);
        using var generator = new OllamaSharpDecisionGenerator(sdk);
        var input = new DecisionInput(JsonSerializer.Deserialize<JsonElement>(state), Fixtures.Input().Questions);
        await Assert.ThrowsAsync<ArgumentException>(() => generator.GenerateAsync(input, cancellationToken: Token));
        Assert.Empty(handler.Requests);
    }

    [Fact]
    public async Task MissingInstructionsCannotBeReplacedByBinaryDescriptions()
    {
        using var handler = new ScriptedHandler(_ => throw new InvalidOperationException("Must not invoke."));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http, Fixtures.Model);
        using var generator = new OllamaSharpDecisionGenerator(sdk);
        foreach (DecisionQuestion question in new DecisionQuestion[]
        {
            new BinaryDecisionQuestion("b", null, "yes", "no"),
            new ChoiceDecisionQuestion("c", " ", [new("a", "a"), new("b", "b")]),
            new ScoreDecisionQuestion("s", " ", ["low", "high"])
        })
            await Assert.ThrowsAsync<ArgumentException>(() => generator.GenerateAsync(new(Fixtures.Input().State, [question]), cancellationToken: Token));
        Assert.Empty(handler.Requests);
    }

    [Theory]
    [InlineData("\"state\"")]
    [InlineData("{\"nested\":[\"café <tag>\",{}]}")]
    [InlineData("[{},\"café <tag>\"]")]
    [InlineData("{}")]
    [InlineData("[]")]
    public async Task StateOwnsDisposedDocumentAndNeverDoubleEncodes(string source)
    {
        DecisionInput input;
        using (var document = JsonDocument.Parse(source)) input = new(document.RootElement, Fixtures.Input().Questions);
        using var handler = new ScriptedHandler(_ => Fixtures.Response(Fixtures.Native(input).ToJsonString()));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http, Fixtures.Model);
        using var generator = new OllamaSharpDecisionGenerator(sdk);
        await generator.GenerateAsync(input, cancellationToken: Token);
        var state = Assert.Single(handler.Requests).Body.GetProperty("state");
        Assert.Equal(input.State.ValueKind, state.ValueKind);
        Assert.True(JsonElement.DeepEquals(input.State, state));
    }

    [Theory]
    [InlineData(null, null)]
    [InlineData(0, "0s")]
    [InlineData(60, "60s")]
    [InlineData(-1, "-1s")]
    [InlineData(int.MaxValue, "2147483647s")]
    [InlineData(int.MinValue, "-2147483648s")]
    public async Task ResidencyInvariantSignedSecondsAndCallerOptionsUnchanged(int? seconds, string? expected)
    {
        var previous = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = new CultureInfo("ar-SA");
            using var handler = new ScriptedHandler(_ => Fixtures.Response(Fixtures.Native().ToJsonString()));
            using var http = Fixtures.Http(handler);
            using var sdk = new OllamaApiClient(http, Fixtures.Model);
            using var generator = new OllamaSharpDecisionGenerator(sdk);
            var options = new OllamaSharpGenerationOptions { KeepAliveSeconds = seconds, AdditionalProperties = new() };
            await generator.GenerateAsync(Fixtures.Input(), options, Token);
            var body = Assert.Single(handler.Requests).Body;
            if (expected is null) Assert.False(body.TryGetProperty("keep_alive", out _));
            else Assert.Equal(expected, body.GetProperty("keep_alive").GetString());
            Assert.Equal(seconds, options.KeepAliveSeconds);
            Assert.Null(options.ModelId);
            Assert.Empty(options.AdditionalProperties);
            Assert.Equal(Fixtures.Model, sdk.SelectedModel);
        }
        finally { CultureInfo.CurrentCulture = previous; }
    }

    [Fact]
    public async Task OptionsCloneBaseOptionsForeignOptionsAndExtensions()
    {
        var value = new object();
        var original = new OllamaSharpGenerationOptions
        {
            ModelId = "override", KeepAliveSeconds = -1, AdditionalProperties = new() { ["extension"] = value }
        };
        var copy = original.Clone();
        Assert.IsType<OllamaSharpGenerationOptions>(copy);
        Assert.NotSame(original, copy);
        Assert.NotSame(original.AdditionalProperties, copy.AdditionalProperties);
        Assert.Same(value, copy.AdditionalProperties!["extension"]);
        copy.AdditionalProperties.Clear();
        Assert.Single(original.AdditionalProperties!);
        Assert.Equal(-1, copy.KeepAliveSeconds);
        Assert.Equal("override", copy.ModelId);
        using var handler = new ScriptedHandler(r => Fixtures.Response(Fixtures.Native(model: r.Body.GetProperty("model").GetString()!).ToJsonString()));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http, Fixtures.Model);
        using var generator = new OllamaSharpDecisionGenerator(sdk);
        await generator.GenerateAsync(Fixtures.Input(), new DecisionGenerationOptions { ModelId = "base-override" }, Token);
        Assert.Equal("base-override", Assert.Single(handler.Requests).Body.GetProperty("model").GetString());
        await Assert.ThrowsAsync<NotSupportedException>(() => generator.GenerateAsync(Fixtures.Input(), original, Token));
        await Assert.ThrowsAsync<NotSupportedException>(() => generator.GenerateAsync(Fixtures.Input(),
            new DecisionGenerationOptions { AdditionalProperties = new() { ["x"] = 1 } }, Token));
        await Assert.ThrowsAsync<NotSupportedException>(() => generator.GenerateAsync(Fixtures.Input(), new ForeignOptions(), Token));
        Assert.Single(handler.Requests);
    }

    [Fact]
    public async Task CapturedDefaultOverridesAreNonstickyAndMetadataUriIsSnapshot()
    {
        using var handler = new ScriptedHandler(r => Fixtures.Response(Fixtures.Native(model: r.Body.GetProperty("model").GetString()!).ToJsonString()));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http, Fixtures.Model);
        using var generator = new OllamaSharpDecisionGenerator(sdk);
        var metadata = generator.GetRequiredService<DecisionGeneratorMetadata>();
        sdk.SelectedModel = "later-sdk-default";
        http.BaseAddress = new("https://caller-owned.invalid/");
        await generator.GenerateAsync(Fixtures.Input(), new() { ModelId = "override" }, Token);
        await generator.GenerateAsync(Fixtures.Input(), cancellationToken: Token);
        Assert.Equal(["override", Fixtures.Model], handler.Requests.Select(r => r.Body.GetProperty("model").GetString()));
        Assert.Equal("later-sdk-default", sdk.SelectedModel);
        Assert.Equal(Fixtures.Model, metadata.DefaultModelId);
        Assert.Equal(new Uri("http://127.0.0.1:11434"), metadata.ProviderUri);
        Assert.All(handler.Requests, r => Assert.Equal("caller-owned.invalid", r.Uri.Host));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    public async Task MissingCapturedModelRequiresPerCallModel(string? initial)
    {
        using var handler = new ScriptedHandler(r => Fixtures.Response(Fixtures.Native(model: r.Body.GetProperty("model").GetString()!).ToJsonString()));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http, initial!);
        using var generator = new OllamaSharpDecisionGenerator(sdk);
        Assert.Null(generator.GetRequiredService<DecisionGeneratorMetadata>().DefaultModelId);
        await Assert.ThrowsAsync<ArgumentNullException>(() => generator.GenerateAsync(Fixtures.Input(), cancellationToken: Token));
        Assert.Empty(handler.Requests);
        await generator.GenerateAsync(Fixtures.Input(), new() { ModelId = "per-call" }, Token);
        Assert.Equal("per-call", Assert.Single(handler.Requests).Body.GetProperty("model").GetString());
    }

    [Theory]
    [InlineData("")]
    [InlineData(" ")]
    [InlineData("\t")]
    public async Task BlankEffectiveModelRejectedBeforeSdk(string model)
    {
        using var handler = new ScriptedHandler(_ => throw new InvalidOperationException("Must not invoke."));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http, Fixtures.Model);
        using var generator = new OllamaSharpDecisionGenerator(sdk);
        await Assert.ThrowsAsync<ArgumentException>(() => generator.GenerateAsync(Fixtures.Input(), new() { ModelId = model }, Token));
        sdk.SelectedModel = model;
        if (model.Length > 0) Assert.Throws<ArgumentException>(() => new OllamaSharpDecisionGenerator(sdk));
        Assert.Empty(handler.Requests);
    }

    [Fact]
    public async Task ConcurrentOverridesUseFreshRequestsAndNeverMutateSdkModel()
    {
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        int calls = 0;
        using var handler = new ScriptedHandler(async (request, token) =>
        {
            if (Interlocked.Increment(ref calls) == 2) entered.SetResult();
            await release.Task.WaitAsync(token);
            return Fixtures.Response(Fixtures.Native(model: request.Body.GetProperty("model").GetString()!).ToJsonString());
        });
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http, Fixtures.Model);
        using var generator = new OllamaSharpDecisionGenerator(sdk);
        var first = generator.GenerateAsync(Fixtures.Input(), new() { ModelId = "one" }, Token);
        var second = generator.GenerateAsync(Fixtures.Input(), new() { ModelId = "two" }, Token);
        await entered.Task.WaitAsync(Token);
        Assert.Equal(Fixtures.Model, sdk.SelectedModel);
        Assert.Equal(["one", "two"], handler.Requests.Select(r => r.Body.GetProperty("model").GetString()));
        release.SetResult();
        var results = await Task.WhenAll(first, second);
        Assert.Equal(["one", "two"], results.Select(r => r.ModelId));
        Assert.Equal(Fixtures.Model, sdk.SelectedModel);
    }

    public static TheoryData<string> InvalidResponses => new()
    {
        "model", "missing-answer", "extra-answer", "unknown-alias", "wrong-kind", "answers-null", "answer-null",
        "usage-null", "choice-null", "probabilities-null", "incomplete-choice", "unknown-candidate", "wrong-winner",
        "tie-wrong-winner", "negative-probability", "over-one", "mass", "confidence", "ordinal-keys",
        "legend-null", "legend-keys", "legend-text", "legend-kind", "native-score", "negative-input", "negative-output"
    };

    [Theory]
    [MemberData(nameof(InvalidResponses))]
    public async Task InvalidKnownNativeInformationRejectsEntireBatch(string mutation)
    {
        var root = Fixtures.Native();
        var answers = root["answers"]!.AsObject();
        var choice = answers["q1"]!.AsObject();
        var score = answers["q2"]!.AsObject();
        switch (mutation)
        {
            case "model": root["model"] = "different"; break;
            case "missing-answer": answers.Remove("q0"); break;
            case "extra-answer": answers["q3"] = answers["q0"]!.DeepClone(); break;
            case "unknown-alias": answers["opaque"] = answers["q0"]!.DeepClone(); answers.Remove("q0"); break;
            case "wrong-kind": answers["q0"] = choice.DeepClone(); break;
            case "answers-null": root["answers"] = null; break;
            case "answer-null": answers["q0"] = null; break;
            case "usage-null": root["usage"] = null; break;
            case "choice-null": choice["choice"] = null; break;
            case "probabilities-null": choice["probabilities"] = null; break;
            case "incomplete-choice": choice["probabilities"]!.AsObject().Remove("0"); break;
            case "unknown-candidate": choice["choice"] = "2"; break;
            case "wrong-winner": choice["choice"] = "0"; break;
            case "tie-wrong-winner":
                choice["probabilities"] = new JsonObject { ["0"] = .5, ["1"] = .5 }; choice["choice"] = "1"; break;
            case "negative-probability": choice["probabilities"]!["0"] = -.1; break;
            case "over-one": answers["q0"]!["noul"] = 1.1; break;
            case "mass": choice["probabilities"]!["0"] = .1; break;
            case "confidence": choice["confidence"] = -1; break;
            case "ordinal-keys": score["probabilities"]!["3"] = 0; break;
            case "legend-null": score["legend"] = null; break;
            case "legend-keys": score["legend"]!.AsObject().Remove("0"); break;
            case "legend-text": score["legend"]!["0"] = "different"; break;
            case "legend-kind": score["legend"]!["0"] = 7; break;
            case "native-score": score["score"] = 0; break;
            case "negative-input": root["usage"]!["input_tokens"] = -1; break;
            case "negative-output": root["usage"]!["output_tokens"] = -1; break;
            default: throw new InvalidOperationException("Unknown test mutation.");
        }
        using var handler = new ScriptedHandler(_ => Fixtures.Response(root.ToJsonString()));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http, Fixtures.Model);
        using var generator = new OllamaSharpDecisionGenerator(sdk);
        var error = await Assert.ThrowsAsync<DecisionProtocolException>(() => generator.GenerateAsync(Fixtures.Input(), cancellationToken: Token));
        Assert.Null(error.InnerException);
        Assert.Single(handler.Requests);
    }

    [Theory]
    [InlineData("null")]
    [InlineData("{}")]
    [InlineData("invalid-sensitive-text")]
    [InlineData("{\"answers\":{\"q0\":{\"type\":\"future\"}}}")]
    [InlineData("{\"model\":\"tev1:0.8b\",\"usage\":{\"input_tokens\":2147483648,\"output_tokens\":0},\"answers\":{}}")]
    public async Task NullOrMalformedSdkResponseBecomesSanitizedProtocolFailure(string source)
    {
        using var handler = new ScriptedHandler(_ => Fixtures.Response(source));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http, Fixtures.Model);
        using var generator = new OllamaSharpDecisionGenerator(sdk);
        var error = await Assert.ThrowsAsync<DecisionProtocolException>(() => generator.GenerateAsync(Fixtures.Input(), cancellationToken: Token));
        Assert.Null(error.InnerException);
        Assert.DoesNotContain("sensitive", error.Message);
        Assert.Single(handler.Requests);
    }

    [Theory]
    [InlineData(0, 0, 0L)]
    [InlineData(int.MaxValue, int.MaxValue, 4294967294L)]
    [InlineData(int.MaxValue, 1, 2147483648L)]
    public async Task UsageCopiesNonnegativeSdkCountersAndSumsAsInt64(int input, int output, long total)
    {
        var body = Fixtures.Native();
        body["usage"]!["input_tokens"] = input;
        body["usage"]!["output_tokens"] = output;
        using var handler = new ScriptedHandler(_ => Fixtures.Response(body.ToJsonString()));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http, Fixtures.Model);
        using var generator = new OllamaSharpDecisionGenerator(sdk);
        var result = await generator.GenerateAsync(Fixtures.Input(), cancellationToken: Token);
        Assert.Equal(input, result.Usage!.InputTokenCount);
        Assert.Equal(output, result.Usage.OutputTokenCount);
        Assert.Equal(total, result.Usage.TotalTokenCount);
        Assert.Null(result.Usage.ReasoningTokenCount);
        Assert.Single(handler.Requests);
    }

    [Fact]
    public async Task ServicesAreExactAndBorrowingDoesNotDisposeStockSdkOrHttp()
    {
        using var handler = new ScriptedHandler(_ => Fixtures.Response(Fixtures.Native().ToJsonString()));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http, Fixtures.Model);
        var generator = new OllamaSharpDecisionGenerator(sdk);
        Assert.Same(generator, generator.GetService(typeof(IDecisionGenerator)));
        Assert.Same(generator, generator.GetService(typeof(OllamaSharpDecisionGenerator)));
        Assert.Same(sdk, generator.GetService(typeof(IOllamaApiClient)));
        Assert.Null(generator.GetService(typeof(OllamaApiClient)));
        Assert.Null(generator.GetService(typeof(object)));
        Assert.Null(generator.GetService(typeof(IChatClient)));
        Assert.Null(generator.GetService(typeof(IEmbeddingGenerator<string, Embedding<float>>)));
        Assert.Null(generator.GetService(typeof(IOllamaApiClient), "key"));
        Assert.Throws<ArgumentNullException>(() => generator.GetService(null!));
        var metadata = generator.GetRequiredService<DecisionGeneratorMetadata>();
        Assert.Equal("ollamasharp", metadata.ProviderName);
        Assert.Equal([DecisionKind.Binary, DecisionKind.Choice, DecisionKind.Score], metadata.SupportedKinds);
        Assert.Empty(handler.Requests);
        generator.Dispose();
        generator.Dispose();
        await Assert.ThrowsAsync<ObjectDisposedException>(() => generator.GenerateAsync(Fixtures.Input(), cancellationToken: Token));
        Assert.Throws<ObjectDisposedException>(() => generator.GetService(typeof(IOllamaApiClient)));
        await sdk.SystemOneAsync(new NativeRequest
        {
            Model = Fixtures.Model, State = Fixtures.Ticket,
            Questions = new() { ["q0"] = new SystemOneNoulQuestion { Instructions = "assess" } }
        }, Token);
        Assert.Single(handler.Requests);
    }

    [Fact]
    public async Task OwnedStockSdkStillDoesNotOwnBorrowedHttpClient()
    {
        using var handler = new ScriptedHandler(_ => Fixtures.Response(Fixtures.Native().ToJsonString()));
        using var http = Fixtures.Http(handler);
        var sdk = new OllamaApiClient(http, Fixtures.Model);
        var generator = new OllamaSharpDecisionGenerator(sdk, ownsClient: true);
        generator.Dispose();
        generator.Dispose();
        using var response = await http.PostAsync("/offline-http-lifetime", new StringContent("{}"), Token);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Single(handler.Requests);
    }

    [Fact]
    public void OwnershipRequiresDisposableAndOwnedClientDisposesExactlyOnce()
    {
        Assert.Throws<ArgumentNullException>(() => new OllamaSharpDecisionGenerator(null!));
        var nondisposable = DispatchProxy.Create<IOllamaApiClient, ContractProxy>();
        Assert.Throws<ArgumentException>(() => new OllamaSharpDecisionGenerator(nondisposable, ownsClient: true));
        var owned = DispatchProxy.Create<IOllamaApiClient, DisposableContractProxy>();
        var generator = new OllamaSharpDecisionGenerator(owned, ownsClient: true);
        generator.Dispose();
        generator.Dispose();
        Assert.Equal(1, ((DisposableContractProxy)owned).Disposals);
        var borrowed = DispatchProxy.Create<IOllamaApiClient, DisposableContractProxy>();
        new OllamaSharpDecisionGenerator(borrowed).Dispose();
        Assert.Equal(0, ((DisposableContractProxy)borrowed).Disposals);
    }

    [Fact]
    public async Task SdkDtoMutationCannotChangePortablePayloadSnapshotsOrUsage()
    {
        using var handler = new ScriptedHandler(_ => Fixtures.Response(Fixtures.Native().ToJsonString()));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http, Fixtures.Model);
        var dto = await sdk.SystemOneAsync(new NativeRequest
        {
            Model = Fixtures.Model, State = Fixtures.Ticket,
            Questions = new() { ["q0"] = new SystemOneNoulQuestion { Instructions = "assess" } }
        }, Token);
        var client = DispatchProxy.Create<IOllamaApiClient, ContractProxy>();
        ((ContractProxy)client).Response = dto;
        using var generator = new OllamaSharpDecisionGenerator(client);
        var result = await generator.GenerateAsync(Fixtures.Input(), cancellationToken: Token);
        string raw = Assert.IsType<JsonElement>(result.RawRepresentation).GetRawText();
        string answerRaw = Assert.IsType<JsonElement>(result.Answers[1].RawRepresentation).GetRawText();
        ((SystemOneChoiceAnswer)dto.Answers["q1"]).Probabilities["0"] = 1;
        ((SystemOneScoreAnswer)dto.Answers["q2"]).Legend["0"] = "mutated";
        dto.Answers.Clear();
        dto.Usage.InputTokens = 0;
        dto.Model = "mutated";
        Assert.Equal(.11920292202211755, result.GetAnswer<ChoiceDecisionAnswer>("support-team").Probabilities["account"]);
        Assert.Equal(Fixtures.Model, result.ModelId);
        Assert.Equal(1836, result.Usage!.InputTokenCount);
        Assert.Equal(raw, ((JsonElement)result.RawRepresentation!).GetRawText());
        Assert.Equal(answerRaw, ((JsonElement)result.Answers[1].RawRepresentation!).GetRawText());
        Assert.Single(handler.Requests);
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    public async Task NonfiniteSdkDtoValuesRejectWithoutParserDetails(double value)
    {
        var client = DispatchProxy.Create<IOllamaApiClient, ContractProxy>();
        ((ContractProxy)client).Response = new()
        {
            Model = Fixtures.Model, Usage = new(),
            Answers = new() { ["q0"] = new SystemOneNoulAnswer { Noul = value } }
        };
        using var generator = new OllamaSharpDecisionGenerator(client);
        var input = new DecisionInput(Fixtures.Input().State, [Fixtures.Input().Questions[0]]);
        var error = await Assert.ThrowsAsync<DecisionProtocolException>(() => generator.GenerateAsync(input, cancellationToken: Token));
        Assert.Null(error.InnerException);
    }

    [Fact]
    public async Task UnsupportedDtoSubclassIsNotSilentlyProjected()
    {
        var client = DispatchProxy.Create<IOllamaApiClient, ContractProxy>();
        ((ContractProxy)client).Response = new()
        {
            Model = Fixtures.Model, Usage = new(), Answers = new() { ["q0"] = new ExtendedNoulAnswer() }
        };
        using var generator = new OllamaSharpDecisionGenerator(client);
        await Assert.ThrowsAsync<DecisionProtocolException>(() => generator.GenerateAsync(Fixtures.Input(), cancellationToken: Token));
    }

    [Theory]
    [InlineData(400)]
    [InlineData(404)]
    [InlineData(500)]
    public async Task GenuineSdkHttpErrorsPropagateWithoutFabricatedAdapterErrors(int status)
    {
        using var handler = new ScriptedHandler(_ => Fixtures.Response("""{"error":"sensitive-fixture-diagnostic"}""", (HttpStatusCode)status));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http, Fixtures.Model);
        using var generator = new OllamaSharpDecisionGenerator(sdk);
        if (status == 400)
        {
            var error = await Assert.ThrowsAsync<OllamaException>(() => generator.GenerateAsync(Fixtures.Input(), cancellationToken: Token));
            Assert.Contains("sensitive-fixture-diagnostic", error.Message);
        }
        else
        {
            var error = await Assert.ThrowsAsync<HttpRequestException>(() => generator.GenerateAsync(Fixtures.Input(), cancellationToken: Token));
            Assert.Equal((HttpStatusCode)status, error.StatusCode);
        }
        Assert.Single(handler.Requests);
    }

    [Fact]
    public async Task GenuineHttpFailureRetainsIdentityAndNeverRetries()
    {
        var expected = new HttpRequestException("offline transport failure");
        using var handler = new ScriptedHandler(_ => throw expected);
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http, Fixtures.Model);
        using var generator = new OllamaSharpDecisionGenerator(sdk);
        var error = await Assert.ThrowsAsync<HttpRequestException>(() => generator.GenerateAsync(Fixtures.Input(), cancellationToken: Token));
        Assert.Same(expected, error);
        Assert.Single(handler.Requests);
    }

    [Fact]
    public async Task PreCancellationDoesNotInvokeAndInFlightCancellationFailsWholeBatch()
    {
        using var source = CancellationTokenSource.CreateLinkedTokenSource(Token);
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var handler = new ScriptedHandler(async (_, token) =>
        {
            entered.SetResult();
            await Task.Delay(Timeout.Infinite, token);
            throw new InvalidOperationException("Expected cancellation.");
        });
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http, Fixtures.Model);
        using var generator = new OllamaSharpDecisionGenerator(sdk);
        using var precancelled = new CancellationTokenSource();
        precancelled.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => generator.GenerateAsync(Fixtures.Input(), cancellationToken: precancelled.Token));
        Assert.Empty(handler.Requests);
        var pending = generator.GenerateAsync(Fixtures.Input(), cancellationToken: source.Token);
        await entered.Task.WaitAsync(Token);
        source.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => pending);
        Assert.Single(handler.Requests);
    }

    [Fact]
    public async Task CancellationDuringSdkDeserializationCannotReturnSuccessfulBatch()
    {
        using var source = CancellationTokenSource.CreateLinkedTokenSource(Token);
        using var handler = new ScriptedHandler(_ => Fixtures.Response(Fixtures.Native().ToJsonString()));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http, Fixtures.Model);
        var converter = new CancellingResponseConverter(source);
        sdk.IncomingJsonSerializerOptions.Converters.Add(converter);
        using var generator = new OllamaSharpDecisionGenerator(sdk);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => generator.GenerateAsync(Fixtures.Input(), cancellationToken: source.Token));
        Assert.Equal(1, converter.Reads);
        Assert.Single(handler.Requests);
    }

    [Theory]
    [InlineData(.5000004, true)]
    [InlineData(.5000011, false)]
    public async Task DistributionToleranceRemainsGlobalOneEMinusSix(double first, bool accepted)
    {
        var root = Fixtures.Native();
        root["answers"]!["q1"]!["probabilities"] = new JsonObject { ["0"] = first, ["1"] = .5 };
        root["answers"]!["q1"]!["choice"] = "0";
        using var handler = new ScriptedHandler(_ => Fixtures.Response(root.ToJsonString()));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http, Fixtures.Model);
        using var generator = new OllamaSharpDecisionGenerator(sdk);
        if (accepted)
        {
            var result = await generator.GenerateAsync(Fixtures.Input(), cancellationToken: Token);
            Assert.Equal(first, result.GetAnswer<ChoiceDecisionAnswer>("support-team").Probabilities["account"]);
        }
        else await Assert.ThrowsAsync<DecisionProtocolException>(() => generator.GenerateAsync(Fixtures.Input(), cancellationToken: Token));
        Assert.Single(handler.Requests);
        Assert.Equal(1e-6, DecisionAnswer.DistributionTolerance);
    }

    [Theory]
    [InlineData("-0.1")]
    [InlineData("1.1")]
    [InlineData("1e400")]
    [InlineData("\"NaN\"")]
    public async Task InvalidSourceNumbersFailWithoutRepair(string number)
    {
        string json = """{"model":"tev1:0.8b","usage":{"input_tokens":0,"output_tokens":0},"answers":{"q0":{"type":"noul","noul":"""
            + number + "}}}";
        using var handler = new ScriptedHandler(_ => Fixtures.Response(json));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http, Fixtures.Model);
        using var generator = new OllamaSharpDecisionGenerator(sdk);
        var input = new DecisionInput(Fixtures.Input().State, [Fixtures.Input().Questions[0]]);
        await Assert.ThrowsAsync<DecisionProtocolException>(() => generator.GenerateAsync(input, cancellationToken: Token));
        Assert.Single(handler.Requests);
    }

    private sealed class CancellingResponseConverter(CancellationTokenSource source) : JsonConverter<SystemOneResponse>
    {
        internal int Reads { get; private set; }
        public override SystemOneResponse? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            Reads++;
            var response = JsonSerializer.Deserialize<SystemOneResponse>(ref reader,
                new JsonSerializerOptions { AllowOutOfOrderMetadataProperties = true });
            source.Cancel();
            return response;
        }
        public override void Write(Utf8JsonWriter writer, SystemOneResponse value, JsonSerializerOptions options) =>
            throw new NotSupportedException("Read-only characterization converter.");
    }

    private sealed class ForeignOptions : DecisionGenerationOptions;
    private sealed class ExtendedNoulAnswer : SystemOneNoulAnswer;

    // Only unreachable interface lifetime/DTO-mutation edges use a proxy; wire behavior uses the stock SDK.
    public class ContractProxy : DispatchProxy
    {
        public SystemOneResponse? Response { get; set; }
        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args) => targetMethod?.Name switch
        {
            "get_SelectedModel" => Fixtures.Model,
            "get_Uri" => new Uri("http://127.0.0.1:11434"),
            "SystemOneAsync" => Task.FromResult(Response!),
            _ => throw new InvalidOperationException("Unexpected SDK interface member.")
        };
    }
    public class DisposableContractProxy : ContractProxy, IDisposable
    {
        public int Disposals { get; private set; }
        public void Dispose() => Disposals++;
    }
}
