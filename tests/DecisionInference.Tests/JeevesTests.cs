using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using DecisionInference.Jeeves;
using Microsoft.Extensions.AI;
using static DecisionInference.Tests.ContractTests;
using static DecisionInference.Tests.OllayaTestFixtures;

namespace DecisionInference.Tests;

// Console capture below must not intercept concurrently running tests.
[CollectionDefinition("Jeeves console isolation", DisableParallelization = true)]
public sealed class JeevesConsoleIsolation;

[Collection("Jeeves console isolation")]
public sealed class JeevesTests
{
    // Pinned PostHog/jeeves f04ec5567301450dcaae0210dd54deeb4f647f87:
    // inference/api.py, types.py, serve.py, engine.py, prep/format.py.
    // Offline protocol fixture, not a checkpoint, model execution or accuracy assertion.
    private const string Envelope = """
        {"model":"jeeves-latest","usage":{"input_tokens":11,"output_tokens":7,"reasoning_tokens":5},"latency_ms":12.5,
         "answers":{
          "q0":{"type":"noul","noul":0.8},
          "q1":{"type":"choice","choice":"1","confidence":0.50,"probabilities":{"0":0.25,"1":0.75}},
          "q2":{"type":"score","score":1.3,"confidence":0.65,"probabilities":{"0":0.2,"1":0.3,"2":0.5},
                "legend":{"0":"low","1":"mid","2":"high"}}}}
        """;

    private static CancellationToken Token => TestContext.Current.CancellationToken;
    private static JsonObject Root() => JsonNode.Parse(Envelope)!.AsObject();
    private static JsonObject Chain(bool thought, bool closed, long tokens, string text) => new()
    {
        ["thought"] = thought, ["closed"] = closed, ["tokens"] = tokens, ["text"] = text
    };

    private static JsonObject WithReasoning()
    {
        var root = Root();
        // At threshold .8, q0's .8 can skip thinking; q1's .75 and q2's .5 cannot.
        // Successful mixed-branch callers explicitly request threshold .8 and budget4.
        root["reasoning"] = new JsonObject
        {
            ["q0"] = Chain(false, false, 0, ""),
            ["q1"] = Chain(true, false, 4, "private rationale far longer than four characters"),
            ["q2"] = Chain(true, true, 1, "") // sole closing token is counted but excluded from text
        };
        return root;
    }

    private static async Task<DecisionResult> DecodeAsync(string body, DecisionGenerationOptions? options = null,
        DecisionInput? input = null)
    {
        using var handler = Script(_ => Response(body));
        using var http = new HttpClient(handler);
        using var generator = new JeevesDecisionGenerator(http);
        var result = await generator.GenerateAsync(input ?? JevTests.AllKinds(), options, Token);
        var sent = Assert.Single(handler.Requests);
        Assert.Equal(HttpMethod.Post, sent.Method);
        Assert.Equal("http://127.0.0.1:8009/v1/systemone", sent.Uri.AbsoluteUri);
        Assert.Null(sent.Authorization);
        return result;
    }

    private static async Task RejectAsync(string body, DecisionGenerationOptions? options = null, string? expectedMessage = null)
    {
        using var stream = new ObservedStream(Encoding.UTF8.GetBytes(body));
        using var content = new TrackingContent(stream);
        using var handler = Script(_ => new(HttpStatusCode.OK) { Content = content });
        using var http = new HttpClient(handler);
        using var generator = new JeevesDecisionGenerator(http);
        var error = await Assert.ThrowsAsync<DecisionProtocolException>(() =>
            generator.GenerateAsync(JevTests.AllKinds(), options, Token));
        if (expectedMessage is not null) Assert.Equal(expectedMessage, error.Message);
        Assert.DoesNotContain("private", error.ToString(), StringComparison.OrdinalIgnoreCase);
        Assert.Null(error.InnerException);
        Assert.Single(handler.Requests);
        Assert.True(content.Disposed);
        Assert.True(stream.Disposed);
    }

    [Fact]
    public async Task NativeEnvelope_UsesOneUnauthenticatedPostAndDecodesAllKinds()
    {
        using var handler = Script(_ => Response(Envelope));
        using var http = new HttpClient(handler)
        {
            Timeout = TimeSpan.FromSeconds(37),
            BaseAddress = new Uri("https://unused-client-base.invalid")
        };
        using var generator = new JeevesDecisionGenerator(http);
        var metadata = generator.GetRequiredService<DecisionGeneratorMetadata>();
        Assert.Equal("jeeves", metadata.ProviderName);
        Assert.Equal(new Uri("http://127.0.0.1:8009"), metadata.ProviderUri);
        Assert.Equal("jeeves-latest", metadata.DefaultModelId);
        Assert.Equal([DecisionKind.Binary, DecisionKind.Choice, DecisionKind.Score], metadata.SupportedKinds);
        Assert.Same(generator, generator.GetService<IDecisionGenerator>());
        Assert.Null(generator.GetService(typeof(IDecisionGenerator), "key"));
        Assert.Null(generator.GetService(typeof(string)));
        Assert.Throws<ArgumentNullException>(() => generator.GetService(null!));
        Assert.Empty(handler.Requests); // metadata must not trigger discovery

        var result = await generator.GenerateAsync(JevTests.AllKinds(), cancellationToken: Token);
        var request = Assert.Single(handler.Requests);
        Assert.Equal(HttpMethod.Post, request.Method);
        Assert.Equal("http://127.0.0.1:8009/v1/systemone", request.Uri.AbsoluteUri);
        Assert.Null(request.Authorization);
        Assert.Equal("application/json", request.MediaType);
        var body = Json(request.Body);
        Assert.Equal(["state", "model", "questions"], body.EnumerateObject().Select(p => p.Name));
        Assert.Equal("state", body.GetProperty("state").GetString());
        Assert.Equal("jeeves-latest", body.GetProperty("model").GetString());
        var questions = body.GetProperty("questions");
        Assert.Equal(["q0", "q1", "q2"], questions.EnumerateObject().Select(p => p.Name));
        Assert.Equal("noul", questions.GetProperty("q0").GetProperty("type").GetString());
        Assert.Equal("is it?", questions.GetProperty("q0").GetProperty("instructions").GetString());
        Assert.Equal("yes text", questions.GetProperty("q0").GetProperty("criteria").GetProperty("true").GetString());
        Assert.Equal("no text", questions.GetProperty("q0").GetProperty("criteria").GetProperty("false").GetString());
        Assert.Equal("choice", questions.GetProperty("q1").GetProperty("type").GetString());
        Assert.Equal("choose", questions.GetProperty("q1").GetProperty("instructions").GetString());
        Assert.Equal(["0", "1"], questions.GetProperty("q1").GetProperty("criteria").EnumerateObject().Select(p => p.Name));
        Assert.Equal(["first", "second"], questions.GetProperty("q1").GetProperty("criteria").EnumerateObject().Select(p => p.Value.GetString()));
        Assert.Equal("score", questions.GetProperty("q2").GetProperty("type").GetString());
        Assert.Equal("rate", questions.GetProperty("q2").GetProperty("instructions").GetString());
        Assert.Equal(["low", "mid", "high"], questions.GetProperty("q2").GetProperty("criteria").EnumerateArray().Select(e => e.GetString()));
        Assert.Equal("jeeves-latest", result.ModelId);
        Assert.Equal(["b", "c", "s"], result.Answers.Select(a => a.QuestionId));
        Assert.All(result.Answers, a => Assert.Equal(DecisionPrecision.TwoDecimalPlaces, a.Precision));
        Assert.Equal(.8, result.GetAnswer<BinaryDecisionAnswer>("b").TrueProbability);
        var choice = result.GetAnswer<ChoiceDecisionAnswer>("c");
        Assert.Equal("A", choice.SelectedCandidateId);
        Assert.Equal(new[] { .25, .75 }, choice.Probabilities.Values);
        Assert.Equal(.50, Assert.IsType<JsonElement>(choice.RawRepresentation).GetProperty("confidence").GetDouble());
        var score = result.GetAnswer<ScoreDecisionAnswer>("s");
        Assert.Equal(new[] { .2, .3, .5 }, score.Probabilities);
        Assert.Equal(1.3, score.NativeScore);
        Assert.Equal(1.3, score.ExpectedIndex, 12);
        Assert.Equal(1.29, score.ExpectedIndexLowerBound);
        Assert.Equal(1.31, score.ExpectedIndexUpperBound);
        Assert.Equal(.65, Assert.IsType<JsonElement>(score.RawRepresentation).GetProperty("confidence").GetDouble());
        Assert.Equal(11, result.Usage!.InputTokenCount);
        Assert.Equal(12, result.Usage.OutputTokenCount);
        Assert.Equal(5, result.Usage.ReasoningTokenCount);
        Assert.Equal(23, result.Usage.TotalTokenCount);
        Assert.Equal(7, result.Usage.AdditionalCounts![JeevesDecisionGenerator.AnswerJsonTokenCountKey]);
        Assert.False(Assert.IsType<JsonElement>(result.RawRepresentation).GetProperty("usage").TryGetProperty("total_tokens", out _));
        var evidence = Assert.IsType<JeevesResultMetadata>(result.AdditionalProperties![JeevesDecisionGenerator.MetadataKey]);
        Assert.Equal(12.5, evidence.LatencyMilliseconds);
        Assert.Null(evidence.Reasoning); // no implicit exposure of diagnostic text
        Assert.Same(metadata, generator.GetRequiredService<DecisionGeneratorMetadata>());
        Assert.Equal(TimeSpan.FromSeconds(37), http.Timeout); // provider does not choose the caller's deadline
        Assert.Equal(new Uri("https://unused-client-base.invalid"), http.BaseAddress);
    }

    [Fact]
    public async Task OpaqueIds_AreMappedOrdinallyWithoutNormalization()
    {
        string[] ids = [" q ", "q", "Q", "\u00e9", "e\u0301"];
        var input = Input(ids.Select(id => (DecisionQuestion)new ChoiceDecisionQuestion(id, "literal \"instruction\"",
            ids.Select(c => new DecisionCandidate(c, $"description {c}")))).ToArray());
        var root = Root();
        var answers = new JsonObject();
        var reasoning = new JsonObject();
        for (int i = ids.Length - 1; i >= 0; i--)
        {
            answers[$"q{i}"] = new JsonObject
            {
                ["type"] = "choice", ["choice"] = i.ToString(), ["confidence"] = 0,
                ["probabilities"] = new JsonObject(Enumerable.Range(0, ids.Length).Reverse()
                    .Select(j => KeyValuePair.Create<string, JsonNode?>(j.ToString(), JsonValue.Create(.2))))
            };
            reasoning[$"q{i}"] = Chain(true, i % 2 == 0, 2, $"private-{i}");
        }
        root["answers"] = answers;
        root["reasoning"] = reasoning;
        root["usage"]!["reasoning_tokens"] = 10;
        using var handler = Script(_ => Response(root.ToJsonString()));
        using var http = new HttpClient(handler);
        using var generator = new JeevesDecisionGenerator(http);
        var result = await generator.GenerateAsync(input, new JeevesGenerationOptions { ReturnReasoning = true, MaxThink = 2 }, Token);
        var sentBody = Json(Assert.Single(handler.Requests).Body);
        Assert.Equal(2, sentBody.GetProperty("options").GetProperty("max_think").GetInt32());
        Assert.True(sentBody.GetProperty("options").GetProperty("return_reasoning").GetBoolean());
        var sent = sentBody.GetProperty("questions");
        Assert.Equal(["q0", "q1", "q2", "q3", "q4"], sent.EnumerateObject().Select(p => p.Name));
        foreach (var q in sent.EnumerateObject())
        {
            Assert.Equal("literal \"instruction\"", q.Value.GetProperty("instructions").GetString());
            Assert.Equal(["0", "1", "2", "3", "4"], q.Value.GetProperty("criteria").EnumerateObject().Select(p => p.Name));
            Assert.Equal(ids.Select(id => $"description {id}"), q.Value.GetProperty("criteria").EnumerateObject().Select(p => p.Value.GetString()));
        }
        Assert.Equal(ids, result.Answers.Select(a => a.QuestionId));
        var evidence = Assert.IsType<JeevesResultMetadata>(result.AdditionalProperties![JeevesDecisionGenerator.MetadataKey]);
        for (int i = 0; i < ids.Length; i++)
        {
            var answer = result.GetAnswer<ChoiceDecisionAnswer>(ids[i]);
            Assert.Equal(ids[i], answer.SelectedCandidateId);
            Assert.Equal(ids, answer.Probabilities.Keys);
            Assert.All(answer.Probabilities.Values, p => Assert.Equal(.2, p));
            Assert.Equal(0, Assert.IsType<JsonElement>(answer.RawRepresentation).GetProperty("confidence").GetDouble());
            Assert.Equal($"private-{i}", evidence.Reasoning![ids[i]].Text);
            Assert.Equal(i % 2 == 0, evidence.Reasoning[ids[i]].Closed);
            Assert.True(evidence.Reasoning[ids[i]].Thought);
            Assert.Equal(2, evidence.Reasoning[ids[i]].Tokens);
        }
        var aliases = Assert.IsAssignableFrom<IReadOnlyDictionary<string, string>>(result.AdditionalProperties["systemone.questionIds"]);
        Assert.Equal(ids, Enumerable.Range(0, 5).Select(i => aliases[$"q{i}"]));
        Assert.Equal(10, result.Usage!.ReasoningTokenCount);
        Assert.Equal(17, result.Usage.OutputTokenCount);
        Assert.Equal(28, result.Usage.TotalTokenCount);
    }

    [Theory]
    [InlineData("https://provider.invalid:8443/", true)]
    [InlineData("http://127.0.0.1:8009/", true)]
    [InlineData("http://127.0.0.2:8009/", true)]
    [InlineData("http://[::1]:8009/", true)]
    [InlineData("http://localhost:8009/", false)]
    [InlineData("http://remote.invalid/", false)]
    [InlineData("http://192.168.1.1/", false)]
    [InlineData("ftp://127.0.0.1/", false)]
    [InlineData("https://provider.invalid/path", false)]
    [InlineData("https://provider.invalid/?query=x", false)]
    [InlineData("https://provider.invalid/#fragment", false)]
    [InlineData("https://user:password@provider.invalid/", false)]
    [InlineData("/relative", false)]
    public void EndpointPolicy_RequiresHttpsOrLiteralLoopbackOrigin(string endpoint, bool accepted)
    {
        using var handler = Script(_ => Response(Envelope));
        using var http = new HttpClient(handler);
        var options = new JeevesDecisionGeneratorOptions { Endpoint = new(endpoint, UriKind.RelativeOrAbsolute) };
        if (accepted)
        {
            using var generator = new JeevesDecisionGenerator(http, options);
            Assert.Equal(options.Endpoint, generator.GetRequiredService<DecisionGeneratorMetadata>().ProviderUri);
            Assert.Equal("jeeves", generator.GetRequiredService<DecisionGeneratorMetadata>().ProviderName);
        }
        else Assert.Throws<ArgumentException>(() => new JeevesDecisionGenerator(http, options));
        Assert.Empty(handler.Requests);
        Assert.Equal(0, handler.DisposeCount);
    }

    [Fact]
    public void Constructor_RejectsNullInvalidLabelsAndBodyLimitsWithoutSending()
    {
        using var handler = Script(_ => Response(Envelope));
        using var http = new HttpClient(handler);
        Assert.Throws<ArgumentNullException>(() => new JeevesDecisionGenerator(null!));
        Assert.Throws<ArgumentNullException>(() => new JeevesDecisionGenerator(http, new() { Endpoint = null! }));
        foreach (string? label in new string?[] { null, "", " \t" })
            Assert.ThrowsAny<ArgumentException>(() => new JeevesDecisionGenerator(http, new() { ModelId = label! }));
        foreach (int maximum in new[] { 0, -1 })
            Assert.Throws<ArgumentOutOfRangeException>(() => new JeevesDecisionGenerator(http, new() { MaximumResponseBytes = maximum }));
        Assert.Empty(handler.Requests);
        Assert.Equal(0, handler.DisposeCount);
    }

    [Fact]
    public async Task Authorization_IsRejectedAtConstructionAndAfterHeaderDrift()
    {
        using var handler = Script(_ => Response(Envelope));
        using var http = new HttpClient(handler);
        http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", "private-key");
        var construction = Assert.Throws<ArgumentException>(() => new JeevesDecisionGenerator(http));
        Assert.DoesNotContain("private-key", construction.ToString());
        http.DefaultRequestHeaders.Authorization = null;
        using var generator = new JeevesDecisionGenerator(http);
        await generator.GenerateAsync(JevTests.AllKinds(), cancellationToken: Token);
        http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Basic", "private-key");
        var drift = await Assert.ThrowsAsync<InvalidOperationException>(() => generator.GenerateAsync(JevTests.AllKinds(), cancellationToken: Token));
        Assert.DoesNotContain("private-key", drift.ToString());
        Assert.Single(handler.Requests);
        Assert.Null(handler.Requests[0].Authorization);
    }

    [Fact]
    public async Task NullAndBaseOptions_OmitOptionsAndModelOverridesAreNonstickyLabels()
    {
        using var handler = Script(r =>
        {
            var root = Root();
            root["model"] = Json(r.Body).GetProperty("model").GetString();
            return Response(root.ToJsonString());
        });
        using var http = new HttpClient(handler);
        using var generator = new JeevesDecisionGenerator(http, new() { ModelId = "configured-label", Endpoint = new("https://provider.invalid:8443") });
        DecisionGenerationOptions?[] options = [null, new(), new() { ModelId = " custom Label ", AdditionalProperties = new() }, new()];
        string[] models = ["configured-label", "configured-label", " custom Label ", "configured-label"];
        for (int i = 0; i < options.Length; i++)
        {
            var result = await generator.GenerateAsync(JevTests.AllKinds(), options[i], Token);
            Assert.Equal(models[i], result.ModelId);
            var sent = Json(handler.Requests[i].Body);
            Assert.False(sent.TryGetProperty("options", out _));
            Assert.Equal(models[i], sent.GetProperty("model").GetString());
            Assert.Equal("https://provider.invalid:8443/v1/systemone", handler.Requests[i].Uri.AbsoluteUri);
        }
        Assert.Equal(4, handler.Requests.Length);
        Assert.Equal("configured-label", generator.GetRequiredService<DecisionGeneratorMetadata>().DefaultModelId);
        Assert.Equal(" custom Label ", options[2]!.ModelId);
        Assert.Empty(options[2]!.AdditionalProperties!);
    }

    [Theory]
    [InlineData("Jeeves-latest")]
    [InlineData("jeeves-latest ")]
    [InlineData("different-weights")]
    [InlineData("")]
    public async Task ResponseModel_MustEchoExactRequestedLabel(string model)
    {
        var root = Root();
        root["model"] = model;
        await RejectAsync(root.ToJsonString());
    }

    [Fact]
    public async Task ExplicitOptions_EmitCompleteDefaultSnapshot()
    {
        using var handler = Script(_ => Response(Envelope));
        using var http = new HttpClient(handler);
        using var generator = new JeevesDecisionGenerator(http);
        var options = new JeevesGenerationOptions();
        await generator.GenerateAsync(JevTests.AllKinds(), options, Token);
        var sent = Json(Assert.Single(handler.Requests).Body).GetProperty("options");
        Assert.Equal(["think", "max_think", "nothink_threshold", "return_reasoning"], sent.EnumerateObject().Select(p => p.Name));
        Assert.True(sent.GetProperty("think").GetBoolean());
        Assert.Equal(2560, sent.GetProperty("max_think").GetInt32());
        Assert.Equal(JsonValueKind.Null, sent.GetProperty("nothink_threshold").ValueKind);
        Assert.False(sent.GetProperty("return_reasoning").GetBoolean());
        Assert.True(options.Think);
        Assert.Equal(2560, options.MaxThink);
        Assert.Null(options.NoThinkThreshold);
        Assert.False(options.ReturnReasoning);
        Assert.Null(options.ModelId);
        Assert.Null(options.AdditionalProperties);
    }

    [Theory]
    [InlineData(null)]
    [InlineData(0d)]
    [InlineData(1d)]
    [InlineData(.25)]
    public async Task ExplicitOptions_EmitMutableValuesAndThresholdEndpoints(double? threshold)
    {
        var root = Root();
        root["usage"]!["reasoning_tokens"] = 0;
        using var handler = Script(_ => Response(root.ToJsonString()));
        using var http = new HttpClient(handler);
        using var generator = new JeevesDecisionGenerator(http);
        var options = new JeevesGenerationOptions { Think = false, MaxThink = 0, NoThinkThreshold = threshold };
        var result = await generator.GenerateAsync(JevTests.AllKinds(), options, Token);
        var sent = Json(Assert.Single(handler.Requests).Body).GetProperty("options");
        Assert.False(sent.GetProperty("think").GetBoolean());
        Assert.Equal(0, sent.GetProperty("max_think").GetInt32());
        Assert.False(sent.GetProperty("return_reasoning").GetBoolean());
        if (threshold is { } value) Assert.Equal(value, sent.GetProperty("nothink_threshold").GetDouble());
        else Assert.Equal(JsonValueKind.Null, sent.GetProperty("nothink_threshold").ValueKind);
        Assert.Equal(0, result.Usage!.ReasoningTokenCount);
        Assert.Equal(7, result.Usage.OutputTokenCount);
        Assert.Equal(18, result.Usage.TotalTokenCount);
        Assert.Equal(threshold, options.NoThinkThreshold);
    }

    [Fact]
    public void Clone_PreservesDerivedAndBasePropertiesWithSeparateShallowDictionary()
    {
        var nested = new List<int> { 42 };
        var original = new JeevesGenerationOptions
        {
            ModelId = "label", Think = false, MaxThink = 17, NoThinkThreshold = .25, ReturnReasoning = true,
            AdditionalProperties = new() { ["nested"] = nested }
        };
        DecisionGenerationOptions asBase = original;
        var clone = Assert.IsType<JeevesGenerationOptions>(asBase.Clone());
        Assert.NotSame(original, clone);
        Assert.Equal("label", clone.ModelId);
        Assert.False(clone.Think);
        Assert.Equal(17, clone.MaxThink);
        Assert.Equal(.25, clone.NoThinkThreshold);
        Assert.True(clone.ReturnReasoning);
        Assert.NotSame(original.AdditionalProperties, clone.AdditionalProperties);
        Assert.Same(nested, clone.AdditionalProperties!["nested"]);
        clone.ModelId = "changed";
        clone.Think = true;
        clone.MaxThink = 0;
        clone.NoThinkThreshold = null;
        clone.ReturnReasoning = false;
        clone.AdditionalProperties.Clear();
        Assert.Equal("label", original.ModelId);
        Assert.False(original.Think);
        Assert.Equal(17, original.MaxThink);
        Assert.Equal(.25, original.NoThinkThreshold);
        Assert.True(original.ReturnReasoning);
        Assert.Same(nested, Assert.Single(original.AdditionalProperties!).Value);
    }

    private sealed class OtherOptions : DecisionGenerationOptions;

    [Theory]
    [InlineData("base-properties")]
    [InlineData("typed-properties")]
    [InlineData("other-derived")]
    [InlineData("negative-budget")]
    [InlineData("negative-threshold")]
    [InlineData("large-threshold")]
    [InlineData("nan-threshold")]
    [InlineData("infinite-threshold")]
    [InlineData("negative-infinite-threshold")]
    [InlineData("empty-model")]
    [InlineData("whitespace-model")]
    public async Task InvalidOptions_AreRejectedBeforeSending(string scenario)
    {
        DecisionGenerationOptions options = scenario switch
        {
            "base-properties" => new() { AdditionalProperties = new() { ["private-key"] = 1 } },
            "typed-properties" => new JeevesGenerationOptions { AdditionalProperties = new() { ["private-key"] = 1 } },
            "other-derived" => new OtherOptions(),
            "negative-budget" => new JeevesGenerationOptions { MaxThink = -1 },
            "negative-threshold" => new JeevesGenerationOptions { NoThinkThreshold = -.01 },
            "large-threshold" => new JeevesGenerationOptions { NoThinkThreshold = 1.01 },
            "nan-threshold" => new JeevesGenerationOptions { NoThinkThreshold = double.NaN },
            "infinite-threshold" => new JeevesGenerationOptions { NoThinkThreshold = double.PositiveInfinity },
            "negative-infinite-threshold" => new JeevesGenerationOptions { NoThinkThreshold = double.NegativeInfinity },
            "empty-model" => new() { ModelId = "" },
            _ => new() { ModelId = " \t" }
        };
        using var handler = Script(_ => Response(Envelope));
        using var http = new HttpClient(handler);
        using var generator = new JeevesDecisionGenerator(http);
        if (scenario is "base-properties" or "typed-properties" or "other-derived")
            await Assert.ThrowsAsync<NotSupportedException>(() => generator.GenerateAsync(JevTests.AllKinds(), options, Token));
        else
            await Assert.ThrowsAnyAsync<ArgumentException>(() => generator.GenerateAsync(JevTests.AllKinds(), options, Token));
        Assert.Empty(handler.Requests);
        if (options.AdditionalProperties is { } properties) Assert.Equal(1, properties["private-key"]);
    }

    [Fact(Timeout = 15000)]
    public async Task OptionsSnapshot_IsCapturedBeforeAwaitAndSurvivesCallerMutation()
    {
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource<HttpResponseMessage>(TaskCreationOptions.RunContinuationsAsynchronously);
        using var handler = new ScriptedHandler(async (_, token) =>
        {
            entered.TrySetResult();
            return await release.Task.WaitAsync(token);
        });
        using var http = new HttpClient(handler);
        using var generator = new JeevesDecisionGenerator(http);
        var options = new JeevesGenerationOptions
        {
            ReturnReasoning = true, MaxThink = 4, NoThinkThreshold = .8,
            ModelId = "before", AdditionalProperties = new()
        };
        var call = generator.GenerateAsync(JevTests.AllKinds(), options, Token);
        try
        {
            await entered.Task.WaitAsync(TimeSpan.FromSeconds(5), Token);
            options.Think = false;
            options.MaxThink = 0;
            options.ReturnReasoning = false;
            options.NoThinkThreshold = double.NaN;
            options.ModelId = "after";
            options.AdditionalProperties["changed"] = true;
            var root = WithReasoning();
            root["model"] = "before";
            release.SetResult(Response(root.ToJsonString()));
            var result = await call;
            Assert.Equal("before", result.ModelId);
            Assert.Equal(5, result.Usage!.ReasoningTokenCount);
            Assert.Equal(4, Assert.IsType<JeevesResultMetadata>(result.AdditionalProperties![JeevesDecisionGenerator.MetadataKey]).Reasoning!["c"].Tokens);
            var sent = Json(Assert.Single(handler.Requests).Body);
            Assert.Equal("before", sent.GetProperty("model").GetString());
            Assert.True(sent.GetProperty("options").GetProperty("think").GetBoolean());
            Assert.True(sent.GetProperty("options").GetProperty("return_reasoning").GetBoolean());
            Assert.Equal(4, sent.GetProperty("options").GetProperty("max_think").GetInt32());
            Assert.Equal(.8, sent.GetProperty("options").GetProperty("nothink_threshold").GetDouble());
            Assert.Equal("after", options.ModelId);
            Assert.True((bool)options.AdditionalProperties["changed"]!);
        }
        finally
        {
            release.TrySetCanceled(Token);
            try { await call; } catch (Exception) { /* drain fixture on assertion failure */ }
        }
    }

    [Theory]
    [InlineData("choice", 1)]
    [InlineData("choice", 255)]
    [InlineData("score", 2)]
    [InlineData("score", 10)]
    public async Task InputLimits_AcceptSystemOneBoundarySizes(string kind, int count)
    {
        var probabilities = new JsonObject();
        for (int i = 0; i < count; i++) probabilities[i.ToString()] = i == count - 1 ? 1d : 0d;
        var root = Root();
        root["answers"] = new JsonObject
        {
            ["q0"] = kind == "choice"
                ? new JsonObject { ["type"] = kind, ["choice"] = (count - 1).ToString(), ["confidence"] = 1, ["probabilities"] = probabilities }
                : new JsonObject { ["type"] = kind, ["score"] = count - 1, ["confidence"] = 0, ["probabilities"] = probabilities,
                    ["legend"] = new JsonObject(Enumerable.Range(0, count).Select(i => KeyValuePair.Create<string, JsonNode?>(i.ToString(), JsonValue.Create($"level{i}")))) }
        };
        DecisionQuestion question = kind == "choice"
            ? new ChoiceDecisionQuestion("opaque", "choose", Enumerable.Range(0, count).Select(i => new DecisionCandidate($"id{i}", $"desc{i}")))
            : new ScoreDecisionQuestion("opaque", "rate", Enumerable.Range(0, count).Select(i => $"level{i}"));
        using var handler = Script(_ => Response(root.ToJsonString()));
        using var http = new HttpClient(handler);
        using var generator = new JeevesDecisionGenerator(http);
        var result = await generator.GenerateAsync(Input(question), cancellationToken: Token);
        var criteria = Json(Assert.Single(handler.Requests).Body).GetProperty("questions").GetProperty("q0").GetProperty("criteria");
        if (kind == "choice")
        {
            Assert.Equal(count, criteria.EnumerateObject().Count());
            Assert.Equal($"desc{count - 1}", criteria.GetProperty((count - 1).ToString()).GetString());
            var answer = result.GetAnswer<ChoiceDecisionAnswer>("opaque");
            Assert.Equal($"id{count - 1}", answer.SelectedCandidateId);
            Assert.Equal(1, answer.Probabilities[$"id{count - 1}"]);
            Assert.Equal(count, answer.Probabilities.Count);
        }
        else
        {
            Assert.Equal(count, criteria.GetArrayLength());
            Assert.Equal($"level{count - 1}", criteria[count - 1].GetString());
            var answer = result.GetAnswer<ScoreDecisionAnswer>("opaque");
            Assert.Equal(count - 1, answer.ExpectedIndex);
            Assert.Equal(count - 1, answer.NativeScore);
            Assert.Equal(count, answer.Probabilities.Count);
        }
    }

    [Theory]
    [InlineData("choice256")]
    [InlineData("choice0")]
    [InlineData("score11")]
    [InlineData("score1")]
    [InlineData("null")]
    [InlineData("numeric-state")]
    [InlineData("bool-state")]
    [InlineData("duplicate-state")]
    public async Task InvalidInputs_FailBeforeRequest(string scenario)
    {
        using var handler = Script(_ => Response(Envelope));
        using var http = new HttpClient(handler);
        using var generator = new JeevesDecisionGenerator(http);
        async Task Call()
        {
            var input = scenario switch
            {
                "choice256" => Input(new ChoiceDecisionQuestion("c", null, Enumerable.Range(0, 256).Select(i => new DecisionCandidate(i.ToString(), "")))),
                "choice0" => Input(new ChoiceDecisionQuestion("c", null, [])),
                "score11" => Input(new ScoreDecisionQuestion("s", null, Enumerable.Repeat("", 11))),
                "score1" => Input(new ScoreDecisionQuestion("s", null, ["only"])),
                "null" => null,
                "numeric-state" => new DecisionInput(Json("42"), [Binary()]),
                "bool-state" => new DecisionInput(Json("true"), [Binary()]),
                _ => new DecisionInput(Json("""{"x":1,"x":2}"""), [Binary()])
            };
            await generator.GenerateAsync(input!, cancellationToken: Token);
        }
        if (scenario == "duplicate-state")
            await Assert.ThrowsAsync<JsonException>(Call);
        else
            await Assert.ThrowsAnyAsync<ArgumentException>(Call);
        Assert.Empty(handler.Requests);
    }

    public static IEnumerable<object[]> InvalidWireCounts()
    {
        foreach (string field in new[] { "input_tokens", "output_tokens", "reasoning_tokens" })
            foreach (string value in new[] { "-1", "true", "false", "0.5", "1.0", "\"1\"", "null", "1e400", "9223372036854775808", "[]", "{}" })
                yield return [field, value];
    }

    [Theory]
    [MemberData(nameof(InvalidWireCounts))]
    public async Task Usage_RequiresNonnegativeStrictInt64ForEveryOriginalCount(string field, string rawValue)
    {
        var root = Root();
        root["usage"]![field] = JsonNode.Parse(rawValue);
        await RejectAsync(root.ToJsonString());
    }

    [Theory]
    [InlineData("input_tokens")]
    [InlineData("output_tokens")]
    [InlineData("reasoning_tokens")]
    [InlineData("usage")]
    public async Task Usage_RequiresEveryOriginalCount(string field)
    {
        var root = Root();
        if (field == "usage") root.Remove(field);
        else root["usage"]!.AsObject().Remove(field);
        await RejectAsync(root.ToJsonString());
    }

    [Theory]
    [InlineData(long.MaxValue, 0, 0, 0)]
    [InlineData(0, long.MaxValue, 0, long.MaxValue)]
    [InlineData(0, 0, long.MaxValue, long.MaxValue)]
    [InlineData(long.MaxValue - 3, 1, 2, 3)]
    public async Task Usage_AcceptsSafeInt64BoundaryAndPreservesOriginalAnswerCount(
        long input, long answerJson, long reasoning, long output)
    {
        var root = Root();
        root["usage"] = new JsonObject { ["input_tokens"] = input, ["output_tokens"] = answerJson, ["reasoning_tokens"] = reasoning };
        var result = await DecodeAsync(root.ToJsonString());
        Assert.Equal(input, result.Usage!.InputTokenCount);
        Assert.Equal(output, result.Usage.OutputTokenCount);
        Assert.Equal(reasoning, result.Usage.ReasoningTokenCount);
        Assert.Equal(long.MaxValue, result.Usage.TotalTokenCount);
        Assert.Equal(answerJson, result.Usage.AdditionalCounts![JeevesDecisionGenerator.AnswerJsonTokenCountKey]);
        Assert.Single(result.Usage.AdditionalCounts);
    }

    [Theory]
    [InlineData(0, long.MaxValue, 1)]
    [InlineData(long.MaxValue, 1, 0)]
    [InlineData(long.MaxValue - 1, 1, 1)]
    public async Task Usage_RejectsBothCheckedAdditionOverflows(long input, long answerJson, long reasoning)
    {
        var root = Root();
        root["usage"] = new JsonObject { ["input_tokens"] = input, ["output_tokens"] = answerJson, ["reasoning_tokens"] = reasoning };
        await RejectAsync(root.ToJsonString());
    }

    [Theory]
    [InlineData(false, 2560, 0, true)]
    [InlineData(false, 2560, 1, false)]
    [InlineData(true, 0, 0, true)]
    [InlineData(true, 0, 1, false)]
    [InlineData(true, 2, 6, true)]
    [InlineData(true, 2, 7, false)]
    [InlineData(true, int.MaxValue, 6442450941L, true)]
    [InlineData(true, int.MaxValue, 6442450942L, false)]
    public async Task TypedBudget_BoundsAggregateAcrossAllQuestions(bool think, int maxThink, long reasoning, bool accepted)
    {
        var root = Root();
        root["usage"]!["reasoning_tokens"] = reasoning;
        var options = new JeevesGenerationOptions { Think = think, MaxThink = maxThink };
        if (!accepted) { await RejectAsync(root.ToJsonString(), options); return; }
        var result = await DecodeAsync(root.ToJsonString(), options);
        Assert.Equal(reasoning, result.Usage!.ReasoningTokenCount);
        Assert.Equal(7 + reasoning, result.Usage.OutputTokenCount);
        Assert.Equal(18 + reasoning, result.Usage.TotalTokenCount);
        Assert.Null(Assert.IsType<JeevesResultMetadata>(result.AdditionalProperties![JeevesDecisionGenerator.MetadataKey]).Reasoning);
    }

    [Fact]
    public async Task Reasoning_ClosingOnlyTruncatedAndNonthinkingChainsAreOwnedImmutableEvidence()
    {
        var root = WithReasoning();
        using var stream = new ObservedStream(Encoding.UTF8.GetBytes(root.ToJsonString()));
        using var content = new TrackingContent(stream);
        using var response = new HttpResponseMessage(HttpStatusCode.OK) { Content = content };
        using var handler = Script(_ => response);
        using var http = new HttpClient(handler);
        using var generator = new JeevesDecisionGenerator(http);
        var result = await generator.GenerateAsync(JevTests.AllKinds(),
            new JeevesGenerationOptions { ReturnReasoning = true, MaxThink = 4, NoThinkThreshold = .8 }, Token);
        Assert.True(content.Disposed);
        Assert.True(stream.Disposed);
        Assert.Throws<ObjectDisposedException>(() => response.Content = new ByteArrayContent([]));
        var evidence = Assert.IsType<JeevesResultMetadata>(result.AdditionalProperties![JeevesDecisionGenerator.MetadataKey]);
        Assert.Equal(12.5, evidence.LatencyMilliseconds);
        var reasoning = evidence.Reasoning!;
        Assert.Equal(["b", "c", "s"], reasoning.Keys);
        Assert.False(reasoning["b"].Thought);
        Assert.False(reasoning["b"].Closed);
        Assert.Equal(0, reasoning["b"].Tokens);
        Assert.Equal("", reasoning["b"].Text);
        Assert.True(reasoning["c"].Thought);
        Assert.False(reasoning["c"].Closed);
        Assert.Equal(4, reasoning["c"].Tokens);
        Assert.Equal("private rationale far longer than four characters", reasoning["c"].Text);
        Assert.True(reasoning["s"].Thought);
        Assert.True(reasoning["s"].Closed);
        Assert.Equal(1, reasoning["s"].Tokens);
        Assert.Equal("", reasoning["s"].Text);
        Assert.Equal(5, result.Usage!.ReasoningTokenCount);
        Assert.Equal(12, result.Usage.OutputTokenCount);
        var sentOptions = Json(Assert.Single(handler.Requests).Body).GetProperty("options");
        Assert.Equal(.8, sentOptions.GetProperty("nothink_threshold").GetDouble());
        Assert.Equal(4, sentOptions.GetProperty("max_think").GetInt32());
        Assert.Throws<NotSupportedException>(() => ((IDictionary<string, JeevesReasoning>)reasoning).Clear());
        Assert.All(typeof(JeevesReasoning).GetProperties(), property => Assert.Null(property.SetMethod));
        Assert.All(typeof(JeevesResultMetadata).GetProperties(), property => Assert.Null(property.SetMethod));
        root["reasoning"]!["q1"]!["text"] = "mutated fixture";
        Assert.Equal("private rationale far longer than four characters", reasoning["c"].Text);
        var raw = Assert.IsType<JsonElement>(result.RawRepresentation);
        Assert.Equal(5, raw.GetProperty("usage").GetProperty("reasoning_tokens").GetInt64());
        Assert.Equal("private rationale far longer than four characters", raw.GetProperty("reasoning").GetProperty("q1").GetProperty("text").GetString());
        Assert.Equal(.8, Assert.IsType<JsonElement>(result.Answers[0].RawRepresentation).GetProperty("noul").GetDouble());
        Assert.DoesNotContain("private rationale", result.ToString()!);
        Assert.DoesNotContain("private rationale", evidence.ToString()!);
        Assert.DoesNotContain("private rationale", reasoning["c"].ToString()!);
    }

    [Theory]
    [InlineData("private impossible closing-only text")]
    [InlineData(" ")]
    [InlineData("\t")]
    public async Task Reasoning_ClosingOnlyNonemptyTextIsRejectedWithOtherwiseConsistentUsage(string text)
    {
        var root = WithReasoning();
        // Only text changes: q2 still has thought=true, closed=true, tokens=1,
        // the per-chain cap is4, and the exact aggregate is still0+4+1=5.
        root["reasoning"]!["q2"]!["text"] = text;
        await RejectAsync(root.ToJsonString(),
            new JeevesGenerationOptions { ReturnReasoning = true, MaxThink = 4, NoThinkThreshold = .8 },
            "Inconsistent reasoning state or budget.");
    }

    [Theory]
    [InlineData(false, 1, "private single open token")]
    [InlineData(false, 1, "")]
    [InlineData(true, 1, "")]
    [InlineData(true, 2, "private token before closing")]
    public async Task Reasoning_SingleTokenAndClosedTextBoundariesRemainValid(bool closed, long tokens, string text)
    {
        var root = WithReasoning();
        root["reasoning"]!["q2"] = Chain(true, closed, tokens, text);
        // A truncated one-token chain can also result from the engine's remaining context cap.
        root["usage"]!["reasoning_tokens"] = 4 + tokens;
        var result = await DecodeAsync(root.ToJsonString(),
            new JeevesGenerationOptions { ReturnReasoning = true, MaxThink = 4, NoThinkThreshold = .8 });
        var evidence = Assert.IsType<JeevesResultMetadata>(result.AdditionalProperties![JeevesDecisionGenerator.MetadataKey]);
        Assert.True(evidence.Reasoning!["s"].Thought);
        Assert.Equal(closed, evidence.Reasoning["s"].Closed);
        Assert.Equal(tokens, evidence.Reasoning["s"].Tokens);
        Assert.Equal(text, evidence.Reasoning["s"].Text);
        Assert.Equal(4, evidence.Reasoning["c"].Tokens);
        Assert.Equal(0, evidence.Reasoning["b"].Tokens);
        Assert.Equal(4 + tokens, result.Usage!.ReasoningTokenCount);
        Assert.Equal(11 + tokens, result.Usage.OutputTokenCount);
        Assert.Equal(22 + tokens, result.Usage.TotalTokenCount);
    }

    [Theory]
    [InlineData(false, 4)]
    [InlineData(true, 0)]
    public async Task Reasoning_ZeroCapUsesEmptyOpenNonthinkingChains(bool think, int maxThink)
    {
        var root = WithReasoning();
        foreach (string id in new[] { "q0", "q1", "q2" }) root["reasoning"]![id] = Chain(false, false, 0, "");
        root["usage"]!["reasoning_tokens"] = 0;
        root["latency_ms"] = 0;
        var result = await DecodeAsync(root.ToJsonString(), new JeevesGenerationOptions { Think = think, MaxThink = maxThink, ReturnReasoning = true });
        var evidence = Assert.IsType<JeevesResultMetadata>(result.AdditionalProperties![JeevesDecisionGenerator.MetadataKey]);
        Assert.Equal(0, evidence.LatencyMilliseconds);
        Assert.Equal(3, evidence.Reasoning!.Count);
        Assert.All(evidence.Reasoning.Values, chain =>
        {
            Assert.False(chain.Thought);
            Assert.False(chain.Closed);
            Assert.Equal(0, chain.Tokens);
            Assert.Equal("", chain.Text);
        });
        Assert.Equal(0, result.Usage!.ReasoningTokenCount);
        Assert.Equal(18, result.Usage.TotalTokenCount);
    }

    [Fact]
    public async Task Reasoning_DiagnosticsAreNeverWrittenToConsoleByDefault()
    {
        var originalOut = Console.Out;
        var originalError = Console.Error;
        using var output = new StringWriter();
        using var error = new StringWriter();
        try
        {
            Console.SetOut(output);
            Console.SetError(error);
            string body = WithReasoning().ToJsonString();
            var result = await DecodeAsync(body,
                new JeevesGenerationOptions { ReturnReasoning = true, MaxThink = 4, NoThinkThreshold = .8 });
            Assert.Equal("private rationale far longer than four characters",
                Assert.IsType<JeevesResultMetadata>(result.AdditionalProperties![JeevesDecisionGenerator.MetadataKey]).Reasoning!["c"].Text);
            await RejectAsync(body); // unsolicited sensitive evidence must not be logged on failure either
            Assert.DoesNotContain("private rationale", output.ToString());
            Assert.DoesNotContain("private rationale", error.ToString());
        }
        finally
        {
            Console.SetOut(originalOut);
            Console.SetError(originalError);
        }
    }

    [Theory]
    [InlineData("no-options")]
    [InlineData("base-options")]
    [InlineData("explicit-false")]
    [InlineData("requested-missing")]
    public async Task Reasoning_PresenceMustMatchExplicitOptIn(string scenario)
    {
        var root = scenario == "requested-missing" ? Root() : WithReasoning();
        DecisionGenerationOptions? options = scenario switch
        {
            "no-options" => null,
            "base-options" => new(),
            "explicit-false" => new JeevesGenerationOptions(),
            _ => new JeevesGenerationOptions { ReturnReasoning = true, MaxThink = 4, NoThinkThreshold = .8 }
        };
        await RejectAsync(root.ToJsonString(), options);
    }

    public static IEnumerable<object[]> InvalidReasoning()
    {
        foreach (string field in new[] { "thought", "closed", "tokens", "text" })
        {
            var root = WithReasoning();
            root["reasoning"]!["q0"]!.AsObject().Remove(field);
            yield return [$"missing-{field}", root.ToJsonString()];
        }
        foreach (var (field, values) in new (string, string[])[]
        {
            ("thought", ["null", "0", "\"true\"", "[]"]),
            ("closed", ["null", "1", "\"false\"", "{}"]),
            ("tokens", ["null", "true", "\"1\"", "1.0", "0.5", "-1", "1e400", "9223372036854775808"]),
            ("text", ["null", "1", "false", "[]", "{}"])
        })
            foreach (string value in values)
            {
                var root = WithReasoning();
                // q2's original count1 keeps relaxed parsing of "1"/1.0 otherwise valid;
                // q1 is open, so text typing cannot be masked by the closing-only guard.
                string id = field == "text" ? "q1" : "q2";
                root["reasoning"]![id]![field] = JsonNode.Parse(value);
                yield return [$"wrong-{field}-{value}", root.ToJsonString()];
            }
        foreach (string scenario in new[] { "null-map", "array-map", "null-entry", "array-entry", "missing-id", "missing-zero-id", "extra-id",
            "case-id", "caller-id", "extra-field", "thought-zero", "nonthought-positive", "nonthought-text",
            "nonthought-positive-open", "nonthought-closed", "sum-too-small", "sum-too-large", "chain-over-budget", "selected-question-kind" })
        {
            var root = WithReasoning();
            var map = root["reasoning"]!.AsObject();
            switch (scenario)
            {
                case "null-map": root["reasoning"] = null; break;
                case "array-map": root["reasoning"] = new JsonArray(); break;
                case "null-entry": map["q0"] = null; break;
                case "array-entry": map["q0"] = new JsonArray(); break;
                case "missing-id": map.Remove("q2"); break;
                case "missing-zero-id": map.Remove("q0"); break; // sum remains correct: isolates completeness
                case "extra-id": map["q3"] = Chain(false, false, 0, ""); break;
                case "case-id": map["Q0"] = map["q0"]!.DeepClone(); map.Remove("q0"); break;
                case "caller-id": map["b"] = map["q0"]!.DeepClone(); map.Remove("q0"); break;
                case "extra-field": map["q0"]!["private-extra"] = 1; break;
                case "thought-zero": map["q0"]!["thought"] = true; break;
                case "nonthought-positive": map["q2"]!["thought"] = false; break;
                case "nonthought-positive-open": map["q2"] = Chain(false, false, 1, ""); break;
                case "nonthought-text": map["q0"]!["text"] = "private"; break;
                case "nonthought-closed": map["q0"]!["closed"] = true; break;
                case "sum-too-small": root["usage"]!["reasoning_tokens"] = 4; break;
                case "sum-too-large": root["usage"]!["reasoning_tokens"] = 6; break;
                case "chain-over-budget":
                    map["q1"]!["tokens"] = 5;
                    root["usage"]!["reasoning_tokens"] = 6; // total fits 3*4; individual does not
                    break;
                case "selected-question-kind": root["answers"]!["q0"] = root["answers"]!["q2"]!.DeepClone(); break;
            }
            yield return [scenario, root.ToJsonString()];
        }
    }

    [Theory]
    [MemberData(nameof(InvalidReasoning))]
    public async Task Reasoning_RejectsMalformedIncompleteInconsistentAndOverBudgetEvidence(string scenario, string body)
    {
        Assert.NotEmpty(scenario);
        await RejectAsync(body, new JeevesGenerationOptions { ReturnReasoning = true, MaxThink = 4, NoThinkThreshold = .8 });
    }

    [Theory]
    [InlineData("-1")]
    [InlineData("1e400")]
    [InlineData("-1e400")]
    [InlineData("null")]
    [InlineData("\"12.5\"")]
    [InlineData("true")]
    [InlineData("[]")]
    [InlineData("missing")]
    public async Task Latency_RequiresFiniteNonnegativeServerMetric(string raw)
    {
        var root = Root();
        if (raw == "missing") root.Remove("latency_ms");
        else root["latency_ms"] = JsonNode.Parse(raw);
        await RejectAsync(root.ToJsonString());
    }

    public static IEnumerable<object[]> InvalidNativeEnvelopes()
    {
        foreach (string body in new[] { "{private", "null", "[]", "true", "{}", Envelope + "{}" })
            yield return ["malformed-root", body];
        foreach (var (id, field) in new[] { ("q0", "type"), ("q0", "noul"), ("q1", "choice"),
            ("q1", "confidence"), ("q1", "probabilities"), ("q2", "score"), ("q2", "confidence"),
            ("q2", "probabilities"), ("q2", "legend") })
        {
            var root = Root();
            root["answers"]![id]!.AsObject().Remove(field);
            yield return [$"missing-{id}-{field}", root.ToJsonString()];
        }
        foreach (string scenario in new[] { "missing-model", "null-model", "missing-answers", "null-answers", "missing-answer",
            "extra-answer", "unknown-answer", "wrong-kind", "unknown-kind", "missing-candidate", "unknown-candidate",
            "unknown-choice", "null-choice", "numeric-choice", "noncontiguous-rubric", "missing-legend",
            "invalid-legend-entry", "native-score-mismatch", "null-usage" })
        {
            var root = Root();
            var answers = root["answers"]!.AsObject();
            switch (scenario)
            {
                case "missing-model": root.Remove("model"); break;
                case "null-model": root["model"] = null; break;
                case "missing-answers": root.Remove("answers"); break;
                case "null-answers": root["answers"] = null; break;
                case "missing-answer": answers.Remove("q0"); break;
                case "extra-answer": answers["q3"] = answers["q0"]!.DeepClone(); break;
                case "unknown-answer": answers["Q0"] = answers["q0"]!.DeepClone(); answers.Remove("q0"); break;
                case "wrong-kind": answers["q0"] = answers["q2"]!.DeepClone(); break;
                case "unknown-kind": answers["q0"]!["type"] = "private"; break;
                case "missing-candidate": answers["q1"]!["probabilities"] = new JsonObject { ["1"] = 1 }; break;
                case "unknown-candidate": answers["q1"]!["probabilities"] = new JsonObject { ["00"] = .25, ["1"] = .75 }; break;
                case "unknown-choice": answers["q1"]!["choice"] = "2"; break;
                case "null-choice": answers["q1"]!["choice"] = null; break;
                case "numeric-choice": answers["q1"]!["choice"] = 1; break;
                case "noncontiguous-rubric": answers["q2"]!["probabilities"] = new JsonObject { ["0"] = .2, ["1"] = .3, ["3"] = .5 }; break;
                case "missing-legend": answers["q2"]!["legend"]!.AsObject().Remove("2"); break;
                case "invalid-legend-entry": answers["q2"]!["legend"]!["0"] = 42; break;
                case "native-score-mismatch": answers["q2"]!["score"] = 1.32; break;
                case "null-usage": root["usage"] = null; break;
            }
            yield return [scenario, root.ToJsonString()];
        }
    }

    [Theory]
    [MemberData(nameof(InvalidNativeEnvelopes))]
    public async Task NativeEnvelope_RejectsMalformedOrMismatchedAnswers(string scenario, string body)
    {
        Assert.NotEmpty(scenario);
        await RejectAsync(body);
    }

    [Theory]
    [InlineData("model")]
    [InlineData("answer-id")]
    [InlineData("answer-field")]
    [InlineData("candidate")]
    [InlineData("usage")]
    [InlineData("latency")]
    [InlineData("reasoning-id")]
    [InlineData("reasoning-field")]
    public async Task DuplicateJsonFields_AreRejectedAtEveryEnvelopeLayer(string scenario)
    {
        string body = WithReasoning().ToJsonString();
        var (find, replace) = scenario switch
        {
            "model" => ("\"model\":", "\"model\":\"jeeves-latest\",\"model\":"),
            "answer-id" => ("\"answers\":{\"q0\":", "\"answers\":{\"q0\":{\"type\":\"noul\",\"noul\":0.8},\"q0\":"),
            "answer-field" => ("\"noul\":0.8", "\"noul\":0.8,\"noul\":0.8"),
            "candidate" => ("\"0\":0.25", "\"0\":0.25,\"0\":0.25"),
            "usage" => ("\"input_tokens\":11", "\"input_tokens\":11,\"input_tokens\":11"),
            "latency" => ("\"latency_ms\":12.5", "\"latency_ms\":12.5,\"latency_ms\":12.5"),
            "reasoning-id" => ("\"reasoning\":{\"q0\":", "\"reasoning\":{\"q0\":{\"thought\":false,\"closed\":false,\"tokens\":0,\"text\":\"\"},\"q0\":"),
            _ => ("\"thought\":true", "\"thought\":true,\"thought\":true")
        };
        Assert.Contains(find, body);
        body = body.Replace(find, replace, StringComparison.Ordinal);
        await RejectAsync(body, new JeevesGenerationOptions { ReturnReasoning = true, MaxThink = 4, NoThinkThreshold = .8 });
    }

    public static IEnumerable<object[]> InvalidRoundedNumbers()
    {
        foreach (string target in new[] { "binary", "choice-probability", "score-probability", "choice-confidence", "score-confidence" })
            foreach (double value in new[] { -.01, 1.01, .001, .755, Math.BitIncrement(.75), Math.BitDecrement(.75) })
                yield return [target, JsonSerializer.Serialize(value)];
        foreach (string target in new[] { "binary", "choice-probability", "score-probability", "choice-confidence", "score-confidence" })
            foreach (string raw in new[] { "1e400", "null", "\"0.75\"", "true" })
                yield return [target, raw];
    }

    [Theory]
    [MemberData(nameof(InvalidRoundedNumbers))]
    public async Task NativeProbabilitiesAndConfidence_RequireCanonicalTwoDecimalUnitValues(string target, string raw)
    {
        var root = Root();
        var value = JsonNode.Parse(raw);
        switch (target)
        {
            case "binary": root["answers"]!["q0"]!["noul"] = value; break;
            case "choice-probability": root["answers"]!["q1"]!["probabilities"]!["1"] = value; break;
            case "score-probability": root["answers"]!["q2"]!["probabilities"]!["2"] = value; break;
            case "choice-confidence": root["answers"]!["q1"]!["confidence"] = value; break;
            default: root["answers"]!["q2"]!["confidence"] = value; break;
        }
        await RejectAsync(root.ToJsonString());
    }

    [Theory]
    [InlineData(0d)]
    [InlineData(1d)]
    public async Task NativeConfidence_AcceptsEndpointsWithoutEquatingItToSelectedProbability(double confidence)
    {
        // Deliberate decoder grid stress, not a claim that these probabilities generate
        // both confidence endpoints in the pinned engine.
        var root = Root();
        root["answers"]!["q1"]!["confidence"] = confidence;
        root["answers"]!["q2"]!["confidence"] = confidence;
        var result = await DecodeAsync(root.ToJsonString());
        foreach (var answer in result.Answers.Skip(1))
            Assert.Equal(confidence, Assert.IsType<JsonElement>(answer.RawRepresentation).GetProperty("confidence").GetDouble());
        Assert.Equal(.75, result.GetAnswer<ChoiceDecisionAnswer>("c").Probabilities["A"]);
        Assert.Equal(1.3, result.GetAnswer<ScoreDecisionAnswer>("s").NativeScore);
    }

    [Theory]
    [InlineData(.33, .33, .33, .99)]
    [InlineData(.34, .33, .34, 1.01)]
    public async Task NativeRoundedDistribution_PreservesMassAndIndependentScore(double first, double middle, double last, double mass)
    {
        // Deliberate rounded-decoder stress; not a claimed engine-output confidence calculation.
        var root = Root();
        root["answers"]!["q1"]!["probabilities"] = new JsonObject { ["0"] = first, ["1"] = middle, ["2"] = last };
        root["answers"]!["q2"]!["probabilities"] = new JsonObject { ["0"] = first, ["1"] = middle, ["2"] = last };
        root["answers"]!["q2"]!["score"] = 1;
        var input = Input(Binary(), new ChoiceDecisionQuestion("c", "choose", [new("a", "a"), new("b", "b"), new("c", "c")]), Score());
        var result = await DecodeAsync(root.ToJsonString(), input: input);
        var choice = result.GetAnswer<ChoiceDecisionAnswer>("c");
        var score = result.GetAnswer<ScoreDecisionAnswer>("s");
        Assert.Equal("b", choice.SelectedCandidateId);
        Assert.Equal(new[] { first, middle, last }, choice.Probabilities.Values);
        Assert.Equal(new[] { first, middle, last }, score.Probabilities);
        Assert.Equal(mass, choice.Probabilities.Values.Sum(), 12);
        Assert.Equal(mass, score.Probabilities.Sum(), 12);
        Assert.Equal(mass, score.ExpectedIndex, 12);
        Assert.Equal(1, score.NativeScore);
        Assert.Equal(.995, score.ExpectedIndexLowerBound);
        Assert.Equal(1.005, score.ExpectedIndexUpperBound);
    }

    [Theory]
    [InlineData(HttpStatusCode.BadRequest, "none", null)]
    [InlineData(HttpStatusCode.TooManyRequests, "typesafe", "fallback-id")]
    [InlineData(HttpStatusCode.InternalServerError, "request", "preferred-id")]
    [InlineData(HttpStatusCode.TooManyRequests, "both", "preferred-id")]
    public async Task ErrorStatus_PreservesStatusAndRequestIdPrecedenceWithoutDiagnosticLeak(
        HttpStatusCode status, string headers, string? expectedId)
    {
        const string diagnostic = "private request context and reasoning";
        using var stream = new ObservedStream(Encoding.UTF8.GetBytes(diagnostic));
        using var content = new TrackingContent(stream);
        using var response = new HttpResponseMessage(status) { Content = content };
        if (headers is "typesafe" or "both") response.Headers.Add("x-typesafe-request-id", "fallback-id");
        if (headers is "request" or "both") response.Headers.Add("x-request-id", "preferred-id");
        using var handler = Script(_ => response);
        using var http = new HttpClient(handler);
        using var generator = new JeevesDecisionGenerator(http);
        var error = await Assert.ThrowsAsync<JeevesApiException>(() => generator.GenerateAsync(JevTests.AllKinds(), cancellationToken: Token));
        Assert.Equal(status, error.StatusCode);
        Assert.Equal(expectedId, error.RequestId);
        Assert.Equal(diagnostic, error.DiagnosticBody);
        Assert.Equal($"Jeeves HTTP operation failed with status {(int)status}.", error.Message);
        Assert.DoesNotContain("private", error.ToString());
        Assert.Null(error.InnerException);
        Assert.Single(handler.Requests);
        Assert.True(content.Disposed);
        Assert.True(stream.Disposed);
        Assert.Throws<ObjectDisposedException>(() => response.Content = new ByteArrayContent([]));
    }

    [Theory]
    [InlineData(false, "fallback-id")]
    [InlineData(true, "preferred-id")]
    public async Task SuccessRequestId_UsesSameHeaderPrecedenceAndOwnedMetadata(bool both, string expected)
    {
        using var response = Response(Envelope);
        response.Headers.Add("x-typesafe-request-id", "fallback-id");
        if (both) response.Headers.Add("x-request-id", "preferred-id");
        using var handler = Script(_ => response);
        using var http = new HttpClient(handler);
        using var generator = new JeevesDecisionGenerator(http);
        var result = await generator.GenerateAsync(JevTests.AllKinds(), cancellationToken: Token);
        Assert.Equal(expected, result.AdditionalProperties!["systemone.requestId"]);
        Assert.Equal(.8, result.GetAnswer<BinaryDecisionAnswer>("b").TrueProbability);
        Assert.Equal(12.5, Assert.IsType<JeevesResultMetadata>(result.AdditionalProperties[JeevesDecisionGenerator.MetadataKey]).LatencyMilliseconds);
        Assert.Throws<ObjectDisposedException>(() => response.Content = new ByteArrayContent([]));
        Assert.Single(handler.Requests);
    }

    [Fact]
    public async Task InvalidUtf8_IsRejectedStrictlyBeforeOtherwiseValidJsonAndDisposesResponse()
    {
        // Use an unknown string property so replacement decoding would otherwise succeed.
        byte[] bytes = [.. Encoding.UTF8.GetBytes("{\"diagnostic\":\""), 0xc3, 0x28, .. Encoding.UTF8.GetBytes("\"," + Envelope[1..])];
        using var stream = new ObservedStream(bytes);
        using var content = new TrackingContent(stream);
        using var response = new HttpResponseMessage(HttpStatusCode.OK) { Content = content };
        using var handler = Script(_ => response);
        using var http = new HttpClient(handler);
        using var generator = new JeevesDecisionGenerator(http);
        var error = await Assert.ThrowsAsync<DecisionProtocolException>(() => generator.GenerateAsync(JevTests.AllKinds(), cancellationToken: Token));
        Assert.Equal("Response is not valid UTF-8.", error.Message);
        Assert.True(stream.Disposed);
        Assert.True(content.Disposed);
        Assert.Throws<ObjectDisposedException>(() => response.Content = new ByteArrayContent([]));
        Assert.Single(handler.Requests);
    }

    [Theory]
    [InlineData(HttpStatusCode.TooManyRequests)]
    [InlineData(HttpStatusCode.InternalServerError)]
    public async Task ErrorUtf8_RetainsReplacementDiagnosticWithoutLeakingInException(HttpStatusCode status)
    {
        using var stream = new ObservedStream([.. Encoding.UTF8.GetBytes("private-"), 0xc3, 0x28]);
        using var content = new TrackingContent(stream);
        using var response = new HttpResponseMessage(status) { Content = content };
        response.Headers.Add("x-request-id", "utf8-id");
        using var handler = Script(_ => response);
        using var http = new HttpClient(handler);
        using var generator = new JeevesDecisionGenerator(http);
        var error = await Assert.ThrowsAsync<JeevesApiException>(() => generator.GenerateAsync(JevTests.AllKinds(), cancellationToken: Token));
        Assert.Equal(status, error.StatusCode);
        Assert.Equal("utf8-id", error.RequestId);
        Assert.Equal("private-\uFFFD(", error.DiagnosticBody);
        Assert.DoesNotContain("private-", error.ToString());
        Assert.Null(error.InnerException);
        Assert.True(content.Disposed);
        Assert.True(stream.Disposed);
        Assert.Single(handler.Requests);
    }

    [Theory]
    [InlineData("declared", HttpStatusCode.OK)]
    [InlineData("streamed", HttpStatusCode.OK)]
    [InlineData("underdeclared", HttpStatusCode.OK)]
    [InlineData("declared", HttpStatusCode.InternalServerError)]
    [InlineData("streamed", HttpStatusCode.InternalServerError)]
    public async Task BodyLimit_RejectsDeclaredAndActualOverflowAndDisposes(string mode, HttpStatusCode status)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(Envelope[..^1] + ",\"unused\":\"\u00e9\u6f22\"}");
        using var stream = new ObservedStream(bytes);
        using var content = new TrackingContent(stream);
        if (mode == "declared") content.Headers.ContentLength = bytes.Length;
        if (mode == "underdeclared") content.Headers.ContentLength = 1;
        using var response = new HttpResponseMessage(status) { Content = content };
        using var handler = Script(_ => response);
        using var http = new HttpClient(handler);
        using var generator = new JeevesDecisionGenerator(http, new() { MaximumResponseBytes = bytes.Length - 1 });
        var error = await Assert.ThrowsAsync<JeevesBodyLimitException>(() => generator.GenerateAsync(JevTests.AllKinds(), cancellationToken: Token));
        Assert.Equal("Jeeves response exceeded the configured buffering limit.", error.Message);
        Assert.True(stream.Disposed);
        Assert.True(content.Disposed);
        if (mode == "declared") Assert.Equal(0, stream.ReadCalls);
        else Assert.True(stream.ReadCalls > 0);
        Assert.Throws<ObjectDisposedException>(() => response.Content = new ByteArrayContent([]));
        Assert.Single(handler.Requests);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task BodyLimit_AcceptsExactlyTheUtf8ByteLimit(bool declared)
    {
        // Multi-byte text distinguishes a byte budget from a character budget.
        var root = Root();
        string body = root.ToJsonString()[..^1] + ",\"unused\":\"\u00e9\u6f22\"}";
        byte[] bytes = Encoding.UTF8.GetBytes(body);
        Assert.True(bytes.Length > body.Length);
        using var stream = new ObservedStream(bytes);
        using var content = new TrackingContent(stream);
        if (declared) content.Headers.ContentLength = bytes.Length;
        using var handler = Script(_ => new(HttpStatusCode.OK) { Content = content });
        using var http = new HttpClient(handler);
        using var generator = new JeevesDecisionGenerator(http, new() { MaximumResponseBytes = bytes.Length });
        var result = await generator.GenerateAsync(JevTests.AllKinds(), cancellationToken: Token);
        Assert.Equal(.8, result.GetAnswer<BinaryDecisionAnswer>("b").TrueProbability);
        Assert.Equal(23, result.Usage!.TotalTokenCount);
        Assert.Equal("\u00e9\u6f22", Assert.IsType<JsonElement>(result.RawRepresentation).GetProperty("unused").GetString());
        Assert.True(content.Disposed);
        Assert.True(stream.Disposed);
        Assert.Single(handler.Requests);
    }

    [Fact]
    public async Task PreCancellation_NeverEntersHandler()
    {
        using var handler = Script(_ => Response(Envelope));
        using var http = new HttpClient(handler);
        using var generator = new JeevesDecisionGenerator(http);
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => generator.GenerateAsync(JevTests.AllKinds(), cancellationToken: cancellation.Token));
        Assert.Empty(handler.Requests);
        Assert.Equal(0, handler.DisposeCount);
    }

    [Fact(Timeout = 15000)]
    public async Task CancellationDuringSend_ReachesHandlerAndCompletesWithoutResponse()
    {
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource<HttpResponseMessage>(TaskCreationOptions.RunContinuationsAsynchronously);
        CancellationToken received = default;
        bool completed = false;
        using var handler = new ScriptedHandler(async (_, token) =>
        {
            received = token;
            entered.TrySetResult();
            try { return await release.Task.WaitAsync(token); }
            finally { completed = true; }
        });
        using var http = new HttpClient(handler) { Timeout = Timeout.InfiniteTimeSpan };
        using var generator = new JeevesDecisionGenerator(http);
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(Token);
        var call = generator.GenerateAsync(JevTests.AllKinds(), cancellationToken: cancellation.Token);
        try
        {
            await entered.Task.WaitAsync(TimeSpan.FromSeconds(5), Token);
            Assert.True(received.CanBeCanceled);
            cancellation.Cancel();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => call.WaitAsync(TimeSpan.FromSeconds(5), Token));
            Assert.True(received.IsCancellationRequested);
            Assert.True(completed);
            Assert.Single(handler.Requests);
            Assert.Equal(0, handler.DisposeCount);
        }
        finally
        {
            release.TrySetCanceled(Token);
            try { await call; } catch (Exception) { /* drain fixture */ }
        }
    }

    [Fact(Timeout = 15000)]
    public async Task DerivedStreamContent_EveryReadReceivesCancellationAndCompletesBeforeDisposal()
    {
        using var stream = new StalledStream();
        using var content = new TrackingContent(stream);
        using var response = new HttpResponseMessage(HttpStatusCode.OK) { Content = content };
        using var handler = Script(_ => response);
        using var http = new HttpClient(handler) { Timeout = Timeout.InfiniteTimeSpan };
        using var generator = new JeevesDecisionGenerator(http);
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(Token);
        var call = generator.GenerateAsync(JevTests.AllKinds(), cancellationToken: cancellation.Token);
        try
        {
            // Watchdog bounds regressions; no sleeps or timing-dependent success criterion.
            await stream.Entered.Task.WaitAsync(TimeSpan.FromSeconds(5), Token);
            Assert.Equal(2, stream.ReadTokens.Count);
            Assert.All(stream.ReadTokens, token => Assert.True(token.CanBeCanceled));
            cancellation.Cancel();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => call.WaitAsync(TimeSpan.FromSeconds(5), Token));
            Assert.All(stream.ReadTokens, token => Assert.True(token.IsCancellationRequested));
            Assert.True(stream.ReadCompleted);
            Assert.False(stream.DisposedWhileReading);
            Assert.True(stream.Disposed);
            Assert.True(content.Disposed);
            Assert.Throws<ObjectDisposedException>(() => response.Content = new ByteArrayContent([]));
            Assert.Single(handler.Requests);
        }
        finally
        {
            stream.Release.TrySetCanceled(Token);
            try { await call; } catch (Exception) { /* drain even a broken fixture operation */ }
        }
    }

    [Fact]
    public async Task BodyReadFailure_IsSanitizedAndDisposesSourceAndResponse()
    {
        using var stream = new FaultingStream();
        using var content = new TrackingContent(stream);
        using var response = new HttpResponseMessage(HttpStatusCode.OK) { Content = content };
        using var handler = Script(_ => response);
        using var http = new HttpClient(handler);
        using var generator = new JeevesDecisionGenerator(http);
        var error = await Assert.ThrowsAsync<HttpRequestException>(() => generator.GenerateAsync(JevTests.AllKinds(), cancellationToken: Token));
        Assert.Equal(HttpRequestError.ResponseEnded, error.HttpRequestError);
        Assert.Equal("System One response body read failed.", error.Message);
        Assert.Null(error.InnerException);
        Assert.DoesNotContain("private-stream-secret", error.ToString());
        Assert.True(stream.Disposed);
        Assert.True(content.Disposed);
        Assert.Throws<ObjectDisposedException>(() => response.Content = new ByteArrayContent([]));
        Assert.Single(handler.Requests);
    }

    [Theory]
    [InlineData("https://other.invalid/v1/systemone")]
    [InlineData("http://127.0.0.1:8009/different")]
    public async Task RedirectedResponseUri_IsRejectedWithoutFollowingAnotherRequest(string actual)
    {
        using var redirected = new HttpRequestMessage(HttpMethod.Post, actual);
        using var content = new TrackingContent(new ObservedStream(Encoding.UTF8.GetBytes(Envelope)));
        using var response = new HttpResponseMessage(HttpStatusCode.OK) { Content = content, RequestMessage = redirected };
        using var handler = Script(_ => response);
        using var http = new HttpClient(handler);
        using var generator = new JeevesDecisionGenerator(http);
        var error = await Assert.ThrowsAsync<DecisionProtocolException>(() => generator.GenerateAsync(JevTests.AllKinds(), cancellationToken: Token));
        Assert.Equal("Automatic redirects are not supported; configure HttpClient to disable them.", error.Message);
        Assert.Equal("/v1/systemone", Assert.Single(handler.Requests).Uri.AbsolutePath);
        Assert.True(content.Disposed);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task Disposal_IsIdempotentAndRespectsHttpClientOwnership(bool owns)
    {
        using var handler = Script(_ => Response(Envelope));
        using var http = new HttpClient(handler);
        var generator = new JeevesDecisionGenerator(http, ownsHttpClient: owns);
        await generator.GenerateAsync(JevTests.AllKinds(), cancellationToken: Token);
        generator.Dispose();
        generator.Dispose();
        Assert.Equal(owns ? 1 : 0, handler.DisposeCount);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => generator.GenerateAsync(JevTests.AllKinds(), cancellationToken: Token));
        Assert.Single(handler.Requests);
        if (owns)
            await Assert.ThrowsAsync<ObjectDisposedException>(() => http.GetAsync("http://127.0.0.1:8009/probe", Token));
        else
        {
            using var response = await http.GetAsync("http://127.0.0.1:8009/probe", Token);
            Assert.Equal(HttpStatusCode.OK, response.StatusCode);
            Assert.Equal(2, handler.Requests.Length);
        }
    }

    [Fact]
    public async Task RepeatedResults_OwnRawAnswersUsageAndAliasEvidenceAfterDisposal()
    {
        int calls = 0;
        var contents = new List<TrackingContent>();
        using var handler = Script(_ =>
        {
            var root = Root();
            root["answers"]!["q0"]!["noul"] = ++calls == 1 ? .8 : .1;
            var content = new TrackingContent(new ObservedStream(Encoding.UTF8.GetBytes(root.ToJsonString())));
            contents.Add(content);
            return new(HttpStatusCode.OK) { Content = content };
        });
        using var http = new HttpClient(handler);
        using var generator = new JeevesDecisionGenerator(http);
        var first = await generator.GenerateAsync(JevTests.AllKinds(), cancellationToken: Token);
        var second = await generator.GenerateAsync(JevTests.AllKinds(), cancellationToken: Token);
        generator.Dispose();
        Assert.Equal(2, handler.Requests.Length);
        Assert.Equal(2, contents.Count);
        Assert.All(contents, c => Assert.True(c.Disposed));
        Assert.Equal(.8, Assert.IsType<JsonElement>(first.RawRepresentation).GetProperty("answers").GetProperty("q0").GetProperty("noul").GetDouble());
        Assert.Equal(.1, Assert.IsType<JsonElement>(second.RawRepresentation).GetProperty("answers").GetProperty("q0").GetProperty("noul").GetDouble());
        Assert.Equal(.8, Assert.IsType<JsonElement>(first.Answers[0].RawRepresentation).GetProperty("noul").GetDouble());
        Assert.Equal(.1, second.GetAnswer<BinaryDecisionAnswer>("b").TrueProbability);
        Assert.Equal("1", Assert.IsType<JsonElement>(first.Answers[1].RawRepresentation).GetProperty("choice").GetString());
        Assert.Equal("high", Assert.IsType<JsonElement>(first.Answers[2].RawRepresentation).GetProperty("legend").GetProperty("2").GetString());
        first.Usage!.ReasoningTokenCount = 99;
        first.Usage.AdditionalCounts![JeevesDecisionGenerator.AnswerJsonTokenCountKey] = 98;
        Assert.Equal(5, second.Usage!.ReasoningTokenCount);
        Assert.Equal(7, second.Usage.AdditionalCounts![JeevesDecisionGenerator.AnswerJsonTokenCountKey]);
        Assert.Equal(23, second.Usage.TotalTokenCount);
        Assert.Equal(5, Assert.IsType<JsonElement>(first.RawRepresentation).GetProperty("usage").GetProperty("reasoning_tokens").GetInt64());
        var aliases = Assert.IsAssignableFrom<IReadOnlyDictionary<string, string>>(first.AdditionalProperties!["systemone.questionIds"]);
        Assert.Equal(new Dictionary<string, string> { ["q0"] = "b", ["q1"] = "c", ["q2"] = "s" }, aliases);
        Assert.Throws<NotSupportedException>(() => ((IDictionary<string, string>)aliases).Clear());
        var candidates = Assert.IsAssignableFrom<IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>>>(first.AdditionalProperties["systemone.candidateIds"]);
        Assert.Equal(new Dictionary<string, string> { ["0"] = " a ", ["1"] = "A" }, candidates["q1"]);
        Assert.Throws<NotSupportedException>(() => ((IDictionary<string, string>)candidates["q1"]).Clear());
        Assert.Equal(0, handler.DisposeCount);
    }
}
