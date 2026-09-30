using System.Globalization;
using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using DecisionInference.Jeeves;
using DecisionInference.Ollama;
using DecisionInference.Ollaya;
using DecisionInference.TypeSafe;
using Microsoft.Extensions.AI;
using static DecisionInference.Tests.ContractTests;
using static DecisionInference.Tests.OllayaTestFixtures;

namespace DecisionInference.Tests;

public sealed class OllamaTests
{
    // Offline fixtures grounded in ollama/ollama v0.35.0,
    // cc4069396f3ad2c370c53eed2e4a42ac13adab84:
    // decision/systemone.go Compile/Answer and llm/llama_server_score.go.
    // No model execution, accuracy claim, discovery, or native library use.
    private const string Model = "tev1:0.8b";
    private static CancellationToken Token => TestContext.Current.CancellationToken;
    private static DecisionInput AllKinds() => Input(
        new BinaryDecisionQuestion("b", "is it?", "yes text", "no text"),
        new ChoiceDecisionQuestion("c", "choose", [new("z", "first"), new("a", "second")]), Score());
    private static OllamaDecisionGenerator Generator(HttpClient http, int maximum = 4 * 1024 * 1024) =>
        new(http, new() { ModelId = Model, MaximumResponseBytes = maximum });

    // Independent server-side response fixture, not a copy of the adapter's decoder.
    private static double[] Softmax(params double[] logits)
    {
        var values = logits.Select(x => Math.Exp(x - logits.Max())).ToArray();
        double sum = values.Sum();
        return values.Select(x => x / sum).ToArray();
    }
    private static JsonObject Distribution(double[] p) => new(p.Select((v, i) =>
        KeyValuePair.Create<string, JsonNode?>(i.ToString(CultureInfo.InvariantCulture), JsonValue.Create(v))));
    private static double Concentration(double[] p) =>
        Math.Clamp(1 + p.Where(x => x > 0).Sum(x => x * Math.Log(x)) / Math.Log(p.Length), 0, 1);
    private static JsonObject Native(DecisionInput? input = null, string model = Model, bool uniform = false)
    {
        input ??= AllKinds();
        var answers = new JsonObject();
        for (int i = 0; i < input.Questions.Count; i++)
        {
            var q = input.Questions[i];
            if (q is BinaryDecisionQuestion)
            {
                answers[$"q{i}"] = new JsonObject { ["type"] = "noul", ["noul"] = Softmax(0, 2)[1] };
                continue;
            }
            int count = q is ChoiceDecisionQuestion c ? c.Candidates.Count : ((ScoreDecisionQuestion)q).Rubric.Count;
            double[] logits = uniform ? new double[count] :
                count == 2 ? [0, 2] : Enumerable.Range(0, count).Select(j => (double)j).ToArray();
            double[] p = Softmax(logits);
            var answer = new JsonObject { ["type"] = q is ChoiceDecisionQuestion ? "choice" : "score",
                ["probabilities"] = Distribution(p), ["confidence"] = Concentration(p) };
            if (q is ChoiceDecisionQuestion)
                answer["choice"] = Array.IndexOf(p, p.Max()).ToString(CultureInfo.InvariantCulture);
            else
            {
                answer["score"] = p.Select((v, j) => v * j).Sum();
                answer["legend"] = new JsonObject(((ScoreDecisionQuestion)q).Rubric.Select((s, j) =>
                    KeyValuePair.Create<string, JsonNode?>(j.ToString(CultureInfo.InvariantCulture), JsonValue.Create(s))));
            }
            answers[$"q{i}"] = answer;
        }
        // Full rendered row counts (601+611+624), not cached-prefix-adjusted input.
        return new JsonObject { ["model"] = model,
            ["usage"] = new JsonObject { ["input_tokens"] = 1836, ["output_tokens"] = 7 }, ["answers"] = answers };
    }
    private static async Task<DecisionResult> Decode(string body, DecisionInput? input = null)
    {
        using var handler = Script(_ => Response(body));
        using var http = new HttpClient(handler);
        using var generator = Generator(http);
        var result = await generator.GenerateAsync(input ?? AllKinds(), cancellationToken: Token);
        var request = Assert.Single(handler.Requests);
        Assert.Equal(HttpMethod.Post, request.Method);
        Assert.Equal("http://127.0.0.1:11434/v1/systemone", request.Uri.AbsoluteUri);
        Assert.Null(request.Authorization);
        return result;
    }
    private static async Task Reject(string body, DecisionInput? input = null)
    {
        using var stream = new ObservedStream(Encoding.UTF8.GetBytes(body));
        using var content = new TrackingContent(stream);
        using var response = new HttpResponseMessage(HttpStatusCode.OK) { Content = content };
        using var handler = Script(_ => response);
        using var http = new HttpClient(handler);
        using var generator = Generator(http);
        var error = await Assert.ThrowsAsync<DecisionProtocolException>(() =>
            generator.GenerateAsync(input ?? AllKinds(), cancellationToken: Token));
        Assert.Null(error.InnerException);
        Assert.Single(handler.Requests);
        Assert.True(content.Disposed);
        Assert.True(stream.Disposed);
        Assert.Throws<ObjectDisposedException>(() => response.Content = new ByteArrayContent([]));
    }

    [Fact]
    public async Task AllKinds_OneUnauthenticatedPostFullSchemaAndUnroundedNativeAnswers()
    {
        using var handler = Script(_ => Response(Native().ToJsonString()));
        using var http = new HttpClient(handler)
            { BaseAddress = new("https://unused.invalid"), Timeout = TimeSpan.FromSeconds(37) };
        using var generator = Generator(http);
        var metadata = generator.GetRequiredService<DecisionGeneratorMetadata>();
        Assert.Equal("ollama", metadata.ProviderName);
        Assert.Equal(new Uri("http://127.0.0.1:11434"), metadata.ProviderUri);
        Assert.Equal(Model, metadata.DefaultModelId);
        Assert.Equal([DecisionKind.Binary, DecisionKind.Choice, DecisionKind.Score], metadata.SupportedKinds);
        Assert.Same(generator, generator.GetService<IDecisionGenerator>());
        Assert.Same(generator, generator.GetService(typeof(OllamaDecisionGenerator)));
        Assert.Null(generator.GetService(typeof(IDecisionGenerator), "key"));
        Assert.Null(generator.GetService(typeof(string)));
        Assert.Throws<ArgumentNullException>(() => generator.GetService(null!));
        Assert.Empty(handler.Requests);

        var result = await generator.GenerateAsync(AllKinds(), cancellationToken: Token);
        var request = Assert.Single(handler.Requests);
        Assert.Equal(HttpMethod.Post, request.Method);
        Assert.Equal("http://127.0.0.1:11434/v1/systemone", request.Uri.AbsoluteUri);
        Assert.Null(request.Authorization);
        Assert.Equal("application/json", request.MediaType);
        var body = Json(request.Body);
        Assert.Equal(["state", "model", "questions"], body.EnumerateObject().Select(p => p.Name));
        Assert.Equal("state", body.GetProperty("state").GetString());
        Assert.Equal(Model, body.GetProperty("model").GetString());
        var questions = body.GetProperty("questions");
        Assert.Equal(["q0", "q1", "q2"], questions.EnumerateObject().Select(p => p.Name));
        Assert.Equal(["noul", "choice", "score"], questions.EnumerateObject().Select(p => p.Value.GetProperty("type").GetString()));
        Assert.Equal(["is it?", "choose", "rate"], questions.EnumerateObject().Select(p => p.Value.GetProperty("instructions").GetString()));
        var criteria = questions.GetProperty("q0").GetProperty("criteria");
        Assert.Equal("yes text", criteria.GetProperty("true").GetString());
        Assert.Equal("no text", criteria.GetProperty("false").GetString());
        Assert.Equal(["0", "1"], questions.GetProperty("q1").GetProperty("criteria").EnumerateObject().Select(p => p.Name));
        Assert.Equal(["first", "second"], questions.GetProperty("q1").GetProperty("criteria").EnumerateObject().Select(p => p.Value.GetString()));
        Assert.Equal(["low", "mid", "high"], questions.GetProperty("q2").GetProperty("criteria").EnumerateArray().Select(p => p.GetString()));
        Assert.Equal(["b", "c", "s"], result.Answers.Select(a => a.QuestionId));
        Assert.Equal(Model, result.ModelId);
        Assert.All(result.Answers, a => Assert.Equal(DecisionPrecision.HighPrecision, a.Precision));
        Assert.Equal(.8807970779778823, result.GetAnswer<BinaryDecisionAnswer>("b").TrueProbability);
        var choice = result.GetAnswer<ChoiceDecisionAnswer>("c");
        Assert.Equal("a", choice.SelectedCandidateId);
        Assert.Equal(["z", "a"], choice.Probabilities.Keys);
        Assert.Equal(.11920292202211755, choice.Probabilities["z"], 15);
        Assert.Equal(.8807970779778823, choice.Probabilities["a"]);
        var score = result.GetAnswer<ScoreDecisionAnswer>("s");
        Assert.Equal(3, score.Probabilities.Count);
        Assert.Equal(.09003057317038046, score.Probabilities[0], 15);
        Assert.Equal(.24472847105479764, score.Probabilities[1], 15);
        Assert.Equal(.6652409557748218, score.Probabilities[2], 15);
        Assert.Equal(1.5752103826044412, score.NativeScore!.Value, 14);
        Assert.Equal(score.NativeScore.Value, score.ExpectedIndex);
        var raw = Assert.IsType<JsonElement>(score.RawRepresentation);
        Assert.Equal(["low", "mid", "high"], raw.GetProperty("legend").EnumerateObject().Select(p => p.Value.GetString()));
        Assert.Equal(Concentration(Softmax(0, 1, 2)), raw.GetProperty("confidence").GetDouble());
        Assert.Equal(1836, result.Usage!.InputTokenCount);
        Assert.Equal(7, result.Usage.OutputTokenCount);
        Assert.Equal(1843, result.Usage.TotalTokenCount);
        Assert.Null(result.Usage.ReasoningTokenCount);
        Assert.Null(result.Usage.CachedInputTokenCount);
        Assert.True(result.Usage.AdditionalCounts is null or { Count: 0 });
        Assert.Equal(["systemone.questionIds", "systemone.candidateIds"], result.AdditionalProperties!.Keys);
        Assert.Same(metadata, generator.GetRequiredService<DecisionGeneratorMetadata>());
        Assert.Equal(TimeSpan.FromSeconds(37), http.Timeout);
        Assert.Equal(new Uri("https://unused.invalid"), http.BaseAddress);
    }

    [Fact]
    public async Task ChoiceAndScore_AllCountsTwoThroughTwentySix_PreserveCompleteUniformDistributions()
    {
        for (int n = 2; n <= 26; n++)
        {
            var input = Input(new ChoiceDecisionQuestion("c", "choose", Enumerable.Range(0, n).Select(i => new DecisionCandidate($"id{i}", $"description{i}"))),
                new ScoreDecisionQuestion("s", "rate", Enumerable.Range(0, n).Select(i => $"level{i}")));
            using var handler = Script(_ => Response(Native(input, uniform: true).ToJsonString()));
            using var http = new HttpClient(handler);
            using var generator = Generator(http);
            var result = await generator.GenerateAsync(input, cancellationToken: Token);
            var choice = result.GetAnswer<ChoiceDecisionAnswer>("c");
            var score = result.GetAnswer<ScoreDecisionAnswer>("s");
            Assert.Equal("id0", choice.SelectedCandidateId);
            Assert.Equal(Enumerable.Range(0, n).Select(i => $"id{i}"), choice.Probabilities.Keys);
            Assert.Equal(n, score.Probabilities.Count);
            Assert.All(choice.Probabilities.Values.Concat(score.Probabilities), p => Assert.Equal(1d / n, p));
            Assert.Equal((n - 1) / 2d, score.NativeScore!.Value, 12);
            Assert.Equal(1, score.Probabilities.Sum(), 12);
            Assert.All(result.Answers, a => Assert.Equal(DecisionPrecision.HighPrecision, a.Precision));
            var sent = Json(Assert.Single(handler.Requests).Body).GetProperty("questions");
            Assert.Equal(n, sent.GetProperty("q0").GetProperty("criteria").EnumerateObject().Count());
            Assert.Equal(Enumerable.Range(0, n).Select(i => $"description{i}"), sent.GetProperty("q0").GetProperty("criteria").EnumerateObject().Select(p => p.Value.GetString()));
            Assert.Equal(Enumerable.Range(0, n).Select(i => $"level{i}"), sent.GetProperty("q1").GetProperty("criteria").EnumerateArray().Select(p => p.GetString()));
        }
    }

    [Theory]
    [InlineData("choice", 1)]
    [InlineData("choice", 27)]
    [InlineData("score", 27)]
    public async Task UnsupportedCandidateCounts_RejectBeforeNetwork(string kind, int count)
    {
        DecisionQuestion q = kind == "choice"
            ? new ChoiceDecisionQuestion("c", "choose", Enumerable.Range(0, count).Select(i => new DecisionCandidate($"id{i}", $"description{i}")))
            : new ScoreDecisionQuestion("s", "rate", Enumerable.Range(0, count).Select(i => $"level{i}"));
        using var handler = Script(_ => throw new InvalidOperationException("Must not send."));
        using var http = new HttpClient(handler);
        using var generator = Generator(http);
        await Assert.ThrowsAsync<ArgumentException>(() => generator.GenerateAsync(Input(q), cancellationToken: Token));
        Assert.Empty(handler.Requests);
    }

    [Fact]
    public async Task QuestionCount64Accepted65Rejected_AndPortableEmptyAndOneLevelRejected()
    {
        // Interacting maxima in one batch: 64 questions, all3 kinds, Choice26 and Score26.
        var input = Input([.. Enumerable.Range(0, 62).Select(i => (DecisionQuestion)new BinaryDecisionQuestion($"id{i}", "assess")),
            new ChoiceDecisionQuestion("id62", "choose", Enumerable.Range(0, 26).Select(i => new DecisionCandidate($"c{i}", $"candidate{i}"))),
            new ScoreDecisionQuestion("id63", "rate", Enumerable.Range(0, 26).Select(i => $"level{i}"))]);
        using var handler = Script(_ => Response(Native(input, uniform: true).ToJsonString()));
        using var http = new HttpClient(handler);
        using var generator = Generator(http);
        var result = await generator.GenerateAsync(input, cancellationToken: Token);
        Assert.Equal(input.Questions.Select(q => q.Id), result.Answers.Select(a => a.QuestionId));
        Assert.All(result.Answers.Take(62), a => Assert.Equal(.8807970779778823, Assert.IsType<BinaryDecisionAnswer>(a).TrueProbability));
        var choice = result.GetAnswer<ChoiceDecisionAnswer>("id62");
        Assert.Equal("c0", choice.SelectedCandidateId);
        Assert.Equal(Enumerable.Range(0, 26).Select(i => $"c{i}"), choice.Probabilities.Keys);
        Assert.All(choice.Probabilities.Values, p => Assert.Equal(1d / 26, p));
        var score = result.GetAnswer<ScoreDecisionAnswer>("id63");
        Assert.Equal(Enumerable.Repeat(1d / 26, 26), score.Probabilities);
        Assert.Equal(12.5, score.NativeScore!.Value, 12);
        var questions = Json(Assert.Single(handler.Requests).Body).GetProperty("questions");
        Assert.Equal(64, questions.EnumerateObject().Count());
        Assert.Equal(26, questions.GetProperty("q62").GetProperty("criteria").EnumerateObject().Count());
        Assert.Equal(Enumerable.Range(0, 26).Select(i => $"level{i}"), questions.GetProperty("q63").GetProperty("criteria").EnumerateArray().Select(v => v.GetString()));
        await Assert.ThrowsAsync<ArgumentException>(() => generator.GenerateAsync(Input([.. input.Questions, new BinaryDecisionQuestion("id64", "assess")]), cancellationToken: Token));
        Assert.Throws<ArgumentException>(() => Input([]));
        Assert.Throws<ArgumentException>(() => new ScoreDecisionQuestion("s", "rate", ["only"]));
        Assert.Single(handler.Requests);
    }

    [Theory]
    [InlineData("typesafe")]
    [InlineData("ollaya")]
    [InlineData("jeeves")]
    public async Task LegacyScoreLimit_TenAcceptedElevenRejectedBeforeEvenDiscovery(string provider)
    {
        var ten = Input(new ScoreDecisionQuestion("s", "rate", Enumerable.Range(0, 10).Select(i => $"level{i}")));
        var root = Native(ten, provider == "ollaya" ? OllayaTestFixtures.Model : provider == "jeeves" ? "jeeves-latest" : "jev-1.13.0");
        // Exact point mass is legitimate at every legacy precision; Jeeves requires
        // its own latency/reasoning usage fields, and Ollaya its own discovery shape.
        root["answers"]!["q0"]!["probabilities"] = Distribution([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
        root["answers"]!["q0"]!["score"] = 9;
        root["answers"]!["q0"]!["confidence"] = 1;
        if (provider == "jeeves") { root["usage"]!["reasoning_tokens"] = 0; root["latency_ms"] = 1; }
        using var handler = Script(r => Response(r.Uri.AbsolutePath == "/api/show" ? NliShow : root.ToJsonString()));
        using var http = new HttpClient(handler);
        using IDecisionGenerator generator = provider switch
        {
            // Only the legacy TypeSafe contract requires a synthetic unit-test credential.
            "typesafe" => new TypeSafeDecisionGenerator(http, "unit-test-only"),
            "ollaya" => new OllayaDecisionGenerator(http, new() { ModelId = OllayaTestFixtures.Model }),
            _ => new JeevesDecisionGenerator(http)
        };
        await Assert.ThrowsAsync<ArgumentException>(() => generator.GenerateAsync(
            Input(new ScoreDecisionQuestion("s", "rate", Enumerable.Range(0, 11).Select(i => $"level{i}"))), cancellationToken: Token));
        Assert.Empty(handler.Requests);
        var result = await generator.GenerateAsync(ten, cancellationToken: Token);
        var score = result.GetAnswer<ScoreDecisionAnswer>("s");
        Assert.Equal(10, score.Probabilities.Count);
        Assert.Equal(Enumerable.Repeat(0d, 9).Append(1d), score.Probabilities);
        Assert.Equal(9, score.NativeScore);
        Assert.Equal(provider == "ollaya" ? 2 : 1, handler.Requests.Length);
        Assert.Equal("/v1/systemone", handler.Requests[^1].Uri.AbsolutePath);
    }

    [Theory]
    [InlineData("\" meaningful \"", true)]
    [InlineData("{}", true)]
    [InlineData("[]", true)]
    [InlineData("{\"text\":\"é\",\"nested\":[1,true]}", true)]
    [InlineData("[\"item\",{}]", true)]
    [InlineData("null", false)]
    [InlineData("true", false)]
    [InlineData("false", false)]
    [InlineData("12", false)]
    [InlineData("1.5", false)]
    [InlineData("\"\"", false)]
    [InlineData("\" \\t\\r\\n\"", false)]
    public async Task StateKinds_OnlyNonblankStringsObjectsAndArraysReachWire(string state, bool accepted)
    {
        using var handler = Script(_ => Response(Native().ToJsonString()));
        using var http = new HttpClient(handler);
        using var generator = Generator(http);
        var input = new DecisionInput(Json(state), AllKinds().Questions);
        if (accepted)
        {
            var result = await generator.GenerateAsync(input, cancellationToken: Token);
            Assert.Equal(.8807970779778823, result.GetAnswer<BinaryDecisionAnswer>("b").TrueProbability);
            Assert.True(JsonElement.DeepEquals(Json(state), Json(Assert.Single(handler.Requests).Body).GetProperty("state")));
        }
        else
        {
            await Assert.ThrowsAsync<ArgumentException>(() => generator.GenerateAsync(input, cancellationToken: Token));
            Assert.Empty(handler.Requests);
        }
    }

    [Theory]
    [InlineData("choice", "")]
    [InlineData("choice", " \t")]
    [InlineData("score", "")]
    [InlineData("score", " \t")]
    [InlineData("binary", null)]
    [InlineData("binary", "")]
    [InlineData("binary", " \t")]
    public async Task BlankInstructions_RejectIncludingDescriptionsOnlyBinary(string kind, string? instructions)
    {
        DecisionQuestion q = kind switch
        {
            "choice" => new ChoiceDecisionQuestion("q", instructions, [new("a", "first"), new("b", "second")]),
            "score" => new ScoreDecisionQuestion("q", instructions, ["low", "high"]),
            _ => new BinaryDecisionQuestion("q", instructions, "yes text", "no text")
        };
        using var handler = Script(_ => throw new InvalidOperationException("Must not send."));
        using var http = new HttpClient(handler);
        using var generator = Generator(http);
        await Assert.ThrowsAsync<ArgumentException>(() => generator.GenerateAsync(Input(q), cancellationToken: Token));
        Assert.Empty(handler.Requests);
    }

    [Theory]
    [InlineData("{\"a\":1,\"a\":2}")]
    [InlineData("[{\"nested\":{\"a\":1,\"a\":1}}]")]
    public async Task DuplicateStateProperties_RejectBeforeNetworkUnchangedSharedBehavior(string state)
    {
        using var handler = Script(_ => throw new InvalidOperationException("Must not send."));
        using var http = new HttpClient(handler);
        using var generator = Generator(http);
        await Assert.ThrowsAsync<JsonException>(() => generator.GenerateAsync(new(Json(state), AllKinds().Questions), cancellationToken: Token));
        Assert.Empty(handler.Requests);
    }

    [Fact]
    public async Task OpaqueIds_ReorderedResponsesAndProbabilityKeysKeepRequestOrderAndFirstTie()
    {
        string[] ids = [" q ", "q", "Q", "\u00e9", "e\u0301"];
        // First request candidate 'z' sorts after 'a'; response reverses both maps.
        string[] candidates = ["z", "a", " A ", "\u00e9", "e\u0301"];
        var input = Input(ids.Select(id => (DecisionQuestion)new ChoiceDecisionQuestion(id, "literal \"instruction\"",
            candidates.Select(c => new DecisionCandidate(c, $"meaning {c}")))).ToArray());
        var root = Native(input, uniform: true);
        var answers = root["answers"]!.AsObject();
        foreach (var (_, answer) in answers)
            answer!["probabilities"] = new JsonObject(answer["probabilities"]!.AsObject().Reverse()
                .Select(p => KeyValuePair.Create<string, JsonNode?>(p.Key, p.Value!.DeepClone())));
        root["answers"] = new JsonObject(answers.Reverse().Select(p => KeyValuePair.Create<string, JsonNode?>(p.Key, p.Value!.DeepClone())));
        using var handler = Script(_ => Response(root.ToJsonString()));
        using var http = new HttpClient(handler);
        using var generator = Generator(http);
        var result = await generator.GenerateAsync(input, cancellationToken: Token);
        var sent = Json(Assert.Single(handler.Requests).Body).GetProperty("questions");
        Assert.Equal(["q0", "q1", "q2", "q3", "q4"], sent.EnumerateObject().Select(p => p.Name));
        foreach (var q in sent.EnumerateObject())
        {
            Assert.Equal("literal \"instruction\"", q.Value.GetProperty("instructions").GetString());
            Assert.Equal(["0", "1", "2", "3", "4"], q.Value.GetProperty("criteria").EnumerateObject().Select(p => p.Name));
            Assert.Equal(candidates.Select(c => $"meaning {c}"), q.Value.GetProperty("criteria").EnumerateObject().Select(p => p.Value.GetString()));
        }
        Assert.Equal(ids, result.Answers.Select(a => a.QuestionId));
        foreach (var answer in result.Answers.Cast<ChoiceDecisionAnswer>())
        {
            Assert.Equal("z", answer.SelectedCandidateId);
            Assert.Equal(candidates, answer.Probabilities.Keys);
            Assert.All(answer.Probabilities.Values, p => Assert.Equal(.2, p));
        }
        var aliases = Assert.IsAssignableFrom<IReadOnlyDictionary<string, string>>(result.AdditionalProperties!["systemone.questionIds"]);
        var candidateAliases = Assert.IsAssignableFrom<IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>>>(result.AdditionalProperties["systemone.candidateIds"]);
        Assert.Equal(ids, Enumerable.Range(0, 5).Select(i => aliases[$"q{i}"]));
        Assert.Equal(candidates, Enumerable.Range(0, 5).Select(i => candidateAliases["q0"][$"{i}"]));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task SelectedChoice_RejectsNonmaximumAndLaterExactTie(bool tied)
    {
        var root = Native(uniform: tied);
        root["answers"]!["q1"]!["choice"] = tied ? "1" : "0";
        await Reject(root.ToJsonString());
    }

    [Theory]
    [InlineData(null)]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(-123)]
    [InlineData(123)]
    [InlineData(int.MinValue)]
    [InlineData(int.MaxValue)]
    public async Task KeepAlive_NullOmittedOtherwiseIntegerSecondsIncludingExtrema(int? seconds)
    {
        using var handler = Script(_ => Response(Native().ToJsonString()));
        using var http = new HttpClient(handler);
        using var generator = Generator(http);
        var options = new OllamaGenerationOptions { KeepAliveSeconds = seconds, AdditionalProperties = new() };
        var result = await generator.GenerateAsync(AllKinds(), options, Token);
        var body = Json(Assert.Single(handler.Requests).Body);
        Assert.Equal(Model, body.GetProperty("model").GetString());
        Assert.Equal(seconds is null ? new[] { "state", "model", "questions" } : ["state", "model", "questions", "keep_alive"],
            body.EnumerateObject().Select(p => p.Name));
        if (seconds is not null)
        {
            var value = body.GetProperty("keep_alive");
            Assert.Equal(JsonValueKind.Number, value.ValueKind);
            Assert.Equal(seconds.Value.ToString(CultureInfo.InvariantCulture), value.GetRawText());
            Assert.Equal(seconds, value.GetInt32());
        }
        Assert.Equal(seconds, options.KeepAliveSeconds);
        Assert.Empty(options.AdditionalProperties);
        Assert.Equal(1843, result.Usage!.TotalTokenCount);
    }

    [Fact]
    public async Task ModelSelection_NullDefaultRequiresPerCallAndOverridesAreNonstickyWithoutControls()
    {
        using var handler = Script(r => Response(Native(model: Json(r.Body).GetProperty("model").GetString()!).ToJsonString()));
        using var http = new HttpClient(handler);
        using (var noDefault = new OllamaDecisionGenerator(http))
        {
            Assert.Null(noDefault.GetRequiredService<DecisionGeneratorMetadata>().DefaultModelId);
            await Assert.ThrowsAnyAsync<ArgumentException>(() => noDefault.GenerateAsync(AllKinds(), cancellationToken: Token));
            Assert.Empty(handler.Requests);
            var result = await noDefault.GenerateAsync(AllKinds(), new() { ModelId = " per Call " }, Token);
            Assert.Equal(" per Call ", result.ModelId);
            Assert.Null(noDefault.GetRequiredService<DecisionGeneratorMetadata>().DefaultModelId);
        }
        using var generator = Generator(http);
        DecisionGenerationOptions?[] options = [null, new(), new() { AdditionalProperties = new() },
            new OllamaGenerationOptions(), new OllamaGenerationOptions { ModelId = " Changed " }, new()];
        string[] expected = [Model, Model, Model, Model, " Changed ", Model];
        for (int i = 0; i < options.Length; i++)
        {
            var result = await generator.GenerateAsync(AllKinds(), options[i], Token);
            Assert.Equal(expected[i], result.ModelId);
            var body = Json(handler.Requests[^1].Body);
            Assert.Equal(expected[i], body.GetProperty("model").GetString());
            Assert.Equal(["state", "model", "questions"], body.EnumerateObject().Select(p => p.Name));
            Assert.Equal(Model, generator.GetRequiredService<DecisionGeneratorMetadata>().DefaultModelId);
        }
        foreach (string blank in new[] { "", " \t" })
            await Assert.ThrowsAnyAsync<ArgumentException>(() => generator.GenerateAsync(AllKinds(), new() { ModelId = blank }, Token));
        Assert.Equal(7, handler.Requests.Length);
    }

    [Fact]
    public void Clone_PreservesDerivedBaseStateAndIndependentShallowExtensionDictionary()
    {
        var shared = new object();
        var original = new OllamaGenerationOptions { ModelId = " clone ", KeepAliveSeconds = -3,
            AdditionalProperties = new() { ["value"] = shared } };
        DecisionGenerationOptions viaBase = original;
        var clone = Assert.IsType<OllamaGenerationOptions>(viaBase.Clone());
        OllamaGenerationOptions covariant = original.Clone();
        Assert.NotSame(original, clone);
        Assert.Equal(" clone ", clone.ModelId);
        Assert.Equal(-3, clone.KeepAliveSeconds);
        Assert.Equal(-3, covariant.KeepAliveSeconds);
        Assert.NotSame(original.AdditionalProperties, clone.AdditionalProperties);
        Assert.Same(shared, clone.AdditionalProperties!["value"]);
        clone.AdditionalProperties["new"] = 2;
        clone.AdditionalProperties.Remove("value");
        clone.ModelId = "other"; clone.KeepAliveSeconds = 9;
        Assert.Equal(" clone ", original.ModelId);
        Assert.Equal(-3, original.KeepAliveSeconds);
        Assert.Equal(["value"], original.AdditionalProperties.Keys);
        Assert.Same(shared, original.AdditionalProperties["value"]);
        Assert.Null(new OllamaGenerationOptions().Clone().AdditionalProperties);
        Assert.Null(new OllamaGenerationOptions().Clone().KeepAliveSeconds);
    }

    private sealed class ForeignOptions : DecisionGenerationOptions;

    [Fact]
    public async Task ForeignOptionsAndNonemptyExtensionBags_RejectBeforeNetwork()
    {
        DecisionGenerationOptions[] cases = [new JeevesGenerationOptions(), new ForeignOptions(),
            new ForeignOptions { AdditionalProperties = new() }, new JeevesGenerationOptions { AdditionalProperties = new() },
            new() { AdditionalProperties = new() { ["temperature"] = 0 } },
            new OllamaGenerationOptions { AdditionalProperties = new() { ["keep_alive"] = 1 } }];
        using var handler = Script(_ => throw new InvalidOperationException("Must not send."));
        using var http = new HttpClient(handler);
        using var generator = Generator(http);
        foreach (var options in cases)
            await Assert.ThrowsAsync<NotSupportedException>(() => generator.GenerateAsync(AllKinds(), options, Token));
        Assert.Empty(handler.Requests);
    }

    [Fact]
    public async Task CallerOptions_SnapshotAcrossAwaitWithoutProviderMutation()
    {
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var handler = new ScriptedHandler(async (r, token) =>
        {
            using var registration = token.Register(() => release.TrySetCanceled(token));
            entered.TrySetResult();
            await release.Task;
            return Response(Native(model: Json(r.Body).GetProperty("model").GetString()!).ToJsonString());
        });
        using var http = new HttpClient(handler);
        using var generator = Generator(http);
        var options = new OllamaGenerationOptions { ModelId = "original", KeepAliveSeconds = -1, AdditionalProperties = new() };
        var call = generator.GenerateAsync(AllKinds(), options, Token);
        await entered.Task;
        Assert.Equal("original", options.ModelId);
        Assert.Equal(-1, options.KeepAliveSeconds);
        Assert.Empty(options.AdditionalProperties);
        options.ModelId = "changed"; options.KeepAliveSeconds = 123; options.AdditionalProperties["later"] = true;
        release.TrySetResult();
        var result = await call;
        Assert.Equal("original", result.ModelId);
        var sent = Json(Assert.Single(handler.Requests).Body);
        Assert.Equal("original", sent.GetProperty("model").GetString());
        Assert.Equal(-1, sent.GetProperty("keep_alive").GetInt32());
        Assert.Equal("changed", options.ModelId);
        Assert.Equal(123, options.KeepAliveSeconds);
        Assert.Equal(true, options.AdditionalProperties["later"]);
        Assert.Equal(Model, generator.GetRequiredService<DecisionGeneratorMetadata>().DefaultModelId);
    }

    // Captures the actual emitted bytes, not a reconstructed request serializer.
    private sealed class ByteHandler(Func<byte[], HttpResponseMessage> reply) : HttpMessageHandler
    {
        internal List<byte[]> Bodies { get; } = [];
        internal List<HttpRequestMessage> Messages { get; } = [];
        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Messages.Add(request);
            var bytes = await request.Content!.ReadAsByteArrayAsync(cancellationToken);
            Bodies.Add(bytes);
            return reply(bytes);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task RequestBytes_Exact65535And65536Accepted65537Rejected_WithEscapingUnicodeAndOptions(bool keepAlive)
    {
        const string marker = "PADMARKER";
        const string complex = "é漢😀 \"quoted\"\n\t\u0001\\";
        DecisionInput Make(int padding) => new(JsonSerializer.SerializeToElement(complex + marker + new string('x', padding)),
            [new ChoiceDecisionQuestion(" q😀 ", complex, [new("z", "first " + complex), new("a", "second " + complex)])]);
        string model = "model " + complex;
        var options = new OllamaGenerationOptions { ModelId = model, KeepAliveSeconds = keepAlive ? int.MinValue : null };
        using var handler = new ByteHandler(_ => Response(Native(Make(0), model).ToJsonString()));
        using var http = new HttpClient(handler);
        using var generator = Generator(http);
        await generator.GenerateAsync(Make(0), options, Token);
        byte[] small = Assert.Single(handler.Bodies);
        string text = Encoding.UTF8.GetString(small);
        Assert.Contains("\\uD83D\\uDE00", text, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("\\u0001", text);
        Assert.Contains("\\n", text);
        Assert.True(text.Contains("\\u0022", StringComparison.OrdinalIgnoreCase) || text.Contains("\\\"", StringComparison.Ordinal));
        Assert.Equal(model, Json(text).GetProperty("model").GetString());
        Assert.Equal(keepAlive, Json(text).TryGetProperty("keep_alive", out _));
        // All complex fields remain fixed. Each ASCII 'x' costs exactly one byte;
        // independently splice the captured wire body at its unique state marker.
        int insertion = text.IndexOf(marker, StringComparison.Ordinal) + marker.Length;
        Assert.True(insertion >= marker.Length);
        foreach (int target in new[] { 65_535, 65_536, 65_537 })
        {
            int padding = target - small.Length;
            byte[] expected = Encoding.UTF8.GetBytes(text.Insert(insertion, new string('x', padding)));
            Assert.Equal(target, expected.Length);
            if (target == 65_537)
            {
                int sends = handler.Bodies.Count;
                await Assert.ThrowsAsync<ArgumentException>(() => generator.GenerateAsync(Make(padding), options, Token));
                Assert.Equal(sends, handler.Bodies.Count);
            }
            else
            {
                var result = await generator.GenerateAsync(Make(padding), options, Token);
                Assert.Equal(expected, handler.Bodies[^1]);
                Assert.Equal(target, handler.Bodies[^1].Length);
                Assert.Equal("a", result.GetAnswer<ChoiceDecisionAnswer>(" q😀 ").SelectedCandidateId);
            }
        }
        Assert.Equal(3, handler.Bodies.Count);
    }

    [Fact]
    public async Task KeepAlivePresence_AloneTipsExactlyAcceptedBodyOverRequestLimit()
    {
        DecisionInput Make(int count) => new(JsonSerializer.SerializeToElement("😀\n\"" + new string('x', count)), [Binary()]);
        using var handler = new ByteHandler(_ => Response(Native(Make(0)).ToJsonString()));
        using var http = new HttpClient(handler);
        using var generator = Generator(http);
        await generator.GenerateAsync(Make(0), cancellationToken: Token);
        int plainLength = handler.Bodies[^1].Length;
        await generator.GenerateAsync(Make(0), new OllamaGenerationOptions { KeepAliveSeconds = 0 }, Token);
        Assert.Equal(plainLength + Encoding.UTF8.GetByteCount(",\"keep_alive\":0"), handler.Bodies[^1].Length);
        var boundary = Make(65_536 - plainLength);
        await generator.GenerateAsync(boundary, cancellationToken: Token);
        Assert.Equal(65_536, handler.Bodies[^1].Length);
        await Assert.ThrowsAsync<ArgumentException>(() => generator.GenerateAsync(boundary, new OllamaGenerationOptions { KeepAliveSeconds = 0 }, Token));
        Assert.Equal(3, handler.Bodies.Count);
    }

    [Theory]
    [InlineData(0L, 0L, false)]
    [InlineData(1836L, 7L, false)]
    [InlineData(1836L, 19L, true)]
    [InlineData(long.MaxValue, 0L, true)]
    [InlineData(1L, long.MaxValue - 1, true)]
    public async Task Usage_RetainsFullRenderedInputAndInternalScoringOutputIncludingZeroAndInt64Limit(
        long input, long output, bool includeTotal)
    {
        var root = Native();
        root["usage"] = new JsonObject { ["input_tokens"] = input, ["output_tokens"] = output };
        if (includeTotal) root["usage"]!["total_tokens"] = checked(input + output);
        var result = await Decode(root.ToJsonString());
        Assert.Equal(input, result.Usage!.InputTokenCount);
        Assert.Equal(output, result.Usage.OutputTokenCount);
        Assert.Equal(checked(input + output), result.Usage.TotalTokenCount);
        Assert.Null(result.Usage.ReasoningTokenCount);
        Assert.Null(result.Usage.CachedInputTokenCount);
        Assert.True(result.Usage.AdditionalCounts is null or { Count: 0 });
        Assert.Equal(3, result.Answers.Count);
        Assert.Equal(2, result.AdditionalProperties!.Count);
        var usage = Assert.IsType<JsonElement>(result.RawRepresentation).GetProperty("usage");
        Assert.Equal(output, usage.GetProperty("output_tokens").GetInt64());
        Assert.Equal(includeTotal, usage.TryGetProperty("total_tokens", out _));
    }

    public static IEnumerable<object[]> InvalidUsage()
    {
        foreach (string field in new[] { "input_tokens", "output_tokens" })
            foreach (string value in new[] { "missing", "-1", "null", "true", "\"7\"", "7.0", "0.5", "[]", "{}", "1e400", "9223372036854775808" })
            {
                var root = Native();
                if (value == "missing") root["usage"]!.AsObject().Remove(field);
                else root["usage"]![field] = JsonNode.Parse(value);
                yield return [$"{field}:{value}", root.ToJsonString()];
            }
        foreach (string value in new[] { "null", "true", "[]", "\"1843\"", "1843.0", "1842", "-1", "1e400" })
        {
            var root = Native();
            root["usage"]!["total_tokens"] = JsonNode.Parse(value);
            yield return [$"total_tokens:{value}", root.ToJsonString()];
        }
        var overflow = Native();
        overflow["usage"]!["input_tokens"] = long.MaxValue;
        overflow["usage"]!["output_tokens"] = 1;
        yield return ["checked-sum-overflow", overflow.ToJsonString()];
    }

    [Theory]
    [MemberData(nameof(InvalidUsage))]
    public async Task Usage_RejectsMissingWrongKindsNonintegerNegativeNonfiniteAndOverflow(string scenario, string body)
    {
        Assert.NotEmpty(scenario);
        await Reject(body);
    }

    [Theory]
    [InlineData("missing")]
    [InlineData("null")]
    [InlineData("\"\"")]
    [InlineData("\" \"")]
    [InlineData("\"TEV1:0.8b\"")]
    [InlineData("\" tev1:0.8b\"")]
    [InlineData("\"tev1:0.8b \"")]
    [InlineData("42")]
    [InlineData("true")]
    [InlineData("[]")]
    [InlineData("{}")]
    public async Task ModelEcho_RequiresExactOrdinalLabelIncludingCaseAndWhitespace(string value)
    {
        var root = Native();
        if (value == "missing") root.Remove("model");
        else root["model"] = JsonNode.Parse(value);
        await Reject(root.ToJsonString());
    }

    public static IEnumerable<object[]> InvalidEnvelopes()
    {
        foreach (string text in new[] { "{private", "null", "[]", "true", "1", "\"text\"", "{}", Native().ToJsonString() + "{}" })
            yield return ["malformed-root:" + text, text];
        foreach (string field in new[] { "usage", "answers" })
            foreach (string value in new[] { "missing", "null", "[]", "true", "1", "\"text\"" })
            {
                var root = Native();
                if (value == "missing") root.Remove(field);
                else root[field] = JsonNode.Parse(value);
                yield return [$"{field}:{value}", root.ToJsonString()];
            }
        foreach (var (id, fields) in new[] {
            ("q0", new[] { "type", "noul" }),
            ("q1", new[] { "type", "choice", "confidence", "probabilities" }),
            ("q2", new[] { "type", "score", "confidence", "probabilities", "legend" }) })
            foreach (string field in fields)
            {
                var root = Native();
                root["answers"]![id]!.AsObject().Remove(field);
                yield return [$"missing:{id}.{field}", root.ToJsonString()];
            }
        foreach (var (id, field) in new[] { ("q0", "type"), ("q1", "choice"), ("q1", "probabilities"),
            ("q2", "probabilities"), ("q2", "legend") })
            foreach (string value in new[] { "null", "true", "42", "[]", "\"text\"" })
            {
                var root = Native();
                root["answers"]![id]![field] = JsonNode.Parse(value);
                yield return [$"shape:{id}.{field}:{value}", root.ToJsonString()];
            }
        foreach (string scenario in new[] { "null-answer", "array-answer", "missing-question", "extra-question", "unknown-question",
            "wrong-kind", "unknown-kind", "missing-candidate", "extra-candidate", "unknown-candidate", "unknown-choice",
            "missing-score-index", "extra-score-index", "noncontiguous-score-index", "noncanonical-score-index",
            "missing-legend-index", "extra-legend-index", "unknown-legend-index", "legend-case", "legend-whitespace",
            "legend-null", "legend-object", "legend-array", "legend-number", "native-score-mismatch", "choice-mass", "score-mass" })
        {
            var root = Native();
            var answers = root["answers"]!.AsObject();
            var cp = answers["q1"]!["probabilities"]!.AsObject();
            var sp = answers["q2"]!["probabilities"]!.AsObject();
            var legend = answers["q2"]!["legend"]!.AsObject();
            switch (scenario)
            {
                case "null-answer": answers["q0"] = null; break;
                case "array-answer": answers["q0"] = new JsonArray(); break;
                case "missing-question": answers.Remove("q0"); break;
                case "extra-question": answers["q3"] = answers["q0"]!.DeepClone(); break;
                case "unknown-question": answers["Q0"] = answers["q0"]!.DeepClone(); answers.Remove("q0"); break;
                case "wrong-kind": answers["q0"] = answers["q2"]!.DeepClone(); break;
                case "unknown-kind": answers["q0"]!["type"] = "binary"; break;
                case "missing-candidate": cp.Remove("0"); cp["1"] = 1; break;
                case "extra-candidate": cp["2"] = 0; break;
                case "unknown-candidate": cp["00"] = cp["0"]!.DeepClone(); cp.Remove("0"); break;
                case "unknown-choice": answers["q1"]!["choice"] = "a"; break; // caller ID is not a wire alias
                case "missing-score-index": sp.Remove("2"); break;
                case "extra-score-index": sp["3"] = 0; break;
                case "noncontiguous-score-index": sp["3"] = sp["2"]!.DeepClone(); sp.Remove("2"); break;
                case "noncanonical-score-index": sp["02"] = sp["2"]!.DeepClone(); sp.Remove("2"); break;
                case "missing-legend-index": legend.Remove("2"); break;
                case "extra-legend-index": legend["3"] = "extra"; break;
                case "unknown-legend-index": legend["02"] = legend["2"]!.DeepClone(); legend.Remove("2"); break;
                case "legend-case": legend["0"] = "Low"; break;
                case "legend-whitespace": legend["0"] = "low "; break;
                case "legend-null": legend["0"] = null; break;
                case "legend-object": legend["0"] = new JsonObject(); break;
                case "legend-array": legend["0"] = new JsonArray(); break;
                case "legend-number": legend["0"] = 0; break;
                case "native-score-mismatch": answers["q2"]!["score"] = 1; break;
                case "choice-mass": cp["0"] = .1; cp["1"] = .8; break;
                case "score-mass": sp["0"] = .1; sp["1"] = .2; sp["2"] = .6; answers["q2"]!["score"] = 1.4; break;
            }
            yield return [scenario, root.ToJsonString()];
        }
        // Same number of questions, internally valid shorter/longer ordinal shapes:
        // these isolate correlation to the REQUESTED rubric length, not mass validation.
        foreach (int count in new[] { 2, 4 })
        {
            var root = Native();
            root["answers"]!["q2"]!["probabilities"] = Distribution(Enumerable.Repeat(1d / count, count).ToArray());
            root["answers"]!["q2"]!["score"] = (count - 1) / 2d;
            root["answers"]!["q2"]!["legend"] = new JsonObject(Enumerable.Range(0, count).Select(i =>
                KeyValuePair.Create<string, JsonNode?>(i.ToString(CultureInfo.InvariantCulture), JsonValue.Create("level"))));
            yield return [$"complete-but-wrong-rubric-size:{count}", root.ToJsonString()];
        }
    }

    [Theory]
    [MemberData(nameof(InvalidEnvelopes))]
    public async Task Envelope_RejectsMalformedIncompleteWrongShapesKindsAliasesAndRubrics(string scenario, string body)
    {
        Assert.NotEmpty(scenario);
        await Reject(body);
    }

    public static IEnumerable<object[]> InvalidNumbers()
    {
        foreach (string target in new[] { "binary", "choice-probability", "score-probability", "choice-confidence", "score-confidence", "native-score" })
            foreach (string value in new[] { "-0.01", "null", "\"0.5\"", "true", "[]", "{}", "1e400", "-1e400", "NaN", "Infinity" })
                yield return [target, value];
        foreach (string target in new[] { "binary", "choice-probability", "score-probability", "choice-confidence", "score-confidence" })
            yield return [target, "1.01"];
        yield return ["native-score", "2.01"];
    }

    [Theory]
    [MemberData(nameof(InvalidNumbers))]
    public async Task NumericFields_RejectNonfiniteWrongTypesAndOutsideDomain(string target, string value)
    {
        var root = Native();
        JsonNode answer = root["answers"]![target == "binary" ? "q0" : target.StartsWith("choice", StringComparison.Ordinal) ? "q1" : "q2"]!;
        var parent = target.EndsWith("probability", StringComparison.Ordinal) ? answer["probabilities"]! : answer;
        string property = target switch
        {
            "binary" => "noul", "native-score" => "score",
            "choice-probability" or "score-probability" => "0", _ => "confidence"
        };
        parent[property] = "NUMERIC_PLACEHOLDER";
        await Reject(root.ToJsonString().Replace("\"NUMERIC_PLACEHOLDER\"", value, StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("model")]
    [InlineData("question")]
    [InlineData("answer-field")]
    [InlineData("candidate")]
    [InlineData("score-index")]
    [InlineData("legend")]
    [InlineData("usage")]
    [InlineData("ignored-nested-array")]
    public async Task DuplicateProperties_RejectAtEveryLevelIncludingUnknownNestedMetadata(string target)
    {
        var root = Native();
        string body = root.ToJsonString();
        var (find, replacement) = target switch
        {
            "model" => ("\"model\":", "\"model\":\"duplicate\",\"model\":"),
            "question" => ("\"answers\":{", "\"answers\":{\"q0\":{\"type\":\"noul\",\"noul\":0.5},"),
            "answer-field" => ("\"noul\":", "\"noul\":0.5,\"noul\":"),
            "candidate" => (root["answers"]!["q1"]!["probabilities"]!.ToJsonString(),
                "{\"0\":0.5," + root["answers"]!["q1"]!["probabilities"]!.ToJsonString()[1..]),
            "score-index" => (root["answers"]!["q2"]!["probabilities"]!.ToJsonString(),
                "{\"0\":0.5," + root["answers"]!["q2"]!["probabilities"]!.ToJsonString()[1..]),
            "legend" => ("\"legend\":{", "\"legend\":{\"0\":\"low\","),
            "usage" => ("\"input_tokens\":", "\"input_tokens\":0,\"input_tokens\":"),
            _ => ("\"model\":", "\"unused\":[{\"a\":1,\"a\":1}],\"model\":")
        };
        Assert.Contains(find, body);
        await Reject(body.Replace(find, replacement, StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("choice")]
    [InlineData("score")]
    public async Task RoundedDocumentationMassPoint9999_IsNotAcceptedAsHighPrecision(string kind)
    {
        var input = Input(new ChoiceDecisionQuestion("c", "choose", [new("a", "first"), new("b", "second"), new("c", "third")]), Score());
        var root = Native(input);
        string id = kind == "choice" ? "q0" : "q1";
        root["answers"]![id]!["probabilities"] = Distribution([.0125, .9781, .0093]);
        if (kind == "choice") root["answers"]![id]!["choice"] = "1";
        else root["answers"]![id]!["score"] = .9781 + 2 * .0093;
        await Reject(root.ToJsonString(), input);
        Assert.Equal(1e-6, DecisionAnswer.DistributionTolerance);
    }

    [Fact]
    public async Task HighPrecision_PreservesEveryDoubleWithoutRoundingOrRenormalization()
    {
        double[] expected = Softmax(0, 1, 2);
        var root = Native();
        var result = await Decode(root.ToJsonString());
        var score = result.GetAnswer<ScoreDecisionAnswer>("s");
        Assert.Equal(expected, score.Probabilities); // exact, not a decimal-place comparison
        Assert.Equal(expected.Select((p, i) => p * i).Sum(), score.NativeScore);
        Assert.Equal(expected.Sum(), score.Probabilities.Sum());
        Assert.Equal(DecisionPrecision.HighPrecision, score.Precision);
        Assert.Equal(root["answers"]!["q2"]!["confidence"]!.GetValue<double>(),
            Assert.IsType<JsonElement>(score.RawRepresentation).GetProperty("confidence").GetDouble());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ProbabilityEndpoints_ExactZerosOnesAndNativeOrdinalExtremesAreAccepted(bool last)
    {
        // Finite large logit gaps can underflow exp to zero in the pinned scorer.
        var root = Native();
        double[] choice = last ? Softmax(-1000, 0) : Softmax(0, -1000);
        double[] score = last ? Softmax(-1000, -1000, 0) : Softmax(0, -1000, -1000);
        root["answers"]!["q0"]!["noul"] = choice[1];
        root["answers"]!["q1"]!["probabilities"] = Distribution(choice);
        root["answers"]!["q1"]!["choice"] = last ? "1" : "0";
        root["answers"]!["q1"]!["confidence"] = Concentration(choice);
        root["answers"]!["q2"]!["probabilities"] = Distribution(score);
        root["answers"]!["q2"]!["score"] = last ? 2 : 0;
        root["answers"]!["q2"]!["confidence"] = Concentration(score);
        var result = await Decode(root.ToJsonString());
        Assert.Equal(last ? 1d : 0d, result.GetAnswer<BinaryDecisionAnswer>("b").TrueProbability);
        Assert.Equal(last ? "a" : "z", result.GetAnswer<ChoiceDecisionAnswer>("c").SelectedCandidateId);
        Assert.Equal(last ? new[] { 0d, 1d } : [1d, 0d], result.GetAnswer<ChoiceDecisionAnswer>("c").Probabilities.Values);
        var answer = result.GetAnswer<ScoreDecisionAnswer>("s");
        Assert.Equal(last ? new[] { 0d, 0d, 1d } : [1d, 0d, 0d], answer.Probabilities);
        Assert.Equal(last ? 2d : 0d, answer.NativeScore);
        Assert.Equal(last ? 2d : 0d, answer.ExpectedIndex);
        Assert.Equal(1, Assert.IsType<JsonElement>(answer.RawRepresentation).GetProperty("confidence").GetDouble());
    }

    [Theory]
    [InlineData(0d)]
    [InlineData(1d)]
    [InlineData(.123456789012345d)]
    public async Task Confidence_IsRetainedRawConcentrationNotCorrectnessOrFormulaRevalidation(double confidence)
    {
        // Deliberate decoder-domain stress, not a claim these particular distributions
        // generate each concentration. Adapter must not invent a formula validator.
        var root = Native();
        root["answers"]!["q1"]!["confidence"] = confidence;
        root["answers"]!["q2"]!["confidence"] = confidence;
        var result = await Decode(root.ToJsonString());
        foreach (var answer in result.Answers.Skip(1))
            Assert.Equal(confidence, Assert.IsType<JsonElement>(answer.RawRepresentation).GetProperty("confidence").GetDouble());
        Assert.Equal(.8807970779778823, result.GetAnswer<ChoiceDecisionAnswer>("c").Probabilities["a"]);
        Assert.Equal(Softmax(0, 1, 2), result.GetAnswer<ScoreDecisionAnswer>("s").Probabilities);
    }

    [Fact]
    public async Task Results_OwnRawEnvelopesAnswersAliasesRequestIdsAndUsageAcrossDisposalAndLaterCalls()
    {
        int calls = 0;
        var contents = new List<TrackingContent>();
        var messages = new List<HttpResponseMessage>();
        using var handler = Script(_ =>
        {
            var root = Native();
            if (++calls == 2)
            {
                root["answers"]!["q0"]!["noul"] = .25;
                root["usage"]!["input_tokens"] = 99;
            }
            using var document = JsonDocument.Parse(root.ToJsonString());
            var content = new TrackingContent(new ObservedStream(Encoding.UTF8.GetBytes(document.RootElement.GetRawText())));
            contents.Add(content);
            var response = new HttpResponseMessage(HttpStatusCode.OK) { Content = content };
            response.Headers.Add("x-request-id", $"request-{calls}");
            messages.Add(response);
            return response;
        });
        using var http = new HttpClient(handler);
        var generator = Generator(http);
        DecisionInput firstInput;
        using (var document = JsonDocument.Parse("{\"state\":\"owned\"}"))
            firstInput = new(document.RootElement, AllKinds().Questions);
        var first = await generator.GenerateAsync(firstInput, cancellationToken: Token);
        var secondInput = Input(new BinaryDecisionQuestion("new-b", "assess"),
            new ChoiceDecisionQuestion("new-c", "choose", [new("new-z", "first"), new("new-a", "second")]),
            new ScoreDecisionQuestion("new-s", "rate", ["low", "mid", "high"]));
        var second = await generator.GenerateAsync(secondInput, cancellationToken: Token);
        generator.Dispose();
        Assert.All(contents, c => Assert.True(c.Disposed));
        Assert.All(messages, r => Assert.Throws<ObjectDisposedException>(() => r.Content = new ByteArrayContent([])));
        Assert.Equal(.8807970779778823, first.GetAnswer<BinaryDecisionAnswer>("b").TrueProbability);
        Assert.Equal(.25, second.GetAnswer<BinaryDecisionAnswer>("new-b").TrueProbability);
        Assert.Equal(.8807970779778823, Assert.IsType<JsonElement>(first.Answers[0].RawRepresentation).GetProperty("noul").GetDouble());
        Assert.Equal(.8807970779778823, Assert.IsType<JsonElement>(first.RawRepresentation).GetProperty("answers").GetProperty("q0").GetProperty("noul").GetDouble());
        Assert.Equal("request-1", first.AdditionalProperties!["systemone.requestId"]);
        Assert.Equal("request-2", second.AdditionalProperties!["systemone.requestId"]);
        var firstAliases = Assert.IsAssignableFrom<IReadOnlyDictionary<string, string>>(first.AdditionalProperties["systemone.questionIds"]);
        var secondAliases = Assert.IsAssignableFrom<IReadOnlyDictionary<string, string>>(second.AdditionalProperties["systemone.questionIds"]);
        Assert.Equal("b", firstAliases["q0"]);
        Assert.Equal("new-b", secondAliases["q0"]);
        var candidates = Assert.IsAssignableFrom<IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>>>(first.AdditionalProperties["systemone.candidateIds"]);
        Assert.Equal("z", candidates["q1"]["0"]);
        Assert.Equal("a", candidates["q1"]["1"]);
        Assert.Equal(1836, first.Usage!.InputTokenCount);
        Assert.Equal(1843, first.Usage.TotalTokenCount);
        Assert.Equal(106, second.Usage!.TotalTokenCount);
        second.Usage.InputTokenCount = 0;
        second.AdditionalProperties["systemone.requestId"] = "modified";
        Assert.Equal(1836, first.Usage.InputTokenCount);
        Assert.Equal("request-1", first.AdditionalProperties["systemone.requestId"]);
        Assert.Equal(2, handler.Requests.Length);
    }

    [Theory]
    [InlineData("https://provider.invalid:8443/", true)]
    [InlineData("http://127.0.0.1:11434/", true)]
    [InlineData("http://127.0.0.2:11434/", true)]
    [InlineData("http://[::1]:11434/", true)]
    [InlineData("http://localhost:11434/", false)]
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
        using var handler = Script(_ => throw new InvalidOperationException("Must not send."));
        using var http = new HttpClient(handler);
        var options = new OllamaDecisionGeneratorOptions { Endpoint = new(endpoint, UriKind.RelativeOrAbsolute) };
        if (accepted)
        {
            using var generator = new OllamaDecisionGenerator(http, options);
            var metadata = generator.GetRequiredService<DecisionGeneratorMetadata>();
            Assert.Equal(options.Endpoint, metadata.ProviderUri);
            Assert.Equal("ollama", metadata.ProviderName);
            Assert.Null(metadata.DefaultModelId);
        }
        else Assert.Throws<ArgumentException>(() => new OllamaDecisionGenerator(http, options));
        Assert.Empty(handler.Requests);
        Assert.Equal(0, handler.DisposeCount);
    }

    [Fact]
    public async Task ConstructorDefaultsAndNullInput_ValidateWithoutOwningBorrowedClient()
    {
        using var handler = Script(_ => throw new InvalidOperationException("Must not send."));
        using var http = new HttpClient(handler);
        var defaults = new OllamaDecisionGeneratorOptions();
        Assert.Equal(new Uri("http://127.0.0.1:11434"), defaults.Endpoint);
        Assert.Null(defaults.ModelId);
        Assert.Equal(4 * 1024 * 1024, defaults.MaximumResponseBytes);
        Assert.Throws<ArgumentNullException>(() => new OllamaDecisionGenerator(null!));
        Assert.Throws<ArgumentNullException>(() => new OllamaDecisionGenerator(http, new() { Endpoint = null! }));
        foreach (string label in new[] { "", " \t" })
            Assert.ThrowsAny<ArgumentException>(() => new OllamaDecisionGenerator(http, new() { ModelId = label }));
        foreach (int maximum in new[] { 0, -1 })
            Assert.Throws<ArgumentOutOfRangeException>(() => new OllamaDecisionGenerator(http, new() { MaximumResponseBytes = maximum }));
        using var generator = Generator(http);
        await Assert.ThrowsAsync<ArgumentNullException>(() => generator.GenerateAsync(null!, cancellationToken: Token));
        Assert.Empty(handler.Requests);
        Assert.Equal(0, handler.DisposeCount);
    }

    [Fact]
    public async Task Authorization_RejectsAmbientHeaderAtConstructionAndAfterDrift()
    {
        using var handler = Script(_ => Response(Native().ToJsonString()));
        using var http = new HttpClient(handler);
        http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", "unit-test-private");
        var construction = Assert.Throws<ArgumentException>(() => new OllamaDecisionGenerator(http));
        Assert.DoesNotContain("unit-test-private", construction.ToString());
        http.DefaultRequestHeaders.Authorization = null;
        using var generator = Generator(http);
        await generator.GenerateAsync(AllKinds(), cancellationToken: Token);
        http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Basic", "unit-test-private");
        var drift = await Assert.ThrowsAsync<InvalidOperationException>(() => generator.GenerateAsync(AllKinds(), cancellationToken: Token));
        Assert.DoesNotContain("unit-test-private", drift.ToString());
        Assert.Null(Assert.Single(handler.Requests).Authorization);
        Assert.Equal(0, handler.DisposeCount);
    }

    [Theory]
    [InlineData(HttpStatusCode.BadRequest, "none", null)]
    [InlineData(HttpStatusCode.NotFound, "fallback", "fallback-id")]
    [InlineData(HttpStatusCode.RequestEntityTooLarge, "preferred", "preferred-id")]
    [InlineData(HttpStatusCode.InternalServerError, "both", "preferred-id")]
    public async Task ApiErrors_PreserveStatusMalformedDiagnosticAndRequestIdWithoutLeaking(
        HttpStatusCode status, string headers, string? requestId)
    {
        const string diagnostic = "{malformed private prompt";
        using var stream = new ObservedStream(Encoding.UTF8.GetBytes(diagnostic));
        using var content = new TrackingContent(stream);
        using var response = new HttpResponseMessage(status) { Content = content };
        if (headers is "fallback" or "both") response.Headers.Add("x-typesafe-request-id", "fallback-id");
        if (headers is "preferred" or "both") response.Headers.Add("x-request-id", "preferred-id");
        using var handler = new ByteHandler(_ => response);
        using var http = new HttpClient(handler);
        using var generator = Generator(http);
        var error = await Assert.ThrowsAsync<OllamaApiException>(() => generator.GenerateAsync(AllKinds(), cancellationToken: Token));
        Assert.Equal(status, error.StatusCode);
        Assert.Equal(requestId, error.RequestId);
        Assert.Equal(diagnostic, error.DiagnosticBody);
        Assert.Equal($"Ollama HTTP operation failed with status {(int)status}.", error.Message);
        Assert.DoesNotContain("private", error.ToString());
        Assert.Null(error.InnerException);
        Assert.True(stream.Disposed);
        Assert.True(content.Disposed);
        Assert.Throws<ObjectDisposedException>(() => response.Content = new ByteArrayContent([]));
        Assert.Throws<ObjectDisposedException>(() => Assert.Single(handler.Messages).Content = new ByteArrayContent([]));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task SuccessfulResponse_RequestIdFallbackAndMessageContentAreOwnedAndDisposed(bool both)
    {
        using var stream = new ObservedStream(Encoding.UTF8.GetBytes(Native().ToJsonString()));
        using var content = new TrackingContent(stream);
        using var response = new HttpResponseMessage(HttpStatusCode.OK) { Content = content };
        response.Headers.Add("x-typesafe-request-id", "fallback-id");
        if (both) response.Headers.Add("x-request-id", "preferred-id");
        using var handler = new ByteHandler(_ => response);
        using var http = new HttpClient(handler);
        using var generator = Generator(http);
        var result = await generator.GenerateAsync(AllKinds(), cancellationToken: Token);
        Assert.Equal(both ? "preferred-id" : "fallback-id", result.AdditionalProperties!["systemone.requestId"]);
        Assert.Equal(.8807970779778823, result.GetAnswer<BinaryDecisionAnswer>("b").TrueProbability);
        Assert.True(content.Disposed);
        Assert.True(stream.Disposed);
        Assert.Throws<ObjectDisposedException>(() => response.Content = new ByteArrayContent([]));
        var request = Assert.Single(handler.Messages);
        Assert.Throws<ObjectDisposedException>(() => request.Content = new ByteArrayContent([]));
        await Assert.ThrowsAsync<ObjectDisposedException>(() => request.Content!.ReadAsByteArrayAsync(Token));
        Assert.Null(request.Headers.Authorization);
    }

    [Theory]
    [InlineData("known", HttpStatusCode.OK)]
    [InlineData("unknown", HttpStatusCode.OK)]
    [InlineData("dishonest", HttpStatusCode.OK)]
    [InlineData("known", HttpStatusCode.InternalServerError)]
    [InlineData("unknown", HttpStatusCode.InternalServerError)]
    [InlineData("dishonest", HttpStatusCode.InternalServerError)]
    public async Task ResponseBodyLimits_KnownUnknownDishonestLengthsRejectOneByteOverflow(
        string lengthMode, HttpStatusCode status)
    {
        string body = Native().ToJsonString()[..^1] + ",\"unused\":\"é漢😀\"}";
        byte[] bytes = Encoding.UTF8.GetBytes(body);
        Assert.True(bytes.Length > body.Length);
        using var stream = new ObservedStream(bytes);
        using var content = new TrackingContent(stream);
        if (lengthMode == "known") content.Headers.ContentLength = bytes.Length;
        if (lengthMode == "dishonest") content.Headers.ContentLength = 1;
        using var response = new HttpResponseMessage(status) { Content = content };
        using var handler = Script(_ => response);
        using var http = new HttpClient(handler);
        using var generator = Generator(http, bytes.Length - 1);
        var error = await Assert.ThrowsAsync<OllamaBodyLimitException>(() => generator.GenerateAsync(AllKinds(), cancellationToken: Token));
        Assert.Equal("Ollama response exceeded the configured buffering limit.", error.Message);
        Assert.True(content.Disposed);
        Assert.True(stream.Disposed);
        if (lengthMode == "known") Assert.Equal(0, stream.ReadCalls);
        else Assert.True(stream.ReadCalls > 0);
        Assert.Single(handler.Requests);
        Assert.Throws<ObjectDisposedException>(() => response.Content = new ByteArrayContent([]));
    }

    [Theory]
    [InlineData("known", HttpStatusCode.OK)]
    [InlineData("unknown", HttpStatusCode.OK)]
    [InlineData("dishonest", HttpStatusCode.OK)]
    [InlineData("known", HttpStatusCode.InternalServerError)]
    [InlineData("unknown", HttpStatusCode.InternalServerError)]
    [InlineData("dishonest", HttpStatusCode.InternalServerError)]
    public async Task ResponseBodyLimits_ExactUtf8LimitAcceptedOnSuccessAndError(string lengthMode, HttpStatusCode status)
    {
        string body = Native().ToJsonString()[..^1] + ",\"unused\":\"é漢😀\"}";
        byte[] bytes = Encoding.UTF8.GetBytes(body);
        using var stream = new ObservedStream(bytes);
        using var content = new TrackingContent(stream);
        if (lengthMode == "known") content.Headers.ContentLength = bytes.Length;
        if (lengthMode == "dishonest") content.Headers.ContentLength = 1;
        using var response = new HttpResponseMessage(status) { Content = content };
        using var handler = Script(_ => response);
        using var http = new HttpClient(handler);
        using var generator = Generator(http, bytes.Length);
        if (status == HttpStatusCode.OK)
        {
            var result = await generator.GenerateAsync(AllKinds(), cancellationToken: Token);
            Assert.Equal("é漢😀", Assert.IsType<JsonElement>(result.RawRepresentation).GetProperty("unused").GetString());
            Assert.Equal(1843, result.Usage!.TotalTokenCount);
        }
        else
        {
            var error = await Assert.ThrowsAsync<OllamaApiException>(() => generator.GenerateAsync(AllKinds(), cancellationToken: Token));
            Assert.Equal(status, error.StatusCode);
            Assert.Equal(body, error.DiagnosticBody);
        }
        Assert.True(content.Disposed);
        Assert.True(stream.Disposed);
        Assert.True(stream.ReadCalls > 0);
        Assert.Single(handler.Requests);
    }

    [Theory]
    [InlineData(HttpStatusCode.OK)]
    [InlineData(HttpStatusCode.InternalServerError)]
    public async Task Utf8_SuccessIsStrictErrorDiagnosticsRemainBoundedReplacementText(HttpStatusCode status)
    {
        // Invalid encoding in an ignored field would pass JSON parsing with replacement decoding.
        byte[] bytes = [.. Encoding.UTF8.GetBytes("{\"unused\":\"private-"), 0xc3, 0x28,
            .. Encoding.UTF8.GetBytes("\"," + Native().ToJsonString()[1..])];
        using var stream = new ObservedStream(bytes);
        using var content = new TrackingContent(stream);
        using var response = new HttpResponseMessage(status) { Content = content };
        using var handler = Script(_ => response);
        using var http = new HttpClient(handler);
        using var generator = Generator(http, bytes.Length);
        if (status == HttpStatusCode.OK)
        {
            var error = await Assert.ThrowsAsync<DecisionProtocolException>(() => generator.GenerateAsync(AllKinds(), cancellationToken: Token));
            Assert.Equal("Response is not valid UTF-8.", error.Message);
        }
        else
        {
            var error = await Assert.ThrowsAsync<OllamaApiException>(() => generator.GenerateAsync(AllKinds(), cancellationToken: Token));
            Assert.Equal(status, error.StatusCode);
            Assert.Equal(Encoding.UTF8.GetString(bytes), error.DiagnosticBody);
            Assert.Contains("private-\uFFFD(", error.DiagnosticBody);
            Assert.DoesNotContain("private", error.ToString());
        }
        Assert.True(content.Disposed);
        Assert.True(stream.Disposed);
        Assert.Single(handler.Requests);
    }

    [Fact]
    public async Task BodyReadFailure_IsSanitizedAndDisposesResponse()
    {
        using var stream = new FaultingStream();
        using var content = new TrackingContent(stream);
        using var response = new HttpResponseMessage(HttpStatusCode.OK) { Content = content };
        using var handler = Script(_ => response);
        using var http = new HttpClient(handler);
        using var generator = Generator(http);
        var error = await Assert.ThrowsAsync<HttpRequestException>(() => generator.GenerateAsync(AllKinds(), cancellationToken: Token));
        Assert.Equal(HttpRequestError.ResponseEnded, error.HttpRequestError);
        Assert.Equal("System One response body read failed.", error.Message);
        Assert.Null(error.InnerException);
        Assert.DoesNotContain("private-stream-secret", error.ToString());
        Assert.True(content.Disposed);
        Assert.True(stream.Disposed);
        Assert.Throws<ObjectDisposedException>(() => response.Content = new ByteArrayContent([]));
        Assert.Single(handler.Requests);
    }

    [Fact]
    public async Task PreCancellation_NeverEntersHandler()
    {
        using var handler = Script(_ => throw new InvalidOperationException("Must not send."));
        using var http = new HttpClient(handler);
        using var generator = Generator(http);
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => generator.GenerateAsync(AllKinds(), cancellationToken: cancellation.Token));
        Assert.Empty(handler.Requests);
        Assert.Equal(0, handler.DisposeCount);
    }

    [Fact(Timeout = 15000)]
    public async Task CancellationDuringHeaders_IsCooperativeAndCompletesHandlerWithoutAbandonment()
    {
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource<HttpResponseMessage>(TaskCreationOptions.RunContinuationsAsynchronously);
        CancellationToken received = default;
        bool completed = false;
        using var handler = new ScriptedHandler(async (_, token) =>
        {
            received = token;
            using var registration = token.Register(() => release.TrySetCanceled(token));
            entered.TrySetResult();
            try { return await release.Task; }
            finally { completed = true; }
        });
        using var http = new HttpClient(handler) { Timeout = Timeout.InfiniteTimeSpan };
        using var generator = Generator(http);
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(Token);
        var call = generator.GenerateAsync(AllKinds(), cancellationToken: cancellation.Token);
        try
        {
            await entered.Task;
            Assert.True(received.CanBeCanceled);
            cancellation.Cancel();
            Assert.True(received.IsCancellationRequested);
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => call);
            Assert.True(completed);
            Assert.Single(handler.Requests);
            Assert.Equal(0, handler.DisposeCount);
        }
        finally
        {
            release.TrySetCanceled(Token);
            try { await call; } catch (OperationCanceledException) { /* drain cooperative fixture */ }
        }
    }

    [Fact(Timeout = 15000)]
    public async Task CancellationDuringBody_EachReadGetsTokenAndFinishesBeforeContentDisposal()
    {
        using var stream = new StalledStream();
        using var content = new TrackingContent(stream);
        using var response = new HttpResponseMessage(HttpStatusCode.OK) { Content = content };
        using var handler = Script(_ => response);
        using var http = new HttpClient(handler) { Timeout = Timeout.InfiniteTimeSpan };
        using var generator = Generator(http);
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(Token);
        var call = generator.GenerateAsync(AllKinds(), cancellationToken: cancellation.Token);
        try
        {
            await stream.Entered.Task;
            Assert.Equal(2, stream.ReadTokens.Count);
            Assert.All(stream.ReadTokens, t => Assert.True(t.CanBeCanceled));
            cancellation.Cancel();
            Assert.All(stream.ReadTokens, t => Assert.True(t.IsCancellationRequested));
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => call);
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
            try { await call; } catch (OperationCanceledException) { /* drain cooperative fixture */ }
        }
    }

    [Theory]
    [InlineData(HttpStatusCode.Redirect)]
    [InlineData(HttpStatusCode.TemporaryRedirect)]
    [InlineData(HttpStatusCode.PermanentRedirect)]
    public async Task RedirectStatus_IsNotFollowedAndNeverSuppliesOffOriginAuthorization(HttpStatusCode status)
    {
        using var response = Response("redirect diagnostic", status);
        response.Headers.Location = new("https://other.invalid/v1/systemone");
        using var handler = Script(_ => response);
        using var http = new HttpClient(handler);
        using var generator = Generator(http);
        var error = await Assert.ThrowsAsync<OllamaApiException>(() => generator.GenerateAsync(AllKinds(), cancellationToken: Token));
        Assert.Equal(status, error.StatusCode);
        Assert.Equal("redirect diagnostic", error.DiagnosticBody);
        var sent = Assert.Single(handler.Requests);
        Assert.Equal("http://127.0.0.1:11434/v1/systemone", sent.Uri.AbsoluteUri);
        Assert.Null(sent.Authorization);
        Assert.Throws<ObjectDisposedException>(() => response.Content = new ByteArrayContent([]));
    }

    [Theory]
    [InlineData("https://other.invalid/v1/systemone")]
    [InlineData("http://127.0.0.1:11434/other")]
    [InlineData("http://127.0.0.1:11435/v1/systemone")]
    public async Task RedirectedEffectiveUri_IsRejectedWithoutAnotherRequestOrAuthorization(string actual)
    {
        using var redirected = new HttpRequestMessage(HttpMethod.Post, actual);
        using var stream = new ObservedStream(Encoding.UTF8.GetBytes(Native().ToJsonString()));
        using var content = new TrackingContent(stream);
        using var response = new HttpResponseMessage(HttpStatusCode.OK) { Content = content, RequestMessage = redirected };
        using var handler = Script(_ => response);
        using var http = new HttpClient(handler);
        using var generator = Generator(http);
        var error = await Assert.ThrowsAsync<DecisionProtocolException>(() => generator.GenerateAsync(AllKinds(), cancellationToken: Token));
        Assert.Equal("Automatic redirects are not supported; configure HttpClient to disable them.", error.Message);
        var sent = Assert.Single(handler.Requests);
        Assert.Equal("http://127.0.0.1:11434/v1/systemone", sent.Uri.AbsoluteUri);
        Assert.Null(sent.Authorization);
        Assert.True(content.Disposed);
        Assert.True(stream.Disposed);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Disposal_IsIdempotentRespectsHttpOwnershipAndRejectsSubsequentCalls(bool owns)
    {
        using var handler = Script(_ => Response(Native().ToJsonString()));
        using var http = new HttpClient(handler);
        var generator = new OllamaDecisionGenerator(http, new() { ModelId = Model }, ownsHttpClient: owns);
        var result = await generator.GenerateAsync(AllKinds(), cancellationToken: Token);
        generator.Dispose(); generator.Dispose();
        Assert.Equal(owns ? 1 : 0, handler.DisposeCount);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => generator.GenerateAsync(AllKinds(), cancellationToken: Token));
        Assert.Single(handler.Requests);
        Assert.Equal(.8807970779778823, result.GetAnswer<BinaryDecisionAnswer>("b").TrueProbability);
        if (owns)
            await Assert.ThrowsAsync<ObjectDisposedException>(() => http.GetAsync("http://127.0.0.1:11434/probe", Token));
        else
        {
            using var response = await http.GetAsync("http://127.0.0.1:11434/probe", Token);
            Assert.Equal(HttpStatusCode.OK, response.StatusCode);
            Assert.Equal(2, handler.Requests.Length);
        }
    }
}
