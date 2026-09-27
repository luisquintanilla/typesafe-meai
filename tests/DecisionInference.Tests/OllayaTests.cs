using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using DecisionInference.Ollaya;
using static DecisionInference.Tests.ContractTests;
using static DecisionInference.Tests.OllayaTestFixtures;

namespace DecisionInference.Tests;

public sealed class OllayaTests
{
    private static OllayaDecisionClient Client(HttpClient http, int maximumBytes = 4 * 1024 * 1024,
        string? key = null, bool owns = false) =>
        new(http, new() { ModelId = Model, MaximumResponseBytes = maximumBytes }, key, owns);

    [Fact]
    public async Task SelectedModel_IsShownOnEveryCallIncludingOverridesWithoutStickyState()
    {
        using var handler = NormalHandler();
        using var http = new HttpClient(handler);
        using var client = Client(http);
        var metadata = client.GetRequiredService<DecisionClientMetadata>();
        Assert.Equal("ollaya", metadata.ProviderName);
        Assert.Equal(new Uri("http://127.0.0.1:11435"), metadata.ProviderUri);
        Assert.Equal(Model, metadata.DefaultModelId);
        Assert.Null(metadata.SupportedKinds);
        var options = new DecisionOptions { ModelId = "nli:override", AdditionalProperties = new() { ["caller"] = 42 } };
        DecisionOptions?[] calls = [null, options, new()];
        string[] expectedModels = [Model, "nli:override", Model];
        var request = JevTests.AllKinds();
        for (int i = 0; i < calls.Length; i++)
        {
            var result = await client.GetResponseAsync(request, calls[i], TestContext.Current.CancellationToken);
            Assert.Equal(expectedModels[i], result.ModelId);
            Assert.Equal(.8, result.GetAnswer<BinaryDecisionAnswer>("b").TrueProbability);
            Assert.Equal("A", result.GetAnswer<ChoiceDecisionAnswer>("c").SelectedCandidateId);
            Assert.Equal(1.3, result.GetAnswer<ScoreDecisionAnswer>("s").ExpectedIndex, 12);
            Assert.All(result.Answers, answer => Assert.Equal(DecisionPrecision.FourDecimalPlaces, answer.Precision));
            Assert.Equal(18, result.Usage!.TotalTokenCount);
            Assert.Equal(handler.Requests[2 * i + 1].Body,
                Assert.IsType<JsonElement>(result.AdditionalProperties!["ollaya.wireRequest"]).GetRawText());
            var modelInfo = Assert.IsType<JsonElement>(result.AdditionalProperties!["ollaya.modelInfo"]);
            Assert.Equal("onnxruntime", modelInfo.GetProperty("model_info").GetProperty("general.engine").GetString());
            Assert.Equal("onnx", modelInfo.GetProperty("details").GetProperty("format").GetString());
            Assert.Same(metadata, client.GetService<DecisionClientMetadata>());
        }
        Assert.Equal(6, handler.Requests.Length);
        for (int i = 0; i < calls.Length; i++)
        {
            var show = handler.Requests[2 * i];
            var infer = handler.Requests[2 * i + 1];
            Assert.Equal("/api/show", show.Uri.AbsolutePath);
            Assert.Equal($$"""{"model":"{{expectedModels[i]}}"}""", show.Body);
            Assert.Equal("/v1/systemone", infer.Uri.AbsolutePath);
            Assert.Equal(expectedModels[i], Json(infer.Body).GetProperty("model").GetString());
            Assert.All(new[] { show, infer }, sent =>
            {
                Assert.Equal(HttpMethod.Post, sent.Method);
                Assert.Equal("application/json", sent.MediaType);
                Assert.Null(sent.Authorization);
            });
        }
        var prepared = Json(handler.Requests[1].Body).GetProperty("questions");
        Assert.Equal(["q0", "q1", "q2"], prepared.EnumerateObject().Select(p => p.Name));
        Assert.Equal("is it?\nyes text", prepared.GetProperty("q0").GetProperty("criteria").GetProperty("true").GetString());
        Assert.Equal("is it?\nno text", prepared.GetProperty("q0").GetProperty("criteria").GetProperty("false").GetString());
        Assert.Equal("choose\nfirst", prepared.GetProperty("q1").GetProperty("criteria").GetProperty("0").GetString());
        Assert.Equal(["choose\nfirst", "choose\nsecond"],
            prepared.GetProperty("q1").GetProperty("criteria").EnumerateObject().Select(p => p.Value.GetString()));
        Assert.Equal("rate\nhigh", prepared.GetProperty("q2").GetProperty("criteria")[2].GetString());
        Assert.Equal(["rate\nlow", "rate\nmid", "rate\nhigh"],
            prepared.GetProperty("q2").GetProperty("criteria").EnumerateArray().Select(p => p.GetString()));
        Assert.Equal("first", ((ChoiceDecisionQuestion)request.Questions[1]).Candidates[0].Description);
        Assert.Equal(["first", "second"], ((ChoiceDecisionQuestion)request.Questions[1]).Candidates.Select(c => c.Description));
        Assert.Equal(["low", "mid", "high"], ((ScoreDecisionQuestion)request.Questions[2]).Rubric);
        Assert.Equal("yes text", ((BinaryDecisionQuestion)request.Questions[0]).TrueDescription);
        Assert.Equal("no text", ((BinaryDecisionQuestion)request.Questions[0]).FalseDescription);
        Assert.Equal(["is it?", "choose", "rate"], request.Questions.Select(q => q.Instructions));
        Assert.Equal("nli:override", options.ModelId);
        Assert.Equal(42, options.AdditionalProperties!["caller"]);
        Assert.Equal(Model, metadata.DefaultModelId);
        Assert.Null(metadata.SupportedKinds);
        Assert.Same(client, client.GetService<IDecisionClient>());
        Assert.Null(client.GetService(typeof(IDecisionClient), "key"));
        Assert.Null(client.GetService(typeof(string)));
        Assert.Equal("serviceType", Assert.Throws<ArgumentNullException>(() => client.GetService(null!)).ParamName);
    }

    [Theory]
    [InlineData("same", true)]
    [InlineData("different-profile", false)]
    [InlineData("guard", false)]
    public async Task ActualModel_IsRediscoveredAndMustMatchTheSelectedProfile(string profile, bool accepted)
    {
        using var handler = Script(sent =>
        {
            if (sent.Uri.AbsolutePath == "/v1/systemone") return Response(Native("canonical-model"));
            string selected = Json(sent.Body).GetProperty("model").GetString()!;
            return Response(selected == Model || profile == "same" ? NliShow : profile == "guard" ? GuardShow : GliclassShow);
        });
        using var http = new HttpClient(handler);
        using var client = Client(http);
        if (accepted)
        {
            var result = await client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: TestContext.Current.CancellationToken);
            Assert.Equal("canonical-model", result.ModelId);
            Assert.Equal(.8, result.GetAnswer<BinaryDecisionAnswer>("b").TrueProbability);
            Assert.Equal(NliShow, Assert.IsType<JsonElement>(result.AdditionalProperties!["ollaya.actualModelInfo"]).GetRawText());
        }
        else if (profile == "guard")
            await Assert.ThrowsAsync<NotSupportedException>(() => client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: TestContext.Current.CancellationToken));
        else
            await Assert.ThrowsAsync<DecisionProtocolException>(() => client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal(["/api/show", "/v1/systemone", "/api/show"], handler.Requests.Select(r => r.Uri.AbsolutePath));
        Assert.Equal("""{"model":"canonical-model"}""", handler.Requests[2].Body);
    }

    [Fact]
    public async Task MutableCatalog_SameModelIsRevalidatedAndUnsafeSecondCallNeverInfers()
    {
        int shows = 0;
        using var handler = Script(sent => sent.Uri.AbsolutePath == "/api/show"
            ? Response(++shows == 1 ? NliShow : GuardShow) : Response(Native()));
        using var http = new HttpClient(handler);
        using var client = Client(http);
        var first = await client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: TestContext.Current.CancellationToken);
        await Assert.ThrowsAsync<NotSupportedException>(() => client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal(["/api/show", "/v1/systemone", "/api/show"], handler.Requests.Select(r => r.Uri.AbsolutePath));
        Assert.Equal(.8, first.GetAnswer<BinaryDecisionAnswer>("b").TrueProbability);
        Assert.Equal("nli", Assert.IsType<JsonElement>(first.AdditionalProperties!["ollaya.modelInfo"]).GetProperty("details").GetProperty("family").GetString());
        Assert.Null(client.GetRequiredService<DecisionClientMetadata>().SupportedKinds);
    }

    [Theory]
    [InlineData("guard")]
    [InlineData("router")]
    [InlineData("laya")]
    [InlineData("unknown-family")]
    [InlineData("unknown-layout")]
    [InlineData("missing-layout")]
    [InlineData("wrong-format")]
    [InlineData("wrong-engine")]
    [InlineData("nli-engine-is-format")]
    [InlineData("gliclass-engine-is-format")]
    [InlineData("restricted-kinds")]
    public async Task ModelCatalog_RejectsUnauditedOrRestrictedProfilesBeforeAnyInference(string defect)
    {
        var show = JsonNode.Parse(defect switch
        {
            "guard" => GuardShow,
            "gliclass-engine-is-format" => GliclassShow,
            _ => NliShow
        })!;
        switch (defect)
        {
            case "router": show["router"] = new JsonObject { ["strategy"] = "fallback", ["default"] = Model, ["routes"] = new JsonObject() }; break;
            case "laya":
                show["details"]!["family"] = "laya";
                show["details"]!["families"] = new JsonArray("laya");
                show["model_info"]!["general.architecture"] = "laya";
                show["model_info"]!.AsObject().Remove("nli.layout");
                show["model_info"]!.AsObject().Remove("nli.context_length");
                // Pinned laya:en decision blob b5bc5074...9291b3c, not an invented "laya" layout.
                show["model_info"]!["laya.layout"] = "laya-markers-v1";
                show["model_info"]!["laya.context_length"] = 512;
                show["model_info"]!["laya.head_max_len"] = 192;
                break;
            case "unknown-family": show["details"]!["family"] = "future"; break;
            case "unknown-layout": show["model_info"]!["nli.layout"] = "future-v2"; break;
            case "missing-layout": show["model_info"]!.AsObject().Remove("nli.layout"); break;
            case "wrong-format": show["details"]!["format"] = "gguf"; break;
            case "wrong-engine": show["model_info"]!["general.engine"] = "llama.cpp"; break;
            case "nli-engine-is-format":
            case "gliclass-engine-is-format":
                show["model_info"]!["general.engine"] = "onnx";
                break;
            case "restricted-kinds": show["capabilities"] = new JsonArray("choice"); break;
        }
        using var content = new TrackingContent(new ObservedStream(Encoding.UTF8.GetBytes(show.ToJsonString())));
        using var handler = Script(_ => new(HttpStatusCode.OK) { Content = content });
        using var http = new HttpClient(handler);
        using var client = Client(http);
        // Choice first: rejection must validate the whole batch, not partially infer it.
        await Assert.ThrowsAsync<NotSupportedException>(() => client.GetResponseAsync(Request(Choice(), Binary(), Score()), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal("/api/show", Assert.Single(handler.Requests).Uri.AbsolutePath);
        Assert.True(content.Disposed);
    }

    public static IEnumerable<object[]> MalformedDiscovery()
    {
        yield return ["malformed-json", "{private-discovery"];
        yield return ["array-root", "[]"];
        foreach (string key in new[] { "router", "details", "model_info", "capabilities" })
        {
            var root = JsonNode.Parse(NliShow)!.AsObject();
            root.Remove(key);
            yield return [$"missing-{key}", root.ToJsonString()];
        }
        foreach (string key in new[] { "details", "model_info", "capabilities" })
        {
            var root = JsonNode.Parse(NliShow)!;
            root[key] = 42;
            yield return [$"wrong-type-{key}", root.ToJsonString()];
        }
        var mismatch = JsonNode.Parse(NliShow)!;
        mismatch["model_info"]!["general.architecture"] = "gliclass";
        yield return ["architecture-mismatch", mismatch.ToJsonString()];
        var badCapability = JsonNode.Parse(NliShow)!;
        badCapability["capabilities"] = new JsonArray("choice", "score", 42);
        yield return ["nonstring-capability", badCapability.ToJsonString()];
        // docs/api.md specifies an array of strings, not nullable strings.
        var nullCapability = JsonNode.Parse(NliShow)!;
        nullCapability["capabilities"]!.AsArray().Add((JsonNode?)null);
        yield return ["null-capability", nullCapability.ToJsonString()];
        yield return ["duplicate-property", NliShow.Replace("\"router\":null", "\"router\":null,\"router\":null", StringComparison.Ordinal)];
    }

    [Theory]
    [MemberData(nameof(MalformedDiscovery))]
    public async Task MalformedDiscovery_IsSanitizedDisposedAndNeverUsedForInference(string scenario, string body)
    {
        using var stream = new ObservedStream(Encoding.UTF8.GetBytes(body));
        using var content = new TrackingContent(stream);
        using var handler = Script(sent => sent.Uri.AbsolutePath == "/api/show"
            ? new(HttpStatusCode.OK) { Content = content } : Response(Native()));
        using var http = new HttpClient(handler);
        using var client = Client(http);
        var error = await Assert.ThrowsAsync<DecisionProtocolException>(() =>
            client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: TestContext.Current.CancellationToken));
        Assert.DoesNotContain("private-discovery", error.ToString(), StringComparison.Ordinal);
        Assert.DoesNotContain(body, error.Message, StringComparison.Ordinal);
        Assert.Equal("/api/show", Assert.Single(handler.Requests).Uri.AbsolutePath);
        Assert.True(content.Disposed, scenario);
        Assert.True(stream.Disposed);
    }

    [Fact]
    public async Task KnownProfile_EmbeddedDefaultsAndOpenCapabilitiesDoNotImplyFixedSchema()
    {
        var show = JsonNode.Parse(NliShow)!;
        show["questions"] = new JsonObject { ["preset"] = new JsonObject { ["type"] = "noul" } };
        show["capabilities"]!.AsArray().Add("future-capability");
        using var handler = NormalHandler(show.ToJsonString());
        using var http = new HttpClient(handler);
        using var client = Client(http);
        var result = await client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(["b", "c", "s"], result.Answers.Select(a => a.QuestionId));
        Assert.Equal(.8, result.GetAnswer<BinaryDecisionAnswer>("b").TrueProbability);
        Assert.Equal(2, handler.Requests.Length);
        Assert.Equal(["q0", "q1", "q2"], Json(handler.Requests[1].Body).GetProperty("questions").EnumerateObject().Select(p => p.Name));
    }

    [Theory]
    [InlineData("nli")]
    [InlineData("gliclass")]
    public async Task ProfileGuards_RejectOneSidedBinaryCriteriaAndPreserveBothSemanticInputs(string family)
    {
        using var handler = NormalHandler(family == "nli" ? NliShow : GliclassShow);
        using var http = new HttpClient(handler);
        using var client = Client(http);
        foreach (bool trueSide in new[] { true, false })
        {
            var request = Request(new BinaryDecisionQuestion("b", "explicit intent", trueSide ? "yes" : null, trueSide ? null : "no"));
            await Assert.ThrowsAsync<NotSupportedException>(() => client.GetResponseAsync(request, cancellationToken: TestContext.Current.CancellationToken));
        }
        Assert.Equal(2, handler.Requests.Length);
        Assert.All(handler.Requests, sent => Assert.Equal("/api/show", sent.Uri.AbsolutePath));
        var result = await client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: TestContext.Current.CancellationToken);
        var questions = Json(handler.Requests[3].Body).GetProperty("questions");
        string prefix = family == "nli" ? "choose\n" : "";
        Assert.Equal(prefix + "first", questions.GetProperty("q1").GetProperty("criteria").GetProperty("0").GetString());
        Assert.Equal("choose", questions.GetProperty("q1").GetProperty("instructions").GetString());
        Assert.Equal("A", result.GetAnswer<ChoiceDecisionAnswer>("c").SelectedCandidateId);
    }

    [Theory]
    [InlineData("nli", "state")]
    [InlineData("nli", "key")]
    [InlineData("nli", "array")]
    [InlineData("nli", "instruction")]
    [InlineData("nli", "candidate")]
    [InlineData("nli", "rubric")]
    [InlineData("nli", "binary")]
    [InlineData("gliclass", "state")]
    [InlineData("gliclass", "key")]
    [InlineData("gliclass", "array")]
    [InlineData("gliclass", "instruction")]
    [InlineData("gliclass", "candidate")]
    [InlineData("gliclass", "rubric")]
    [InlineData("gliclass", "binary")]
    public async Task ProfileGuards_RejectReservedMarkersInEverySemanticPosition(string family, string position)
    {
        string marker = family == "nli" ? "[CLS]" : "<<LABEL>>";
        var state = position switch
        {
            "state" => JsonSerializer.SerializeToElement(marker),
            "key" => JsonSerializer.SerializeToElement(new Dictionary<string, string> { [marker] = "value" }),
            "array" => JsonSerializer.SerializeToElement(new[] { marker }),
            _ => Json("\"state\"")
        };
        DecisionQuestion question = position switch
        {
            "rubric" => new ScoreDecisionQuestion("s", "rate", ["low", marker]),
            "binary" => new BinaryDecisionQuestion("b", "intent", "true text", marker),
            _ => new ChoiceDecisionQuestion("c", position == "instruction" ? marker : "choose",
                [new("a", "first"), new("b", position == "candidate" ? marker : "second")])
        };
        using var handler = NormalHandler(family == "nli" ? NliShow : GliclassShow);
        using var http = new HttpClient(handler);
        using var client = Client(http);
        await Assert.ThrowsAsync<NotSupportedException>(() => client.GetResponseAsync(new(state, [question]), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal("/api/show", Assert.Single(handler.Requests).Uri.AbsolutePath);
    }

    [Theory]
    [InlineData("", "choose")]
    [InlineData("BEL>>", "<<LA")]
    public async Task Gliclass_RejectsEmptyStateAndMarkersSpanningInstructionStateBoundary(string state, string instruction)
    {
        using var handler = NormalHandler(GliclassShow);
        using var http = new HttpClient(handler);
        using var client = Client(http);
        var request = new DecisionRequest(JsonSerializer.SerializeToElement(state),
            [new ChoiceDecisionQuestion("c", instruction, [new("a", "first"), new("b", "second")])]);
        await Assert.ThrowsAsync<NotSupportedException>(() => client.GetResponseAsync(request, cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal("/api/show", Assert.Single(handler.Requests).Uri.AbsolutePath);
    }

    [Theory]
    [InlineData("null")]
    [InlineData("true")]
    [InlineData("42")]
    public async Task InvalidState_IsRejectedBeforeDiscovery(string state)
    {
        using var handler = NormalHandler();
        using var http = new HttpClient(handler);
        using var client = Client(http);
        var error = await Assert.ThrowsAsync<ArgumentException>(() =>
            client.GetResponseAsync(new(Json(state), [Binary()]), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal("request", error.ParamName);
        Assert.Empty(handler.Requests);
    }

    [Theory]
    [InlineData("questions", 1, true)]
    [InlineData("questions", 256, true)]
    [InlineData("questions", 257, false)]
    [InlineData("choice", 1, false)]
    [InlineData("choice", 2, true)]
    [InlineData("choice", 255, true)]
    [InlineData("choice", 256, false)]
    [InlineData("score", 2, true)]
    [InlineData("score", 10, true)]
    [InlineData("score", 11, false)]
    public async Task RequestLimits_EnforceQuestionCandidateAndRubricBoundaries(string kind, int count, bool accepted)
    {
        var answers = new JsonObject();
        DecisionRequest request;
        if (kind == "questions")
        {
            request = Request(Enumerable.Range(0, count).Select(i => (DecisionQuestion)Binary($"id-{i}")).ToArray());
            for (int i = 0; i < count; i++) answers[$"q{i}"] = new JsonObject { ["type"] = "noul", ["noul"] = .8 };
        }
        else
        {
            var probabilities = new JsonObject(Enumerable.Range(0, count).Select(i =>
                KeyValuePair.Create<string, JsonNode?>(i.ToString(), JsonValue.Create(i == count - 1 ? 1d : 0d))));
            if (kind == "choice")
            {
                request = Request(new ChoiceDecisionQuestion("c", "choose", Enumerable.Range(0, count).Select(i => new DecisionCandidate($"id-{i}", $"option {i}"))));
                answers["q0"] = new JsonObject { ["type"] = "choice", ["choice"] = (count - 1).ToString(), ["confidence"] = 1, ["probabilities"] = probabilities };
            }
            else
            {
                request = Request(new ScoreDecisionQuestion("s", "rate", Enumerable.Range(0, count).Select(i => $"level {i}")));
                answers["q0"] = new JsonObject { ["type"] = "score", ["score"] = count - 1, ["confidence"] = 1, ["probabilities"] = probabilities,
                    ["legend"] = new JsonObject(Enumerable.Range(0, count).Select(i => KeyValuePair.Create<string, JsonNode?>(i.ToString(), JsonValue.Create($"rate\nlevel {i}")))) };
            }
        }
        string body = new JsonObject { ["model"] = Model, ["answers"] = answers,
            ["usage"] = new JsonObject { ["input_tokens"] = 7, ["output_tokens"] = 0 } }.ToJsonString();
        using var handler = Script(sent => Response(sent.Uri.AbsolutePath == "/api/show" ? NliShow : body));
        using var http = new HttpClient(handler);
        using var client = Client(http);
        if (!accepted)
        {
            await Assert.ThrowsAsync<ArgumentException>(() => client.GetResponseAsync(request, cancellationToken: TestContext.Current.CancellationToken));
            Assert.Empty(handler.Requests);
            return;
        }
        var result = await client.GetResponseAsync(request, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(["/api/show", "/v1/systemone"], handler.Requests.Select(r => r.Uri.AbsolutePath));
        Assert.Equal(request.Questions.Select(q => q.Id), result.Answers.Select(a => a.QuestionId));
        Assert.Equal(7, result.Usage!.InputTokenCount);
        if (kind == "questions")
            Assert.All(result.Answers, answer => Assert.Equal(.8, Assert.IsType<BinaryDecisionAnswer>(answer).TrueProbability));
        else if (kind == "choice")
        {
            var answer = result.GetAnswer<ChoiceDecisionAnswer>("c");
            Assert.Equal($"id-{count - 1}", answer.SelectedCandidateId);
            Assert.Equal(count, answer.Probabilities.Count);
            Assert.Equal(1, answer.Probabilities[$"id-{count - 1}"]);
        }
        else
        {
            var answer = result.GetAnswer<ScoreDecisionAnswer>("s");
            Assert.Equal(count, answer.Probabilities.Count);
            Assert.Equal(count - 1, answer.ExpectedIndex);
            Assert.Equal(1, answer.Probabilities[^1]);
        }
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(0)]
    [InlineData(1)]
    public async Task RequestByteLimit_CountsSerializedEscapedUnicodeAtExactEightMiB(int delta)
    {
        const int limit = 8 * 1024 * 1024;
        const string prefix = "{\"state\":\"";
        const string suffix = "\",\"model\":\"nli:modernbert-large\",\"questions\":{\"q0\":{\"type\":\"noul\",\"instructions\":\"Is the condition satisfied?\"}}}";
        int filler = limit + delta - Encoding.UTF8.GetByteCount(prefix + suffix) - 6; // é is written as \u00E9.
        string state = new string('x', filler) + "\u00e9";
        using var handler = Script(sent => sent.Uri.AbsolutePath == "/api/show" ? Response(NliShow) :
            Response($$$"""{"model":"{{{Model}}}","answers":{"q0":{"type":"noul","noul":0.8}},"usage":{"input_tokens":3,"output_tokens":0}}"""));
        using var http = new HttpClient(handler);
        using var client = Client(http);
        var request = new DecisionRequest(JsonSerializer.SerializeToElement(state), [Binary()]);
        if (delta > 0)
        {
            await Assert.ThrowsAsync<ArgumentException>(() => client.GetResponseAsync(request, cancellationToken: TestContext.Current.CancellationToken));
            Assert.Empty(handler.Requests);
            return;
        }
        var result = await client.GetResponseAsync(request, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(.8, result.GetAnswer<BinaryDecisionAnswer>("b").TrueProbability);
        Assert.Equal(2, handler.Requests.Length);
        Assert.Equal(limit + delta, Encoding.UTF8.GetByteCount(handler.Requests[1].Body));
        Assert.Equal(state, Json(handler.Requests[1].Body).GetProperty("state").GetString());
    }

    [Fact]
    public async Task NliPreparation_RechecksExpandedBodyLimitBeforeInference()
    {
        var request = Request(new ChoiceDecisionQuestion("c", new string('x', 3 * 1024 * 1024),
            [new("a", "first"), new("b", "second")]));
        using var handler = NormalHandler();
        using var http = new HttpClient(handler);
        using var client = Client(http);
        await Assert.ThrowsAsync<ArgumentException>(() => client.GetResponseAsync(request, cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal("/api/show", Assert.Single(handler.Requests).Uri.AbsolutePath);
        Assert.Equal("first", ((ChoiceDecisionQuestion)request.Questions[0]).Candidates[0].Description);
    }

    [Theory]
    [InlineData("http://127.0.0.1:11435/", true)]
    [InlineData("http://127.0.0.2:11435/", true)]
    [InlineData("http://[::1]:11435/", true)]
    [InlineData("https://remote.invalid:8443/", true)]
    [InlineData("http://localhost:11435/", false)]
    [InlineData("http://127.0.0.1.remote.invalid/", false)]
    [InlineData("http://192.0.2.1/", false)]
    [InlineData("ftp://127.0.0.1/", false)]
    [InlineData("https://remote.invalid/path", false)]
    [InlineData("https://remote.invalid/?query=x", false)]
    [InlineData("https://remote.invalid/#fragment", false)]
    [InlineData("https://user:password@remote.invalid/", false)]
    [InlineData("/relative", false)]
    public void EndpointPolicy_AllowsLiteralLoopbackHttpAndRemoteHttpsOnly(string endpoint, bool accepted)
    {
        using var handler = NormalHandler();
        using var http = new HttpClient(handler);
        var options = new OllayaDecisionOptions { Endpoint = new(endpoint, UriKind.RelativeOrAbsolute), ModelId = Model };
        if (accepted)
        {
            using var client = new OllayaDecisionClient(http, options);
            Assert.Equal(options.Endpoint, client.GetRequiredService<DecisionClientMetadata>().ProviderUri);
            Assert.Equal(Model, client.GetRequiredService<DecisionClientMetadata>().DefaultModelId);
        }
        else Assert.Throws<ArgumentException>(() => new OllayaDecisionClient(http, options));
        Assert.Empty(handler.Requests);
        Assert.Equal(0, handler.DisposeCount);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("fixture-key")]
    public async Task Authentication_IsOptionalExplicitAndRequestLocalOnBothEndpoints(string? key)
    {
        using var handler = NormalHandler();
        using var http = new HttpClient(handler);
        http.DefaultRequestHeaders.Add("x-caller", "unchanged");
        using var client = Client(http, key: key);
        var result = await client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(.8, result.GetAnswer<BinaryDecisionAnswer>("b").TrueProbability);
        Assert.Equal(2, handler.Requests.Length);
        Assert.All(handler.Requests, sent => Assert.Equal(key is null ? null : "Bearer fixture-key", sent.Authorization));
        Assert.Null(http.DefaultRequestHeaders.Authorization);
        Assert.Equal("unchanged", Assert.Single(http.DefaultRequestHeaders.GetValues("x-caller")));
    }

    [Theory]
    [InlineData("")]
    [InlineData(" \t")]
    [InlineData("has space")]
    [InlineData("line\r\nbreak")]
    public void Authentication_RejectsInvalidExplicitKeysBeforeSending(string key)
    {
        using var handler = NormalHandler();
        using var http = new HttpClient(handler);
        Assert.Equal("apiKey", Assert.Throws<ArgumentException>(() => Client(http, key: key)).ParamName);
        Assert.Empty(handler.Requests);
        Assert.Null(http.DefaultRequestHeaders.Authorization);
    }

    [Fact]
    public async Task Authentication_RejectsAmbientHeadersAtConstructionAndBeforeEachUnauthenticatedCall()
    {
        using var handler = NormalHandler();
        using var http = new HttpClient(handler);
        http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", "ambient");
        Assert.Equal("httpClient", Assert.Throws<ArgumentException>(() => Client(http)).ParamName);
        Assert.Equal("httpClient", Assert.Throws<ArgumentException>(() => Client(http, key: "explicit")).ParamName);
        http.DefaultRequestHeaders.Authorization = null;
        using var client = Client(http);
        http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", "later");
        await Assert.ThrowsAsync<InvalidOperationException>(() => client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Empty(handler.Requests);
        Assert.Equal("later", http.DefaultRequestHeaders.Authorization!.Parameter);
    }

    [Fact]
    public async Task ModelAndClientArgumentGuards_DoNotSendOrTakeOwnershipOnFailure()
    {
        using var handler = NormalHandler();
        using var http = new HttpClient(handler);
        Assert.Throws<ArgumentNullException>(() => new OllayaDecisionClient(null!));
        Assert.Throws<ArgumentNullException>(() => new OllayaDecisionClient(http, new() { Endpoint = null! }));
        foreach (int maximum in new[] { 0, -1 })
            Assert.Throws<ArgumentOutOfRangeException>(() => Client(http, maximum));
        foreach (string model in new[] { "", " \t" })
            Assert.Throws<ArgumentException>(() => new OllayaDecisionClient(http, new() { ModelId = model }));
        using var client = new OllayaDecisionClient(http); // No default model is honestly reported.
        Assert.Null(client.GetRequiredService<DecisionClientMetadata>().DefaultModelId);
        await Assert.ThrowsAsync<ArgumentNullException>(() => client.GetResponseAsync(null!, cancellationToken: TestContext.Current.CancellationToken));
        await Assert.ThrowsAnyAsync<ArgumentException>(() => client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: TestContext.Current.CancellationToken));
        foreach (string model in new[] { "", " \t" })
            await Assert.ThrowsAsync<ArgumentException>(() => client.GetResponseAsync(JevTests.AllKinds(), new() { ModelId = model }, TestContext.Current.CancellationToken));
        Assert.Empty(handler.Requests);
        Assert.Equal(0, handler.DisposeCount);
    }

    [Theory]
    [InlineData(false, true, false)]
    [InlineData(false, false, false)]
    [InlineData(true, true, false)]
    [InlineData(true, false, false)]
    [InlineData(false, true, true)]
    [InlineData(false, false, true)]
    [InlineData(true, true, true)]
    [InlineData(true, false, true)]
    public async Task ResponseBody_EnforcesExactUtf8ByteLimitForDeclaredAndStreamedBodies(bool inference, bool declared, bool overflow)
    {
        string body = (inference ? Native() : NliShow).Insert(1, "\"note\":\"é\",").PadRight(2048);
        byte[] bytes = Encoding.UTF8.GetBytes(body);
        Assert.Equal(body.Length + 1, bytes.Length);
        using var stream = new ObservedStream(bytes);
        using var content = new TrackingContent(stream);
        if (declared) content.Headers.ContentLength = bytes.Length;
        using var handler = Script(sent => (sent.Uri.AbsolutePath == "/v1/systemone") == inference
            ? new(HttpStatusCode.OK) { Content = content } : Response(sent.Uri.AbsolutePath == "/api/show" ? NliShow : Native()));
        using var http = new HttpClient(handler);
        using var client = Client(http, bytes.Length - (overflow ? 1 : 0));
        if (overflow)
            await Assert.ThrowsAsync<OllayaBodyLimitException>(() => client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: TestContext.Current.CancellationToken));
        else
        {
            var result = await client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: TestContext.Current.CancellationToken);
            Assert.Equal(.8, result.GetAnswer<BinaryDecisionAnswer>("b").TrueProbability);
            Assert.Equal(18, result.Usage!.TotalTokenCount);
        }
        Assert.Equal(overflow && !inference ? 1 : 2, handler.Requests.Length);
        Assert.True(content.Disposed);
        Assert.True(stream.Disposed);
        if (declared && overflow) Assert.Equal(0, stream.ReadCalls);
        else Assert.True(stream.ReadCalls > 0);
    }

    [Theory]
    [InlineData(false, 429)]
    [InlineData(false, 500)]
    [InlineData(true, 429)]
    [InlineData(true, 500)]
    public async Task ErrorStatus_InvalidUtf8RetainsStatusAndOptInDiagnosticsWithoutLeaking(bool inference, int status)
    {
        using var stream = new ObservedStream([.. Encoding.UTF8.GetBytes("private-"), 0xc3, 0x28, .. Encoding.UTF8.GetBytes("-body")]);
        using var content = new TrackingContent(stream);
        using var response = new HttpResponseMessage((HttpStatusCode)status) { Content = content };
        response.Headers.Add("x-typesafe-request-id", "fixture-error");
        using var handler = Script(sent => (sent.Uri.AbsolutePath == "/v1/systemone") == inference ? response : Response(NliShow));
        using var http = new HttpClient(handler);
        using var client = Client(http);
        var error = await Assert.ThrowsAsync<OllayaApiException>(() => client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal((HttpStatusCode)status, error.StatusCode);
        Assert.Equal("fixture-error", error.RequestId);
        Assert.Equal("private-\uFFFD(-body", error.DiagnosticBody);
        Assert.Equal($"Ollaya HTTP operation failed with status {status}.", error.Message);
        Assert.Null(error.Code);
        Assert.Null(error.InnerException);
        Assert.DoesNotContain("private-", error.ToString(), StringComparison.Ordinal);
        Assert.True(content.Disposed);
        Assert.True(stream.Disposed);
        Assert.Equal(inference ? 2 : 1, handler.Requests.Length);
        Assert.Throws<ObjectDisposedException>(() => response.Content = new ByteArrayContent([]));
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public async Task ErrorBody_PreservesStructuredCodeOrMalformedTextWithoutErasingStatus(bool inference, bool malformed)
    {
        string body = malformed ? "{private-error" : """{"error":"private-state","code":"STATE_TRUNCATED","detail":[{"loc":["body","state"],"type":"state_truncated"}]}""";
        using var handler = Script(sent => (sent.Uri.AbsolutePath == "/v1/systemone") == inference
            ? Response(body, HttpStatusCode.UnprocessableEntity) : Response(NliShow));
        using var http = new HttpClient(handler);
        using var client = Client(http);
        var error = await Assert.ThrowsAsync<OllayaApiException>(() => client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal(HttpStatusCode.UnprocessableEntity, error.StatusCode);
        Assert.Equal(body, error.DiagnosticBody);
        Assert.Equal(malformed ? null : "STATE_TRUNCATED", error.Code);
        Assert.DoesNotContain("private-", error.ToString(), StringComparison.Ordinal);
        Assert.Equal(inference ? 2 : 1, handler.Requests.Length);
    }

    [Theory]
    [InlineData("null", null)]
    [InlineData("17", null)]
    [InlineData("{}", null)]
    [InlineData("[]", null)]
    [InlineData("true", null)]
    [InlineData("\"\"", "")]
    [InlineData("\"FUTURE_CODE\"", "FUTURE_CODE")]
    public async Task ErrorBody_CodeIsParsedOnlyForJsonStringsWithoutLosingStatus(string codeJson, string? expectedCode)
    {
        string body = $$"""{"error":"private-fixture","code":{{codeJson}}}""";
        using var stream = new ObservedStream(Encoding.UTF8.GetBytes(body));
        using var content = new TrackingContent(stream);
        using var response = new HttpResponseMessage(HttpStatusCode.BadRequest) { Content = content };
        response.Headers.Add("x-typesafe-request-id", "code-type-fixture");
        using var handler = Script(sent => sent.Uri.AbsolutePath == "/api/show" ? Response(NliShow) : response);
        using var http = new HttpClient(handler);
        using var client = Client(http);
        var error = await Assert.ThrowsAsync<OllayaApiException>(() =>
            client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal(expectedCode, error.Code);
        Assert.Equal(HttpStatusCode.BadRequest, error.StatusCode);
        Assert.Equal("code-type-fixture", error.RequestId);
        Assert.Equal(body, error.DiagnosticBody);
        Assert.Equal("Ollaya HTTP operation failed with status 400.", error.Message);
        Assert.DoesNotContain("private-fixture", error.ToString(), StringComparison.Ordinal);
        Assert.Equal(["/api/show", "/v1/systemone"], handler.Requests.Select(r => r.Uri.AbsolutePath));
        Assert.True(content.Disposed);
        Assert.True(stream.Disposed);
        Assert.Throws<ObjectDisposedException>(() => response.Content = new ByteArrayContent([]));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task SuccessfulInvalidUtf8_IsRejectedBeforeParsingAndDisposed(bool inference)
    {
        string body = (inference ? Native() : NliShow).Insert(1, "\"note\":\"PLACEHOLDER\",");
        string[] parts = body.Split("PLACEHOLDER", StringSplitOptions.None);
        using var stream = new ObservedStream([.. Encoding.UTF8.GetBytes(parts[0]), 0xc3, 0x28, .. Encoding.UTF8.GetBytes(parts[1])]);
        using var content = new TrackingContent(stream);
        using var response = new HttpResponseMessage(HttpStatusCode.OK) { Content = content };
        using var handler = Script(sent => (sent.Uri.AbsolutePath == "/v1/systemone") == inference ? response : Response(NliShow));
        using var http = new HttpClient(handler);
        using var client = Client(http);
        var error = await Assert.ThrowsAsync<DecisionProtocolException>(() => client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal("Response is not valid UTF-8.", error.Message);
        Assert.True(content.Disposed);
        Assert.True(stream.Disposed);
        Assert.Equal(inference ? 2 : 1, handler.Requests.Length);
        Assert.Throws<ObjectDisposedException>(() => response.Content = new ByteArrayContent([]));
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public async Task RedirectResponses_AreRejectedAndDisposedWithoutClaimingRedirectPrevention(bool inference, bool followed)
    {
        using var content = new TrackingContent(new ObservedStream(Encoding.UTF8.GetBytes(inference ? Native() : NliShow)));
        using var response = new HttpResponseMessage(followed ? HttpStatusCode.OK : HttpStatusCode.Found) { Content = content };
        using var finalRequest = new HttpRequestMessage(HttpMethod.Post, "https://redirect.invalid/");
        if (followed) response.RequestMessage = finalRequest;
        else response.Headers.Location = finalRequest.RequestUri;
        using var handler = Script(sent => (sent.Uri.AbsolutePath == "/v1/systemone") == inference ? response : Response(NliShow));
        using var http = new HttpClient(handler);
        using var client = Client(http);
        if (followed)
            await Assert.ThrowsAsync<DecisionProtocolException>(() => client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: TestContext.Current.CancellationToken));
        else
        {
            var error = await Assert.ThrowsAsync<OllayaApiException>(() => client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: TestContext.Current.CancellationToken));
            Assert.Equal(HttpStatusCode.Found, error.StatusCode);
        }
        Assert.Equal(inference ? 2 : 1, handler.Requests.Length);
        Assert.True(content.Disposed);
        Assert.Throws<ObjectDisposedException>(() => response.Content = new ByteArrayContent([]));
        // This tests detection of a mocked final URI, not prevention of HttpClient forwarding.
    }

    [Theory(Timeout = 15000)]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Cancellation_ReachesEveryShowAndInferenceReadBeforeDisposal(bool inference)
    {
        using var stream = new StalledStream();
        using var content = new TrackingContent(stream);
        using var response = new HttpResponseMessage(HttpStatusCode.OK) { Content = content };
        using var handler = Script(sent => (sent.Uri.AbsolutePath == "/v1/systemone") == inference ? response : Response(NliShow));
        using var http = new HttpClient(handler) { Timeout = Timeout.InfiniteTimeSpan };
        using var client = Client(http);
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        var call = client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: cancellation.Token);
        try
        {
            await stream.Entered.Task.WaitAsync(TimeSpan.FromSeconds(5), TestContext.Current.CancellationToken);
            Assert.Equal(2, stream.ReadTokens.Count);
            Assert.All(stream.ReadTokens, token => Assert.True(token.CanBeCanceled));
            cancellation.Cancel();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => call.WaitAsync(TimeSpan.FromSeconds(5), TestContext.Current.CancellationToken));
            Assert.All(stream.ReadTokens, token => Assert.True(token.IsCancellationRequested));
            Assert.True(stream.ReadCompleted);
            Assert.False(stream.DisposedWhileReading);
            Assert.True(content.Disposed);
            Assert.True(stream.Disposed);
            Assert.Equal(inference ? 2 : 1, handler.Requests.Length);
            Assert.Throws<ObjectDisposedException>(() => response.Content = new ByteArrayContent([]));
        }
        finally
        {
            stream.Release.TrySetCanceled(TestContext.Current.CancellationToken);
            try { await call; } catch (Exception) { /* Drain the fixture, including regressions. */ }
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task BodyIOException_IsSanitizedAndDisposesSourceAndResponse(bool inference)
    {
        using var stream = new FaultingStream();
        using var content = new TrackingContent(stream);
        using var response = new HttpResponseMessage(HttpStatusCode.OK) { Content = content };
        using var handler = Script(sent => (sent.Uri.AbsolutePath == "/v1/systemone") == inference ? response : Response(NliShow));
        using var http = new HttpClient(handler);
        using var client = Client(http);
        var error = await Assert.ThrowsAsync<HttpRequestException>(() => client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal(HttpRequestError.ResponseEnded, error.HttpRequestError);
        Assert.Equal("System One response body read failed.", error.Message);
        Assert.Null(error.InnerException);
        Assert.DoesNotContain("private-stream-secret", error.ToString(), StringComparison.Ordinal);
        Assert.True(content.Disposed);
        Assert.True(stream.Disposed);
        Assert.Equal(inference ? 2 : 1, handler.Requests.Length);
        Assert.Throws<ObjectDisposedException>(() => response.Content = new ByteArrayContent([]));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task HttpClientOwnership_PreservesBorrowedAndDisposesOwnedClientOnce(bool owns)
    {
        using var handler = NormalHandler();
        using var http = new HttpClient(handler);
        using var client = Client(http, owns: owns);
        using var canceled = new CancellationTokenSource();
        canceled.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: canceled.Token));
        Assert.Empty(handler.Requests);
        var result = await client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: TestContext.Current.CancellationToken);
        client.Dispose();
        Assert.Equal(Model, client.GetRequiredService<DecisionClientMetadata>().DefaultModelId);
        client.Dispose();
        Assert.Equal(owns ? 1 : 0, handler.DisposeCount);
        Assert.Equal(.8, result.GetAnswer<BinaryDecisionAnswer>("b").TrueProbability);
        Assert.Equal(.8, Assert.IsType<JsonElement>(result.RawRepresentation).GetProperty("answers").GetProperty("q0").GetProperty("noul").GetDouble());
        await Assert.ThrowsAsync<ObjectDisposedException>(() => client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal(2, handler.Requests.Length);
        if (owns)
            await Assert.ThrowsAsync<ObjectDisposedException>(() => http.PostAsync("http://127.0.0.1:11435/api/show", new StringContent("{}"), TestContext.Current.CancellationToken));
        else
        {
            using var response = await http.PostAsync("http://127.0.0.1:11435/api/show", new StringContent("{}"), TestContext.Current.CancellationToken);
            Assert.Equal(HttpStatusCode.OK, response.StatusCode);
            Assert.Equal(3, handler.Requests.Length);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task OllayaWire_RetainsRoundedThirdsAndIndependentNativeScoreWithBounds(bool independentScore)
    {
        var native = JsonNode.Parse(Native())!;
        var score = native["answers"]!["q2"]!;
        score["probabilities"] = independentScore
            ? new JsonObject { ["0"] = .1234, ["1"] = .2344, ["2"] = .6421 }
            : new JsonObject { ["0"] = .3333, ["1"] = .3333, ["2"] = .3333 };
        score["score"] = independentScore ? 1.5187 : 1;
        using var handler = Script(sent => Response(sent.Uri.AbsolutePath == "/api/show" ? NliShow : native.ToJsonString()));
        using var http = new HttpClient(handler);
        using var client = Client(http);
        var result = await client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: TestContext.Current.CancellationToken);
        var answer = result.GetAnswer<ScoreDecisionAnswer>("s");
        Assert.Equal(independentScore ? new double[] { .1234, .2344, .6421 } : [.3333, .3333, .3333], answer.Probabilities);
        Assert.Equal(.9999, answer.Probabilities.Sum(), 12);
        Assert.Equal(independentScore ? 1.5186 : .9999, answer.ExpectedIndex, 12);
        Assert.Equal(independentScore ? 1.51865 : .99995, answer.ExpectedIndexLowerBound, 12);
        Assert.Equal(independentScore ? 1.51875 : 1.00005, answer.ExpectedIndexUpperBound, 12);
        Assert.Equal(independentScore ? 1.5187 : 1, answer.NativeScore);
        Assert.Equal(DecisionPrecision.FourDecimalPlaces, answer.Precision);
        client.Dispose();
        var raw = Assert.IsType<JsonElement>(answer.RawRepresentation);
        Assert.Equal(independentScore ? .1234 : .3333, raw.GetProperty("probabilities").GetProperty("0").GetDouble());
        Assert.Equal(independentScore ? 1.5187 : 1, raw.GetProperty("score").GetDouble());
        Assert.Equal(["q0", "q1", "q2"], Assert.IsType<JsonElement>(result.RawRepresentation).GetProperty("answers").EnumerateObject().Select(p => p.Name));
        Assert.Equal(2, handler.Requests.Length);
    }

    [Theory]
    [InlineData("infeasible")]
    [InlineData("disjoint-native")]
    [InlineData("off-grid-native")]
    [InlineData("off-grid-binary")]
    public async Task OllayaWire_RejectsImpossibleOrOffGridObservedAnswers(string defect)
    {
        var native = JsonNode.Parse(Native())!;
        switch (defect)
        {
            case "infeasible": native["answers"]!["q2"]!["probabilities"] = new JsonObject { ["0"] = .3332, ["1"] = .3332, ["2"] = .3332 }; break;
            case "disjoint-native": native["answers"]!["q2"]!["score"] = 1.5; break;
            case "off-grid-native": native["answers"]!["q2"]!["score"] = 1.30001; break;
            case "off-grid-binary": native["answers"]!["q0"]!["noul"] = .80001; break;
        }
        using var content = new TrackingContent(new ObservedStream(Encoding.UTF8.GetBytes(native.ToJsonString())));
        using var handler = Script(sent => sent.Uri.AbsolutePath == "/api/show" ? Response(NliShow) : new(HttpStatusCode.OK) { Content = content });
        using var http = new HttpClient(handler);
        using var client = Client(http);
        await Assert.ThrowsAsync<DecisionProtocolException>(() => client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal(2, handler.Requests.Length);
        Assert.True(content.Disposed);
    }

    [Theory]
    [MemberData(nameof(JevTests.InvalidNative), MemberType = typeof(JevTests))]
    public async Task MalformedInference_RejectsAliasedCorrelationAndEnvelopeDefects(string scenario, string body)
    {
        using var content = new TrackingContent(new ObservedStream(Encoding.UTF8.GetBytes(body)));
        using var handler = Script(sent => sent.Uri.AbsolutePath == "/api/show" ? Response(NliShow) : new(HttpStatusCode.OK) { Content = content });
        using var http = new HttpClient(handler);
        using var client = Client(http);
        var error = await Assert.ThrowsAsync<DecisionProtocolException>(() => client.GetResponseAsync(JevTests.AllKinds(), cancellationToken: TestContext.Current.CancellationToken));
        Assert.DoesNotContain("sensitive", error.ToString(), StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(body, error.Message, StringComparison.Ordinal);
        Assert.Equal(["/api/show", "/v1/systemone"], handler.Requests.Select(r => r.Uri.AbsolutePath));
        Assert.True(content.Disposed, scenario);
    }

    [Theory(Timeout = 15000)]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ConcurrentRequests_KeepModelsAliasesAndCancellationIndependent(bool cancelFirst)
    {
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var handler = new ScriptedHandler(async (sent, token) =>
        {
            if (sent.Uri.AbsolutePath == "/api/show") return Response(NliShow);
            string model = Json(sent.Body).GetProperty("model").GetString()!;
            if (model == "nli:first")
            {
                entered.TrySetResult();
                await release.Task.WaitAsync(token);
            }
            return Response(ChoiceNative(model));
        });
        using var http = new HttpClient(handler) { Timeout = Timeout.InfiniteTimeSpan };
        using var client = Client(http);
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        var firstRequest = Request(new ChoiceDecisionQuestion(" q ", "choose", [new("left", "first"), new("\u00e9", "second")]));
        var secondRequest = Request(new ChoiceDecisionQuestion("Q", "choose", [new("right", "first"), new("e\u0301", "second")]));
        var first = client.GetResponseAsync(firstRequest, new() { ModelId = "nli:first" }, cancellation.Token);
        try
        {
            await entered.Task.WaitAsync(TimeSpan.FromSeconds(5), TestContext.Current.CancellationToken);
            var second = await client.GetResponseAsync(secondRequest, new() { ModelId = "nli:second" }, TestContext.Current.CancellationToken);
            Assert.Equal("nli:second", second.ModelId);
            Assert.Equal("e\u0301", second.GetAnswer<ChoiceDecisionAnswer>("Q").SelectedCandidateId);
            Assert.Equal(.75, second.GetAnswer<ChoiceDecisionAnswer>("Q").Probabilities["e\u0301"]);
            if (cancelFirst)
            {
                cancellation.Cancel();
                await Assert.ThrowsAnyAsync<OperationCanceledException>(() => first.WaitAsync(TimeSpan.FromSeconds(5), TestContext.Current.CancellationToken));
            }
            else
            {
                release.TrySetResult();
                var result = await first;
                Assert.Equal("nli:first", result.ModelId);
                Assert.Equal("\u00e9", result.GetAnswer<ChoiceDecisionAnswer>(" q ").SelectedCandidateId);
                Assert.Equal(.75, result.GetAnswer<ChoiceDecisionAnswer>(" q ").Probabilities["\u00e9"]);
                var firstMap = Assert.IsAssignableFrom<IReadOnlyDictionary<string, string>>(result.AdditionalProperties!["systemone.questionIds"]);
                Assert.Equal(" q ", firstMap["q0"]);
                Assert.NotSame(result.AdditionalProperties["systemone.questionIds"], second.AdditionalProperties!["systemone.questionIds"]);
            }
            client.Dispose();
            Assert.Equal("Q", Assert.IsAssignableFrom<IReadOnlyDictionary<string, string>>(second.AdditionalProperties!["systemone.questionIds"])["q0"]);
            Assert.Equal("nli:second", Assert.IsType<JsonElement>(second.RawRepresentation).GetProperty("model").GetString());
            Assert.Equal("1", Assert.IsType<JsonElement>(second.Answers[0].RawRepresentation).GetProperty("choice").GetString());
            Assert.Equal(["nli:first", "nli:first", "nli:second", "nli:second"],
                handler.Requests.Select(r => Json(r.Body).GetProperty("model").GetString()));
        }
        finally
        {
            release.TrySetResult();
            try { await first; } catch (Exception) { /* Always drain the owned pending operation. */ }
        }
    }
}
