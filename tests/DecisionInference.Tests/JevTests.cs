using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using DecisionInference.TypeSafe;
using Microsoft.Extensions.AI;
using static DecisionInference.Tests.ContractTests;

namespace DecisionInference.Tests;

public sealed class JevTests
{
    internal const string Native = """
        {"model":"observed-model","usage":{"input_tokens":11,"output_tokens":7,"total_tokens":18},
         "answers":{
           "q0":{"type":"noul","noul":0.8},
           "q1":{"type":"choice","choice":"1","confidence":0.75,"probabilities":{"0":0.25,"1":0.75}},
           "q2":{"type":"score","score":1.3,"confidence":0.5,"probabilities":{"0":0.2,"1":0.3,"2":0.5},"legend":{"0":"low","1":"mid","2":"high"}}}}
        """;
    internal static DecisionRequest AllKinds() =>
        Request(new BinaryDecisionQuestion("b", "is it?", "yes text", "no text"), Choice(), Score());

    [Fact]
    public async Task TypedRequest_UsesNativeStringEnvelopeAndDecodesAllThreeKinds()
    {
        using var handler = new RecordingHandler(() => Response(Native));
        using var http = new HttpClient(handler);
        using var client = new TypeSafeDecisionClient(http, "fixture-key");
        var result = await client.GetResponseAsync(AllKinds(), new() { ModelId = "override" }, TestContext.Current.CancellationToken);
        var sent = Json(handler.Body!);

        Assert.Equal(HttpMethod.Post, handler.Method);
        Assert.Equal("https://api.typesafe.ai/v1/systemone", handler.Uri!.AbsoluteUri);
        Assert.Equal("Bearer fixture-key", handler.Authorization);
        Assert.Equal("application/json", handler.MediaType);
        Assert.Equal(["state", "model", "questions"], sent.EnumerateObject().Select(p => p.Name));
        Assert.Equal("state", sent.GetProperty("state").GetString());
        Assert.Equal("override", sent.GetProperty("model").GetString());
        var questions = sent.GetProperty("questions");
        Assert.Equal(["q0", "q1", "q2"], questions.EnumerateObject().Select(p => p.Name));
        Assert.Equal("noul", questions.GetProperty("q0").GetProperty("type").GetString());
        Assert.Equal("is it?", questions.GetProperty("q0").GetProperty("instructions").GetString());
        Assert.Equal("yes text", questions.GetProperty("q0").GetProperty("criteria").GetProperty("true").GetString());
        Assert.Equal("no text", questions.GetProperty("q0").GetProperty("criteria").GetProperty("false").GetString());
        Assert.Equal("choice", questions.GetProperty("q1").GetProperty("type").GetString());
        Assert.Equal("choose", questions.GetProperty("q1").GetProperty("instructions").GetString());
        Assert.Equal("first", questions.GetProperty("q1").GetProperty("criteria").GetProperty("0").GetString());
        Assert.Equal("second", questions.GetProperty("q1").GetProperty("criteria").GetProperty("1").GetString());
        Assert.Equal("score", questions.GetProperty("q2").GetProperty("type").GetString());
        Assert.Equal("rate", questions.GetProperty("q2").GetProperty("instructions").GetString());
        Assert.Equal(["low", "mid", "high"], questions.GetProperty("q2").GetProperty("criteria").EnumerateArray().Select(e => e.GetString()));
        Assert.Equal("observed-model", result.ModelId);
        Assert.Equal(.8, Assert.IsType<BinaryDecisionAnswer>(result.Answers[0]).TrueProbability);
        var choice = Assert.IsType<ChoiceDecisionAnswer>(result.Answers[1]);
        Assert.Equal("A", choice.SelectedCandidateId);
        Assert.Equal(Distribution(), choice.Probabilities);
        var score = Assert.IsType<ScoreDecisionAnswer>(result.Answers[2]);
        Assert.Equal(new[] { .2, .3, .5 }, score.Probabilities);
        Assert.Equal(1.3, score.ExpectedIndex, 12);
        Assert.Equal(1.3, score.NativeScore);
        Assert.Equal(11, result.Usage!.InputTokenCount);
        Assert.Equal(7, result.Usage.OutputTokenCount);
        Assert.Equal(18, result.Usage.TotalTokenCount);
        Assert.All(result.Answers, answer => Assert.Equal(DecisionPrecision.HighPrecision, answer.Precision));
    }

    [Fact]
    public async Task NativeEnvelope_PreservesOpaqueQuestionAndCandidateIdsExactly()
    {
        string[] ids = [" q ", "q", "Q", "\u00e9", "e\u0301"];
        var questions = ids.Select(id => (DecisionQuestion)new ChoiceDecisionQuestion(id, "literal \"instruction\"",
            ids.Select(c => new DecisionCandidate(c, $"description {c}")))).ToArray();
        var answerMap = new JsonObject();
        for (int i = ids.Length - 1; i >= 0; i--)
            answerMap[$"q{i}"] = new JsonObject
            {
                ["type"] = "choice", ["choice"] = "4", ["confidence"] = .2,
                ["probabilities"] = new JsonObject(Enumerable.Range(0, ids.Length).Reverse().Select(i => KeyValuePair.Create<string, JsonNode?>(i.ToString(), JsonValue.Create(.2))))
            };
        var native = new JsonObject
        {
            ["model"] = "fixture", ["usage"] = new JsonObject { ["input_tokens"] = 1, ["output_tokens"] = 0 },
            ["answers"] = answerMap
        };
        using var handler = new RecordingHandler(() => Response(native.ToJsonString()));
        using var http = new HttpClient(handler);
        using var client = new TypeSafeDecisionClient(http, "key");
        var result = await client.GetResponseAsync(Request(questions), cancellationToken: TestContext.Current.CancellationToken);
        var sent = Json(handler.Body!).GetProperty("questions");
        Assert.Equal(["q0", "q1", "q2", "q3", "q4"], sent.EnumerateObject().Select(p => p.Name));
        foreach (var q in sent.EnumerateObject())
        {
            Assert.Equal("literal \"instruction\"", q.Value.GetProperty("instructions").GetString());
            Assert.Equal(["0", "1", "2", "3", "4"], q.Value.GetProperty("criteria").EnumerateObject().Select(p => p.Name));
            Assert.Equal(ids.Select(id => $"description {id}"), q.Value.GetProperty("criteria").EnumerateObject().Select(p => p.Value.GetString()));
        }
        Assert.Equal(ids, result.Answers.Select(a => a.QuestionId));
        Assert.All(result.Answers.Cast<ChoiceDecisionAnswer>(), a =>
        {
            Assert.Equal("e\u0301", a.SelectedCandidateId);
            Assert.Equal(ids, a.Probabilities.Keys);
            Assert.All(a.Probabilities.Values, p => Assert.Equal(.2, p));
        });
    }

    public static IEnumerable<object[]> InvalidNative()
    {
        yield return ["malformed", "{sensitive-json"];
        yield return ["duplicate-root", Native.Replace("\"model\":", "\"model\":\"duplicate\",\"model\":", StringComparison.Ordinal)];
        yield return ["duplicate-answer", Native.Replace("\"noul\":0.8", "\"noul\":0.8,\"noul\":0.2", StringComparison.Ordinal)];
        yield return ["duplicate-answer-id", Native.Replace("\"q0\":", "\"q0\":{\"type\":\"noul\",\"noul\":0.2},\"q0\":", StringComparison.Ordinal)];
        yield return ["duplicate-candidate", Native.Replace("\"0\":0.25", "\"0\":0.25,\"0\":0.25", StringComparison.Ordinal)];
        foreach (string scenario in new[] { "missing-answer", "extra-answer", "unknown-answer", "wrong-kind", "missing-candidate",
            "unknown-candidate", "unknown-choice", "native-score", "negative-usage", "negative-output", "overflow-usage", "overflow-json-integer", "wrong-total",
            "missing-usage", "noncontiguous-rubric", "wrong-legend", "bad-confidence", "wrong-json-kind" })
        {
            var root = JsonNode.Parse(Native)!;
            var answers = root["answers"]!.AsObject();
            switch (scenario)
            {
                case "missing-answer": answers.Remove("q0"); break;
                case "extra-answer": answers["extra"] = answers["q0"]!.DeepClone(); break;
                case "unknown-answer": answers["Q0"] = answers["q0"]!.DeepClone(); answers.Remove("q0"); break;
                case "wrong-kind": answers["q0"] = answers["q2"]!.DeepClone(); break;
                case "missing-candidate": answers["q1"]!["probabilities"] = new JsonObject { ["1"] = 1 }; break;
                case "unknown-candidate": answers["q1"]!["probabilities"] = new JsonObject { ["00"] = .25, ["1"] = .75 }; break;
                case "unknown-choice": answers["q1"]!["choice"] = "missing"; break;
                case "native-score": answers["q2"]!["score"] = 1.5; break;
                case "negative-usage": root["usage"]!["input_tokens"] = -1; root["usage"]!["total_tokens"] = 6; break;
                case "negative-output": root["usage"]!["output_tokens"] = -1; root["usage"]!["total_tokens"] = 10; break;
                case "overflow-usage": root["usage"]!["input_tokens"] = long.MaxValue; break;
                case "overflow-json-integer": root["usage"]!["input_tokens"] = ulong.MaxValue; break;
                case "wrong-total": root["usage"]!["total_tokens"] = 999; break;
                case "missing-usage": root.AsObject().Remove("usage"); break;
                case "noncontiguous-rubric": answers["q2"]!["probabilities"] = new JsonObject { ["0"] = .2, ["1"] = .3, ["3"] = .5 }; break;
                case "wrong-legend": answers["q2"]!["legend"]!.AsObject().Remove("2"); break;
                case "bad-confidence": answers["q1"]!["confidence"] = 1.1; break;
                case "wrong-json-kind": answers["q0"]!["noul"] = "sensitive-payload"; break;
            }
            yield return [scenario, root.ToJsonString()];
        }
    }

    [Theory]
    [MemberData(nameof(InvalidNative))]
    public async Task InvalidNativeEnvelope_IsRejectedWithoutPayloadLeak(string scenario, string body)
    {
        using var content = new TrackingContent(new MemoryStream(Encoding.UTF8.GetBytes(body)));
        using var handler = new RecordingHandler(() => new(HttpStatusCode.OK) { Content = content });
        using var http = new HttpClient(handler);
        using var client = new TypeSafeDecisionClient(http, "key");
        var error = await Assert.ThrowsAsync<DecisionProtocolException>(() =>
            client.GetResponseAsync(AllKinds(), cancellationToken: TestContext.Current.CancellationToken));
        Assert.DoesNotContain("sensitive", error.ToString(), StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(body, error.Message, StringComparison.Ordinal);
        Assert.True(content.Disposed, scenario);
    }

    [Fact]
    public async Task InvalidUtf8_IsRejectedAndResponseContentDisposed()
    {
        // Replacement decoding would turn this into a valid native response with model "�(".
        // A 200 response must instead fail strict UTF-8 decoding, before protocol parsing.
        var parts = Native.Split("observed-model", StringSplitOptions.None);
        using var stream = new NonSeekableStream([.. Encoding.UTF8.GetBytes(parts[0]), 0xc3, 0x28, .. Encoding.UTF8.GetBytes(parts[1])]);
        using var content = new TrackingContent(stream);
        using var response = new HttpResponseMessage(HttpStatusCode.OK) { Content = content };
        using var handler = new RecordingHandler(() => response);
        using var http = new HttpClient(handler);
        using var client = new TypeSafeDecisionClient(http, "key");
        var error = await Assert.ThrowsAsync<DecisionProtocolException>(() => client.GetResponseAsync(AllKinds(), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal("Response is not valid UTF-8.", error.Message);
        Assert.True(content.Disposed);
        Assert.True(stream.Disposed);
        Assert.Throws<ObjectDisposedException>(() => response.Content = new ByteArrayContent([]));
    }

    [Theory]
    [InlineData(HttpStatusCode.TooManyRequests)]
    [InlineData(HttpStatusCode.InternalServerError)]
    public async Task ErrorStatus_InvalidUtf8PreservesStatusAndReplacementDiagnosticBody(HttpStatusCode status)
    {
        using var stream = new NonSeekableStream([.. Encoding.UTF8.GetBytes("private-"), 0xc3, 0x28, .. Encoding.UTF8.GetBytes("-body")]);
        using var content = new TrackingContent(stream);
        using var response = new HttpResponseMessage(status) { Content = content };
        response.Headers.Add("x-request-id", "invalid-utf8-fixture");
        using var handler = new RecordingHandler(() => response);
        using var http = new HttpClient(handler);
        using var client = new TypeSafeDecisionClient(http, "key");

        var error = await Assert.ThrowsAsync<TypeSafeApiException>(() =>
            client.GetResponseAsync(AllKinds(), cancellationToken: TestContext.Current.CancellationToken));

        Assert.Equal(status, error.StatusCode);
        Assert.Equal("invalid-utf8-fixture", error.RequestId);
        Assert.Equal("private-\uFFFD(-body", error.DiagnosticBody);
        Assert.Equal($"TypeSafe HTTP operation failed with status {(int)status}.", error.Message);
        Assert.DoesNotContain("private-", error.ToString(), StringComparison.Ordinal);
        Assert.Null(error.InnerException);
        Assert.True(content.Disposed);
        Assert.True(stream.Disposed);
        Assert.Throws<ObjectDisposedException>(() => response.Content = new ByteArrayContent([]));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task BodyLimit_RejectsDeclaredOrActualOverflowAndDisposes(bool declared)
    {
        using var stream = new NonSeekableStream(Encoding.UTF8.GetBytes(Native));
        using var content = new TrackingContent(stream);
        if (declared) content.Headers.ContentLength = 1_000;
        using var handler = new RecordingHandler(() => new(HttpStatusCode.OK) { Content = content });
        using var http = new HttpClient(handler);
        using var client = new TypeSafeDecisionClient(http, "key", new() { MaximumResponseBytes = 32 });
        await Assert.ThrowsAsync<TypeSafeBodyLimitException>(() => client.GetResponseAsync(AllKinds(), cancellationToken: TestContext.Current.CancellationToken));
        Assert.True(content.Disposed);
        Assert.True(stream.Disposed);
        if (declared) Assert.Equal(0, stream.ReadCalls);
        else Assert.True(stream.ReadCalls > 0);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task BodyLimit_AcceptsExactlyLimitBytes(bool declared)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(Native);
        using var handler = new RecordingHandler(() =>
        {
            var content = new StreamContent(new NonSeekableStream(bytes));
            if (declared) content.Headers.ContentLength = bytes.Length;
            return new(HttpStatusCode.OK) { Content = content };
        });
        using var http = new HttpClient(handler);
        using var client = new TypeSafeDecisionClient(http, "key", new() { MaximumResponseBytes = bytes.Length });
        var result = await client.GetResponseAsync(AllKinds(), cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(.8, Assert.IsType<BinaryDecisionAnswer>(result.Answers[0]).TrueProbability);
        Assert.Equal(18, result.Usage!.TotalTokenCount);
    }

    [Fact]
    public async Task BadStatus_SanitizesExceptionButRetainsOptInBodyAndRequestId()
    {
        const string secret = "private diagnostic fixture";
        using var handler = new RecordingHandler(() =>
        {
            var response = new HttpResponseMessage(HttpStatusCode.TooManyRequests) { Content = new StringContent(secret) };
            response.Headers.Add("x-request-id", "fixture-request");
            return response;
        });
        using var http = new HttpClient(handler);
        using var client = new TypeSafeDecisionClient(http, "key");
        var error = await Assert.ThrowsAsync<TypeSafeApiException>(() => client.GetResponseAsync(AllKinds(), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal(HttpStatusCode.TooManyRequests, error.StatusCode);
        Assert.Equal("fixture-request", error.RequestId);
        Assert.Equal(secret, error.DiagnosticBody);
        Assert.Equal("TypeSafe HTTP operation failed with status 429.", error.Message);
        Assert.DoesNotContain(secret, error.ToString(), StringComparison.Ordinal);
    }

    [Fact(Timeout = 15000)]
    public async Task DerivedStreamContent_StalledReadReceivesCancellationAndCompletesBeforeDisposal()
    {
        using var stream = new StalledStream();
        using var content = new TrackingContent(stream);
        using var response = new HttpResponseMessage(HttpStatusCode.OK) { Content = content };
        using var handler = new RecordingHandler(() => response);
        using var http = new HttpClient(handler) { Timeout = Timeout.InfiniteTimeSpan };
        using var client = new TypeSafeDecisionClient(http, "key");
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        var call = client.GetResponseAsync(AllKinds(), cancellationToken: cancellation.Token);
        try
        {
            // Test-only watchdog bounds a broken implementation; no sleeps or live timing assumptions.
            await stream.Entered.Task.WaitAsync(TimeSpan.FromSeconds(5), TestContext.Current.CancellationToken);
            Assert.True(stream.ReadToken.CanBeCanceled, "The underlying per-read token must be cancellable, including for StreamContent subclasses.");
            Assert.Equal(2, stream.TokensReceived.Count);
            Assert.All(stream.TokensReceived, token => Assert.True(token.CanBeCanceled));
            cancellation.Cancel();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => call.WaitAsync(TimeSpan.FromSeconds(5), TestContext.Current.CancellationToken));
            Assert.True(stream.ReadToken.IsCancellationRequested);
            Assert.All(stream.TokensReceived, token => Assert.True(token.IsCancellationRequested));
            Assert.True(stream.ReadCompleted);
            Assert.True(content.Disposed);
            Assert.True(stream.Disposed);
            Assert.False(stream.DisposedWhileReading, "Cancellation must finish the source read, not abandon it.");
            Assert.Throws<ObjectDisposedException>(() => response.Content = new ByteArrayContent([]));
        }
        finally
        {
            // Drain even a regressed implementation; never leave the fixture's operation abandoned.
            stream.Release.TrySetCanceled(TestContext.Current.CancellationToken);
            try { await call; } catch (Exception) { /* expected fault/cancellation under test */ }
        }
    }

    [Fact]
    public async Task BodyIOException_IsSanitizedAndSourceAndResponseDisposed()
    {
        using var stream = new FaultingStream();
        using var content = new TrackingContent(stream);
        using var response = new HttpResponseMessage(HttpStatusCode.OK) { Content = content };
        using var handler = new RecordingHandler(() => response);
        using var http = new HttpClient(handler);
        using var client = new TypeSafeDecisionClient(http, "key");
        var error = await Assert.ThrowsAsync<HttpRequestException>(() => client.GetResponseAsync(AllKinds(), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal(HttpRequestError.ResponseEnded, error.HttpRequestError);
        Assert.Equal("System One response body read failed.", error.Message);
        Assert.Null(error.InnerException);
        Assert.DoesNotContain("private-stream-secret", error.ToString(), StringComparison.Ordinal);
        Assert.True(content.Disposed);
        Assert.True(stream.Disposed);
    }

    [Fact]
    public async Task BorrowedHttpClient_SurvivesDisposalAndRepeatedResponsesOwnEvidence()
    {
        int call = 0;
        var contents = new List<TrackingContent>();
        var streams = new List<NonSeekableStream>();
        using var handler = new RecordingHandler(() =>
        {
            var stream = new NonSeekableStream(Encoding.UTF8.GetBytes(Native.Replace("0.8", ++call == 1 ? "0.8" : "0.1", StringComparison.Ordinal)));
            var content = new TrackingContent(stream);
            streams.Add(stream);
            contents.Add(content);
            return new(HttpStatusCode.OK) { Content = content };
        });
        using var http = new HttpClient(handler);
        var client = new TypeSafeDecisionClient(http, "key");
        var first = await client.GetResponseAsync(AllKinds(), cancellationToken: TestContext.Current.CancellationToken);
        var second = await client.GetResponseAsync(AllKinds(), cancellationToken: TestContext.Current.CancellationToken);
        client.Dispose();
        client.Dispose();
        Assert.False(handler.Disposed);
        Assert.Equal(.8, Assert.IsType<JsonElement>(first.Answers[0].RawRepresentation).GetProperty("noul").GetDouble());
        Assert.Equal(.1, Assert.IsType<JsonElement>(second.Answers[0].RawRepresentation).GetProperty("noul").GetDouble());
        Assert.Equal(.8, Assert.IsType<JsonElement>(first.RawRepresentation).GetProperty("answers").GetProperty("q0").GetProperty("noul").GetDouble());
        Assert.Equal(2, contents.Count);
        Assert.All(contents, content => Assert.True(content.Disposed));
        Assert.All(streams, stream => Assert.True(stream.Disposed));
        foreach (var result in new[] { first, second })
        {
            var choiceRaw = Assert.IsType<JsonElement>(result.Answers[1].RawRepresentation);
            Assert.Equal("1", choiceRaw.GetProperty("choice").GetString());
            Assert.Equal(.25, choiceRaw.GetProperty("probabilities").GetProperty("0").GetDouble());
            Assert.Equal(.75, choiceRaw.GetProperty("probabilities").GetProperty("1").GetDouble());
            var questionIds = Assert.IsAssignableFrom<IReadOnlyDictionary<string, string>>(result.AdditionalProperties!["systemone.questionIds"]);
            Assert.Equal(new Dictionary<string, string> { ["q0"] = "b", ["q1"] = "c", ["q2"] = "s" }, questionIds);
            var candidateIds = Assert.IsAssignableFrom<IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>>>(result.AdditionalProperties["systemone.candidateIds"]);
            Assert.Equal(new Dictionary<string, string> { ["0"] = " a ", ["1"] = "A" }, candidateIds["q1"]);
            var scoreRaw = Assert.IsType<JsonElement>(result.Answers[2].RawRepresentation);
            Assert.Equal(1.3, scoreRaw.GetProperty("score").GetDouble());
            Assert.Equal("high", scoreRaw.GetProperty("legend").GetProperty("2").GetString());
            Assert.Equal(.5, scoreRaw.GetProperty("probabilities").GetProperty("2").GetDouble());
            var root = Assert.IsType<JsonElement>(result.RawRepresentation);
            Assert.Equal("observed-model", root.GetProperty("model").GetString());
            Assert.Equal(18, root.GetProperty("usage").GetProperty("total_tokens").GetInt64());
        }
        Assert.Equal(3, first.Answers.Count);
        using var remaining = await http.GetAsync("https://fixture.invalid/", TestContext.Current.CancellationToken);
        Assert.Equal(HttpStatusCode.OK, remaining.StatusCode);
        Assert.Equal(3, handler.Calls);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => client.GetResponseAsync(AllKinds(), cancellationToken: TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task OwnedHttpClient_DisposesAndPreCanceledRequestNeverSends()
    {
        using var handler = new RecordingHandler(() => Response(Native));
        var http = new HttpClient(handler);
        var client = new TypeSafeDecisionClient(http, "key", ownsHttpClient: true);
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => client.GetResponseAsync(AllKinds(), cancellationToken: cancellation.Token));
        Assert.Equal(0, handler.Calls);
        var metadata = Assert.IsType<DecisionClientMetadata>(client.GetService(typeof(DecisionClientMetadata)));
        Assert.Null(metadata.SupportedKinds);
        Assert.Same(client, client.GetService(typeof(IDecisionClient)));
        Assert.Null(client.GetService(typeof(IDecisionClient), "key"));
        client.Dispose();
        Assert.True(handler.Disposed);
    }

    [Fact]
    public async Task Metadata_DefaultOverrideAndObservedModelRemainDistinct()
    {
        var endpoint = new Uri("https://provider.invalid:8443/");
        using var handler = new RecordingHandler(() => Response(Native));
        using var http = new HttpClient(handler);
        using var client = new TypeSafeDecisionClient(http, "key", new() { Endpoint = endpoint, ModelId = "configured-default" });
        var metadata = client.GetRequiredService<DecisionClientMetadata>();
        Assert.Equal("typesafe", metadata.ProviderName);
        Assert.Equal(endpoint, metadata.ProviderUri);
        Assert.Equal("configured-default", metadata.DefaultModelId);
        Assert.Null(metadata.SupportedKinds);
        var nested = new List<int> { 7 };
        var extensions = new AdditionalPropertiesDictionary { ["nested"] = nested };
        DecisionOptions?[] calls = [null, new() { ModelId = "per-call", AdditionalProperties = extensions }, new() { ModelId = null }];
        string[] expectedModels = ["configured-default", "per-call", "configured-default"];
        for (int i = 0; i < calls.Length; i++)
        {
            var response = await client.GetResponseAsync(AllKinds(), calls[i], TestContext.Current.CancellationToken);
            Assert.Equal(expectedModels[i], Json(handler.Body!).GetProperty("model").GetString());
            Assert.Equal("https://provider.invalid:8443/v1/systemone", handler.Uri!.AbsoluteUri);
            Assert.Equal("observed-model", response.ModelId);
            Assert.Equal(.8, Assert.IsType<BinaryDecisionAnswer>(response.Answers[0]).TrueProbability);
            Assert.Same(metadata, client.GetService<DecisionClientMetadata>());
            Assert.Equal("configured-default", metadata.DefaultModelId);
        }
        Assert.Equal(3, handler.Calls);
        Assert.Equal("per-call", calls[1]!.ModelId);
        Assert.Same(extensions, calls[1]!.AdditionalProperties);
        Assert.Same(nested, calls[1]!.AdditionalProperties!["nested"]);
        Assert.Equal(new[] { 7 }, nested);
        Assert.Null(calls[2]!.ModelId);
        Assert.Null(client.GetService(typeof(DecisionClientMetadata), new object()));
        Assert.Null(client.GetService(typeof(string)));
        Assert.Equal("serviceType", Assert.Throws<ArgumentNullException>(() => client.GetService(null!)).ParamName);
    }

    [Theory]
    [InlineData("")]
    [InlineData(" \t")]
    public async Task InvalidModel_IsRejectedBeforeSending(string model)
    {
        using var handler = new RecordingHandler(() => Response(Native));
        using var http = new HttpClient(handler);
        using var client = new TypeSafeDecisionClient(http, "key");
        var options = new DecisionOptions { ModelId = model };
        await Assert.ThrowsAsync<ArgumentException>(() => client.GetResponseAsync(AllKinds(), options, TestContext.Current.CancellationToken));
        Assert.Equal(0, handler.Calls);
        Assert.Null(handler.Body);
        Assert.Equal(model, options.ModelId);
    }

    [Theory]
    [InlineData("http://127.0.0.1:11435/", false)]
    [InlineData("http://[::1]:11435/", false)]
    [InlineData("http://remote.invalid/", false)]
    [InlineData("https://remote.invalid:8443/", true)]
    public void EndpointPolicy_RemainsHttpsOnlyAfterSharingOllayaTransport(string endpoint, bool accepted)
    {
        using var handler = new RecordingHandler(() => Response(Native));
        using var http = new HttpClient(handler);
        var options = new TypeSafeDecisionOptions { Endpoint = new(endpoint) };
        if (accepted)
        {
            using var client = new TypeSafeDecisionClient(http, "fixture-key", options);
            Assert.Equal(options.Endpoint, client.GetRequiredService<DecisionClientMetadata>().ProviderUri);
            Assert.Equal("typesafe", client.GetRequiredService<DecisionClientMetadata>().ProviderName);
        }
        else Assert.Throws<ArgumentException>(() => new TypeSafeDecisionClient(http, "fixture-key", options));
        Assert.Equal(0, handler.Calls);
        Assert.False(handler.Disposed);
    }

    [Fact]
    public async Task SingleCandidateChoice_RemainsSupportedWithoutOllayaMinimum()
    {
        const string body = """
            {"model":"observed-model","usage":{"input_tokens":2,"output_tokens":0},
             "answers":{"q0":{"type":"choice","choice":"0","confidence":1,"probabilities":{"0":1}}}}
            """;
        using var handler = new RecordingHandler(() => Response(body));
        using var http = new HttpClient(handler);
        using var client = new TypeSafeDecisionClient(http, "fixture-key");
        var request = Request(new ChoiceDecisionQuestion(" q ", "choose", [new(" \u00e9 ", "only option")]));
        var result = await client.GetResponseAsync(request, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(1, handler.Calls);
        Assert.Equal(HttpMethod.Post, handler.Method);
        Assert.Equal("/v1/systemone", handler.Uri!.AbsolutePath);
        var wire = Json(handler.Body!).GetProperty("questions");
        Assert.Equal("q0", Assert.Single(wire.EnumerateObject()).Name);
        Assert.Equal("only option", wire.GetProperty("q0").GetProperty("criteria").GetProperty("0").GetString());
        Assert.Equal("0", Assert.Single(wire.GetProperty("q0").GetProperty("criteria").EnumerateObject()).Name);
        var answer = result.GetAnswer<ChoiceDecisionAnswer>(" q ");
        Assert.Equal(" \u00e9 ", answer.SelectedCandidateId);
        Assert.Equal(new KeyValuePair<string, double>(" \u00e9 ", 1), Assert.Single(answer.Probabilities));
        Assert.Equal(DecisionPrecision.HighPrecision, answer.Precision);
        Assert.Equal(2, result.Usage!.TotalTokenCount);
        Assert.Equal("0", Assert.IsType<JsonElement>(answer.RawRepresentation).GetProperty("choice").GetString());
    }

    private static HttpResponseMessage Response(string body) => new(HttpStatusCode.OK) { Content = new StringContent(body) };

    private sealed class RecordingHandler(Func<HttpResponseMessage> response) : HttpMessageHandler
    {
        public string? Body, Authorization, MediaType;
        public HttpMethod? Method;
        public Uri? Uri;
        public int Calls;
        public bool Disposed;
        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Calls++;
            Method = request.Method;
            Uri = request.RequestUri;
            Authorization = request.Headers.Authorization?.ToString();
            MediaType = request.Content?.Headers.ContentType?.MediaType;
            Body = request.Content is null ? null : await request.Content.ReadAsStringAsync(cancellationToken);
            return response();
        }
        protected override void Dispose(bool disposing) { Disposed = true; base.Dispose(disposing); }
    }

    // Deliberately a subclass: overriding only Dispose must not lose per-read cancellation.
    private sealed class TrackingContent(Stream stream) : StreamContent(stream)
    {
        public bool Disposed;
        protected override void Dispose(bool disposing) { Disposed = true; base.Dispose(disposing); }
    }

    private class NonSeekableStream(byte[] bytes) : Stream
    {
        private readonly MemoryStream _inner = new(bytes);
        public bool Disposed;
        public int ReadCalls;
        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override int Read(byte[] buffer, int offset, int count) { ReadCalls++; return _inner.Read(buffer, offset, count); }
        public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
            ReadAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();
        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        { ReadCalls++; return _inner.ReadAsync(buffer, cancellationToken); }
        public override void Flush() => throw new NotSupportedException();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        protected override void Dispose(bool disposing) { Disposed = true; _inner.Dispose(); base.Dispose(disposing); }
    }

    private sealed class StalledStream() : NonSeekableStream([])
    {
        public TaskCompletionSource Entered { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource<int> Release { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public CancellationToken ReadToken { get; private set; }
        public List<CancellationToken> TokensReceived { get; } = [];
        public bool ReadCompleted, DisposedWhileReading;
        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            ReadToken = cancellationToken;
            TokensReceived.Add(cancellationToken);
            if (TokensReceived.Count == 1)
            {
                buffer.Span[0] = (byte)' ';
                return 1; // Second read must still receive cancellation; no abandoned operation.
            }
            using var registration = cancellationToken.Register(() => Release.TrySetCanceled(cancellationToken));
            Entered.TrySetResult();
            try { return await Release.Task; }
            finally { ReadCompleted = true; }
        }
        protected override void Dispose(bool disposing)
        { DisposedWhileReading |= Entered.Task.IsCompleted && !ReadCompleted; base.Dispose(disposing); }
    }

    private sealed class FaultingStream() : NonSeekableStream([])
    {
        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default) =>
            ValueTask.FromException<int>(new IOException("private-stream-secret"));
    }
}
