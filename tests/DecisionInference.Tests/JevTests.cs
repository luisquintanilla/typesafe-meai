using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using DecisionInference.TypeSafe;
using static DecisionInference.Tests.ContractTests;

namespace DecisionInference.Tests;

public sealed class JevTests
{
    private const string Native = """
        {"model":"observed-model","usage":{"input_tokens":11,"output_tokens":7,"total_tokens":18},
         "answers":{
           "b":{"type":"noul","noul":0.8},
           "c":{"type":"choice","choice":"A","confidence":0.75,"probabilities":{" a ":0.25,"A":0.75}},
           "s":{"type":"score","score":1.3,"confidence":0.5,"probabilities":{"0":0.2,"1":0.3,"2":0.5},"legend":{"0":"low","1":"mid","2":"high"}}}}
        """;
    private static DecisionRequest AllKinds() =>
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
        Assert.Equal(["b", "c", "s"], questions.EnumerateObject().Select(p => p.Name));
        Assert.Equal("noul", questions.GetProperty("b").GetProperty("type").GetString());
        Assert.Equal("is it?", questions.GetProperty("b").GetProperty("instructions").GetString());
        Assert.Equal("yes text", questions.GetProperty("b").GetProperty("criteria").GetProperty("true").GetString());
        Assert.Equal("no text", questions.GetProperty("b").GetProperty("criteria").GetProperty("false").GetString());
        Assert.Equal("choice", questions.GetProperty("c").GetProperty("type").GetString());
        Assert.Equal("choose", questions.GetProperty("c").GetProperty("instructions").GetString());
        Assert.Equal("first", questions.GetProperty("c").GetProperty("criteria").GetProperty(" a ").GetString());
        Assert.Equal("second", questions.GetProperty("c").GetProperty("criteria").GetProperty("A").GetString());
        Assert.Equal("score", questions.GetProperty("s").GetProperty("type").GetString());
        Assert.Equal("rate", questions.GetProperty("s").GetProperty("instructions").GetString());
        Assert.Equal(["low", "mid", "high"], questions.GetProperty("s").GetProperty("criteria").EnumerateArray().Select(e => e.GetString()));
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
    }

    [Fact]
    public async Task NativeEnvelope_PreservesOpaqueQuestionAndCandidateIdsExactly()
    {
        string[] ids = [" q ", "q", "Q", "\u00e9", "e\u0301"];
        var questions = ids.Select(id => (DecisionQuestion)new ChoiceDecisionQuestion(id, "literal \"instruction\"",
            ids.Select(c => new DecisionCandidate(c, $"description {c}")))).ToArray();
        var answerMap = new JsonObject();
        foreach (string id in ids)
            answerMap[id] = new JsonObject
            {
                ["type"] = "choice", ["choice"] = "e\u0301", ["confidence"] = .2,
                ["probabilities"] = new JsonObject(ids.Select(c => KeyValuePair.Create<string, JsonNode?>(c, JsonValue.Create(.2))))
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
        Assert.Equal(ids, sent.EnumerateObject().Select(p => p.Name));
        foreach (var q in sent.EnumerateObject())
        {
            Assert.Equal("literal \"instruction\"", q.Value.GetProperty("instructions").GetString());
            Assert.Equal(ids, q.Value.GetProperty("criteria").EnumerateObject().Select(p => p.Name));
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
        yield return ["duplicate-answer-id", Native.Replace("\"b\":", "\"b\":{\"type\":\"noul\",\"noul\":0.2},\"b\":", StringComparison.Ordinal)];
        yield return ["duplicate-candidate", Native.Replace("\" a \":0.25", "\" a \":0.25,\" a \":0.25", StringComparison.Ordinal)];
        foreach (string scenario in new[] { "missing-answer", "extra-answer", "unknown-answer", "wrong-kind", "missing-candidate",
            "unknown-candidate", "unknown-choice", "native-score", "negative-usage", "negative-output", "overflow-usage", "overflow-json-integer", "wrong-total",
            "missing-usage", "noncontiguous-rubric", "wrong-legend", "bad-confidence", "wrong-json-kind" })
        {
            var root = JsonNode.Parse(Native)!;
            var answers = root["answers"]!.AsObject();
            switch (scenario)
            {
                case "missing-answer": answers.Remove("b"); break;
                case "extra-answer": answers["extra"] = answers["b"]!.DeepClone(); break;
                case "unknown-answer": answers["B"] = answers["b"]!.DeepClone(); answers.Remove("b"); break;
                case "wrong-kind": answers["b"] = answers["s"]!.DeepClone(); break;
                case "missing-candidate": answers["c"]!["probabilities"] = new JsonObject { ["A"] = 1 }; break;
                case "unknown-candidate": answers["c"]!["probabilities"] = new JsonObject { ["a"] = .25, ["A"] = .75 }; break;
                case "unknown-choice": answers["c"]!["choice"] = "missing"; break;
                case "native-score": answers["s"]!["score"] = 1.5; break;
                case "negative-usage": root["usage"]!["input_tokens"] = -1; root["usage"]!["total_tokens"] = 6; break;
                case "negative-output": root["usage"]!["output_tokens"] = -1; root["usage"]!["total_tokens"] = 10; break;
                case "overflow-usage": root["usage"]!["input_tokens"] = long.MaxValue; break;
                case "overflow-json-integer": root["usage"]!["input_tokens"] = ulong.MaxValue; break;
                case "wrong-total": root["usage"]!["total_tokens"] = 999; break;
                case "missing-usage": root.AsObject().Remove("usage"); break;
                case "noncontiguous-rubric": answers["s"]!["probabilities"] = new JsonObject { ["0"] = .2, ["1"] = .3, ["3"] = .5 }; break;
                case "wrong-legend": answers["s"]!["legend"]!.AsObject().Remove("2"); break;
                case "bad-confidence": answers["c"]!["confidence"] = 1.1; break;
                case "wrong-json-kind": answers["b"]!["noul"] = "sensitive-payload"; break;
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
        Assert.Equal("TypeSafe response body read failed.", error.Message);
        Assert.Null(error.InnerException);
        Assert.DoesNotContain("private-stream-secret", error.ToString(), StringComparison.Ordinal);
        Assert.True(content.Disposed);
        Assert.True(stream.Disposed);
    }

    [Fact]
    public async Task BorrowedHttpClient_SurvivesDisposalAndRepeatedResponsesOwnEvidence()
    {
        int call = 0;
        using var handler = new RecordingHandler(() => Response(Native.Replace("0.8", ++call == 1 ? "0.8" : "0.1", StringComparison.Ordinal)));
        using var http = new HttpClient(handler);
        var client = new TypeSafeDecisionClient(http, "key");
        var first = await client.GetResponseAsync(AllKinds(), cancellationToken: TestContext.Current.CancellationToken);
        var second = await client.GetResponseAsync(AllKinds(), cancellationToken: TestContext.Current.CancellationToken);
        client.Dispose();
        client.Dispose();
        Assert.False(handler.Disposed);
        Assert.Equal(.8, first.Answers[0].ProviderEvidence.GetProperty("noul").GetDouble());
        Assert.Equal(.1, second.Answers[0].ProviderEvidence.GetProperty("noul").GetDouble());
        Assert.Equal(.8, first.ProviderEvidence.GetProperty("answers").GetProperty("b").GetProperty("noul").GetDouble());
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
        Assert.Equal([DecisionKind.Binary, DecisionKind.Choice, DecisionKind.Score], metadata.SupportedKinds);
        Assert.Same(client, client.GetService(typeof(IDecisionClient)));
        Assert.Null(client.GetService(typeof(IDecisionClient), "key"));
        client.Dispose();
        Assert.True(handler.Disposed);
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
