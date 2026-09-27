using System.Collections.Concurrent;
using System.Net;
using System.Text;
using System.Text.Json;
using static DecisionInference.Tests.ContractTests;

namespace DecisionInference.Tests;

// Minimal relevant discovery projections in local deterministic handlers, NOT a
// server or trained-model run. Unused optional metadata is intentionally omitted.
// Shape: ollaya-dev/ollaya@152ad20c88f8ea9b6d1acf3ed0e06b002d38b2b4:
// docs/api.md, crates/ollaya-server/src/views.rs, crates/ollaya-decision/src/answer.rs.
// general.engine is "onnxruntime" (not details.format "onnx"), also independently
// corroborated by the parent's retained api-show.json. Unused license text is omitted.
internal static class OllayaTestFixtures
{
    internal const string Model = "nli:modernbert-large";
    internal const string NliShow = """
        {"questions":null,"router":null,
         "details":{"parent_model":"","format":"onnx","family":"nli","families":["nli"],"parameter_size":"396M","quantization_level":"F32"},
         "model_info":{"general.architecture":"nli","general.engine":"onnxruntime","general.languages":["en"],
                       "nli.layout":"nli-pairs-v1","nli.context_length":512},
         "capabilities":["choice","score","noul"]}
        """;
    internal const string GliclassShow = """
        {"questions":null,"router":null,
         "details":{"format":"onnx","family":"gliclass","families":["gliclass"]},
         "model_info":{"general.architecture":"gliclass","general.engine":"onnxruntime",
                       "gliclass.layout":"gliclass-uni-v1"},
         "capabilities":["choice","score","noul"]}
        """;
    // Config-derived restricted profile, not a live guard run. Embedded schema verified
    // against pinned blob b7ed6e0fac05095ada9799393ef79cbd72879a5a3ddb2d9ded68806941deb328.
    // The server advertises all three kinds despite this fixed-question restriction.
    internal const string GuardShow = """
        {"questions":{
           "safety":{"type":"choice","instructions":"Qwen3Guard safety level of the user text","criteria":{"safe":null,"controversial":null,"unsafe":null}},
           "unsafe":{"type":"noul","instructions":"Qwen3Guard: the user text is unsafe"},
           "unsafe_strict":{"type":"noul","instructions":"Qwen3Guard: the user text is unsafe or controversial"},
           "category":{"type":"choice","instructions":"Qwen3Guard: first unsafe category, given unsafe","criteria":{
             "none":null,"violent":null,"non_violent_illegal":null,"sexual":null,"pii":null,"suicide_self_harm":null,
             "unethical":null,"politically_sensitive":null,"copyright":null,"jailbreak":null}}},
         "router":null,"details":{"format":"onnx","family":"qwen3guard","families":["qwen3guard"]},
         "model_info":{"general.architecture":"qwen3guard","general.engine":"onnxruntime",
                       "qwen3guard.layout":"qwen3guard-gen-v1","qwen3guard.context_length":32768},
         "capabilities":["choice","score","noul"]}
        """;

    internal static string Native(string model = Model) =>
        JevTests.Native.Replace("observed-model", model, StringComparison.Ordinal);

    internal static string ChoiceNative(string model = Model) => JsonSerializer.Serialize(new
    {
        model,
        answers = new { q0 = new { type = "choice", choice = "1", confidence = .75, probabilities = new Dictionary<string, double> { ["0"] = .25, ["1"] = .75 } } },
        usage = new { input_tokens = 11, output_tokens = 7 }
    });

    internal static HttpResponseMessage Response(string body, HttpStatusCode status = HttpStatusCode.OK) =>
        new(status) { Content = new StringContent(body, Encoding.UTF8, "application/json") };

    internal static ScriptedHandler Script(Func<RecordedRequest, HttpResponseMessage> reply) =>
        new((request, _) => Task.FromResult(reply(request)));

    internal static ScriptedHandler NormalHandler(string show = NliShow) => Script(request =>
        request.Uri.AbsolutePath switch
        {
            "/api/show" => Response(show),
            "/v1/systemone" => Response(Native(Json(request.Body).GetProperty("model").GetString()!)),
            _ => throw new InvalidOperationException("Unexpected fixture route.")
        });

    internal sealed record RecordedRequest(Uri Uri, HttpMethod Method, string? Authorization, string? MediaType, string Body);

    internal sealed class ScriptedHandler(Func<RecordedRequest, CancellationToken, Task<HttpResponseMessage>> reply) : HttpMessageHandler
    {
        private readonly ConcurrentQueue<RecordedRequest> _requests = new();
        private int _disposeCount;
        internal RecordedRequest[] Requests => _requests.ToArray();
        internal int DisposeCount => Volatile.Read(ref _disposeCount);
        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            var captured = new RecordedRequest(request.RequestUri!, request.Method,
                request.Headers.Authorization?.ToString(), request.Content?.Headers.ContentType?.MediaType,
                request.Content is null ? "" : await request.Content.ReadAsStringAsync(cancellationToken));
            _requests.Enqueue(captured);
            return await reply(captured, cancellationToken);
        }
        protected override void Dispose(bool disposing)
        {
            if (disposing) Interlocked.Increment(ref _disposeCount);
            base.Dispose(disposing);
        }
    }

    // A derived StreamContent is important: cancellation must reach the underlying ReadAsync.
    internal sealed class TrackingContent(Stream stream) : StreamContent(stream)
    {
        internal bool Disposed { get; private set; }
        protected override void Dispose(bool disposing) { Disposed = true; base.Dispose(disposing); }
    }

    internal class ObservedStream(byte[] bytes) : Stream
    {
        private readonly MemoryStream _inner = new(bytes);
        internal bool Disposed { get; private set; }
        internal int ReadCalls { get; private set; }
        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override int Read(byte[] buffer, int offset, int count) { ReadCalls++; return _inner.Read(buffer, offset, count); }
        public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
            ReadAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();
        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            ReadCalls++;
            return _inner.ReadAsync(buffer, cancellationToken);
        }
        public override void Flush() => throw new NotSupportedException();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        protected override void Dispose(bool disposing) { Disposed = true; _inner.Dispose(); base.Dispose(disposing); }
    }

    internal sealed class StalledStream() : ObservedStream([])
    {
        internal TaskCompletionSource Entered { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        internal TaskCompletionSource<int> Release { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        internal List<CancellationToken> ReadTokens { get; } = [];
        internal bool ReadCompleted { get; private set; }
        internal bool DisposedWhileReading { get; private set; }
        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            ReadTokens.Add(cancellationToken);
            if (ReadTokens.Count == 1) { buffer.Span[0] = (byte)' '; return 1; }
            using var registration = cancellationToken.Register(() => Release.TrySetCanceled(cancellationToken));
            Entered.TrySetResult();
            try { return await Release.Task; }
            finally { ReadCompleted = true; }
        }
        protected override void Dispose(bool disposing)
        {
            DisposedWhileReading |= Entered.Task.IsCompleted && !ReadCompleted;
            base.Dispose(disposing);
        }
    }

    internal sealed class FaultingStream() : ObservedStream([])
    {
        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default) =>
            ValueTask.FromException<int>(new IOException("private-stream-secret"));
    }
}
