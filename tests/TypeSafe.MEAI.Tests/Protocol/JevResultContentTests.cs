using System.Text;
using System.Runtime.InteropServices;
using System.Text.Json;
using Microsoft.Extensions.AI;
using TypeSafe.MEAI.Tests.Fixtures;
using Xunit;

namespace TypeSafe.MEAI.Tests.Protocol;

[Trait("Stage", "A")]
public sealed class JevResultContentTests
{
    [Fact]
    public async Task Batch_ClonesSharedEvidenceForIndependentConcurrentResponses()
    {
        var evidence = new JevResultContent(Encoding.UTF8.GetBytes(" \n" + NativeFixture.Envelope + "\t"))
        {
            Name = "native.json",
            AdditionalProperties = new() { ["metadata"] = new List<string> { "original" } },
            Annotations = [new CitationAnnotation { Title = "original" }]
        };
        using var shared = new SharedEvidenceClient(evidence);
        using var client = new TypeSafeChatClient(shared, NativeFixture.Settings);
        var results = await Task.WhenAll(Enumerable.Range(0, 16).Select(_ =>
            client.GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), TestContext.Current.CancellationToken)));
        var copies = results.Select(result => Assert.IsType<JevResultContent>(Assert.Single(result.Messages[0].Contents))).ToArray();
        copies[0].AdditionalProperties!["metadata"] = "changed";
        Assert.IsType<CitationAnnotation>(Assert.Single(copies[0].Annotations!)).Title = "changed";

        Assert.Equal(16, shared.Calls);
        Assert.Equal(16, copies.Distinct(ReferenceEqualityComparer.Instance).Count());
        Assert.False(evidence.AdditionalProperties.ContainsKey("typesafe.v1.batch-role"));
        Assert.Equal("original", Assert.IsType<CitationAnnotation>(Assert.Single(evidence.Annotations)).Title);
        Assert.All(copies.Skip(1), copy =>
        {
            Assert.NotSame(evidence, copy);
            Assert.Same(evidence.Native, copy.Native);
            Assert.Equal(evidence.Data.ToArray(), copy.Data.ToArray());
            Assert.Equal("native", copy.AdditionalProperties!["typesafe.v1.batch-role"]);
            Assert.Equal("original", Assert.IsType<JsonElement>(copy.AdditionalProperties["metadata"])[0].GetString());
            Assert.Equal("original", Assert.IsType<CitationAnnotation>(Assert.Single(copy.Annotations!)).Title);
            Assert.Equal("native.json", copy.Name);
            NativeFixture.AssertNative(copy);
        });
    }

    [Fact]
    public void Copy_ReusesImmutableNativeViewsButOwnsBytesAndDeepMutableMetadata()
    {
        var nested = new List<string> { "original" };
        var annotation = new CitationAnnotation
        {
            Title = "original",
            AdditionalProperties = new() { ["tags"] = nested },
            AnnotatedRegions = [new TextSpanAnnotatedRegion { StartIndex = 1, EndIndex = 3 }],
            RawRepresentation = new object()
        };
        var original = new JevResultContent(Encoding.UTF8.GetBytes(" \r\n" + NativeFixture.Envelope + "\t"))
        {
            Name = "native.json",
            AdditionalProperties = new() { ["nested"] = new Dictionary<string, object> { ["tags"] = nested } },
            Annotations = [annotation],
            RawRepresentation = new object()
        };
        var copy = new JevResultContent(original);
        Assert.Same(original.Native, copy.Native);
        Assert.Same(original.Answers, copy.Answers);
        Assert.Equal(original.Data.ToArray(), copy.Data.ToArray());
        Assert.True(MemoryMarshal.TryGetArray(original.Data, out var originalBytes));
        Assert.True(MemoryMarshal.TryGetArray(copy.Data, out var copyBytes));
        Assert.NotSame(originalBytes.Array, copyBytes.Array);
        Assert.NotSame(original.AdditionalProperties, copy.AdditionalProperties);
        Assert.NotSame(original.Annotations, copy.Annotations);
        Assert.Null(copy.RawRepresentation);
        Assert.Equal("native.json", copy.Name);
        nested[0] = "mutated";
        annotation.Title = "mutated";
        Assert.IsType<TextSpanAnnotatedRegion>(annotation.AnnotatedRegions[0]).EndIndex = 99;
        original.Annotations.Clear();
        original.AdditionalProperties.Clear();
        var copiedAnnotation = Assert.IsType<CitationAnnotation>(Assert.Single(copy.Annotations!));
        Assert.NotSame(annotation, copiedAnnotation);
        Assert.Null(copiedAnnotation.RawRepresentation);
        Assert.Equal("original", copiedAnnotation.Title);
        Assert.Equal(3, Assert.IsType<TextSpanAnnotatedRegion>(Assert.Single(copiedAnnotation.AnnotatedRegions!)).EndIndex);
        Assert.Equal("original", Assert.IsType<JsonElement>(copiedAnnotation.AdditionalProperties!["tags"])[0].GetString());
        Assert.Equal("original", Assert.IsType<JsonElement>(copy.AdditionalProperties!["nested"]).GetProperty("tags")[0].GetString());
        NativeFixture.AssertNative(copy);
    }

    [Fact]
    public void Constructor_OwnsExactUtf8AndImmutableAnswersWithoutQuestionSnapshot()
    {
        byte[] input = Encoding.UTF8.GetBytes(NativeFixture.Envelope);
        byte[] expected = input.ToArray();
        var evidence = new JevResultContent(input);
        Array.Fill(input, (byte)'x');
        Assert.Equal(expected, evidence.Data.ToArray());
        NativeFixture.AssertNative(evidence);
        var answers = Assert.IsAssignableFrom<IDictionary<string, TypeSafeAnswer>>(evidence.Answers);
        Assert.Throws<NotSupportedException>(() => answers.Remove("quality"));
        Assert.Equal(3, evidence.Answers.Count);
        Assert.Equal(1.25, Assert.IsType<ScoreAnswer>(evidence.Answers["quality"]).Score);
        Assert.Null(evidence.RawRepresentation);
    }

    [Theory]
    [InlineData("AIContent")]
    [InlineData("DataContent")]
    [InlineData("ChatResponse")]
    public void RegisteredJson_RoundtripPreservesSubtypeExactPayloadAnnotationsAndAdditiveFields(string declaredType)
    {
        var evidence = new JevResultContent(Encoding.UTF8.GetBytes(NativeFixture.Envelope))
        {
            AdditionalProperties = new() { ["typesafe.v1.batch-role"] = "native", ["host-metadata"] = NativeFixture.Json("""{"tenant":"synthetic","sequence":17}""") },
            Annotations = [new CitationAnnotation { Title = "Synthetic source", FileId = "fixture-17", Snippet = "owned evidence" }]
        };
        var json = NativeFixture.ContentJson();
        JevResultContent restored;
        string serialized;
        if (declaredType == "ChatResponse")
        {
            var response = new ChatResponse(new ChatMessage(ChatRole.Assistant, [evidence]))
            { ModelId = evidence.Model, Usage = new() { InputTokenCount = 11, OutputTokenCount = 7, TotalTokenCount = 18 } };
            serialized = JsonSerializer.Serialize(response, json);
            var roundtrip = JsonSerializer.Deserialize<ChatResponse>(serialized, json)!;
            restored = Assert.IsType<JevResultContent>(Assert.Single(Assert.Single(roundtrip.Messages).Contents));
            Assert.Equal("", roundtrip.Text);
            Assert.Equal("jev-1.13.0", roundtrip.ModelId);
            Assert.Equal(18, roundtrip.Usage!.TotalTokenCount);
        }
        else if (declaredType == "DataContent")
        {
            serialized = JsonSerializer.Serialize<DataContent>(evidence, json);
            restored = Assert.IsType<JevResultContent>(JsonSerializer.Deserialize<DataContent>(serialized, json));
        }
        else
        {
            serialized = JsonSerializer.Serialize<AIContent>(evidence, json);
            restored = Assert.IsType<JevResultContent>(JsonSerializer.Deserialize<AIContent>(serialized, json));
        }
        Assert.Contains(JevResultContent.TypeDiscriminator, serialized);
        Assert.NotSame(evidence, restored);
        Assert.Equal(evidence.Data.ToArray(), restored.Data.ToArray());
        NativeFixture.AssertNative(restored);
        Assert.Equal("native", Assert.IsType<JsonElement>(restored.AdditionalProperties!["typesafe.v1.batch-role"]).GetString());
        Assert.Equal(17, Assert.IsType<JsonElement>(restored.AdditionalProperties["host-metadata"]).GetProperty("sequence").GetInt32());
        var annotation = Assert.IsType<CitationAnnotation>(Assert.Single(restored.Annotations!));
        Assert.Equal("Synthetic source", annotation.Title);
        Assert.Equal("fixture-17", annotation.FileId);
        Assert.Equal("owned evidence", annotation.Snippet);
        Assert.Null(restored.RawRepresentation);
    }

    [Fact]
    public void Json_DefaultOptionsRemainUnregisteredAndFailExplicitly()
    {
        AIContent evidence = new JevResultContent(Encoding.UTF8.GetBytes(NativeFixture.Envelope));
        _ = NativeFixture.ContentJson();
        Assert.Throws<NotSupportedException>(() => JsonSerializer.Serialize(evidence, AIJsonUtilities.DefaultOptions));
        // An unregistered DataContent contract has no polymorphic map at all.
        // Its bytes survive, but applications must register the subtype to retain it.
        var unregisteredData = JsonSerializer.Deserialize<DataContent>(
            JsonSerializer.Serialize((DataContent)evidence, AIJsonUtilities.DefaultOptions), AIJsonUtilities.DefaultOptions);
        Assert.IsType<DataContent>(unregisteredData);
        Assert.Equal(((JevResultContent)evidence).Data.ToArray(), unregisteredData.Data.ToArray());
        Assert.Equal(JevResultContent.ContentMediaType, unregisteredData.MediaType);
        var encoded = JsonSerializer.Serialize((JevResultContent)evidence, NativeFixture.ContentJson());
        var payload = NativeFixture.Json(encoded);
        Assert.False(payload.TryGetProperty("answers", out _));
        Assert.False(payload.TryGetProperty("model", out _));
        Assert.False(payload.TryGetProperty("json", out _));
        Assert.True(payload.TryGetProperty("uri", out _));
    }

    [Fact]
    public void Constructor_ValidatesMediaMalformedEnvelopeAndStrictUtf8()
    {
        var wrong = new DataContent(Encoding.UTF8.GetBytes(NativeFixture.Envelope), "application/json");
        Assert.Throws<ArgumentException>(() => new JevResultContent(wrong.Uri));
        Assert.Throws<TypeSafeProtocolException>(() => new JevResultContent("{}"u8.ToArray()));
        var error = Assert.Throws<TypeSafeProtocolException>(() => new JevResultContent(new byte[] { 0x22, 0xff, 0x22 }));
        Assert.Equal("Native result must contain valid UTF-8 JSON.", error.Message);
        Assert.Null(error.InnerException);
        Assert.DoesNotContain(nameof(DecoderFallbackException), error.ToString());
        var right = new DataContent(Encoding.UTF8.GetBytes(NativeFixture.Envelope), JevResultContent.ContentMediaType);
        NativeFixture.AssertNative(new JevResultContent(right.Uri));
    }

    private sealed class SharedEvidenceClient(JevResultContent evidence) : IChatClient
    {
        private int _calls;
        internal int Calls => Volatile.Read(ref _calls);
        public async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            Interlocked.Increment(ref _calls);
            await Task.Yield();
            return new ChatResponse(new ChatMessage(ChatRole.Assistant, [evidence]));
        }
        public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages,
            ChatOptions? options = null, CancellationToken cancellationToken = default) => throw new NotSupportedException();
        public object? GetService(Type serviceType, object? serviceKey = null) => throw new InvalidOperationException("No service bypass.");
        public void Dispose() { }
    }
}
