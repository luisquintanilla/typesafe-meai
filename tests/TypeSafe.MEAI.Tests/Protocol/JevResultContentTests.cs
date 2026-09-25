using System.Text;
using System.Text.Json;
using Microsoft.Extensions.AI;
using TypeSafe.MEAI.Tests.Fixtures;
using Xunit;

namespace TypeSafe.MEAI.Tests.Protocol;

[Trait("Stage", "A")]
public sealed class JevResultContentTests
{
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
}
