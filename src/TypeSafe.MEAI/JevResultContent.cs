using System.Text;
using System.Text.Json.Serialization;
using Microsoft.Extensions.AI;

namespace TypeSafe.MEAI;

/// <summary>Owned, self-contained native evidence. Typed views are derived from the inherited data payload, never a second serialized truth.</summary>
public sealed class JevResultContent : DataContent
{
    /// <summary>Versioned integration-owned media type; not an official service media type.</summary>
    public const string ContentMediaType = "application/vnd.typesafe-meai.jev-result.v1+json";
    /// <summary>Register using JsonSerializerOptions.AddAIContentType to preserve this subtype through MEAI serialization.</summary>
    public const string TypeDiscriminator = "typesafe.jevResult.v1";
    internal TypeSafeResponse Native { get; }
    /// <summary>Full owned envelope, including unknown additive fields.</summary>
    [JsonIgnore]
    public System.Text.Json.JsonElement Json => Native.Json;
    /// <summary>Actual native model, not the requested alias.</summary>
    [JsonIgnore]
    public string Model => Native.Model;
    /// <summary>Read-only, validated native answer views.</summary>
    [JsonIgnore]
    public IReadOnlyDictionary<string, TypeSafeAnswer> Answers => Native.Answers;
    /// <summary>Original inference input tokens (also on cache hits).</summary>
    [JsonIgnore]
    public long InputTokens => Native.InputTokens;
    /// <summary>Original inference output tokens.</summary>
    [JsonIgnore]
    public long OutputTokens => Native.OutputTokens;
    /// <summary>Original total, or input plus output when the total was absent.</summary>
    [JsonIgnore]
    public long TotalTokens => Native.TotalTokens;

    /// <summary>Copies and validates a complete UTF-8 native envelope without requiring the original questions.</summary>
    public JevResultContent(ReadOnlyMemory<byte> data) : base(data.ToArray(), ContentMediaType)
    {
        try { Native = TypeSafeResponse.Parse(new UTF8Encoding(false, true).GetString(Data.Span)); }
        catch (DecoderFallbackException) { throw new TypeSafeProtocolException("Native result must contain valid UTF-8 JSON."); }
    }

    /// <summary>Reconstructs the inherited serialized data URI, including strict media type and payload validation.</summary>
    [JsonConstructor]
    public JevResultContent(string uri) : this(ReadUri(uri)) { }

    private static ReadOnlyMemory<byte> ReadUri(string uri)
    {
        var content = new DataContent(uri);
        if (!string.Equals(content.MediaType, ContentMediaType, StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("Expected the versioned Jev result media type.", nameof(uri));
        return content.Data;
    }
}
