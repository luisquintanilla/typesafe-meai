using System.Text.Json;

namespace TypeSafe.MEAI;

/// <summary>Model discovery through the shared transport, not a synthetic chat turn.</summary>
public sealed class TypeSafeModelClient
{
    private readonly TypeSafeHttpTransport _transport;
    /// <summary>Creates a model client borrowing a configured HttpClient.</summary>
    public TypeSafeModelClient(HttpClient httpClient, TypeSafeClientOptions options, string apiKey) =>
        _transport = new(httpClient, options, apiKey);

    /// <summary>Returns the complete owned models wrapper, including additive fields.</summary>
    public async Task<JsonElement> GetModelsAsync(CancellationToken cancellationToken = default)
    {
        var bytes = await _transport.SendAsync("/v1/models", null, cancellationToken).ConfigureAwait(false);
        try
        {
            string text = new System.Text.UTF8Encoding(false, true).GetString(bytes);
            using var doc = JsonDocument.Parse(text);
            TypeSafeJson.UniqueProperties(doc.RootElement);
            if (doc.RootElement.GetProperty("models").ValueKind != JsonValueKind.Array)
                throw new TypeSafeProtocolException("Missing models array.");
            return doc.RootElement.Clone();
        }
        catch (Exception e) when (e is JsonException or InvalidOperationException or KeyNotFoundException or System.Text.DecoderFallbackException)
        {
            throw new TypeSafeProtocolException("Malformed model listing wrapper.");
        }
    }
}
