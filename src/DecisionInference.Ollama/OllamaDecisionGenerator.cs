using System.Globalization;
using System.Net;
using System.Text.Json;
using DecisionInference.SystemOne;

namespace DecisionInference.Ollama;

/// <summary>Connection settings for an independently managed Ollama server.</summary>
public sealed record OllamaDecisionGeneratorOptions
{
    public Uri Endpoint { get; init; } = new("http://127.0.0.1:11434");
    /// <summary>Default local model selection. Null requires a per-call ModelId; the response label is not digest attestation.</summary>
    public string? ModelId { get; init; }
    public int MaximumResponseBytes { get; init; } = 4 * 1024 * 1024;
}

/// <summary>Thin System One adapter for Ollama v0.35.0 at cc4069396f3ad2c370c53eed2e4a42ac13adab84.</summary>
/// <remarks>
/// Supports 1..64 questions, Choice/Score 2..26, and at most 65,536 serialized UTF-8 request bytes.
/// The server owns model compatibility and nontruncating context validation. No discovery or model management occurs.
/// Inject an isolated HttpClient with redirects/cookies disabled and an explicit deadline.
/// Cancellation stops HTTP waiting; it does not guarantee interruption of server inference.
/// </remarks>
public sealed class OllamaDecisionGenerator : IDecisionGenerator
{
    private const int MaximumRequestBytes = 65_536;
    private readonly SystemOneTransport _transport;
    private readonly DecisionGeneratorMetadata _metadata;

    public OllamaDecisionGenerator(HttpClient httpClient, OllamaDecisionGeneratorOptions? options = null,
        bool ownsHttpClient = false)
    {
        ArgumentNullException.ThrowIfNull(httpClient);
        options ??= new();
        SystemOneTransport.ValidateEndpoint(options.Endpoint, allowLoopbackHttp: true);
        if (options.ModelId is not null) ArgumentException.ThrowIfNullOrWhiteSpace(options.ModelId);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.MaximumResponseBytes);
        if (httpClient.DefaultRequestHeaders.Authorization is not null)
            throw new ArgumentException("Ollama requires an isolated HttpClient without ambient authorization.", nameof(httpClient));
        _metadata = new("ollama", options.Endpoint, options.ModelId,
            [DecisionKind.Binary, DecisionKind.Choice, DecisionKind.Score]);
        _transport = new(httpClient, options.Endpoint, null, options.MaximumResponseBytes, ownsHttpClient,
            (status, requestId, body) => new OllamaApiException(status, requestId, body), () => new OllamaBodyLimitException());
    }

    public async Task<DecisionResult> GenerateAsync(DecisionInput input, DecisionGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        _transport.ThrowIfDisposed();
        cancellationToken.ThrowIfCancellationRequested();
        ArgumentNullException.ThrowIfNull(input);
        DecisionGenerationOptions? snapshot = options switch
        {
            null => null,
            OllamaGenerationOptions typed => typed.Clone(),
            _ when options.GetType() == typeof(DecisionGenerationOptions) => options.Clone(),
            _ => throw new NotSupportedException("Unsupported generation options type for Ollama.")
        };
        if (snapshot?.AdditionalProperties is { Count: > 0 })
            throw new NotSupportedException("Ollama does not accept AdditionalProperties input keys.");
        string? model = snapshot?.ModelId ?? _metadata.DefaultModelId;
        ArgumentException.ThrowIfNullOrWhiteSpace(model);
        if (input.State.ValueKind is not (JsonValueKind.String or JsonValueKind.Object or JsonValueKind.Array) ||
            (input.State.ValueKind == JsonValueKind.String && string.IsNullOrWhiteSpace(input.State.GetString())))
            throw new ArgumentException("Ollama state must be a nonblank string, object or array.", nameof(input));
        if (input.Questions.Count > 64) throw new ArgumentException("Ollama supports at most 64 questions.", nameof(input));
        foreach (var question in input.Questions)
        {
            if (string.IsNullOrWhiteSpace(question.Instructions))
                throw new ArgumentException("Ollama requires nonblank instructions for every question.", nameof(input));
            if (question is ChoiceDecisionQuestion choice && choice.Candidates.Count is < 2 or > 26)
                throw new ArgumentException("Ollama Choice supports 2 through 26 candidates.", nameof(input));
        }
        var wire = SystemOneProtocol.Encode(input, model, maximumScoreLevels: 26);
        if (snapshot is OllamaGenerationOptions { KeepAliveSeconds: { } seconds })
        {
            var body = JsonSerializer.SerializeToNode(wire.Body)!.AsObject();
            body["keep_alive"] = seconds;
            wire = wire with { Body = JsonSerializer.SerializeToElement(body) };
        }
        // The shared transport checks the exact final body before constructing/sending the HTTP request.
        var response = await _transport.PostAsync("/v1/systemone", wire.Body, cancellationToken, MaximumRequestBytes).ConfigureAwait(false);
        var result = SystemOneProtocol.Decode(wire, response.Body, DecisionPrecision.HighPrecision);
        if (!string.Equals(result.ModelId, model, StringComparison.Ordinal))
            throw new DecisionProtocolException("Ollama response model does not match the requested model label.");
        for (int i = 0; i < input.Questions.Count; i++)
        {
            if (input.Questions[i] is ChoiceDecisionQuestion choice && result.Answers[i] is ChoiceDecisionAnswer answer)
            {
                double peak = answer.Probabilities.Values.Max();
                string firstWinner = choice.Candidates.First(c => answer.Probabilities[c.Id] == peak).Id;
                if (!string.Equals(answer.SelectedCandidateId, firstWinner, StringComparison.Ordinal))
                    throw new DecisionProtocolException("Ollama choice must select the first maximum in request order.");
            }
            else if (input.Questions[i] is ScoreDecisionQuestion score)
            {
                var rawAnswer = (JsonElement)result.Answers[i].RawRepresentation!;
                var legend = rawAnswer.GetProperty("legend");
                for (int level = 0; level < score.Rubric.Count; level++)
                {
                    var description = legend.GetProperty(level.ToString(CultureInfo.InvariantCulture));
                    if (description.ValueKind != JsonValueKind.String ||
                        !string.Equals(description.GetString(), score.Rubric[level], StringComparison.Ordinal))
                        throw new DecisionProtocolException("Ollama score legend differs from the requested rubric.");
                }
            }
        }
        if (response.RequestId is not null) result.AdditionalProperties!["systemone.requestId"] = response.RequestId;
        cancellationToken.ThrowIfCancellationRequested();
        return result;
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null) return null;
        return serviceType == typeof(DecisionGeneratorMetadata) ? _metadata : serviceType.IsInstanceOfType(this) ? this : null;
    }

    public void Dispose() => _transport.Dispose();
}

public sealed class OllamaApiException : HttpRequestException
{
    public string? RequestId { get; }
    /// <summary>Owned, opt-in diagnostic text. Never logged automatically; it may contain sensitive data.</summary>
    public string DiagnosticBody { get; }
    internal OllamaApiException(HttpStatusCode status, string? requestId, string body)
        : base($"Ollama HTTP operation failed with status {(int)status}.", null, status)
        => (RequestId, DiagnosticBody) = (requestId, body);
}

public sealed class OllamaBodyLimitException() : Exception("Ollama response exceeded the configured buffering limit.");
