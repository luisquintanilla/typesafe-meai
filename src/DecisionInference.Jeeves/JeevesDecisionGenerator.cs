using System.Collections.ObjectModel;
using System.Net;
using System.Text.Json;
using System.Text.Json.Nodes;
using DecisionInference.SystemOne;
using Microsoft.Extensions.AI;

namespace DecisionInference.Jeeves;

/// <summary>Connection settings for an independently managed server, not model or daemon lifecycle options.</summary>
public sealed record JeevesDecisionGeneratorOptions
{
    public Uri Endpoint { get; init; } = new("http://127.0.0.1:8009");
    /// <summary>A request/echo label, not verified checkpoint identity or a weight-selection control.</summary>
    public string ModelId { get; init; } = "jeeves-latest";
    public int MaximumResponseBytes { get; init; } = 4 * 1024 * 1024;
}

/// <summary>Thin adapter for PostHog/jeeves at f04ec5567301450dcaae0210dd54deeb4f647f87.</summary>
/// <remarks>
/// Uses the existing System One subset (Choice at most 255; Score 2..10). No discovery or authentication
/// is invented for the unauthenticated pinned server. Use an isolated HttpClient with redirects and
/// cookies disabled and an explicit deadline. Cancellation stops HTTP waiting, not server GPU work.
/// </remarks>
public sealed class JeevesDecisionGenerator : IDecisionGenerator
{
    public const string MetadataKey = "jeeves.metadata";
    /// <summary>UsageDetails.AdditionalCounts key for upstream output_tokens: answer-JSON tokens, excluding reasoning.</summary>
    public const string AnswerJsonTokenCountKey = "jeeves.answerJsonTokens";
    private readonly SystemOneTransport _transport;
    private readonly DecisionGeneratorMetadata _metadata;

    public JeevesDecisionGenerator(HttpClient httpClient, JeevesDecisionGeneratorOptions? options = null,
        bool ownsHttpClient = false)
    {
        ArgumentNullException.ThrowIfNull(httpClient);
        options ??= new();
        SystemOneTransport.ValidateEndpoint(options.Endpoint, allowLoopbackHttp: true);
        ArgumentException.ThrowIfNullOrWhiteSpace(options.ModelId);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.MaximumResponseBytes);
        if (httpClient.DefaultRequestHeaders.Authorization is not null)
            throw new ArgumentException("Jeeves requires an isolated HttpClient without ambient authorization.", nameof(httpClient));
        _metadata = new("jeeves", options.Endpoint, options.ModelId,
            [DecisionKind.Binary, DecisionKind.Choice, DecisionKind.Score]);
        _transport = new(httpClient, options.Endpoint, null, options.MaximumResponseBytes, ownsHttpClient,
            (status, requestId, body) => new JeevesApiException(status, requestId, body), () => new JeevesBodyLimitException());
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
            JeevesGenerationOptions typed => typed.Clone(),
            _ when options.GetType() == typeof(DecisionGenerationOptions) => options.Clone(),
            _ => throw new NotSupportedException("Unsupported generation options type for Jeeves.")
        };
        if (snapshot?.AdditionalProperties is { Count: > 0 })
            throw new NotSupportedException("Jeeves does not accept AdditionalProperties input keys.");
        string model = snapshot?.ModelId ?? _metadata.DefaultModelId!;
        ArgumentException.ThrowIfNullOrWhiteSpace(model);
        var reasoningOptions = snapshot as JeevesGenerationOptions;
        var wire = SystemOneProtocol.Encode(input, model);
        if (reasoningOptions is { } settings)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(settings.MaxThink);
            if (settings.NoThinkThreshold is { } threshold && (!double.IsFinite(threshold) || threshold is < 0 or > 1))
                throw new ArgumentOutOfRangeException(nameof(options), "The no-think threshold must be finite and in [0,1].");
            var body = JsonSerializer.SerializeToNode(wire.Body)!.AsObject();
            body["options"] = new JsonObject
            {
                ["think"] = settings.Think,
                ["max_think"] = settings.MaxThink,
                ["nothink_threshold"] = settings.NoThinkThreshold,
                ["return_reasoning"] = settings.ReturnReasoning
            };
            wire = wire with { Body = JsonSerializer.SerializeToElement(body) };
        }
        var response = await _transport.PostAsync("/v1/systemone", wire.Body, cancellationToken).ConfigureAwait(false);
        var result = Decode(wire, response.Body, model, reasoningOptions);
        if (response.RequestId is not null) result.AdditionalProperties!["systemone.requestId"] = response.RequestId;
        cancellationToken.ThrowIfCancellationRequested();
        return result;
    }

    private static DecisionResult Decode(SystemOneRequest wire, JsonElement root, string model, JeevesGenerationOptions? options)
    {
        try
        {
            if (!string.Equals(root.GetProperty("model").GetString(), model, StringComparison.Ordinal))
                throw new DecisionProtocolException("Jeeves response model does not match the request label.");
            var rawUsage = root.GetProperty("usage");
            long input = Count(rawUsage, "input_tokens"), answerJson = Count(rawUsage, "output_tokens"),
                reasoning = Count(rawUsage, "reasoning_tokens");
            long output = checked(answerJson + reasoning), total = checked(input + output);
            if (options is not null)
            {
                long budget = options.Think ? checked((long)options.MaxThink * wire.QuestionIds.Count) : 0;
                if (reasoning > budget) throw new DecisionProtocolException("Reported reasoning exceeds the requested budget.");
            }
            var usage = new UsageDetails
            {
                InputTokenCount = input, OutputTokenCount = output, ReasoningTokenCount = reasoning, TotalTokenCount = total,
                AdditionalCounts = new() { [AnswerJsonTokenCountKey] = answerJson }
            };
            double latency = root.GetProperty("latency_ms").GetDouble();
            if (!double.IsFinite(latency) || latency < 0)
                throw new DecisionProtocolException("Invalid Jeeves server latency.");
            var metadata = new JeevesResultMetadata(latency, ReadReasoning(wire, root, options, reasoning));
            var result = SystemOneProtocol.Decode(wire, root, DecisionPrecision.TwoDecimalPlaces, usage);
            result.AdditionalProperties![MetadataKey] = metadata;
            return result;
        }
        catch (Exception e) when (e is JsonException or InvalidOperationException or KeyNotFoundException or
            FormatException or OverflowException or ArgumentException)
        {
            throw new DecisionProtocolException("Malformed Jeeves response envelope.");
        }
    }

    private static long Count(JsonElement value, string property)
    {
        long count = value.GetProperty(property).GetInt64();
        if (count < 0) throw new DecisionProtocolException("Invalid Jeeves token count.");
        return count;
    }

    private static IReadOnlyDictionary<string, JeevesReasoning>? ReadReasoning(
        SystemOneRequest wire, JsonElement root, JeevesGenerationOptions? options, long expectedTokens)
    {
        bool present = root.TryGetProperty("reasoning", out var raw);
        if (present != (options?.ReturnReasoning == true))
            throw new DecisionProtocolException("Reasoning must be returned exactly when explicitly requested.");
        if (!present) return null;
        Dictionary<string, JeevesReasoning> snapshots = new(StringComparer.Ordinal);
        long tokens = 0;
        foreach (var item in raw.EnumerateObject())
        {
            if (!wire.QuestionIds.TryGetValue(item.Name, out var questionId))
                throw new DecisionProtocolException("Unknown reasoning question ID.");
            var entry = item.Value;
            if (!entry.EnumerateObject().Select(p => p.Name).ToHashSet(StringComparer.Ordinal)
                .SetEquals(["thought", "closed", "tokens", "text"]))
                throw new DecisionProtocolException("Invalid reasoning fields.");
            bool thought = entry.GetProperty("thought").GetBoolean(), closed = entry.GetProperty("closed").GetBoolean();
            long count = Count(entry, "tokens");
            string text = entry.GetProperty("text").GetString() ?? throw new DecisionProtocolException("Missing reasoning text.");
            if (thought != (count > 0) || (!thought && (closed || text.Length != 0)) ||
                (closed && count == 1 && text.Length != 0) || count > options!.MaxThink)
                throw new DecisionProtocolException("Inconsistent reasoning state or budget.");
            tokens = checked(tokens + count);
            if (!snapshots.TryAdd(questionId, new(thought, closed, count, text)))
                throw new DecisionProtocolException("Duplicate reasoning question ID.");
        }
        if (snapshots.Count != wire.QuestionIds.Count || tokens != expectedTokens)
            throw new DecisionProtocolException("Incomplete reasoning or inconsistent aggregate token count.");
        return new ReadOnlyDictionary<string, JeevesReasoning>(snapshots);
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null) return null;
        return serviceType == typeof(DecisionGeneratorMetadata) ? _metadata : serviceType.IsInstanceOfType(this) ? this : null;
    }

    public void Dispose() => _transport.Dispose();
}

public sealed class JeevesApiException : HttpRequestException
{
    public string? RequestId { get; }
    /// <summary>Owned, opt-in diagnostic body. Never logged automatically; may contain sensitive data.</summary>
    public string DiagnosticBody { get; }

    internal JeevesApiException(HttpStatusCode status, string? requestId, string body)
        : base($"Jeeves HTTP operation failed with status {(int)status}.", null, status)
        => (RequestId, DiagnosticBody) = (requestId, body);
}

public sealed class JeevesBodyLimitException() : Exception("Jeeves response exceeded the configured buffering limit.");
