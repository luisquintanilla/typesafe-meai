using System.Globalization;
using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.AI;

namespace OllamaDecisions;

/// <summary>An <see cref="IDecisionClient"/> for Ollama's decision endpoint (<c>POST /v1/systemone</c>).</summary>
/// <remarks>
/// Requires Ollama v0.35.0 or later and a decision model such as nimble. Image input requires a Clef model and v0.35.1 or later.
/// Set <see cref="DecisionOptions.RawRepresentationFactory"/> to return a <see cref="JsonObject"/> to start the request
/// body from it, for example to set <c>keep_alive</c>; the client then sets <c>model</c>, <c>state</c>, <c>images</c> and <c>questions</c>.
/// </remarks>
public sealed class OllamaDecisionClient : IDecisionClient
{
    /// <summary>The key into <see cref="DecisionAnswer.AdditionalProperties"/> used to report Ollama's confidence.</summary>
    public const string ConfidenceKey = "confidence";

    private static readonly Uri DefaultEndpoint = new("http://localhost:11434");
    private static readonly string[] SupportedImageTypes = ["image/png", "image/jpeg", "image/webp"];

    private readonly HttpClient _httpClient;
    private readonly bool _disposeHttpClient;
    private readonly Uri _requestUri;
    private readonly string? _defaultModelId;
    private readonly DecisionClientMetadata _metadata;

    /// <summary>Initializes a new instance of the <see cref="OllamaDecisionClient"/> class.</summary>
    /// <param name="endpoint">The Ollama server, or <see langword="null"/> for http://localhost:11434.</param>
    /// <param name="defaultModelId">The model to use when <see cref="DecisionOptions.ModelId"/> is not set, such as nimble.</param>
    /// <param name="httpClient">An HTTP client to use. The caller keeps ownership of a supplied client.</param>
    public OllamaDecisionClient(Uri? endpoint = null, string? defaultModelId = null, HttpClient? httpClient = null)
    {
        endpoint ??= DefaultEndpoint;
        if (!endpoint.IsAbsoluteUri)
        {
            throw new ArgumentException("The endpoint must be an absolute URI.", nameof(endpoint));
        }

        _httpClient = httpClient ?? new HttpClient { Timeout = Timeout.InfiniteTimeSpan };
        _disposeHttpClient = httpClient is null;
        _requestUri = new Uri(endpoint, "v1/systemone");
        _defaultModelId = defaultModelId;
        _metadata = new("ollama", endpoint, defaultModelId, [DecisionKind.Binary, DecisionKind.Choice, DecisionKind.Score]);
    }

    /// <inheritdoc />
    public async Task<DecisionResponse> GetResponseAsync(
        DecisionRequest request, DecisionOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);

        JsonObject body = ToOllamaRequest(request, options);
        using HttpRequestMessage message = new(HttpMethod.Post, _requestUri)
        {
            Content = new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json"),
        };

        string json;
        try
        {
            using HttpResponseMessage response = await _httpClient.SendAsync(message, cancellationToken).ConfigureAwait(false);
            json = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            if (!response.IsSuccessStatusCode)
            {
                throw new DecisionClientException(
                    $"Ollama returned {(int)response.StatusCode} ({response.StatusCode}): {GetErrorMessage(json)}",
                    isTransient: IsTransient(response.StatusCode));
            }
        }
        catch (HttpRequestException ex)
        {
            throw new DecisionClientException($"The Ollama request failed: {ex.Message}", isTransient: true, ex);
        }
        catch (OperationCanceledException ex) when (!cancellationToken.IsCancellationRequested)
        {
            throw new DecisionClientException("The Ollama request timed out.", isTransient: true, ex);
        }

        return FromOllamaResponse(request, json);
    }

    /// <inheritdoc />
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);

        return
            serviceKey is not null ? null :
            serviceType == typeof(DecisionClientMetadata) ? _metadata :
            serviceType == typeof(HttpClient) ? _httpClient :
            serviceType.IsInstanceOfType(this) ? this :
            null;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposeHttpClient)
        {
            _httpClient.Dispose();
        }
    }

    private static bool IsTransient(HttpStatusCode status) => (int)status is 408 or 429 or >= 500;

    private static string GetErrorMessage(string json)
    {
        try
        {
            using JsonDocument document = JsonDocument.Parse(json);
            if (document.RootElement.ValueKind == JsonValueKind.Object &&
                document.RootElement.TryGetProperty("error", out JsonElement error) &&
                error.ValueKind == JsonValueKind.String)
            {
                return error.GetString()!;
            }
        }
        catch (JsonException)
        {
        }

        return json.Length > 200 ? json[..200] : json;
    }

    private JsonObject ToOllamaRequest(DecisionRequest request, DecisionOptions? options)
    {
        JsonObject body = options?.RawRepresentationFactory?.Invoke(this) is JsonObject raw ? (JsonObject)raw.DeepClone() : [];

        body["model"] = options?.ModelId ?? _defaultModelId ??
            throw new InvalidOperationException("No model was specified. Provide a default model when creating the client or set DecisionOptions.ModelId.");

        // Ollama takes one state plus a separate list of images shared by all questions.
        StringBuilder? text = null;
        JsonArray? images = null;
        foreach (AIContent content in request.Contents)
        {
            switch (content)
            {
                case TextContent textContent:
                    (text ??= new()).Append(textContent.Text);
                    break;

                case DataContent data when Array.Exists(SupportedImageTypes, t => string.Equals(t, data.MediaType, StringComparison.OrdinalIgnoreCase)):
                    (images ??= []).Add(Convert.ToBase64String(data.Data.Span));
                    break;

                default:
                    throw new NotSupportedException(
                        $"Content of type '{content.GetType().Name}'{(content is DataContent d ? $" with media type '{d.MediaType}'" : "")} is not supported " +
                        "as decision input by Ollama. Use text, or inline PNG, JPEG or WebP image data.");
            }
        }

        if (request.State is JsonElement state)
        {
            if (text is not null)
            {
                throw new NotSupportedException("Ollama accepts a single state. Put the text in the JSON state, or send text contents without state.");
            }

            body["state"] = JsonNode.Parse(state.GetRawText());
        }
        else if (text is not null)
        {
            body["state"] = text.ToString();
        }
        else
        {
            throw new NotSupportedException("Ollama requires text or JSON state alongside images.");
        }

        if (images is not null)
        {
            body["images"] = images;
        }

        JsonObject questions = [];
        foreach (DecisionQuestion question in request.Questions)
        {
            questions[question.Id] = ToOllamaQuestion(question);
        }

        body["questions"] = questions;
        return body;
    }

    private static JsonObject ToOllamaQuestion(DecisionQuestion question)
    {
        if (question.Instructions.Count == 0)
        {
            throw new NotSupportedException($"Ollama requires instructions for every question, but question '{question.Id}' has none.");
        }

        JsonObject result = new() { ["instructions"] = ToText(question.Instructions, "question instructions") };
        switch (question)
        {
            case BinaryDecisionQuestion binary:
                result["type"] = "noul";
                if (binary.TrueDescription is not null || binary.FalseDescription is not null)
                {
                    JsonObject criteria = [];
                    if (binary.FalseDescription is not null)
                    {
                        criteria["false"] = ToText(binary.FalseDescription, "binary outcome descriptions");
                    }

                    if (binary.TrueDescription is not null)
                    {
                        criteria["true"] = ToText(binary.TrueDescription, "binary outcome descriptions");
                    }

                    result["criteria"] = criteria;
                }

                break;

            case ChoiceDecisionQuestion choice:
                result["type"] = "choice";
                JsonObject candidates = [];
                foreach (DecisionCandidate candidate in choice.Candidates)
                {
                    candidates[candidate.Id] = ToText(candidate.Description, "candidate descriptions");
                }

                result["criteria"] = candidates;
                break;

            case ScoreDecisionQuestion score:
                // Ollama scores against descriptions ordered from lowest to highest; level IDs are not sent.
                result["type"] = "score";
                JsonArray levels = [];
                foreach (DecisionScoreLevel level in score.Levels)
                {
                    levels.Add(ToText(level.Description, "score level descriptions"));
                }

                result["criteria"] = levels;
                break;

            default:
                throw new NotSupportedException($"Decision questions of type '{question.GetType().Name}' are not supported by Ollama.");
        }

        return result;
    }

    private static string ToText(IReadOnlyList<AIContent> contents, string usage)
    {
        StringBuilder builder = new();
        foreach (AIContent content in contents)
        {
            if (content is not TextContent text)
            {
                throw new NotSupportedException($"Content of type '{content.GetType().Name}' is not supported in {usage} by Ollama, which accepts only text.");
            }

            builder.Append(text.Text);
        }

        return builder.ToString();
    }

    private static DecisionResponse FromOllamaResponse(DecisionRequest request, string json)
    {
        JsonElement root;
        try
        {
            using JsonDocument document = JsonDocument.Parse(json);
            root = document.RootElement.Clone();
        }
        catch (JsonException ex)
        {
            throw new DecisionProtocolException("The Ollama decision response is not valid JSON.", ex);
        }

        if (root.ValueKind != JsonValueKind.Object ||
            !root.TryGetProperty("answers", out JsonElement answersElement) || answersElement.ValueKind != JsonValueKind.Object)
        {
            throw new DecisionProtocolException("The Ollama decision response has no answers object.");
        }

        Dictionary<string, JsonElement> nativeAnswers = new(StringComparer.Ordinal);
        foreach (JsonProperty property in answersElement.EnumerateObject())
        {
            if (!nativeAnswers.TryAdd(property.Name, property.Value))
            {
                throw new DecisionProtocolException($"The Ollama decision response answers question '{property.Name}' more than once.");
            }
        }

        List<DecisionAnswer> answers = new(request.Questions.Count);
        foreach (DecisionQuestion question in request.Questions)
        {
            if (!nativeAnswers.TryGetValue(question.Id, out JsonElement native))
            {
                throw new DecisionProtocolException($"The Ollama decision response does not answer question '{question.Id}'.");
            }

            DecisionAnswer answer;
            try
            {
                answer = (question, GetString(native, "type")) switch
                {
                    (BinaryDecisionQuestion, "noul") => new BinaryDecisionAnswer(question.Id, GetNumber(native, "noul")),
                    (ChoiceDecisionQuestion choice, "choice") => FromChoiceAnswer(choice, native),
                    (ScoreDecisionQuestion score, "score") => FromScoreAnswer(score, native),
                    _ => throw new DecisionProtocolException($"The Ollama decision answer for question '{question.Id}' does not match the question kind."),
                };
            }
            catch (Exception ex) when (ex is ArgumentException or InvalidOperationException or FormatException)
            {
                throw new DecisionProtocolException($"The Ollama decision answer for question '{question.Id}' is invalid: {ex.Message}", ex);
            }

            answer.RawRepresentation = native;
            answers.Add(answer);
        }

        if (nativeAnswers.Count != answers.Count)
        {
            throw new DecisionProtocolException("The Ollama decision response contains answers for questions that were not asked.");
        }

        return new DecisionResponse(answers)
        {
            ModelId = root.TryGetProperty("model", out JsonElement model) && model.ValueKind == JsonValueKind.String ? model.GetString() : null,
            Usage = ToUsage(root),
            RawRepresentation = root,
        };
    }

    private static ChoiceDecisionAnswer FromChoiceAnswer(ChoiceDecisionQuestion question, JsonElement native)
    {
        DecisionProbability[] probabilities = ToDistribution(question.Id, question.Candidates.Count, i => question.Candidates[i].Id, i => question.Candidates[i].Id, native);
        return new ChoiceDecisionAnswer(question.Id, GetString(native, "choice") ?? throw new FormatException("'choice' is missing."), probabilities)
        {
            AdditionalProperties = GetConfidence(native),
        };
    }

    private static ScoreDecisionAnswer FromScoreAnswer(ScoreDecisionQuestion question, JsonElement native)
    {
        // Ollama keys score probabilities by zero-based level index.
        DecisionProbability[] probabilities = ToDistribution(question.Id, question.Levels.Count, i => i.ToString(CultureInfo.InvariantCulture), i => question.Levels[i].Id, native);

        // Clamp to the level scale to absorb rounding in the server-computed expectation.
        double score = Math.Clamp(GetNumber(native, "score"), 0d, question.Levels.Count - 1);
        return new ScoreDecisionAnswer(question.Id, score, probabilities)
        {
            AdditionalProperties = GetConfidence(native),
        };
    }

    /// <summary>Builds a distribution in request order, treating options omitted by the server as having zero probability.</summary>
    private static DecisionProbability[] ToDistribution(
        string questionId, int count, Func<int, string> getNativeKey, Func<int, string> getId, JsonElement native)
    {
        if (!native.TryGetProperty("probabilities", out JsonElement reportedElement) || reportedElement.ValueKind != JsonValueKind.Object)
        {
            throw new FormatException("'probabilities' is missing.");
        }

        Dictionary<string, double> reported = new(StringComparer.Ordinal);
        foreach (JsonProperty property in reportedElement.EnumerateObject())
        {
            if (property.Value.ValueKind != JsonValueKind.Number || !reported.TryAdd(property.Name, property.Value.GetDouble()))
            {
                throw new DecisionProtocolException($"The Ollama decision answer for question '{questionId}' reports option '{property.Name}' more than once or without a number.");
            }
        }

        DecisionProbability[] probabilities = new DecisionProbability[count];
        int matched = 0;
        for (int i = 0; i < count; i++)
        {
            if (reported.TryGetValue(getNativeKey(i), out double value))
            {
                matched++;
            }

            probabilities[i] = new DecisionProbability(getId(i), value);
        }

        if (matched != reported.Count)
        {
            throw new DecisionProtocolException($"The Ollama decision answer for question '{questionId}' reports options that were not requested.");
        }

        return probabilities;
    }

    private static AdditionalPropertiesDictionary? GetConfidence(JsonElement native) =>
        native.TryGetProperty(ConfidenceKey, out JsonElement confidence) && confidence.ValueKind == JsonValueKind.Number ?
            new() { [ConfidenceKey] = confidence.GetDouble() } :
            null;

    private static string? GetString(JsonElement element, string name) =>
        element.ValueKind == JsonValueKind.Object && element.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.String ?
            value.GetString() :
            null;

    private static double GetNumber(JsonElement element, string name) =>
        element.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.Number ?
            value.GetDouble() :
            throw new FormatException($"'{name}' is missing or not a number.");

    private static UsageDetails? ToUsage(JsonElement root)
    {
        if (!root.TryGetProperty("usage", out JsonElement usage) || usage.ValueKind != JsonValueKind.Object)
        {
            return null;
        }

        long? input = GetCount(usage, "input_tokens");
        long? output = GetCount(usage, "output_tokens");
        return new UsageDetails
        {
            InputTokenCount = input,
            OutputTokenCount = output,
            TotalTokenCount = input + output,
            CachedInputTokenCount = GetCount(root, "prompt_eval_cached_count"),
        };
    }

    private static long? GetCount(JsonElement element, string name) =>
        element.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.Number && value.TryGetInt64(out long count) ? count : null;
}
