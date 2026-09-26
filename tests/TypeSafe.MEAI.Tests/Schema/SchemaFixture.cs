using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using Microsoft.Extensions.AI;
using TypeSafe.MEAI.Tests.Fixtures;

namespace TypeSafe.MEAI.Tests.Schema;

internal static class SchemaFixture
{
    internal static readonly JsonSerializerOptions Serializer = new(AIJsonUtilities.DefaultOptions);
    internal static JsonElement Schema => ChatResponseFormat.ForJsonSchema<DecisionDto>(Serializer).Schema!.Value;
    internal static TypeSafeState State => TypeSafeState.Parse(NativeFixture.State);
    internal static Dictionary<string, TypeSafeFieldOptions> Bindings(double confidence = 0.8,
        string scoreInstructions = "Assess the synthetic draft.", IEnumerable<JsonElement>? rubric = null) => new()
    {
        ["/followup"] = Boolean(),
        ["/details/destination"] = Choice(confidence),
        ["/details/quality"] = Score(confidence, scoreInstructions, rubric)
    };
    internal static TypeSafeFieldOptions Boolean(string? instructions = "Does the synthetic ticket need follow-up?") =>
        new(instructions, booleanPolicy: new(0.2, 0.8));
    internal static TypeSafeFieldOptions Choice(double confidence = 0.8, IEnumerable<string>? labels = null) =>
        new("Select the appropriate synthetic support team.", minimumConfidence: confidence,
            choices: (labels ?? ["Billing", "Technical"]).Select(label => KeyValuePair.Create(label, TypeSafeJson.Entry("Team " + label))));
    internal static TypeSafeFieldOptions Score(double confidence = 0.8, string instructions = "Assess the synthetic draft.",
        IEnumerable<JsonElement>? rubric = null) =>
        new(instructions, minimumConfidence: confidence,
            rubric: rubric ?? [TypeSafeJson.Entry("poor"), NativeFixture.Json("""{"level":"partial","examples":["a",null]}"""), NativeFixture.Json("null")]);
    internal static JsonElement ObjectSchema(string leaf) =>
        NativeFixture.Json("""{"type":"object","additionalProperties":false,"properties":{"value":LEAF},"required":["value"]}""".Replace("LEAF", leaf, StringComparison.Ordinal));
    internal static ChatOptions Options(JsonElement schema, string? shared = null) =>
        new() { ResponseFormat = ChatResponseFormat.ForJsonSchema(schema), Instructions = shared };

    internal static RecordingHandler Handler(double noul = 0.9, double confidence = 0.9, double probability = 0.2, double score = 1.25) =>
        new(async (request, _, token) => NativeFixture.Response(Envelope(
            NativeFixture.Json(await request.Content!.ReadAsStringAsync(token)), noul, confidence, probability, score)));

    internal static string Envelope(JsonElement request, double noul = 0.9, double confidence = 0.9, double probability = 0.2, double score = 1.25)
    {
        JsonObject answers = new();
        foreach (var question in request.GetProperty("questions").EnumerateObject())
        {
            string type = question.Value.GetProperty("type").GetString()!;
            JsonObject answer = new() { ["type"] = type, ["vendor_answer"] = "synthetic-preserved" };
            if (type == "noul") answer["noul"] = noul;
            else
            {
                answer["confidence"] = confidence;
                JsonObject probabilities = new();
                if (type == "choice")
                {
                    var labels = question.Value.GetProperty("criteria").EnumerateObject().Select(p => p.Name).ToArray();
                    answer["choice"] = labels[0];
                    for (int i = 0; i < labels.Length; i++)
                        probabilities[labels[i]] = labels.Length == 1 ? 1 : i == 0 ? probability : i == labels.Length - 1 ? 1 - probability : 0;
                }
                else
                {
                    answer["score"] = score;
                    var levels = question.Value.GetProperty("criteria").EnumerateArray().ToArray();
                    JsonObject legend = new();
                    for (int i = 0; i < levels.Length; i++)
                    {
                        string key = i.ToString(System.Globalization.CultureInfo.InvariantCulture);
                        probabilities[key] = i == 0 ? probability : i == levels.Length - 1 ? 1 - probability : 0;
                        legend[key] = JsonNode.Parse(levels[i].GetRawText());
                    }
                    answer["legend"] = legend;
                }
                answer["probabilities"] = probabilities;
            }
            answers[question.Name] = answer;
        }
        return new JsonObject
        {
            ["model"] = request.GetProperty("model").GetString(),
            ["usage"] = new JsonObject { ["input_tokens"] = 11, ["output_tokens"] = 7, ["vendor_usage"] = "retained" },
            ["answers"] = answers,
            ["vendor_envelope"] = new JsonObject { ["revision"] = 17, ["structured"] = new JsonArray(null, "retained") }
        }.ToJsonString();
    }
}

internal sealed class SchemaSpy(IChatClient inner) : DelegatingChatClient(inner)
{
    internal int Calls { get; private set; }
    internal int Lookups { get; private set; }
    internal List<ChatOptions?> Options { get; } = [];
    internal List<ChatResponse> Responses { get; } = [];
    public override async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        Calls++;
        Options.Add(options);
        var response = await InnerClient.GetResponseAsync(messages, options, cancellationToken);
        Responses.Add(response);
        return response;
    }
    public override object? GetService(Type serviceType, object? serviceKey = null)
    {
        Lookups++;
        return base.GetService(serviceType, serviceKey);
    }
}

[JsonUnmappedMemberHandling(JsonUnmappedMemberHandling.Disallow)]
public sealed class DecisionDto
{
    [JsonPropertyName("followup")]
    public required bool Followup { get; init; }
    [JsonPropertyName("details")]
    public required DecisionDetails Details { get; init; }
}

[JsonUnmappedMemberHandling(JsonUnmappedMemberHandling.Disallow)]
public sealed class DecisionDetails
{
    [JsonPropertyName("destination")]
    public required TestDestination Destination { get; init; }
    [JsonPropertyName("quality")]
    public required double Quality { get; init; }
}

[JsonConverter(typeof(JsonStringEnumConverter<TestDestination>))]
public enum TestDestination { Billing, Technical }
