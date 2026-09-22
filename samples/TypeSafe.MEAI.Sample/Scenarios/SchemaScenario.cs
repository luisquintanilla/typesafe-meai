using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.AI;

namespace TypeSafe.MEAI.Sample;

internal static class SchemaScenario
{
    internal static Dictionary<string, TypeSafeFieldBinding> Bindings(double choiceThreshold = 0.8) => new()
    {
        ["/followup"] = new("Does this ticket need follow-up?", booleanPolicy: new(0.2, 0.8)),
        ["/destination"] = new("Which team should handle the ticket?", minimumConfidence: choiceThreshold, choices:
            new Dictionary<string, JsonElement> { ["Billing"] = TypeSafeJson.Entry("Payments"), ["Technical"] = TypeSafeJson.Entry("Technical problems"), ["General"] = TypeSafeJson.Entry("General support") }),
        ["/quality"] = new("Rate the draft's usefulness as a fractional rubric index.", minimumConfidence: 0.8,
            rubric: [TypeSafeJson.Entry("Unhelpful"), TypeSafeJson.Entry("Partly useful"), TypeSafeJson.Entry("Actionable")])
    };

    internal static async Task RunAsync(SampleEnvironment environment)
    {
        var schema = new TypeSafeSchemaChatClient(environment.Client, Bindings());
        var response = await schema.GetResponseAsync<SupportDecision>(SampleEnvironment.State.ToMessage());
        if (!response.Result.Followup || response.Result.Destination != Destination.Billing || response.Result.Quality != 1.05)
            throw new InvalidOperationException("MEAI typed-output proof failed.");
        Console.WriteLine(JsonSerializer.Serialize(response.Result));
        Console.WriteLine(response.GetTypeSafeEvidence().Json);
        var stricter = new TypeSafeSchemaChatClient(environment.Client, Bindings(0.95));
        var decision = await stricter.GetTypeSafeDecisionAsync<SupportDecision>(SampleEnvironment.State);
        if (decision.Accepted || environment.Handler.Sends != 1 || decision.Rejections.Single().Pointer != "/destination")
            throw new InvalidOperationException("Projection/cache separation proof failed.");
        Console.WriteLine("Changed only confidence threshold: abstained, same native distribution, one HTTP send.");
        var boolean = new TypeSafeSchemaChatClient(environment.Client,
            new Dictionary<string, TypeSafeFieldBinding> { ["/data"] = new("Does this ticket need follow-up?", booleanPolicy: new(0.2, 0.8)) });
        var root = await boolean.GetResponseAsync<bool>(SampleEnvironment.State.ToMessage());
        if (!root.Result) throw new InvalidOperationException("MEAI primitive data-wrapper proof failed.");
        Console.WriteLine("Actual MEAI primitive bool root wrapper accepted. Thresholds are illustrative, not calibrated.");
    }
}

[JsonUnmappedMemberHandling(JsonUnmappedMemberHandling.Disallow)]
internal sealed class SupportDecision
{
    [JsonPropertyName("followup")]
    public required bool Followup { get; init; }
    [JsonPropertyName("destination")]
    public required Destination Destination { get; init; }
    [JsonPropertyName("quality")]
    public required double Quality { get; init; }
}

[JsonConverter(typeof(JsonStringEnumConverter<Destination>))]
internal enum Destination { Billing, Technical, General }
