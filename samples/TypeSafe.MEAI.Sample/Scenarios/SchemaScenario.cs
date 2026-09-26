using System.ComponentModel;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.AI;

namespace TypeSafe.MEAI.Sample;

internal static class SchemaScenario
{
    internal static Dictionary<string, TypeSafeFieldOptions> Fields(bool strict = false)
    {
        Dictionary<string, TypeSafeFieldOptions> fields = new()
        {
            ["/quality"] = new(rubric: [TypeSafeJson.Entry("Unhelpful"), TypeSafeJson.Entry("Partly useful"), TypeSafeJson.Entry("Actionable")])
        };
        if (strict) fields["/destination"] = new(minimumConfidence: 0.95);
        return fields;
    }

    internal static async Task RunAsync(SampleEnvironment environment)
    {
        var response = await environment.Client.GetResponseAsync<SupportDecision>(SampleEnvironment.State);
        if (!response.Result.Followup || response.Result.Destination != Destination.Billing || response.Result.Quality != 1.05)
            throw new InvalidOperationException("MEAI typed-output proof failed.");
        Console.WriteLine(JsonSerializer.Serialize(response.Result));
        Console.WriteLine(response.Messages.Last().Contents.OfType<JevResultContent>().Single().Json);
        try
        {
            await environment.StrictClient.GetResponseAsync<SupportDecision>(SampleEnvironment.State);
            throw new InvalidOperationException("The stricter confidence policy should abstain.");
        }
        catch (TypeSafeAbstentionException ex) when (ex.Rejections.Single().Pointer == "/destination")
        {
            if (environment.Handler.Sends != 1 || ex.Evidence.Count != 1)
                throw new InvalidOperationException("Projection/cache separation proof failed.");
        }
        Console.WriteLine("Changed only confidence threshold: abstained, same native distribution, one HTTP send.");
        var root = await environment.RootClient.GetResponseAsync<bool>("Does this synthetic payment ticket need follow-up?");
        if (!root.Result) throw new InvalidOperationException("MEAI primitive data-wrapper proof failed.");
        Console.WriteLine("Actual MEAI primitive bool root wrapper accepted. Thresholds are illustrative, not calibrated.");
    }
}

internal sealed class SupportDecision
{
    [JsonPropertyName("followup"), Description("Does this ticket need follow-up?")]
    public bool Followup { get; init; }
    [JsonPropertyName("destination"), Description("Which team should handle the ticket?")]
    public Destination Destination { get; init; }
    [JsonPropertyName("quality"), Description("Rate the draft's usefulness as a fractional rubric index.")]
    public double Quality { get; init; }
}

[JsonConverter(typeof(JsonStringEnumConverter<Destination>))]
internal enum Destination { Billing, Technical, General }
