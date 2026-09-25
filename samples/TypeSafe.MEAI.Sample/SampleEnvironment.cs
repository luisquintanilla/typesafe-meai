using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace TypeSafe.MEAI.Sample;

internal sealed class SampleEnvironment : IDisposable
{
    internal const string State = "Synthetic ticket: payouts failed. Please preview a billing queue assignment.";
    internal static readonly TypeSafeQuestion[] Questions =
    [
        new NoulQuestion("followup", TypeSafeJson.Entry("Does this ticket need follow-up?")),
        new ChoiceQuestion("destination", TypeSafeJson.Entry("Which team should handle this ticket?"),
            new Dictionary<string, JsonElement> { ["billing"] = TypeSafeJson.Entry("Payments"), ["technical"] = TypeSafeJson.Entry("Technical problems"), ["general"] = TypeSafeJson.Entry("Other support") }),
        new ScoreQuestion("quality", TypeSafeJson.Entry("Rate the fixed draft's usefulness."),
            new[] { TypeSafeJson.Entry("Unhelpful"), TypeSafeJson.Entry("Partly useful"), TypeSafeJson.Entry("Useful and actionable") })
    ];
    internal SyntheticHandler Handler { get; } = new();
    internal ServiceProvider Services { get; }
    internal IChatClient Client => Services.GetRequiredKeyedService<IChatClient>("jev");
    internal IChatClient StrictClient => Services.GetRequiredKeyedService<IChatClient>("jev-strict");
    internal IChatClient RootClient => Services.GetRequiredKeyedService<IChatClient>("jev-root");
    internal static ChatOptions NativeOptions() => new()
    {
        AdditionalProperties = new()
        {
            [TypeSafeChatClient.QuestionsKey] = JsonSerializer.SerializeToElement(Questions.ToDictionary(q => q.Id, q => q.ToJson()))
        }
    };

    internal SampleEnvironment(bool useCache = false)
    {
        ServiceCollection services = new();
        services.AddDistributedMemoryCache();
        services.AddLogging(b => b.AddConsole().SetMinimumLevel(LogLevel.Information));
        foreach (string key in new[] { "jev", "jev-strict", "jev-root" })
            services.AddTypeSafeChatClient(key, "SYNTHETIC-NOT-A-REAL-KEY",
                new()
                {
                    ModelId = "jev-1.13.0", CacheNamespace = "synthetic-local-proof",
                    Policy = new(new(0.2, 0.8), 0.8),
                    Fields = key == "jev-root" ? new Dictionary<string, TypeSafeFieldOptions>() : SchemaScenario.Fields(key == "jev-strict")
                }, useCache)
                .ConfigurePrimaryHttpMessageHandler(() => Handler);
        Services = services.BuildServiceProvider();
    }
    public void Dispose() => Services.Dispose();
}

internal sealed class SyntheticHandler : HttpMessageHandler
{
    private int _sends;
    internal int Sends => Volatile.Read(ref _sends);
    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        Interlocked.Increment(ref _sends);
        if (request.Method == HttpMethod.Get)
            return Json(new JsonObject { ["models"] = new JsonArray("jev-latest") });
        using var body = JsonDocument.Parse(await request.Content!.ReadAsStringAsync(cancellationToken));
        JsonObject answers = new();
        foreach (var q in body.RootElement.GetProperty("questions").EnumerateObject())
        {
            string type = q.Value.GetProperty("type").GetString()!;
            JsonObject answer = new() { ["type"] = type, ["synthetic_extra"] = "preserved" };
            if (type == "noul") answer["noul"] = 0.9;
            else if (type == "choice")
            {
                var keys = q.Value.GetProperty("criteria").EnumerateObject().Select(p => p.Name).ToArray();
                JsonObject probabilities = new();
                for (int i = 0; i < keys.Length; i++) probabilities[keys[i]] = keys.Length == 1 ? 1 : i == 0 ? 0.88 : i == 1 ? 0.12 : 0;
                answer["choice"] = keys[0];
                answer["probabilities"] = probabilities;
                answer["confidence"] = 0.81;
            }
            else
            {
                var levels = q.Value.GetProperty("criteria").EnumerateArray().ToArray();
                JsonObject legend = new(), probabilities = new();
                for (int i = 0; i < levels.Length; i++)
                {
                    string key = i.ToString(System.Globalization.CultureInfo.InvariantCulture);
                    legend[key] = JsonNode.Parse(levels[i].GetRawText());
                    probabilities[key] = levels.Length == 2 ? (i == 0 ? 0.05 : 0.95) : i == 1 ? 0.95 : i == 2 ? 0.05 : 0;
                }
                answer["score"] = levels.Length == 2 ? 0.95 : 1.05;
                answer["confidence"] = 0.92;
                answer["legend"] = legend;
                answer["probabilities"] = probabilities;
            }
            answers[q.Name] = answer;
        }
        return Json(new JsonObject
        {
            ["model"] = "jev-1.13.0", ["answers"] = answers,
            ["usage"] = new JsonObject { ["input_tokens"] = 304, ["output_tokens"] = 38 },
            ["synthetic"] = true
        });
    }
    private static HttpResponseMessage Json(JsonNode node) => new(HttpStatusCode.OK)
    {
        Content = new StringContent(node.ToJsonString(), Encoding.UTF8, "application/json")
    };
}
