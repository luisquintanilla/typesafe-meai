#:package Microsoft.Extensions.AI@11.0.0-decisions.1
#:package Microsoft.Extensions.AI.OpenAI@11.0.0-decisions.1
#:package Azure.Identity@1.17.1
#:property TargetFramework=net10.0
#:property PublishAot=false
#:property RestorePackagesWithLockFile=false
#:property NoWarn=$(NoWarn);MEAI001;OPENAI001

using System.ClientModel.Primitives;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Text.Json.Serialization;
using Azure.Identity;
using Microsoft.Extensions.AI;
using OpenAI;

const string usage = """
    Azure OpenAI decisions: dotnet run --file samples\official\AzureOpenAI\AzureOpenAI.cs
    Environment:
      AZURE_OPENAI_ENDPOINT             Resource endpoint, e.g. https://<resource>.cognitiveservices.azure.com
      AZURE_OPENAI_DECISION_DEPLOYMENT  Decision model deployment, e.g. gpt-6-luna
    Authentication uses DefaultAzureCredential (for example, az login).
    """;
if (args is ["--help"]) { Console.WriteLine(usage); return 0; }

string? endpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT");
string? deployment = Environment.GetEnvironmentVariable("AZURE_OPENAI_DECISION_DEPLOYMENT");
if (!Uri.TryCreate(endpoint, UriKind.Absolute, out Uri? endpointUri) || endpointUri.Scheme != Uri.UriSchemeHttps ||
    string.IsNullOrWhiteSpace(deployment))
{
    Console.Error.WriteLine(usage);
    return 2;
}

OpenAIClient openAI = new(
    new BearerTokenPolicy(new DefaultAzureCredential(), "https://ai.azure.com/.default"),
    new OpenAIClientOptions { Endpoint = new Uri(endpointUri, "/openai/v1") });
using IDecisionClient client = openAI.GetDecisionsClient().AsIDecisionClient(deployment);

DecisionDefinition<TicketDecision> definition = DecisionDefinition<TicketDecision>.Create(
    builder =>
    {
        builder.Choice(result => result.Category);
        builder.BinaryProbability(result => result.RefundProbability, "Is the customer asking for a refund?");
        builder.ExpectedScore(result => result.Urgency,
        [
            new("low", "Can wait for normal processing."),
            new("medium", "Should be handled today."),
            new("high", "Needs immediate attention."),
        ]);
    },
    TicketJson.Default.Options);

using var deadline = new CancellationTokenSource(TimeSpan.FromMinutes(2));

// 1. Typed JSON state.
var ticket = new SupportTicket("T-1001", "I cannot sign in after resetting my password, and my team is blocked.");
Print("Typed state", await client.GetDecisionAsync(ticket, TicketJson.Default.SupportTicket, definition,
    cancellationToken: deadline.Token));

// 2. Ordered contents: text plus an image.
byte[] screenshot = await File.ReadAllBytesAsync(Path.Combine(SourceDirectory(), "declined-payment.png"), deadline.Token);
Print("Text + image", await client.GetDecisionAsync(
    [new TextContent("The customer attached this screenshot and wrote: 'Please look at this.'"), new DataContent(screenshot, "image/png")],
    definition,
    cancellationToken: deadline.Token));

return 0;

static string SourceDirectory([CallerFilePath] string path = "") => Path.GetDirectoryName(path)!;

static void Print(string label, DecisionResponse<TicketDecision> response)
{
    TicketDecision decision = response.Result;
    Console.WriteLine($"== {label} (model {response.Response.ModelId})");
    Console.WriteLine($"Category: {decision.Category}");
    foreach (var (category, probability) in response.GetDistribution(result => result.Category))
    {
        Console.WriteLine($"  {category,-10} {probability:P1}");
    }

    Console.WriteLine($"Refund probability: {decision.RefundProbability:P1}");
    Console.WriteLine($"Expected urgency (0=low, 2=high): {decision.Urgency:F2}");
    Console.WriteLine($"Tokens: {response.Response.Usage?.InputTokenCount} in, {response.Response.Usage?.OutputTokenCount} out");
}

public sealed record SupportTicket(string Id, string Text);

public sealed record TicketDecision(
    [property: Description("Classify the customer's main concern.")] TicketCategory Category,
    double RefundProbability,
    double Urgency);

[JsonConverter(typeof(JsonStringEnumConverter<TicketCategory>))]
public enum TicketCategory
{
    [Description("Payment, invoice, or charge concerns.")] Billing,
    [Description("Product defects, errors, or outages.")] Technical,
    [Description("Sign-in, profile, or account access.")] Account,
}

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
[JsonSerializable(typeof(SupportTicket))]
[JsonSerializable(typeof(TicketDecision))]
internal sealed partial class TicketJson : JsonSerializerContext;
