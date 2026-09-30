#:project ..\..\..\src\DecisionInference.Ollama\DecisionInference.Ollama.csproj
#:property TargetFramework=net10.0
#:property PublishAot=false
#:property RestorePackagesWithLockFile=false

using System.Globalization;
using System.Net;
using System.Text.Json.Serialization;
using DecisionInference;
using DecisionInference.Ollama;

const string usage = "Ollama: dotnet run --file samples\\quickstarts\\Ollama\\Ollama.cs -- [--check-config] <endpoint> <installed-model>";
if (args is ["--help"]) { Console.WriteLine(usage); return 0; }
bool checkConfig = args.Length > 0 && args[0] == "--check-config";
string[] config = checkConfig ? args[1..] : args;
if (config.Length != 2 || string.IsNullOrWhiteSpace(config[1]) ||
    !Uri.TryCreate(config[0], UriKind.Absolute, out var endpoint) ||
    !(endpoint.Scheme == "https" || (endpoint.Scheme == "http" &&
        IPAddress.TryParse(endpoint.IdnHost.Trim('[', ']'), out var address) && IPAddress.IsLoopback(address))) ||
    !string.IsNullOrEmpty(endpoint.UserInfo) || !string.IsNullOrEmpty(endpoint.Query) ||
    !string.IsNullOrEmpty(endpoint.Fragment) || endpoint.AbsolutePath != "/")
{
    Console.Error.WriteLine($"Supply an HTTPS origin or literal-loopback HTTP (e.g. http://127.0.0.1:11434, not localhost) and an explicit installed model.\n{usage}");
    return 2;
}
if (checkConfig)
{
    Console.WriteLine("Configuration shape accepted. No HTTP call; server version, model presence and scoring compatibility were NOT checked.");
    return 0;
}

using var http = new HttpClient(new SocketsHttpHandler { AllowAutoRedirect = false, UseCookies = false })
{
    Timeout = Timeout.InfiniteTimeSpan
};
using IDecisionGenerator generator = new OllamaDecisionGenerator(http, new() { Endpoint = endpoint, ModelId = config[1] });
using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(30));
string ticket = "I cannot sign in after resetting my password.";
DecisionQuestion[] questions =
[
    new BinaryDecisionQuestion("account-access", "Does this ticket concern account access?",
        trueDescription: "The customer needs help signing in.",
        falseDescription: "The customer does not need help signing in."),
    new ChoiceDecisionQuestion("support-team", "Which support team should handle this ticket?",
        [new("account", "Account access and password problems"),
         new("billing", "Invoices, charges and payment problems"),
         new("technical", "Software errors and broken features")]),
    new ScoreDecisionQuestion("urgency", "Assess the urgency using only the stated facts.",
        ["Routine: no blocked task is described.",
         "Blocked: the customer cannot complete a task.",
         "Critical: a widespread outage or immediate harm is described."])
];
var options = new OllamaGenerationOptions { KeepAliveSeconds = 60 };
DecisionResult result = await generator.GenerateAsync(ticket, questions,
    TicketJson.Default.String, options, deadline.Token);
var account = result.GetAnswer<BinaryDecisionAnswer>("account-access");
var team = result.GetAnswer<ChoiceDecisionAnswer>("support-team");
var urgency = result.GetAnswer<ScoreDecisionAnswer>("urgency");
Console.WriteLine($"Provider: {generator.GetRequiredService<DecisionGeneratorMetadata>().ProviderName}");
Console.WriteLine($"Selected local model label (not a verified digest): {result.ModelId}");
Console.WriteLine($"Account access, P(true): {account.TrueProbability.ToString("G17", CultureInfo.InvariantCulture)}");
Console.WriteLine($"Selected team ID: {team.SelectedCandidateId}");
foreach (var (id, probability) in team.Probabilities)
    Console.WriteLine($"{id}: {probability.ToString("G17", CultureInfo.InvariantCulture)}");
for (int i = 0; i < urgency.Probabilities.Count; i++)
    Console.WriteLine($"Urgency level {i}: {urgency.Probabilities[i].ToString("G17", CultureInfo.InvariantCulture)}");
Console.WriteLine($"Expected urgency index: {urgency.ExpectedIndex.ToString("G17", CultureInfo.InvariantCulture)} (not a percentage)");
Console.WriteLine($"Input tokens: {result.Usage?.InputTokenCount}; internal scoring output tokens: {result.Usage?.OutputTokenCount}; total: {result.Usage?.TotalTokenCount}");
Console.WriteLine("Prediction only: no ticket was routed. Probabilities are not calibrated correctness.");
Console.WriteLine("The synthetic ticket and full question schema were sent to the specified server. Residency is not the HTTP deadline.");
return 0;

[JsonSerializable(typeof(string))]
internal partial class TicketJson : JsonSerializerContext;
