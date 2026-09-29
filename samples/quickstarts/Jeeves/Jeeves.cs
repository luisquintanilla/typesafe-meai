#:project ..\..\..\src\DecisionInference.Jeeves\DecisionInference.Jeeves.csproj
#:property TargetFramework=net10.0
#:property PublishAot=false
#:property RestorePackagesWithLockFile=false

using System.Globalization;
using System.Net;
using System.Text.Json.Serialization;
using DecisionInference;
using DecisionInference.Jeeves;

const string usage = "Jeeves: dotnet run --file samples\\quickstarts\\Jeeves\\Jeeves.cs -- <endpoint>";
if (args is ["--help"]) { Console.WriteLine(usage); return 0; }
if (args.Length != 1 || !Uri.TryCreate(args[0], UriKind.Absolute, out var endpoint) ||
    !(endpoint.Scheme == "https" || (endpoint.Scheme == "http" &&
        IPAddress.TryParse(endpoint.IdnHost.Trim('[', ']'), out var address) && IPAddress.IsLoopback(address))) ||
    !string.IsNullOrEmpty(endpoint.UserInfo) || !string.IsNullOrEmpty(endpoint.Query) ||
    !string.IsNullOrEmpty(endpoint.Fragment) || endpoint.AbsolutePath != "/")
{
    Console.Error.WriteLine($"Supply an HTTPS origin or literal-loopback HTTP, e.g. http://127.0.0.1:8009 (not localhost).\n{usage}");
    return 2;
}

using var http = new HttpClient(new SocketsHttpHandler { AllowAutoRedirect = false, UseCookies = false })
{
    Timeout = Timeout.InfiniteTimeSpan
};
using IDecisionGenerator generator = new JeevesDecisionGenerator(http, new() { Endpoint = endpoint });
using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(30));
string ticket = "I cannot sign in after resetting my password.";
DecisionQuestion[] questions =
[
    new ChoiceDecisionQuestion("support-team", "Which support team should handle this ticket?",
        [new("account", "Account access and password problems"),
         new("billing", "Invoices, charges and payment problems"),
         new("technical", "Software errors and broken features")])
];
var options = new JeevesGenerationOptions { Think = false };
DecisionResult result = await generator.GenerateAsync(ticket, questions,
    TicketJson.Default.String, options, deadline.Token);
ChoiceDecisionAnswer answer = result.GetAnswer<ChoiceDecisionAnswer>("support-team");
Console.WriteLine($"Provider: {generator.GetRequiredService<DecisionGeneratorMetadata>().ProviderName}");
Console.WriteLine($"Model label (not verified weights): {result.ModelId}\nSelected ID: {answer.SelectedCandidateId}");
foreach (var (id, probability) in answer.Probabilities)
    Console.WriteLine($"{id}: {probability.ToString("F2", CultureInfo.InvariantCulture)}");
Console.WriteLine("No thinking requested; two-decimal observations are not calibrated correctness.");
Console.WriteLine("Prediction only: no ticket was routed. The synthetic ticket was sent to the specified server.");
return 0;

[JsonSerializable(typeof(string))]
internal partial class TicketJson : JsonSerializerContext;
