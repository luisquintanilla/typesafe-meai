#:project ..\..\..\src\DecisionInference.TypeSafe\DecisionInference.TypeSafe.csproj
#:property TargetFramework=net10.0
#:property PublishAot=false
#:property RestorePackagesWithLockFile=false

using System.Globalization;
using System.Text.Json.Serialization;
using DecisionInference;
using DecisionInference.TypeSafe;

const string usage = "Jev: set TYPESAFE_API_KEY privately, then dotnet run --file samples\\quickstarts\\Jev\\Jev.cs";
if (args is ["--help"]) { Console.WriteLine(usage); return 0; }
if (args.Length != 0) { Console.Error.WriteLine(usage); return 2; }
string? apiKey = Environment.GetEnvironmentVariable("TYPESAFE_API_KEY");
if (string.IsNullOrWhiteSpace(apiKey) || apiKey.Any(char.IsWhiteSpace))
{
    Console.Error.WriteLine($"TYPESAFE_API_KEY must contain your key with no whitespace. It is never printed.\n{usage}");
    return 2;
}

using var http = new HttpClient(new SocketsHttpHandler { AllowAutoRedirect = false, UseCookies = false })
{
    Timeout = Timeout.InfiniteTimeSpan
};
// Defaults are https://api.typesafe.ai and jev-1.13.0. The generator borrows http.
using IDecisionGenerator generator = new TypeSafeDecisionGenerator(http, apiKey);
using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(30));
string ticket = "I cannot sign in after resetting my password.";
DecisionQuestion[] questions =
[
    new ChoiceDecisionQuestion("support-team", "Which support team should handle this ticket?",
        [new("account", "Account access and password problems"),
         new("billing", "Invoices, charges and payment problems"),
         new("technical", "Software errors and broken features")])
];
DecisionResult result = await generator.GenerateAsync(ticket, questions,
    TicketJson.Default.String, cancellationToken: deadline.Token);
ChoiceDecisionAnswer answer = result.GetAnswer<ChoiceDecisionAnswer>("support-team");
Console.WriteLine($"Provider: {generator.GetRequiredService<DecisionGeneratorMetadata>().ProviderName}");
Console.WriteLine($"Model: {result.ModelId}\nSelected ID: {answer.SelectedCandidateId}");
foreach (var (id, probability) in answer.Probabilities)
    Console.WriteLine($"{id}: {probability.ToString("F4", CultureInfo.InvariantCulture)}");
Console.WriteLine("Prediction only: no ticket was routed. The synthetic ticket was sent to the hosted service.");
return 0;

[JsonSerializable(typeof(string))]
internal partial class TicketJson : JsonSerializerContext;
