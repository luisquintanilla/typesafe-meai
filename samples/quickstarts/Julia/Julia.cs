#:project ..\..\..\src\DecisionInference.Julia\DecisionInference.Julia.csproj
#:package Microsoft.ML.OnnxRuntime@1.23.2
#:property TargetFramework=net10.0
#:property PublishAot=false
#:property RestorePackagesWithLockFile=false

using System.Globalization;
using System.Text.Json.Serialization;
using DecisionInference;
using DecisionInference.Julia;

const string usage = "Julia: dotnet run --file samples\\quickstarts\\Julia\\Julia.cs -- <asset-directory>";
if (args is ["--help"]) { Console.WriteLine(usage); return 0; }
if (args.Length != 1 || string.IsNullOrWhiteSpace(args[0]) || !Directory.Exists(args[0]))
{
    Console.Error.WriteLine($"Provide an existing Julia asset directory. See adjacent README.md.\n{usage}");
    return 2;
}

using IDecisionGenerator generator = JuliaDecisionGenerator.LoadFromDirectory(args[0]);
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
Console.WriteLine("Prediction only: no ticket was routed. These probabilities are not accuracy guarantees.");
return 0;

[JsonSerializable(typeof(string))]
internal partial class TicketJson : JsonSerializerContext;
