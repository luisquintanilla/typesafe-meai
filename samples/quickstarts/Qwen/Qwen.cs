#:project ..\..\..\src\DecisionInference.TensorSharp\DecisionInference.TensorSharp.csproj
#:property TargetFramework=net10.0
#:property PublishAot=false
#:property RestorePackagesWithLockFile=false

using System.Globalization;
using System.Text.Json.Serialization;
using DecisionInference;
using DecisionInference.TensorSharp;

const string usage = "Qwen: dotnet run --file samples\\quickstarts\\Qwen\\Qwen.cs -- <pinned-model.gguf>";
if (args is ["--help"]) { Console.WriteLine(usage); return 0; }
if (args.Length != 1 || string.IsNullOrWhiteSpace(args[0]) || !File.Exists(args[0]))
{
    Console.Error.WriteLine($"Provide the hash-pinned Qwen GGUF file. See adjacent README.md.\n{usage}");
    return 2;
}

// TensorSharp reads this process-local setting during model initialization.
Environment.SetEnvironmentVariable("MAX_CONTEXT", "512");
using IDecisionGenerator generator = QwenDecisionGenerator.Load(args[0]);
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
Console.WriteLine("Prediction only: no ticket was routed. Scores are conditional on the allowed answer tokens.");
return 0;

[JsonSerializable(typeof(string))]
internal partial class TicketJson : JsonSerializerContext;
