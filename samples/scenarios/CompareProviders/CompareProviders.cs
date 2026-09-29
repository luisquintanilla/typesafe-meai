#:project ..\..\..\src\DecisionInference.Julia\DecisionInference.Julia.csproj
#:project ..\..\..\src\DecisionInference.Laya\DecisionInference.Laya.csproj
#:package Microsoft.ML.OnnxRuntime@1.23.2
#:property TargetFramework=net10.0
#:property PublishAot=false
#:property RestorePackagesWithLockFile=false

using System.Globalization;
using System.Text.Json.Serialization;
using DecisionInference;
using DecisionInference.Julia;
using DecisionInference.Laya;

const string usage = "Compare: dotnet run --file samples\\scenarios\\CompareProviders\\CompareProviders.cs -- <julia-directory> <laya-directory>";
if (args is ["--help"]) { Console.WriteLine(usage); return 0; }
if (args.Length != 2 || args.Any(path => string.IsNullOrWhiteSpace(path) || !Directory.Exists(path)))
{
    Console.Error.WriteLine($"Provide both existing local asset directories. No other providers or credentials are needed.\n{usage}");
    return 2;
}

string ticket = "I cannot sign in after resetting my password.";
DecisionQuestion[] questions =
[
    new ChoiceDecisionQuestion("support-team", "Which support team should handle this ticket?",
        [new("account", "Account access and password problems"),
         new("billing", "Invoices, charges and payment problems"),
         new("technical", "Software errors and broken features")])
];
using var deadline = new CancellationTokenSource(TimeSpan.FromMinutes(2));

// The scopes are intentionally sequential: release Julia before loading Laya.
using (IDecisionGenerator generator = JuliaDecisionGenerator.LoadFromDirectory(args[0]))
    await ShowDecisionAsync(generator, ticket, questions, deadline.Token);
using (IDecisionGenerator generator = LayaDecisionGenerator.LoadFromDirectory(args[1]))
    await ShowDecisionAsync(generator, ticket, questions, deadline.Token);
Console.WriteLine("Same C# contract, different models. This is not an accuracy, calibration or speed ranking.");
return 0;

static async Task ShowDecisionAsync(IDecisionGenerator generator, string ticket,
    DecisionQuestion[] questions, CancellationToken cancellationToken)
{
    DecisionResult result = await generator.GenerateAsync(ticket, questions,
        TicketJson.Default.String, cancellationToken: cancellationToken);
    ChoiceDecisionAnswer answer = result.GetAnswer<ChoiceDecisionAnswer>("support-team");
    Console.WriteLine($"\nProvider: {generator.GetRequiredService<DecisionGeneratorMetadata>().ProviderName}");
    Console.WriteLine($"Model: {result.ModelId}\nSelected ID: {answer.SelectedCandidateId}");
    foreach (var (id, probability) in answer.Probabilities)
        Console.WriteLine($"{id}: {probability.ToString("F4", CultureInfo.InvariantCulture)}");
}

[JsonSerializable(typeof(string))]
internal partial class TicketJson : JsonSerializerContext;
