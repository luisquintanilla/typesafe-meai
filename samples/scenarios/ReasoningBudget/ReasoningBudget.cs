#:project ..\..\..\src\DecisionInference.Jeeves\DecisionInference.Jeeves.csproj
#:property TargetFramework=net10.0
#:property PublishAot=false
#:property RestorePackagesWithLockFile=false

using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Text.Json.Serialization;
using DecisionInference;
using DecisionInference.Jeeves;

const string usage = "ReasoningBudget: dotnet run --file samples\\scenarios\\ReasoningBudget\\ReasoningBudget.cs -- <endpoint> [--show-reasoning]";
if (args is ["--help"]) { Console.WriteLine(usage); return 0; }
if (args.Length is < 1 or > 2 || (args.Length == 2 && args[1] != "--show-reasoning") ||
    !Uri.TryCreate(args[0], UriKind.Absolute, out var endpoint) ||
    !(endpoint.Scheme == "https" || (endpoint.Scheme == "http" &&
        IPAddress.TryParse(endpoint.IdnHost.Trim('[', ']'), out var address) && IPAddress.IsLoopback(address))) ||
    !string.IsNullOrEmpty(endpoint.UserInfo) || !string.IsNullOrEmpty(endpoint.Query) ||
    !string.IsNullOrEmpty(endpoint.Fragment) || endpoint.AbsolutePath != "/")
{
    Console.Error.WriteLine($"Supply an HTTPS origin or literal-loopback HTTP, e.g. http://127.0.0.1:8009 (not localhost).\n{usage}");
    return 2;
}

bool showReasoning = args.Length == 2;
const int thinkingBudget = 128;
using var http = new HttpClient(new SocketsHttpHandler { AllowAutoRedirect = false, UseCookies = false })
{
    Timeout = Timeout.InfiniteTimeSpan
};
using IDecisionGenerator generator = new JeevesDecisionGenerator(http, new() { Endpoint = endpoint });
using var deadline = new CancellationTokenSource(TimeSpan.FromMinutes(2));
string ticket = "I cannot sign in after resetting my password.";
DecisionQuestion[] questions =
[
    new ChoiceDecisionQuestion("support-team", "Which support team should handle this ticket?",
        [new("account", "Account access and password problems"),
         new("billing", "Invoices, charges and payment problems"),
         new("technical", "Software errors and broken features")])
];
Console.WriteLine("Two authorized HTTP calls, one synthetic ticket; this is not a benchmark.");
foreach (bool think in new[] { false, true })
{
    var options = new JeevesGenerationOptions
    {
        Think = think, MaxThink = thinkingBudget, NoThinkThreshold = null, ReturnReasoning = showReasoning
    };
    var clock = Stopwatch.StartNew();
    DecisionResult result = await generator.GenerateAsync(ticket, questions, TicketJson.Default.String, options, deadline.Token);
    clock.Stop();
    var evidence = (JeevesResultMetadata)result.AdditionalProperties![JeevesDecisionGenerator.MetadataKey]!;
    ChoiceDecisionAnswer answer = result.GetAnswer<ChoiceDecisionAnswer>("support-team");
    Console.WriteLine($"\nRequested Think={think}, cap={thinkingBudget} per question; model label={result.ModelId}");
    Console.WriteLine($"Selected ID: {answer.SelectedCandidateId}");
    foreach (var (id, probability) in answer.Probabilities)
        Console.WriteLine($"{id}: {probability.ToString("F2", CultureInfo.InvariantCulture)}");
    Console.WriteLine($"Reported input tokens: {result.Usage!.InputTokenCount}");
    Console.WriteLine($"Reported answer-JSON tokens: {result.Usage.AdditionalCounts![JeevesDecisionGenerator.AnswerJsonTokenCountKey]}");
    Console.WriteLine($"Reported reasoning tokens: {result.Usage.ReasoningTokenCount}");
    Console.WriteLine($"MEAI output (answer JSON + reasoning): {result.Usage.OutputTokenCount}; total: {result.Usage.TotalTokenCount}");
    Console.WriteLine($"Server inference ms: {evidence.LatencyMilliseconds.ToString("F1", CultureInfo.InvariantCulture)}");
    Console.WriteLine($"Whole GenerateAsync elapsed ms: {clock.Elapsed.TotalMilliseconds.ToString("F1", CultureInfo.InvariantCulture)}");
    if (showReasoning)
    {
        var reasoning = evidence.Reasoning!["support-team"];
        Console.WriteLine($"Diagnostic thought={reasoning.Thought}, closed={reasoning.Closed}, chain tokens={reasoning.Tokens}");
        Console.WriteLine("Opt-in diagnostic text, not a faithful-explanation guarantee:");
        Console.WriteLine(reasoning.Text);
    }
}
Console.WriteLine("\nNo ticket was routed. Two observations establish neither improvement, speed advantage nor calibration.");
return 0;

[JsonSerializable(typeof(string))]
internal partial class TicketJson : JsonSerializerContext;
