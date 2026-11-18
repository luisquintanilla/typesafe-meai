#:project ../OllamaDecisions/OllamaDecisions.csproj
#:property TargetFramework=net10.0
#:property PublishAot=false
#:property RestorePackagesWithLockFile=false
#:property NoWarn=$(NoWarn);MEAI001

using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.AI;
using OllamaDecisions;

const string usage = """
    Ollama decisions: dotnet run --file samples\official\Ollama\Ollama.cs
    Environment (all optional):
      OLLAMA_ENDPOINT              Ollama server, default http://localhost:11434
      OLLAMA_DECISION_MODEL        Decision model for text state, default nimble
      OLLAMA_IMAGE_DECISION_MODEL  Decision model that accepts images, default clef-flash
    Pull the models first, for example: ollama pull nimble; ollama pull clef-flash
    """;
if (args is ["--help"]) { Console.WriteLine(usage); return 0; }

string endpoint = Environment.GetEnvironmentVariable("OLLAMA_ENDPOINT") is { Length: > 0 } e ? e : "http://localhost:11434";
string model = Environment.GetEnvironmentVariable("OLLAMA_DECISION_MODEL") is { Length: > 0 } m ? m : "nimble";
string imageModel = Environment.GetEnvironmentVariable("OLLAMA_IMAGE_DECISION_MODEL") is { Length: > 0 } im ? im : "clef-flash";
if (!Uri.TryCreate(endpoint, UriKind.Absolute, out Uri? endpointUri))
{
    Console.Error.WriteLine(usage);
    return 2;
}

using IDecisionClient client = new OllamaDecisionClient(endpointUri, model);

DecisionQuestion[] questions =
[
    new BinaryDecisionQuestion("refund", "Is the customer asking for a refund?",
        trueDescription: "The customer explicitly wants money back.",
        falseDescription: "The customer wants something other than a refund."),
    new ChoiceDecisionQuestion("category", "Classify the customer's main concern.",
    [
        new("billing", "Payment, invoice, or charge concerns."),
        new("technical", "Product defects, errors, or outages."),
        new("account", "Sign-in, profile, or account access."),
    ]),
    new ScoreDecisionQuestion("urgency", "How urgent is this ticket?",
    [
        new("low", "Can wait for normal processing."),
        new("medium", "Should be handled today."),
        new("high", "Needs immediate attention."),
    ]),
];

using var deadline = new CancellationTokenSource(TimeSpan.FromMinutes(5));

// 1. JSON state.
JsonElement ticket = JsonSerializer.SerializeToElement(new
{
    id = "T-1001",
    text = "I cannot sign in after resetting my password, and my team is blocked.",
});
Print("JSON state", await client.GetResponseAsync(new DecisionRequest(ticket, questions), cancellationToken: deadline.Token));

// 2. Text plus an image, judged by a model that accepts images.
byte[] screenshot = await File.ReadAllBytesAsync(
    Path.Combine(SourceDirectory(), "..", "AzureOpenAI", "declined-payment.png"), deadline.Token);
Print("Text + image", await client.GetResponseAsync(
    new DecisionRequest(
        state: null,
        questions,
        [new TextContent("The customer attached this screenshot and wrote: 'Please look at this.'"), new DataContent(screenshot, "image/png")]),
    new DecisionOptions { ModelId = imageModel },
    deadline.Token));

return 0;

static string SourceDirectory([CallerFilePath] string path = "") => Path.GetDirectoryName(path)!;

static void Print(string label, DecisionResponse response)
{
    Console.WriteLine($"== {label} (model {response.ModelId})");
    foreach (DecisionAnswer answer in response.Answers)
    {
        switch (answer)
        {
            case BinaryDecisionAnswer binary:
                Console.WriteLine($"{binary.QuestionId}: P(true) = {binary.TrueProbability:P1}");
                break;
            case ChoiceDecisionAnswer choice:
                Console.WriteLine($"{choice.QuestionId}: {choice.SelectedCandidateId}{Confidence(choice)}");
                PrintDistribution(choice.Probabilities);
                break;
            case ScoreDecisionAnswer score:
                Console.WriteLine($"{score.QuestionId}: expected score {score.ExpectedScore:F2} (0=low, 2=high){Confidence(score)}");
                PrintDistribution(score.Probabilities);
                break;
            case UnansweredDecisionAnswer unanswered:
                Console.WriteLine($"{unanswered.QuestionId}: unanswered ({unanswered.Reason})");
                break;
        }
    }

    Console.WriteLine($"Tokens: {response.Usage?.InputTokenCount} in, {response.Usage?.OutputTokenCount} out");
}

static void PrintDistribution(IReadOnlyList<DecisionProbability> probabilities)
{
    foreach (DecisionProbability probability in probabilities)
    {
        Console.WriteLine($"  {probability.Id,-10} {probability.Value:P1}");
    }
}

static string Confidence(DecisionAnswer answer) =>
    answer.AdditionalProperties?.TryGetValue(OllamaDecisionClient.ConfidenceKey, out object? value) is true ? $" (confidence {value:F2})" : "";
