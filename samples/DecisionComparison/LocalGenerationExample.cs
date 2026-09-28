using System.Text.Json.Serialization;
using DecisionInference;
using DecisionInference.Julia;
using DecisionInference.Laya;

internal static class LocalGenerationExample
{
    internal static async Task RunAsync(string provider, string directory, bool configured, CancellationToken cancellationToken)
    {
        using IDecisionGenerator generator = (provider, configured) switch
        {
            ("julia", false) => JuliaDecisionGenerator.LoadFromDirectory(directory),
            ("laya", false) => LayaDecisionGenerator.LoadFromDirectory(directory),
            ("julia", true) => JuliaDecisionGenerator.Load(new()
            {
                ModelPath = Path.Combine(directory, "model.onnx"),
                TokenizerJsonPath = Path.Combine(directory, "tokenizer.json"),
                ConfigureSession = options => { options.IntraOpNumThreads = 2; options.InterOpNumThreads = 1; }
            }),
            ("laya", true) => LayaDecisionGenerator.Load(new()
            {
                ModelPath = Path.Combine(directory, "laya.onnx"),
                TokenizerJsonPath = Path.Combine(directory, "tokenizer", "tokenizer.json"),
                ProfilePath = Path.Combine(directory, "laya_config.json"),
                ConfigureSession = options => { options.IntraOpNumThreads = 2; options.InterOpNumThreads = 1; }
            }),
            _ => throw new ArgumentException("Choose julia or laya.", nameof(provider))
        };
        DecisionQuestion[] questions =
        [
            new ChoiceDecisionQuestion("team", "Which team should handle this?",
                [new("account", "Account support"), new("billing", "Billing")])
        ];
        DecisionResult result = await generator.GenerateAsync(
            "The customer asks to reset their password.", questions,
            LocalGenerationJson.Default.String, cancellationToken: cancellationToken);
        var answer = result.GetAnswer<ChoiceDecisionAnswer>("team");
        Console.WriteLine($"Selected team: {answer.SelectedCandidateId}");
        foreach (var (id, probability) in answer.Probabilities)
            Console.WriteLine($"{id}: {probability:G17}");
    }
}

[JsonSerializable(typeof(string))]
internal partial class LocalGenerationJson : JsonSerializerContext;
