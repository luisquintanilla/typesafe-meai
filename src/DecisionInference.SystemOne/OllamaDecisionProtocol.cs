using System.Globalization;
using System.Text.Json;

namespace DecisionInference.SystemOne;

internal static class OllamaDecisionProtocol
{
    internal const int MaximumQuestions = 64;
    internal const int MaximumCandidates = 26;
    internal const int MaximumScoreLevels = 26;

    internal static void ValidateInput(DecisionInput input)
    {
        if (input.State.ValueKind is not (JsonValueKind.String or JsonValueKind.Object or JsonValueKind.Array) ||
            (input.State.ValueKind == JsonValueKind.String && string.IsNullOrWhiteSpace(input.State.GetString())))
            throw new ArgumentException("Ollama state must be a nonblank string, object or array.", nameof(input));
        if (input.Questions.Count > MaximumQuestions)
            throw new ArgumentException("Ollama supports at most 64 questions.", nameof(input));
        foreach (var question in input.Questions)
        {
            if (string.IsNullOrWhiteSpace(question.Instructions))
                throw new ArgumentException("Ollama requires nonblank instructions for every question.", nameof(input));
            if (question is ChoiceDecisionQuestion choice && choice.Candidates.Count is < 2 or > MaximumCandidates)
                throw new ArgumentException("Ollama Choice supports 2 through 26 candidates.", nameof(input));
        }
    }

    internal static void ValidateResult(DecisionInput input, string model, DecisionResult result)
    {
        if (!string.Equals(result.ModelId, model, StringComparison.Ordinal))
            throw new DecisionProtocolException("Ollama response model does not match the requested model label.");
        for (int i = 0; i < input.Questions.Count; i++)
        {
            if (input.Questions[i] is ChoiceDecisionQuestion choice && result.Answers[i] is ChoiceDecisionAnswer answer)
            {
                double peak = answer.Probabilities.Values.Max();
                string firstWinner = choice.Candidates.First(c => answer.Probabilities[c.Id] == peak).Id;
                if (!string.Equals(answer.SelectedCandidateId, firstWinner, StringComparison.Ordinal))
                    throw new DecisionProtocolException("Ollama choice must select the first maximum in request order.");
            }
            else if (input.Questions[i] is ScoreDecisionQuestion score)
            {
                var rawAnswer = (JsonElement)result.Answers[i].RawRepresentation!;
                var legend = rawAnswer.GetProperty("legend");
                for (int level = 0; level < score.Rubric.Count; level++)
                {
                    var description = legend.GetProperty(level.ToString(CultureInfo.InvariantCulture));
                    if (description.ValueKind != JsonValueKind.String ||
                        !string.Equals(description.GetString(), score.Rubric[level], StringComparison.Ordinal))
                        throw new DecisionProtocolException("Ollama score legend differs from the requested rubric.");
                }
            }
        }
    }
}
