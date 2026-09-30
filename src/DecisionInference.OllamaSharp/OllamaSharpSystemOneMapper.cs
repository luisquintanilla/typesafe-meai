using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using OllamaSharp.Models;
using PreparedRequest = DecisionInference.SystemOne.SystemOneRequest;
using NativeRequest = OllamaSharp.Models.SystemOneRequest;

namespace DecisionInference.OllamaSharp;

internal static class OllamaSharpSystemOneMapper
{
    internal static NativeRequest Request(PreparedRequest wire, int? keepAliveSeconds)
    {
        Dictionary<string, SystemOneQuestion> questions = new(StringComparer.Ordinal);
        foreach (var property in wire.Body.GetProperty("questions").EnumerateObject())
        {
            var body = property.Value;
            SystemOneQuestion question = body.GetProperty("type").GetString() switch
            {
                "noul" => new SystemOneNoulQuestion
                {
                    Criteria = body.TryGetProperty("criteria", out var criteria) ? new()
                    {
                        No = criteria.TryGetProperty("false", out var no) ? no.GetString() : null,
                        Yes = criteria.TryGetProperty("true", out var yes) ? yes.GetString() : null
                    } : null
                },
                "choice" => new SystemOneChoiceQuestion
                {
                    Criteria = body.GetProperty("criteria").EnumerateObject()
                        .ToDictionary(p => p.Name, p => p.Value.GetString(), StringComparer.Ordinal)
                },
                "score" => new SystemOneScoreQuestion
                {
                    Criteria = body.GetProperty("criteria").EnumerateArray().Select(p => p.GetString()!).ToList()
                },
                _ => throw new NotSupportedException("Unsupported prepared native question.")
            };
            question.Instructions = body.GetProperty("instructions").GetString()!;
            questions.Add(property.Name, question);
        }
        return new()
        {
            Model = wire.Body.GetProperty("model").GetString()!,
            State = wire.Input.State.Clone(),
            Questions = questions,
            KeepAlive = keepAliveSeconds is { } seconds ? seconds.ToString(CultureInfo.InvariantCulture) + "s" : null
        };
    }

    internal static JsonElement Response(SystemOneResponse? response)
    {
        if (response is null || response.Answers is null || response.Usage is null)
            throw new DecisionProtocolException("Missing OllamaSharp native response fields.");
        try
        {
            var answers = new JsonObject();
            foreach (var (id, answer) in response.Answers)
            {
                if (answer is null) throw new DecisionProtocolException("Missing OllamaSharp native answer.");
                JsonObject snapshot;
                // Exact DTO types keep unknown subclasses from silently losing additional response semantics.
                if (answer.GetType() == typeof(SystemOneNoulAnswer))
                    snapshot = new() { ["type"] = "noul", ["noul"] = ((SystemOneNoulAnswer)answer).Noul };
                else if (answer.GetType() == typeof(SystemOneChoiceAnswer))
                {
                    var choice = (SystemOneChoiceAnswer)answer;
                    if (choice.Probabilities is null) throw new DecisionProtocolException("Missing native probabilities.");
                    snapshot = new()
                    {
                        ["type"] = "choice", ["choice"] = choice.Choice, ["confidence"] = choice.Confidence,
                        ["probabilities"] = Probabilities(choice.Probabilities)
                    };
                }
                else if (answer.GetType() == typeof(SystemOneScoreAnswer))
                {
                    var score = (SystemOneScoreAnswer)answer;
                    if (score.Probabilities is null || score.Legend is null)
                        throw new DecisionProtocolException("Missing native ordinal fields.");
                    snapshot = new()
                    {
                        ["type"] = "score", ["score"] = score.Score, ["confidence"] = score.Confidence,
                        ["probabilities"] = Probabilities(score.Probabilities),
                        ["legend"] = new JsonObject(score.Legend.Select(p => KeyValuePair.Create<string, JsonNode?>(p.Key, JsonValue.Create(p.Value))))
                    };
                }
                else throw new DecisionProtocolException("Unsupported OllamaSharp native answer subtype.");
                answers.Add(id, snapshot);
            }
            return JsonSerializer.SerializeToElement(new JsonObject
            {
                ["model"] = response.Model,
                ["answers"] = answers,
                ["usage"] = new JsonObject
                {
                    ["input_tokens"] = response.Usage.InputTokens,
                    ["output_tokens"] = response.Usage.OutputTokens
                }
            });
        }
        catch (Exception error) when (error is JsonException or InvalidOperationException or ArgumentException)
        {
            throw new DecisionProtocolException("Malformed OllamaSharp native response fields.");
        }
    }

    private static JsonObject Probabilities(Dictionary<string, double> values) =>
        new(values.Select(p => KeyValuePair.Create<string, JsonNode?>(p.Key, JsonValue.Create(p.Value))));
}
