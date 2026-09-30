using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using DecisionInference.Laya;
using DecisionInference.Ollaya;
using DecisionInference.TypeSafe;
using Microsoft.ML.OnnxRuntime;
using static DecisionInference.Tests.ContractTests;
using static DecisionInference.Tests.OllayaTestFixtures;

namespace DecisionInference.Tests;

public sealed class DecisionSubstitutionTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task WireAliases_IdRenamingAndReorderingPreserveSemanticContentExactMapsAndOwnedRaw(bool ollaya)
    {
        string[][] questionIds =
        [
            [" q ", "q", "Q", "\u00e9", "e\u0301", "q0"],
            ["0", " q0 ", "[CLS]", "<<LABEL>>", "漢字", "new"],
            ["new", "漢字", "<<LABEL>>", "[CLS]", " q0 ", "0"]
        ];
        DecisionCandidate[][] candidates =
        [
            [new(" a ", "red"), new("\u00e9", "blue")],
            [new("e\u0301", "red"), new("q0", "blue")],
            [new("q0", "blue"), new("e\u0301", "red")]
        ];
        double[] firstProbabilities = [.1, .2, .3, .4, .5, .6];
        var answers = new JsonObject();
        for (int i = 5; i >= 0; i--)
            answers[$"q{i}"] = new JsonObject
            {
                ["type"] = "choice", ["choice"] = i < 4 ? "1" : "0", ["confidence"] = .5,
                ["probabilities"] = new JsonObject { ["1"] = 1 - firstProbabilities[i], ["0"] = firstProbabilities[i] }
            };
        string body = new JsonObject { ["model"] = Model, ["answers"] = answers,
            ["usage"] = new JsonObject { ["input_tokens"] = 17, ["output_tokens"] = 0 } }.ToJsonString();
        var contents = new List<TrackingContent>();
        using var handler = Script(sent =>
        {
            var content = new TrackingContent(new ObservedStream(Encoding.UTF8.GetBytes(sent.Uri.AbsolutePath == "/api/show" ? NliShow : body)));
            contents.Add(content);
            var response = new HttpResponseMessage(HttpStatusCode.OK) { Content = content };
            response.Headers.Add("x-typesafe-request-id", $"request-{contents.Count}");
            return response;
        });
        using var http = new HttpClient(handler);
        using IDecisionGenerator generator = ollaya
            ? new OllayaDecisionGenerator(http, new() { ModelId = Model })
            : new TypeSafeDecisionGenerator(http, "fixture-key", new() { ModelId = Model });
        var results = new List<DecisionResult>();
        for (int call = 0; call < 3; call++)
        {
            var input = Input(questionIds[call].Select(id => (DecisionQuestion)new ChoiceDecisionQuestion(id, "choose color", candidates[call])).ToArray());
            results.Add(await generator.GenerateAsync(input, cancellationToken: TestContext.Current.CancellationToken));
        }
        generator.Dispose();
        http.Dispose();
        Assert.All(contents, content => Assert.True(content.Disposed));
        var inference = handler.Requests.Where(sent => sent.Uri.AbsolutePath == "/v1/systemone").ToArray();
        Assert.Equal(3, inference.Length);
        Assert.Equal(inference[0].Body, inference[1].Body);
        string prefix = ollaya ? "choose color\n" : "";
        for (int call = 0; call < 3; call++)
        {
            var result = results[call];
            var wireQuestions = Json(inference[call].Body).GetProperty("questions");
            Assert.Equal(["q0", "q1", "q2", "q3", "q4", "q5"], wireQuestions.EnumerateObject().Select(p => p.Name));
            Assert.Equal(questionIds[call], result.Answers.Select(a => a.QuestionId));
            Assert.Equal(17, result.Usage!.InputTokenCount);
            Assert.Equal($"request-{(call + 1) * (ollaya ? 2 : 1)}", result.AdditionalProperties!["systemone.requestId"]);
            var questionMap = Assert.IsAssignableFrom<IReadOnlyDictionary<string, string>>(result.AdditionalProperties!["systemone.questionIds"]);
            var candidateMaps = Assert.IsAssignableFrom<IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>>>(result.AdditionalProperties["systemone.candidateIds"]);
            Assert.Equal(6, questionMap.Count);
            Assert.Equal(6, candidateMaps.Count);
            for (int i = 0; i < 6; i++)
            {
                string alias = $"q{i}";
                var sent = wireQuestions.GetProperty(alias);
                Assert.Equal("choice", sent.GetProperty("type").GetString());
                Assert.Equal("choose color", sent.GetProperty("instructions").GetString());
                Assert.Equal(["0", "1"], sent.GetProperty("criteria").EnumerateObject().Select(p => p.Name));
                Assert.Equal(prefix + (call == 2 ? "blue" : "red"), sent.GetProperty("criteria").GetProperty("0").GetString());
                Assert.Equal(prefix + (call == 2 ? "red" : "blue"), sent.GetProperty("criteria").GetProperty("1").GetString());
                Assert.Equal(questionIds[call][i], questionMap[alias]);
                Assert.Equal(candidates[call].Select(c => c.Id), candidateMaps[alias].Values);
                var answer = result.GetAnswer<ChoiceDecisionAnswer>(questionIds[call][i]);
                Assert.Equal(candidates[call][i < 4 ? 1 : 0].Id, answer.SelectedCandidateId);
                Assert.Equal(firstProbabilities[i], answer.Probabilities[candidates[call][0].Id], 12);
                Assert.Equal(1 - firstProbabilities[i], answer.Probabilities[candidates[call][1].Id], 12);
                Assert.Equal(ollaya ? DecisionPrecision.FourDecimalPlaces : DecisionPrecision.HighPrecision, answer.Precision);
                var raw = Assert.IsType<JsonElement>(answer.RawRepresentation);
                Assert.Equal(i < 4 ? "1" : "0", raw.GetProperty("choice").GetString());
                Assert.Equal(["1", "0"], raw.GetProperty("probabilities").EnumerateObject().Select(p => p.Name));
            }
            Assert.Equal(["q5", "q4", "q3", "q2", "q1", "q0"],
                Assert.IsType<JsonElement>(result.RawRepresentation).GetProperty("answers").EnumerateObject().Select(p => p.Name));
            Assert.Throws<NotSupportedException>(() => ((IDictionary<string, string>)questionMap)["q0"] = "changed");
            Assert.Throws<NotSupportedException>(() => ((IDictionary<string, IReadOnlyDictionary<string, string>>)candidateMaps).Clear());
            Assert.Throws<NotSupportedException>(() => ((IDictionary<string, string>)candidateMaps["q0"])["0"] = "changed");
            if (ollaya)
            {
                Assert.Equal("onnxruntime", Assert.IsType<JsonElement>(result.AdditionalProperties["ollaya.modelInfo"]).GetProperty("model_info").GetProperty("general.engine").GetString());
                Assert.Equal(prefix + (call == 2 ? "blue" : "red"),
                    Assert.IsType<JsonElement>(result.AdditionalProperties["ollaya.wireRequest"]).GetProperty("questions").GetProperty("q0").GetProperty("criteria").GetProperty("0").GetString());
            }
        }
        Assert.NotSame(results[0].AdditionalProperties!["systemone.questionIds"], results[1].AdditionalProperties!["systemone.questionIds"]);
        Assert.NotSame(results[0].AdditionalProperties!["systemone.candidateIds"], results[1].AdditionalProperties!["systemone.candidateIds"]);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task NullChoiceScoreAndDescribedBinary_EncodeIntentWithoutUsingIds(bool ollaya)
    {
        const string body = """
            {"model":"nli:modernbert-large","usage":{"input_tokens":7,"output_tokens":0},"answers":{
              "q2":{"type":"noul","noul":0.8},
              "q1":{"type":"score","score":0.75,"confidence":0.5,"probabilities":{"0":0.25,"1":0.75},"legend":{"0":"low","1":"high"}},
              "q0":{"type":"choice","choice":"1","confidence":0.5,"probabilities":{"0":0.25,"1":0.75}}}}
            """;
        using var handler = Script(sent => Response(sent.Uri.AbsolutePath == "/api/show" ? NliShow : body));
        using var http = new HttpClient(handler);
        using IDecisionGenerator generator = ollaya
            ? new OllayaDecisionGenerator(http, new() { ModelId = Model })
            : new TypeSafeDecisionGenerator(http, "fixture-key", new() { ModelId = Model });
        var input = Input(
            new ChoiceDecisionQuestion("Do not use this ID as choice intent", null, [new("é", "red"), new("e\u0301", "blue")]),
            new ScoreDecisionQuestion("Do not use this ID as score intent", null, ["low", "high"]),
            new BinaryDecisionQuestion("Do not use this ID as binary intent", null, "condition met", "condition absent"));
        var result = await generator.GenerateAsync(input, cancellationToken: TestContext.Current.CancellationToken);
        var wire = Json(handler.Requests[^1].Body).GetProperty("questions");
        Assert.Equal("Choose one option.", wire.GetProperty("q0").GetProperty("instructions").GetString());
        Assert.Equal("Assess the state against the ordered rubric.", wire.GetProperty("q1").GetProperty("instructions").GetString());
        Assert.Equal("", wire.GetProperty("q2").GetProperty("instructions").GetString());
        Assert.Equal("condition met", wire.GetProperty("q2").GetProperty("criteria").GetProperty("true").GetString());
        Assert.Equal("condition absent", wire.GetProperty("q2").GetProperty("criteria").GetProperty("false").GetString());
        Assert.Equal(ollaya ? "Choose one option.\nred" : "red", wire.GetProperty("q0").GetProperty("criteria").GetProperty("0").GetString());
        Assert.Equal(ollaya ? "Assess the state against the ordered rubric.\nhigh" : "high", wire.GetProperty("q1").GetProperty("criteria")[1].GetString());
        Assert.DoesNotContain("Do not use", handler.Requests[^1].Body, StringComparison.Ordinal);
        Assert.Equal(input.Questions.Select(q => q.Id), result.Answers.Select(a => a.QuestionId));
        Assert.Equal("e\u0301", result.GetAnswer<ChoiceDecisionAnswer>(input.Questions[0].Id).SelectedCandidateId);
        Assert.Equal(.75, result.GetAnswer<ScoreDecisionAnswer>(input.Questions[1].Id).ExpectedIndex);
        Assert.Equal(.8, result.GetAnswer<BinaryDecisionAnswer>(input.Questions[2].Id).TrueProbability);
    }

    [Fact]
    public async Task ChoiceWorkflow_UsesTheSameTypedConsumerForJevOllayaAndEmbeddedLaya()
    {
        using var jevHandler = Script(_ => Response(ChoiceNative()));
        using var ollayaHandler = Script(sent => Response(sent.Uri.AbsolutePath == "/api/show" ? NliShow : ChoiceNative()));
        using var jevHttp = new HttpClient(jevHandler);
        using var ollayaHttp = new HttpClient(ollayaHandler);
        using IDecisionGenerator jev = new TypeSafeDecisionGenerator(jevHttp, "fixture-key");
        using IDecisionGenerator ollaya = new OllayaDecisionGenerator(ollayaHttp, new() { ModelId = Model });
        using var options = new SessionOptions { IntraOpNumThreads = 1, InterOpNumThreads = 1, LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR };
        using var session = new InferenceSession(SyntheticOnnx.Create(), options);
        using var scorer = new ScoreOnnxDecisionModel(session);
        using IDecisionGenerator laya = new LayaDecisionGenerator(LayaStagesTests.Prepare(), scorer, new DecodeDecisions(LayaStagesTests.Profile()));
        // Composition knows the provider; this application's typed workflow does not.
        foreach (var (generator, expectedProbability) in new[] { (jev, .75), (ollaya, .75), (laya, 1d / (1 + Math.Exp(-8))) })
        {
            var decision = await ChooseAsync(generator, TestContext.Current.CancellationToken);
            Assert.Equal("SECOND", decision.Selected);
            Assert.Equal(expectedProbability, decision.Probability, 12);
            Assert.Equal(2, decision.CandidateCount);
        }
        Assert.Single(jevHandler.Requests);
        Assert.Equal(["/api/show", "/v1/systemone"], ollayaHandler.Requests.Select(r => r.Uri.AbsolutePath));
        // Actual Qwen requires a loaded model; no helper or fake is represented as that proof.
    }

    private static async Task<(string Selected, double Probability, int CandidateCount)> ChooseAsync(IDecisionGenerator generator, CancellationToken token)
    {
        var input = Input(new ChoiceDecisionQuestion("decision", "pick", [new(" first ", "red"), new("SECOND", "blue")]));
        var result = await generator.GenerateAsync(input, cancellationToken: token);
        var answer = result.GetAnswer<ChoiceDecisionAnswer>("decision");
        return (answer.SelectedCandidateId, answer.Probabilities[answer.SelectedCandidateId], answer.Probabilities.Count);
    }
}
