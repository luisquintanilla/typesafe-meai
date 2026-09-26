using System.Text.Json;
using Microsoft.Extensions.AI;

namespace DecisionInference.Tests;

public sealed class ContractTests
{
    internal static JsonElement Json(string text) => JsonSerializer.Deserialize<JsonElement>(text);
    internal static DecisionRequest Request(params DecisionQuestion[] questions) => new(Json("\"state\""), questions);
    internal static ChoiceDecisionQuestion Choice(string id = "c") =>
        new(id, "choose", [new(" a ", "first"), new("A", "second")]);
    internal static ScoreDecisionQuestion Score(string id = "s") => new(id, "rate", ["low", "mid", "high"]);
    internal static Dictionary<string, double> Distribution(double first = .25) =>
        new(StringComparer.Ordinal) { [" a "] = first, ["A"] = 1 - first };

    [Fact]
    public void OpaqueIds_PreserveSpacesCaseAndDistinctUnicodeNormalization()
    {
        string[] ids = [" q ", "q", "Q", "\u00e9", "e\u0301"];
        var candidates = ids.Select(id => new DecisionCandidate(id, "description")).ToArray();
        var request = Request(ids.Select(id => (DecisionQuestion)new ChoiceDecisionQuestion(id, "string", candidates)).ToArray());
        var response = new DecisionResponse(request, "model", ids.Reverse().Select(id =>
            new ChoiceDecisionAnswer(id, "e\u0301", ids.Select(c => KeyValuePair.Create(c, .2)))));

        Assert.Equal(ids, request.Questions.Select(q => q.Id));
        Assert.Equal(ids, response.Answers.Select(a => a.QuestionId));
        foreach (var answer in response.Answers.Cast<ChoiceDecisionAnswer>())
        {
            Assert.Equal("e\u0301", answer.SelectedCandidateId);
            Assert.Equal(ids, answer.Probabilities.Keys);
            Assert.All(answer.Probabilities.Values, p => Assert.Equal(.2, p));
        }
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData(" \t")]
    public void NullEmptyWhitespaceIds_AreRejectedAcrossContracts(string? id)
    {
        Assert.ThrowsAny<ArgumentException>(() => new BinaryDecisionQuestion(id!));
        Assert.ThrowsAny<ArgumentException>(() => new ChoiceDecisionQuestion(id!, null, [new("a", "")]));
        Assert.ThrowsAny<ArgumentException>(() => new ScoreDecisionQuestion(id!, null, ["", ""]));
        Assert.ThrowsAny<ArgumentException>(() => new DecisionCandidate(id!, ""));
        Assert.ThrowsAny<ArgumentException>(() => new BinaryDecisionAnswer(id!, .5));
        Assert.ThrowsAny<ArgumentException>(() => new ChoiceDecisionAnswer("c", id!, Distribution()));
        Assert.ThrowsAny<ArgumentException>(() => new ChoiceDecisionAnswer("c", "a", [new(id!, 1)]));
    }

    [Fact]
    public void RequestCollections_RejectNullEmptyAndDuplicateEntries()
    {
        Assert.Throws<ArgumentException>(() => new DecisionRequest(default, [new BinaryDecisionQuestion("b")]));
        Assert.Throws<ArgumentNullException>(() => new DecisionRequest(Json("null"), null!));
        Assert.Throws<ArgumentException>(() => Request());
        Assert.Throws<ArgumentException>(() => Request([null!]));
        Assert.Throws<ArgumentException>(() => Request(new BinaryDecisionQuestion("b"), new BinaryDecisionQuestion("b")));
        Assert.Throws<ArgumentNullException>(() => new ChoiceDecisionQuestion("c", null, null!));
        Assert.Throws<ArgumentException>(() => new ChoiceDecisionQuestion("c", null, []));
        Assert.Throws<ArgumentException>(() => new ChoiceDecisionQuestion("c", null, [null!]));
        Assert.Throws<ArgumentException>(() => new ChoiceDecisionQuestion("c", null, [new("a", ""), new("a", "")]));
        Assert.Throws<ArgumentNullException>(() => new DecisionCandidate("a", null!));
        Assert.Throws<ArgumentNullException>(() => new ScoreDecisionQuestion("s", null, null!));
        Assert.Throws<ArgumentException>(() => new ScoreDecisionQuestion("s", null, ["one"]));
        Assert.Throws<ArgumentException>(() => new ScoreDecisionQuestion("s", null, ["one", null!]));
    }

    [Fact]
    public void Request_CopiesStateQuestionsCandidatesAndRubric()
    {
        var document = JsonDocument.Parse("""{"nested":{"value":42}}""");
        DecisionCandidate[] candidates = [new("one", "original"), new("two", "second")];
        string[] rubric = ["poor", "good"];
        var choice = new ChoiceDecisionQuestion("c", "literal", candidates);
        var score = new ScoreDecisionQuestion("s", "string rubric", rubric);
        DecisionQuestion[] questions = [choice, score];
        var request = new DecisionRequest(document.RootElement, questions);
        document.Dispose();
        candidates[0] = new("changed", "changed");
        rubric[0] = "changed";
        questions[0] = new BinaryDecisionQuestion("changed");

        Assert.Equal(42, request.State.GetProperty("nested").GetProperty("value").GetInt32());
        Assert.Equal(["c", "s"], request.Questions.Select(q => q.Id));
        Assert.Equal("one", choice.Candidates[0].Id);
        Assert.Equal("original", choice.Candidates[0].Description);
        Assert.Equal(["poor", "good"], score.Rubric);
        Assert.Throws<NotSupportedException>(() => ((IList<DecisionQuestion>)request.Questions)[0] = questions[0]);
    }

    [Fact]
    public void Response_OwnsAnswerDistributionsEvidenceAndUsageSnapshots()
    {
        var doc = JsonDocument.Parse("""{"nested":[13]}""");
        var probabilities = Distribution();
        double[] ordinal = [.2, .3, .5];
        var choice = new ChoiceDecisionAnswer("c", "A", probabilities, doc.RootElement);
        var score = new ScoreDecisionAnswer("s", ordinal, 1.3, doc.RootElement);
        DecisionAnswer[] answers = [score, choice];
        var usage = new UsageDetails
        {
            InputTokenCount = 11, OutputTokenCount = 7, TotalTokenCount = 18,
            AdditionalCounts = new() { ["cached"] = 3 }
        };
        var response = new DecisionResponse(Request(Choice(), Score()), "model", answers, usage, doc.RootElement);
        doc.Dispose();
        probabilities["A"] = 0;
        ordinal[2] = 0;
        answers[0] = new BinaryDecisionAnswer("bad", 0);
        usage.InputTokenCount = 99;
        usage.AdditionalCounts["cached"] = 99;

        Assert.Equal(["c", "s"], response.Answers.Select(a => a.QuestionId));
        Assert.Equal(.75, choice.Probabilities["A"]);
        Assert.Equal(new[] { .2, .3, .5 }, score.Probabilities);
        Assert.Equal(1.3, score.ExpectedIndex, 12);
        Assert.Equal(13, score.ProviderEvidence.GetProperty("nested")[0].GetInt32());
        Assert.Equal(13, choice.ProviderEvidence.GetProperty("nested")[0].GetInt32());
        Assert.Equal(13, response.ProviderEvidence.GetProperty("nested")[0].GetInt32());
        Assert.Equal(11, response.Usage!.InputTokenCount);
        Assert.Equal(7, response.Usage.OutputTokenCount);
        Assert.Equal(18, response.Usage.TotalTokenCount);
        Assert.Equal(3, response.Usage.AdditionalCounts!["cached"]);
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(-.000001)]
    [InlineData(1.000001)]
    public void Probabilities_RejectNonfiniteAndOutsideUnitInterval(double value)
    {
        Assert.Throws<DecisionProtocolException>(() => new BinaryDecisionAnswer("b", value));
        Assert.Throws<DecisionProtocolException>(() => new ChoiceDecisionAnswer("c", "a", [new("a", value)]));
        Assert.Throws<DecisionProtocolException>(() => new ScoreDecisionAnswer("s", [value, 0]));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void BinaryProbability_UnitEndpointsAreAccepted(double p) =>
        Assert.Equal(p, new BinaryDecisionAnswer("b", p).TrueProbability);

    [Theory]
    [InlineData(.0000009, true)]
    [InlineData(-.0000009, true)]
    [InlineData(.0000011, false)]
    [InlineData(-.0000011, false)]
    public void DistributionSum_UsesAbsoluteOneMicroTolerance(double delta, bool accepted)
    {
        double[] p = [.4, .6 + delta];
        if (accepted)
        {
            Assert.Equal(p, new ScoreDecisionAnswer("s", p).Probabilities);
            Assert.Equal(p[1], new ChoiceDecisionAnswer("c", "a", [new("a", p[0]), new("b", p[1])]).Probabilities["b"]);
        }
        else
        {
            Assert.Throws<DecisionProtocolException>(() => new ScoreDecisionAnswer("s", p));
            Assert.Throws<DecisionProtocolException>(() => new ChoiceDecisionAnswer("c", "a", [new("a", p[0]), new("b", p[1])]));
        }
    }

    [Fact]
    public void ChoiceDistribution_RejectsDuplicateUnknownSelectedAndEmpty()
    {
        Assert.Throws<DecisionProtocolException>(() => new ChoiceDecisionAnswer("c", "a", [new("a", .5), new("a", .5)]));
        Assert.Throws<DecisionProtocolException>(() => new ChoiceDecisionAnswer("c", "missing", Distribution()));
        Assert.Throws<DecisionProtocolException>(() => new ChoiceDecisionAnswer("c", "a", []));
        Assert.Throws<DecisionProtocolException>(() => new ScoreDecisionAnswer("s", [1]));
    }

    [Theory]
    [InlineData(1.3, true)]
    [InlineData(1.3000019, true)]
    [InlineData(1.2999981, true)]
    [InlineData(1.3000021, false)]
    [InlineData(1.2999979, false)]
    [InlineData(-1, false)]
    [InlineData(3, false)]
    [InlineData(double.NaN, false)]
    [InlineData(double.PositiveInfinity, false)]
    public void Score_ExpectedIndexAndNativeToleranceScaleWithRubric(double native, bool accepted)
    {
        if (accepted)
        {
            var answer = new ScoreDecisionAnswer("s", [.2, .3, .5], native);
            Assert.Equal(1.3, answer.ExpectedIndex, 12);
            Assert.Equal(native, answer.NativeScore);
        }
        else
            Assert.Throws<DecisionProtocolException>(() => new ScoreDecisionAnswer("s", [.2, .3, .5], native));
    }

    [Theory]
    [InlineData("missing")]
    [InlineData("extra")]
    [InlineData("duplicate")]
    [InlineData("wrong-kind")]
    [InlineData("unknown")]
    [InlineData("null")]
    public void Response_RejectsMissingExtraDuplicateWrongKindUnknownOrNullAnswer(string scenario)
    {
        DecisionAnswer[] answers = scenario switch
        {
            "missing" => [],
            "extra" => [new BinaryDecisionAnswer("b", .5), new BinaryDecisionAnswer("extra", .5)],
            "duplicate" => [new BinaryDecisionAnswer("b", .5), new BinaryDecisionAnswer("b", .5)],
            "wrong-kind" => [new ScoreDecisionAnswer("b", [.5, .5])],
            "unknown" => [new BinaryDecisionAnswer("B", .5)],
            _ => [null!]
        };
        var request = scenario == "duplicate" ? Request(new BinaryDecisionQuestion("b"), new BinaryDecisionQuestion("other")) : Request(new BinaryDecisionQuestion("b"));
        Assert.Throws<DecisionProtocolException>(() => new DecisionResponse(request, "m", answers));
    }

    [Theory]
    [InlineData("missing")]
    [InlineData("extra")]
    [InlineData("case")]
    public void Response_RequiresExactCompleteCandidateAndRubricDistributions(string scenario)
    {
        KeyValuePair<string, double>[] pairs = scenario switch
        {
            "missing" => [new(" a ", 1)],
            "extra" => [new(" a ", .2), new("A", .3), new("third", .5)],
            _ => [new(" a ", .5), new("a", .5)]
        };
        Assert.Throws<DecisionProtocolException>(() => new DecisionResponse(Request(Choice()), "m", [new ChoiceDecisionAnswer("c", " a ", pairs)]));
        double[] rubric = scenario == "extra" ? [.1, .2, .3, .4] : [.5, .5];
        Assert.Throws<DecisionProtocolException>(() => new DecisionResponse(Request(Score()), "m", [new ScoreDecisionAnswer("s", rubric)]));
    }

    [Theory]
    [InlineData(-1, 0, 0)]
    [InlineData(-1, 1, 0)]
    [InlineData(0, -1, 0)]
    [InlineData(1, -1, 0)]
    [InlineData(0, 0, -1)]
    [InlineData(2, 3, 4)]
    [InlineData(long.MaxValue, 1, long.MaxValue)]
    public void Response_RejectsInvalidUsage(long input, long output, long total) =>
        Assert.Throws<DecisionProtocolException>(() => new DecisionResponse(Request(new BinaryDecisionQuestion("b")), "m",
            [new BinaryDecisionAnswer("b", .5)], new() { InputTokenCount = input, OutputTokenCount = output, TotalTokenCount = total }));

    [Fact]
    public void Response_RejectsNegativePartialUsageWithoutRelyingOnSumMismatch()
    {
        UsageDetails[] invalid = [new() { InputTokenCount = -1 }, new() { OutputTokenCount = -1 }, new() { TotalTokenCount = -1 }];
        foreach (var usage in invalid)
            Assert.Throws<DecisionProtocolException>(() => new DecisionResponse(Request(new BinaryDecisionQuestion("b")), "m",
                [new BinaryDecisionAnswer("b", .5)], usage));
        var result = new DecisionResponse(Request(new BinaryDecisionQuestion("b")), "m", [new BinaryDecisionAnswer("b", .5)],
            new() { InputTokenCount = 9 });
        Assert.Equal(9, result.Usage!.InputTokenCount);
        Assert.Null(result.Usage.OutputTokenCount);
        Assert.Null(result.Usage.TotalTokenCount);
    }

    [Fact]
    public void Metadata_CopiesDistinctSupportedKindsAndEnforcesExactFixedModel()
    {
        DecisionKind[] kinds = [DecisionKind.Choice, DecisionKind.Choice];
        var metadata = new DecisionClientMetadata("provider", "Model ", kinds);
        kinds[0] = DecisionKind.Binary;
        Assert.Equal([DecisionKind.Choice], metadata.SupportedKinds);
        Assert.Equal("provider", metadata.ProviderName);
        Assert.Equal("Model ", metadata.ModelId);
        metadata.Validate(Request(Choice()), new() { ModelId = "Model " }, fixedModel: true);
        metadata.Validate(Request(Choice()), new() { ModelId = "other" });
        Assert.Throws<NotSupportedException>(() => metadata.Validate(Request(Choice()), new() { ModelId = "model " }, true));
        Assert.Throws<NotSupportedException>(() => metadata.Validate(Request(new BinaryDecisionQuestion("b")), null));
        Assert.Throws<ArgumentException>(() => metadata.Validate(Request(Choice()), new() { ModelId = "" }));
        Assert.Throws<ArgumentException>(() => new DecisionClientMetadata("p", "m", []));
        Assert.Throws<ArgumentException>(() => new DecisionClientMetadata("p", "m", (DecisionKind)99));
    }
}
