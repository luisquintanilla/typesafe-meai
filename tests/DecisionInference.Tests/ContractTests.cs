using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Json.Serialization.Metadata;
using Microsoft.Extensions.AI;

namespace DecisionInference.Tests;

public sealed class ContractTests
{
    internal static JsonElement Json(string text) => JsonSerializer.Deserialize<JsonElement>(text);
    internal static DecisionInput Input(params DecisionQuestion[] questions) => new(Json("\"state\""), questions);
    internal static BinaryDecisionQuestion Binary(string id = "b") => new(id, "Is the condition satisfied?");
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
        var input = Input(ids.Select(id => (DecisionQuestion)new ChoiceDecisionQuestion(id, "string", candidates)).ToArray());
        var result = new DecisionResult(input, "model", ids.Reverse().Select(id =>
            new ChoiceDecisionAnswer(id, "e\u0301", ids.Select(c => KeyValuePair.Create(c, .2)))));

        Assert.Equal(ids, input.Questions.Select(q => q.Id));
        Assert.Equal(ids, result.Answers.Select(a => a.QuestionId));
        foreach (var answer in result.Answers.Cast<ChoiceDecisionAnswer>())
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
        Assert.ThrowsAny<ArgumentException>(() => new BinaryDecisionQuestion(id!, "Is it true?"));
        Assert.ThrowsAny<ArgumentException>(() => new ChoiceDecisionQuestion(id!, null, [new("a", "first")]));
        Assert.ThrowsAny<ArgumentException>(() => new ScoreDecisionQuestion(id!, null, ["low", "high"]));
        Assert.ThrowsAny<ArgumentException>(() => new DecisionCandidate(id!, "first"));
        Assert.ThrowsAny<ArgumentException>(() => new BinaryDecisionAnswer(id!, .5));
        Assert.ThrowsAny<ArgumentException>(() => new ChoiceDecisionAnswer("c", id!, Distribution()));
        Assert.ThrowsAny<ArgumentException>(() => new ChoiceDecisionAnswer("c", "a", [new(id!, 1)]));
    }

    [Fact]
    public void InputCollections_RejectNullEmptyAndDuplicateEntries()
    {
        Assert.Throws<ArgumentException>(() => new DecisionInput(default, [Binary()]));
        Assert.Throws<ArgumentNullException>(() => new DecisionInput(Json("null"), null!));
        Assert.Throws<ArgumentException>(() => Input());
        Assert.Throws<ArgumentException>(() => Input([null!]));
        Assert.Throws<ArgumentException>(() => Input(Binary(), Binary()));
        Assert.Throws<ArgumentNullException>(() => new ChoiceDecisionQuestion("c", null, null!));
        Assert.Throws<ArgumentException>(() => new ChoiceDecisionQuestion("c", null, []));
        Assert.Throws<ArgumentException>(() => new ChoiceDecisionQuestion("c", null, [null!]));
        Assert.Throws<ArgumentException>(() => new ChoiceDecisionQuestion("c", null, [new("a", "first"), new("a", "second")]));
        Assert.Throws<ArgumentNullException>(() => new DecisionCandidate("a", null!));
        Assert.Throws<ArgumentNullException>(() => new ScoreDecisionQuestion("s", null, null!));
        Assert.Throws<ArgumentException>(() => new ScoreDecisionQuestion("s", null, ["one"]));
        Assert.Throws<ArgumentException>(() => new ScoreDecisionQuestion("s", null, ["one", null!]));
    }

    [Fact]
    public void Input_CopiesStateQuestionsCandidatesAndRubric()
    {
        var document = JsonDocument.Parse("""{"nested":{"value":42}}""");
        DecisionCandidate[] candidates = [new("one", "original"), new("two", "second")];
        string[] rubric = ["poor", "good"];
        var choice = new ChoiceDecisionQuestion("c", "literal", candidates);
        var score = new ScoreDecisionQuestion("s", "string rubric", rubric);
        DecisionQuestion[] questions = [choice, score];
        var input = new DecisionInput(document.RootElement, questions);
        document.Dispose();
        candidates[0] = new("changed", "changed");
        rubric[0] = "changed";
        questions[0] = Binary("changed");

        Assert.Equal(42, input.State.GetProperty("nested").GetProperty("value").GetInt32());
        Assert.Equal(["c", "s"], input.Questions.Select(q => q.Id));
        Assert.Equal("one", choice.Candidates[0].Id);
        Assert.Equal("original", choice.Candidates[0].Description);
        Assert.Equal(["poor", "good"], score.Rubric);
        Assert.Throws<NotSupportedException>(() => ((IList<DecisionQuestion>)input.Questions)[0] = questions[0]);
    }

    [Fact]
    public void Result_OwnsAnswerDistributionsEvidenceAndUsageSnapshots()
    {
        var doc = JsonDocument.Parse("""{"nested":[13]}""");
        var probabilities = Distribution();
        double[] ordinal = [.2, .3, .5];
        // Callers own arbitrary raw references; only providers promise their own snapshots.
        var choice = new ChoiceDecisionAnswer("c", "A", probabilities) { RawRepresentation = doc.RootElement.Clone() };
        var score = new ScoreDecisionAnswer("s", ordinal, 1.3) { RawRepresentation = doc.RootElement.Clone() };
        DecisionAnswer[] answers = [score, choice];
        var usage = new UsageDetails
        {
            InputTokenCount = 11, OutputTokenCount = 7, TotalTokenCount = 18,
            AdditionalCounts = new() { ["cached"] = 3 }
        };
        var result = new DecisionResult(Input(Choice(), Score()), "model", answers, usage) { RawRepresentation = doc.RootElement.Clone() };
        doc.Dispose();
        probabilities["A"] = 0;
        ordinal[2] = 0;
        answers[0] = new BinaryDecisionAnswer("bad", 0);
        usage.InputTokenCount = 99;
        usage.AdditionalCounts["cached"] = 99;

        Assert.Equal(["c", "s"], result.Answers.Select(a => a.QuestionId));
        Assert.Equal(.75, choice.Probabilities["A"]);
        Assert.Equal(new[] { .2, .3, .5 }, score.Probabilities);
        Assert.Equal(1.3, score.ExpectedIndex, 12);
        Assert.Equal(13, Assert.IsType<JsonElement>(score.RawRepresentation).GetProperty("nested")[0].GetInt32());
        Assert.Equal(13, Assert.IsType<JsonElement>(choice.RawRepresentation).GetProperty("nested")[0].GetInt32());
        Assert.Equal(13, Assert.IsType<JsonElement>(result.RawRepresentation).GetProperty("nested")[0].GetInt32());
        Assert.Equal(11, result.Usage!.InputTokenCount);
        Assert.Equal(7, result.Usage.OutputTokenCount);
        Assert.Equal(18, result.Usage.TotalTokenCount);
        Assert.Equal(3, result.Usage.AdditionalCounts!["cached"]);
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
    public void BinaryProbability_UnitEndpointsAreAccepted(double p)
    {
        Assert.Equal(p, new BinaryDecisionAnswer("b", p).TrueProbability);
        var rounded = new BinaryDecisionAnswer("b", p, DecisionPrecision.FourDecimalPlaces);
        Assert.Equal(p, rounded.TrueProbability);
        Assert.Equal(DecisionPrecision.FourDecimalPlaces, rounded.Precision);
        var score = new ScoreDecisionAnswer("s", [1 - p, p], p, DecisionPrecision.FourDecimalPlaces);
        Assert.Equal(p, score.NativeScore);
        Assert.Equal(p, score.ExpectedIndex);
        Assert.Equal(p == 0 ? 0 : .99995, score.ExpectedIndexLowerBound, 12);
        Assert.Equal(p == 0 ? .00005 : 1, score.ExpectedIndexUpperBound, 12);
    }

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
    public void Result_RejectsMissingExtraDuplicateWrongKindUnknownOrNullAnswer(string scenario)
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
        var input = scenario == "duplicate" ? Input(Binary(), Binary("other")) : Input(Binary());
        Assert.Throws<DecisionProtocolException>(() => new DecisionResult(input, "m", answers));
    }

    [Theory]
    [InlineData("missing")]
    [InlineData("extra")]
    [InlineData("case")]
    public void Result_RequiresExactCompleteCandidateAndRubricDistributions(string scenario)
    {
        KeyValuePair<string, double>[] pairs = scenario switch
        {
            "missing" => [new(" a ", 1)],
            "extra" => [new(" a ", .2), new("A", .3), new("third", .5)],
            _ => [new(" a ", .5), new("a", .5)]
        };
        Assert.Throws<DecisionProtocolException>(() => new DecisionResult(Input(Choice()), "m", [new ChoiceDecisionAnswer("c", " a ", pairs)]));
        double[] rubric = scenario == "extra" ? [.1, .2, .3, .4] : [.5, .5];
        Assert.Throws<DecisionProtocolException>(() => new DecisionResult(Input(Score()), "m", [new ScoreDecisionAnswer("s", rubric)]));
    }

    [Theory]
    [InlineData(-1, 0, 0)]
    [InlineData(-1, 1, 0)]
    [InlineData(0, -1, 0)]
    [InlineData(1, -1, 0)]
    [InlineData(0, 0, -1)]
    [InlineData(2, 3, 4)]
    [InlineData(long.MaxValue, 1, long.MaxValue)]
    public void Result_RejectsInvalidUsage(long input, long output, long total) =>
        Assert.Throws<DecisionProtocolException>(() => new DecisionResult(Input(Binary()), "m",
            [new BinaryDecisionAnswer("b", .5)], new() { InputTokenCount = input, OutputTokenCount = output, TotalTokenCount = total }));

    [Fact]
    public void Result_RejectsNegativePartialUsageWithoutRelyingOnSumMismatch()
    {
        UsageDetails[] invalid = [new() { InputTokenCount = -1 }, new() { OutputTokenCount = -1 }, new() { TotalTokenCount = -1 }];
        foreach (var usage in invalid)
            Assert.Throws<DecisionProtocolException>(() => new DecisionResult(Input(Binary()), "m",
                [new BinaryDecisionAnswer("b", .5)], usage));
        var result = new DecisionResult(Input(Binary()), "m", [new BinaryDecisionAnswer("b", .5)],
            new() { InputTokenCount = 9 });
        Assert.Equal(9, result.Usage!.InputTokenCount);
        Assert.Null(result.Usage.OutputTokenCount);
        Assert.Null(result.Usage.TotalTokenCount);
    }

    [Fact]
    public void Metadata_CopiesDistinctSupportedKindsAndDescribesDefaultModel()
    {
        DecisionKind[] kinds = [DecisionKind.Choice, DecisionKind.Choice];
        var metadata = new DecisionGeneratorMetadata("provider", defaultModelId: "Model ", supportedKinds: kinds);
        kinds[0] = DecisionKind.Binary;
        Assert.Equal([DecisionKind.Choice], metadata.SupportedKinds);
        Assert.Equal("provider", metadata.ProviderName);
        Assert.Equal("Model ", metadata.DefaultModelId);
        Assert.Throws<NotSupportedException>(() => ((IList<DecisionKind>)metadata.SupportedKinds!)[0] = DecisionKind.Score);
        // Metadata is descriptive now: fixed-model/blank-model checks live in the provider tests.
        // Unknown versus empty capabilities is tested separately; neither invokes provider policy.
        Assert.Empty(new DecisionGeneratorMetadata("p", defaultModelId: "m", supportedKinds: []).SupportedKinds!);
        Assert.Throws<ArgumentException>(() => new DecisionGeneratorMetadata("p", defaultModelId: "m", supportedKinds: [(DecisionKind)99]));
    }

    [Fact]
    public void Options_DefaultAndNullCopyConstructorsPreserveNullableState()
    {
        DecisionGenerationOptions[] options = [new(), new DerivedOptions(null)];
        foreach (var original in options)
        {
            Assert.Null(original.ModelId);
            Assert.Null(original.AdditionalProperties);
            var copy = original.Clone();
            Assert.NotSame(original, copy);
            Assert.Null(copy.ModelId);
            Assert.Null(copy.AdditionalProperties);
            original.ModelId = "changed";
            original.AdditionalProperties = new() { ["key"] = 7 };
            Assert.Equal("changed", original.ModelId);
            Assert.Equal(7, original.AdditionalProperties["key"]);
            Assert.Null(copy.ModelId);
            Assert.Null(copy.AdditionalProperties);
        }
    }

    [Fact]
    public void Options_CloneSeparatesDictionaryEntriesButRetainsNestedReferences()
    {
        var nested = new List<int> { 13 };
        var original = new DecisionGenerationOptions
        {
            ModelId = "Model ",
            AdditionalProperties = new() { ["nested"] = nested, ["replace"] = 7, ["remove"] = 9 }
        };
        var copy = original.Clone();
        Assert.NotSame(original, copy);
        Assert.NotSame(original.AdditionalProperties, copy.AdditionalProperties);
        Assert.Equal("Model ", copy.ModelId);
        Assert.Same(nested, copy.AdditionalProperties!["nested"]);
        copy.ModelId = "override";
        copy.AdditionalProperties["replace"] = 8;
        copy.AdditionalProperties.Remove("remove");
        copy.AdditionalProperties["copy-only"] = true;
        original.AdditionalProperties["original-only"] = true;
        nested.Add(42);
        Assert.Equal("Model ", original.ModelId);
        Assert.Equal(7, original.AdditionalProperties["replace"]);
        Assert.Equal(9, original.AdditionalProperties["remove"]);
        Assert.False(original.AdditionalProperties.ContainsKey("copy-only"));
        Assert.False(copy.AdditionalProperties.ContainsKey("original-only"));
        Assert.Equal(new[] { 13, 42 }, Assert.IsType<List<int>>(copy.AdditionalProperties["nested"]));
        original.ModelId = null;
        original.AdditionalProperties = null;
        Assert.Equal("override", copy.ModelId);
        Assert.Equal(8, copy.AdditionalProperties["replace"]);
    }

    [Fact]
    public void Options_DerivedCloneDispatchesVirtuallyAndPreservesDerivedState()
    {
        var nested = new List<int> { 3 };
        DecisionGenerationOptions original = new DerivedOptions(null)
        {
            Label = "derived", ModelId = "base-model", AdditionalProperties = new() { ["nested"] = nested }
        };
        var copy = Assert.IsType<DerivedOptions>(original.Clone());
        Assert.NotSame(original, copy);
        Assert.Equal("derived", copy.Label);
        Assert.Equal("base-model", copy.ModelId);
        Assert.NotSame(original.AdditionalProperties, copy.AdditionalProperties);
        Assert.Same(nested, copy.AdditionalProperties!["nested"]);
        copy.Label = "copy";
        copy.AdditionalProperties["copy-only"] = 2;
        nested.Add(5);
        Assert.Equal("derived", ((DerivedOptions)original).Label);
        Assert.False(original.AdditionalProperties!.ContainsKey("copy-only"));
        Assert.Equal(new[] { 3, 5 }, Assert.IsType<List<int>>(copy.AdditionalProperties["nested"]));
    }

    [Fact]
    public void Metadata_DistinguishesUnknownFromExplicitEmptyCapabilities()
    {
        var unknown = new DecisionGeneratorMetadata();
        Assert.Null(unknown.ProviderName);
        Assert.Null(unknown.ProviderUri);
        Assert.Null(unknown.DefaultModelId);
        Assert.Null(unknown.SupportedKinds);
        var uri = new Uri("https://provider.invalid/");
        var none = new DecisionGeneratorMetadata("descriptive", uri, "Model ", []);
        Assert.Equal("descriptive", none.ProviderName);
        Assert.Equal(uri, none.ProviderUri);
        Assert.Equal("Model ", none.DefaultModelId);
        Assert.Empty(none.SupportedKinds!);
        // Descriptive values are not provider policy or per-call validation.
        var descriptive = new DecisionGeneratorMetadata("", defaultModelId: "");
        Assert.Equal("", descriptive.ProviderName);
        Assert.Equal("", descriptive.DefaultModelId);
    }

    [Fact]
    public void RawRepresentation_IsIgnoredByJsonSerializationForResultAndEveryAnswerKind()
    {
        DecisionAnswer[] answers =
        [
            new BinaryDecisionAnswer("b", .8),
            new ChoiceDecisionAnswer("c", "A", Distribution()),
            new ScoreDecisionAnswer("s", [.2, .3, .5], 1.3)
        ];
        var unreadable = new UnserializableRaw();
        foreach (var answer in answers)
        {
            answer.RawRepresentation = unreadable;
            answer.AdditionalProperties = new() { ["label"] = "kept" };
            var json = JsonSerializer.SerializeToElement(answer, answer.GetType());
            Assert.False(json.TryGetProperty("RawRepresentation", out _));
            Assert.Equal(answer.QuestionId, json.GetProperty("QuestionId").GetString());
            Assert.Equal("kept", json.GetProperty("AdditionalProperties").GetProperty("label").GetString());
            switch (answer)
            {
                case BinaryDecisionAnswer:
                    Assert.Equal(.8, json.GetProperty("TrueProbability").GetDouble());
                    break;
                case ChoiceDecisionAnswer:
                    Assert.Equal("A", json.GetProperty("SelectedCandidateId").GetString());
                    Assert.Equal(.75, json.GetProperty("Probabilities").GetProperty("A").GetDouble());
                    break;
                case ScoreDecisionAnswer:
                    Assert.Equal(1.3, json.GetProperty("ExpectedIndex").GetDouble(), 12);
                    Assert.Equal(new[] { .2, .3, .5 }, json.GetProperty("Probabilities").EnumerateArray().Select(p => p.GetDouble()));
                    break;
            }
        }
        var result = new DecisionResult(Input(Binary(), Choice(), Score()), "model", answers,
            new() { InputTokenCount = 5 })
        {
            RawRepresentation = unreadable, AdditionalProperties = new() { ["label"] = "response" }
        };
        var root = JsonSerializer.SerializeToElement(result);
        Assert.False(root.TryGetProperty("RawRepresentation", out _));
        Assert.Equal("model", root.GetProperty("ModelId").GetString());
        Assert.Equal(5, root.GetProperty("Usage").GetProperty("InputTokenCount").GetInt64());
        Assert.Equal("response", root.GetProperty("AdditionalProperties").GetProperty("label").GetString());
        Assert.Equal(["b", "c", "s"], root.GetProperty("Answers").EnumerateArray().Select(a => a.GetProperty("QuestionId").GetString()));
        Assert.All(root.GetProperty("Answers").EnumerateArray(), a => Assert.False(a.TryGetProperty("RawRepresentation", out _)));
        Assert.Equal(0, unreadable.Reads);
    }

    [Fact]
    public void Extensions_RetainMutableCallerReferencesWithoutChangingValidatedPayload()
    {
        var raw = new List<int> { 13 };
        var nested = new List<int> { 7 };
        var properties = new AdditionalPropertiesDictionary { ["nested"] = nested };
        DecisionAnswer[] answers =
        [
            new BinaryDecisionAnswer("b", .8),
            new ChoiceDecisionAnswer("c", "A", Distribution()),
            new ScoreDecisionAnswer("s", [.2, .3, .5], 1.3)
        ];
        var result = new DecisionResult(Input(Binary(), Choice(), Score()), "model", answers);
        Assert.Null(result.RawRepresentation);
        Assert.Null(result.AdditionalProperties);
        foreach (var answer in answers)
        {
            Assert.Null(answer.RawRepresentation);
            Assert.Null(answer.AdditionalProperties);
            answer.RawRepresentation = raw;
            answer.AdditionalProperties = properties;
        }
        result.RawRepresentation = raw;
        result.AdditionalProperties = properties;
        raw.Add(42);
        nested.Add(9);
        properties["added"] = "visible";
        foreach (var answer in answers)
        {
            Assert.Same(raw, answer.RawRepresentation);
            Assert.Equal(new[] { 13, 42 }, Assert.IsType<List<int>>(answer.RawRepresentation));
            Assert.Same(properties, answer.AdditionalProperties);
            Assert.Same(nested, answer.AdditionalProperties!["nested"]);
            Assert.Equal(new[] { 7, 9 }, Assert.IsType<List<int>>(answer.AdditionalProperties["nested"]));
            Assert.Equal("visible", answer.AdditionalProperties["added"]);
            answer.RawRepresentation = "replacement";
            answer.AdditionalProperties = new() { ["own"] = 1 };
            Assert.Equal("replacement", answer.RawRepresentation);
            Assert.Equal(1, answer.AdditionalProperties["own"]);
            answer.RawRepresentation = null;
            answer.AdditionalProperties = null;
            Assert.Null(answer.RawRepresentation);
            Assert.Null(answer.AdditionalProperties);
        }
        Assert.Same(raw, result.RawRepresentation);
        Assert.Equal(new[] { 13, 42 }, Assert.IsType<List<int>>(result.RawRepresentation));
        Assert.Same(properties, result.AdditionalProperties);
        Assert.Same(nested, result.AdditionalProperties!["nested"]);
        Assert.Equal("visible", result.AdditionalProperties["added"]);
        result.RawRepresentation = "response-replacement";
        result.AdditionalProperties = new() { ["own"] = 2 };
        Assert.Equal("response-replacement", result.RawRepresentation);
        Assert.Equal(2, result.AdditionalProperties["own"]);
        result.RawRepresentation = null;
        result.AdditionalProperties = null;
        Assert.Null(result.RawRepresentation);
        Assert.Null(result.AdditionalProperties);
        Assert.Equal("model", result.ModelId);
        Assert.Equal(["b", "c", "s"], result.Answers.Select(a => a.QuestionId));
        Assert.Equal(.8, ((BinaryDecisionAnswer)answers[0]).TrueProbability);
        Assert.Equal(.75, ((ChoiceDecisionAnswer)answers[1]).Probabilities["A"]);
        Assert.Equal(new[] { .2, .3, .5 }, ((ScoreDecisionAnswer)answers[2]).Probabilities);
        Assert.Equal(1.3, ((ScoreDecisionAnswer)answers[2]).ExpectedIndex, 12);
        Assert.Throws<NotSupportedException>(() => ((IList<DecisionAnswer>)result.Answers)[0] = answers[1]);
        Assert.Throws<NotSupportedException>(() => ((IDictionary<string, double>)((ChoiceDecisionAnswer)answers[1]).Probabilities)["A"] = 0);
        Assert.Throws<NotSupportedException>(() => ((IList<double>)((ScoreDecisionAnswer)answers[2]).Probabilities)[0] = 0);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ServiceExtensions_ForwardExactTypeAndKeyAndReturnSameService(bool keyed)
    {
        object? key = keyed ? new object() : null;
        var service = new List<int> { 7 };
        using var generator = new RecordingGenerator { Service = service };
        Assert.Same(service, generator.GetService<IList<int>>(key));
        Assert.Equal((typeof(IList<int>), key), Assert.Single(generator.ServiceCalls));
        Assert.Same(key, generator.ServiceCalls[0].Key);
        generator.ServiceCalls.Clear();
        Assert.Same(service, generator.GetRequiredService<IList<int>>(key));
        Assert.Equal((typeof(IList<int>), key), Assert.Single(generator.ServiceCalls));
        Assert.Same(key, generator.ServiceCalls[0].Key);
        generator.ServiceCalls.Clear();
        Assert.Same(service, generator.GetRequiredService(typeof(IList<int>), key));
        Assert.Equal((typeof(IList<int>), key), Assert.Single(generator.ServiceCalls));
        Assert.Same(key, generator.ServiceCalls[0].Key);
        generator.Service = 17;
        Assert.Equal(17, generator.GetService<int>(key));
        Assert.Equal(17, generator.GetRequiredService<int>(key));
        Assert.Equal(17, generator.GetRequiredService(typeof(int), key));
    }

    [Fact]
    public void ServiceExtensions_RejectNullGeneratorOrTypeBeforeDelegating()
    {
        IDecisionGenerator missing = null!;
        Assert.Equal("generator", Assert.Throws<ArgumentNullException>(() => missing.GetService<object>()).ParamName);
        Assert.Equal("generator", Assert.Throws<ArgumentNullException>(() => missing.GetRequiredService<object>()).ParamName);
        Assert.Equal("generator", Assert.Throws<ArgumentNullException>(() => missing.GetRequiredService(typeof(object))).ParamName);
        using var generator = new RecordingGenerator();
        Assert.Equal("serviceType", Assert.Throws<ArgumentNullException>(() => generator.GetRequiredService(null!)).ParamName);
        Assert.Empty(generator.ServiceCalls);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ServiceExtensions_HandleMissingAndWrongTypeServices(bool wrongType)
    {
        using var generator = new RecordingGenerator { Service = wrongType ? "wrong-type" : null };
        object key = new();
        Assert.Null(generator.GetService<IList<int>>(key));
        Assert.Equal(0, generator.GetService<int>(key));
        Assert.Equal($"A service of type '{typeof(IList<int>)}' is not available for the specified key.",
            Assert.Throws<InvalidOperationException>(() => generator.GetRequiredService<IList<int>>(key)).Message);
        Assert.Equal($"A service of type '{typeof(IList<int>)}' is not available for the specified key.",
            Assert.Throws<InvalidOperationException>(() => generator.GetRequiredService(typeof(IList<int>), key)).Message);
        Assert.Throws<InvalidOperationException>(() => generator.GetRequiredService<int>(key));
        Assert.Equal(5, generator.ServiceCalls.Count);
        Assert.Equal([typeof(IList<int>), typeof(int), typeof(IList<int>), typeof(IList<int>), typeof(int)],
            generator.ServiceCalls.Select(c => c.Type));
        Assert.All(generator.ServiceCalls, c => Assert.Same(key, c.Key));
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public async Task TypedStateExtension_UsesSuppliedMetadataIncludingNullAndForwardsExactCallArguments(bool nullState, bool withOptions)
    {
        var converter = new WireStateConverter();
        JsonTypeInfo<WireState?> metadata = WireMetadata(converter);
        using var generator = new RecordingGenerator();
        DecisionGenerationOptions? options = withOptions ? new() { ModelId = "per-call", AdditionalProperties = new() { ["key"] = 13 } } : null;
        DecisionQuestion[] questions = [new BinaryDecisionQuestion("b", "original"), new BinaryDecisionQuestion("B", "second")];
        var originals = questions.ToArray();
        using var cancellation = new CancellationTokenSource();
        var result = await generator.GenerateAsync(nullState ? null : new WireState(7), questions, metadata, options, cancellation.Token);
        questions[0] = Binary("changed");
        var call = Assert.Single(generator.GenerationCalls);
        Assert.Equal(nullState ? """{"wire":"typed-null"}""" : """{"wire":"v:21"}""", call.Input.State.GetRawText());
        Assert.False(call.Input.State.TryGetProperty("Value", out _));
        Assert.Equal(1, converter.Writes);
        Assert.Equal(["b", "B"], call.Input.Questions.Select(q => q.Id));
        Assert.Same(originals[0], call.Input.Questions[0]);
        Assert.Same(originals[1], call.Input.Questions[1]);
        Assert.Same(options, call.Options);
        Assert.Equal(cancellation.Token, call.Token);
        if (withOptions)
        {
            Assert.Equal("per-call", options!.ModelId);
            Assert.Equal(13, options.AdditionalProperties!["key"]);
        }
        Assert.Same(generator.Result, result);
        Assert.Equal(.6, Assert.IsType<BinaryDecisionAnswer>(result.Answers[0]).TrueProbability);
    }

    [Fact]
    public async Task TypedStateExtension_PreCancellationDoesNotSerializeOrCallGenerator()
    {
        var converter = new WireStateConverter();
        var metadata = WireMetadata(converter);
        using var generator = new RecordingGenerator();
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        var error = await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            generator.GenerateAsync(new WireState(7), [Binary()], metadata, cancellationToken: cancellation.Token));
        Assert.Equal(cancellation.Token, error.CancellationToken);
        Assert.Equal(0, converter.Writes);
        Assert.Empty(generator.GenerationCalls);
    }

    [Fact]
    public async Task StateExtensions_ValidateInputsWithoutCallingGenerator()
    {
        var converter = new WireStateConverter();
        var metadata = WireMetadata(converter);
        using var generator = new RecordingGenerator();
        IDecisionGenerator missing = null!;
        DecisionQuestion[] questions = [Binary()];
        var token = TestContext.Current.CancellationToken;
        Assert.Equal("generator", (await Assert.ThrowsAsync<ArgumentNullException>(() => missing.GenerateAsync(new WireState(7), questions, metadata, cancellationToken: token))).ParamName);
        Assert.Equal("questions", (await Assert.ThrowsAsync<ArgumentNullException>(() => generator.GenerateAsync(new WireState(7), null!, metadata, cancellationToken: token))).ParamName);
        Assert.Equal("stateTypeInfo", (await Assert.ThrowsAsync<ArgumentNullException>(() => generator.GenerateAsync<WireState?>(new(7), questions, null!, cancellationToken: token))).ParamName);
        Assert.Equal("generator", (await Assert.ThrowsAsync<ArgumentNullException>(() => missing.GenerateAsync(Json("null"), questions, cancellationToken: token))).ParamName);
        Assert.Equal("questions", (await Assert.ThrowsAsync<ArgumentNullException>(() => generator.GenerateAsync(Json("null"), null!, cancellationToken: token))).ParamName);
        Assert.Equal("state", (await Assert.ThrowsAsync<ArgumentException>(() => generator.GenerateAsync(default(JsonElement), questions, cancellationToken: token))).ParamName);
        Assert.Equal(0, converter.Writes);
        Assert.Empty(generator.GenerationCalls);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    public async Task JsonStateExtension_SnapshotsDocumentAndForwardsExactCallArguments(bool withOptions, bool nullStateAndCanceledToken)
    {
        using var document = JsonDocument.Parse(nullStateAndCanceledToken ? "null" : """{"nested":{"value":42}}""");
        using var generator = new RecordingGenerator();
        DecisionGenerationOptions? options = withOptions ? new() { ModelId = "override" } : null;
        DecisionQuestion[] questions = [new BinaryDecisionQuestion("b", "original"), new BinaryDecisionQuestion("B", "second")];
        var originals = questions.ToArray();
        using var cancellation = new CancellationTokenSource();
        // The JSON convenience forwards even a canceled token; the provider owns cancellation.
        if (nullStateAndCanceledToken) cancellation.Cancel();
        var result = await generator.GenerateAsync(document.RootElement, questions, options, cancellation.Token);
        document.Dispose();
        questions[0] = Binary("changed");
        var call = Assert.Single(generator.GenerationCalls);
        if (nullStateAndCanceledToken) Assert.Equal(JsonValueKind.Null, call.Input.State.ValueKind);
        else Assert.Equal(42, call.Input.State.GetProperty("nested").GetProperty("value").GetInt32());
        Assert.Equal(["b", "B"], call.Input.Questions.Select(q => q.Id));
        Assert.Same(originals[0], call.Input.Questions[0]);
        Assert.Same(originals[1], call.Input.Questions[1]);
        Assert.Same(options, call.Options);
        Assert.Equal(cancellation.Token, call.Token);
        Assert.Same(generator.Result, result);
        Assert.Equal(.6, Assert.IsType<BinaryDecisionAnswer>(result.Answers[0]).TrueProbability);
    }

    [Fact]
    public void AnswerExtensions_UseExactIdsIncludingCaseSpacesAndUnicode()
    {
        string[] ids = [" q ", "q", "Q", "\u00e9", "e\u0301"];
        var input = Input(ids.Select(Binary).ToArray());
        var answers = ids.Select((id, i) => new BinaryDecisionAnswer(id, (i + 1) / 10d)).ToArray();
        var result = new DecisionResult(input, "model", answers.Reverse());
        for (int i = 0; i < ids.Length; i++)
        {
            Assert.Same(answers[i], result.GetAnswer(ids[i]));
            Assert.Same(answers[i], result.GetAnswer<BinaryDecisionAnswer>(ids[i]));
            Assert.Equal((i + 1) / 10d, result.GetAnswer<BinaryDecisionAnswer>(ids[i]).TrueProbability);
        }
    }

    [Theory]
    [InlineData("q")]
    [InlineData(" q")]
    [InlineData("q ")]
    [InlineData("\u00c9")]
    [InlineData("e\u0301")]
    [InlineData("missing")]
    public void AnswerExtensions_ReportMissingExactIdsWithoutNormalizing(string id)
    {
        var result = new DecisionResult(Input(Binary(" q "), Binary("Q"), Binary("\u00e9")),
            "model", [new BinaryDecisionAnswer(" q ", .2), new BinaryDecisionAnswer("Q", .4), new BinaryDecisionAnswer("\u00e9", .6)]);
        string message = $"No answer exists for question ID '{id}'.";
        Assert.Equal(message, Assert.Throws<KeyNotFoundException>(() => result.GetAnswer(id)).Message);
        Assert.Equal(message, Assert.Throws<KeyNotFoundException>(() => result.GetAnswer<BinaryDecisionAnswer>(id)).Message);
    }

    [Fact]
    public void AnswerExtensions_ReportWrongTypesAndReturnExactMatchingSubtypes()
    {
        var binary = new BinaryDecisionAnswer("b", .8);
        var choice = new ChoiceDecisionAnswer("c", "A", Distribution());
        var score = new ScoreDecisionAnswer("s", [.2, .3, .5], 1.3);
        var result = new DecisionResult(Input(Binary(), Choice(), Score()), "model", [binary, choice, score]);
        Assert.Same(binary, result.GetAnswer<BinaryDecisionAnswer>("b"));
        Assert.Same(choice, result.GetAnswer<ChoiceDecisionAnswer>("c"));
        Assert.Same(score, result.GetAnswer<ScoreDecisionAnswer>("s"));
        Assert.Same(choice, result.GetAnswer<DecisionAnswer>("c"));
        Assert.Equal("Answer 'b' is not a ChoiceDecisionAnswer.", Assert.Throws<InvalidOperationException>(() => result.GetAnswer<ChoiceDecisionAnswer>("b")).Message);
        Assert.Equal("Answer 'c' is not a ScoreDecisionAnswer.", Assert.Throws<InvalidOperationException>(() => result.GetAnswer<ScoreDecisionAnswer>("c")).Message);
        Assert.Equal("Answer 's' is not a BinaryDecisionAnswer.", Assert.Throws<InvalidOperationException>(() => result.GetAnswer<BinaryDecisionAnswer>("s")).Message);
    }

    [Fact]
    public void AnswerExtensions_RejectNullResultAndNullOrWhitespaceIds()
    {
        DecisionResult missing = null!;
        Assert.Equal("result", Assert.Throws<ArgumentNullException>(() => missing.GetAnswer("b")).ParamName);
        Assert.Equal("result", Assert.Throws<ArgumentNullException>(() => missing.GetAnswer<BinaryDecisionAnswer>("b")).ParamName);
        var result = new DecisionResult(Input(Binary()), "model", [new BinaryDecisionAnswer("b", .6)]);
        Assert.Equal("questionId", Assert.Throws<ArgumentNullException>(() => result.GetAnswer(null!)).ParamName);
        Assert.Equal("questionId", Assert.Throws<ArgumentNullException>(() => result.GetAnswer<BinaryDecisionAnswer>(null!)).ParamName);
        foreach (string id in new[] { "", " \t" })
        {
            Assert.Equal("questionId", Assert.Throws<ArgumentException>(() => result.GetAnswer(id)).ParamName);
            Assert.Equal("questionId", Assert.Throws<ArgumentException>(() => result.GetAnswer<BinaryDecisionAnswer>(id)).ParamName);
        }
    }

    [Fact]
    public void Precision_DefaultsToHighPrecisionWithoutDiscardingNonGridValues()
    {
        double[] probabilities = [.123449, .234449, .642102];
        var binary = new BinaryDecisionAnswer("b", probabilities[0]);
        var choice = new ChoiceDecisionAnswer("c", "2", IndexedDistribution(probabilities));
        var score = new ScoreDecisionAnswer("s", probabilities, 1.518653);
        Assert.Equal(.123449, binary.TrueProbability);
        Assert.Equal(probabilities, choice.Probabilities.Values);
        Assert.Equal(probabilities, score.Probabilities);
        Assert.Equal(1.518653, score.ExpectedIndex, 12);
        Assert.Equal(1.518653, score.NativeScore);
        Assert.Equal(1.518651, score.ExpectedIndexLowerBound, 12);
        Assert.Equal(1.518655, score.ExpectedIndexUpperBound, 12);
        Assert.All(new DecisionAnswer[] { binary, choice, score },
            answer => Assert.Equal(DecisionPrecision.HighPrecision, answer.Precision));
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(3)]
    [InlineData(int.MaxValue)]
    public void Precision_RejectsUndefinedValuesForEveryAnswerKind(int value)
    {
        var precision = (DecisionPrecision)value;
        Assert.Equal("precision", Assert.Throws<ArgumentOutOfRangeException>(() => new BinaryDecisionAnswer("b", .5, precision)).ParamName);
        Assert.Equal("precision", Assert.Throws<ArgumentOutOfRangeException>(() => new ChoiceDecisionAnswer("c", "A", Distribution(), precision)).ParamName);
        Assert.Equal("precision", Assert.Throws<ArgumentOutOfRangeException>(() => new ScoreDecisionAnswer("s", [.5, .5], precision: precision)).ParamName);
    }

    [Theory]
    [InlineData(.25, true)]
    [InlineData(.2500000000005, false)]
    [InlineData(.2499999999995, false)]
    public void FourDecimalGrid_RequiresCanonicalValuesWithoutRepresentationSlack(double p, bool accepted)
    {
        const DecisionPrecision precision = DecisionPrecision.FourDecimalPlaces;
        if (!accepted)
        {
            Assert.Throws<DecisionProtocolException>(() => new BinaryDecisionAnswer("b", p, precision));
            Assert.Throws<DecisionProtocolException>(() => new ChoiceDecisionAnswer("c", "1", IndexedDistribution([.75, p]), precision));
            Assert.Throws<DecisionProtocolException>(() => new ScoreDecisionAnswer("s", [.75, p], precision: precision));
            Assert.Throws<DecisionProtocolException>(() => new ScoreDecisionAnswer("s", [.75, .25], p, precision));
            return;
        }

        var binary = new BinaryDecisionAnswer("b", p, precision);
        var choice = new ChoiceDecisionAnswer("c", "1", IndexedDistribution([1 - p, p]), precision);
        var score = new ScoreDecisionAnswer("s", [1 - p, p], p, precision);
        Assert.Equal(p, binary.TrueProbability);
        Assert.Equal(p, choice.Probabilities["1"]);
        Assert.Equal(new[] { 1 - p, p }, score.Probabilities);
        Assert.Equal(p, score.ExpectedIndex);
        Assert.Equal(p, score.NativeScore);
        Assert.Equal(.24995, score.ExpectedIndexLowerBound, 12);
        Assert.Equal(.25005, score.ExpectedIndexUpperBound, 12);
        Assert.All(new DecisionAnswer[] { binary, choice, score }, a => Assert.Equal(precision, a.Precision));
    }

    [Fact]
    public void FourDecimalGrid_AcceptsJsonLiteralButRejectsAdjacentDoublesAndSubGridOffsets()
    {
        const DecisionPrecision precision = DecisionPrecision.FourDecimalPlaces;
        using var document = JsonDocument.Parse("0.3333");
        double canonical = document.RootElement.GetDouble();
        var binary = new BinaryDecisionAnswer("b", canonical, precision);
        var choice = new ChoiceDecisionAnswer("c", "0", IndexedDistribution([.6667, canonical]), precision);
        var score = new ScoreDecisionAnswer("s", [.6667, canonical], canonical, precision);

        Assert.Equal(.3333, binary.TrueProbability);
        Assert.Equal(new double[] { .6667, .3333 }, choice.Probabilities.Values);
        Assert.Equal("0", choice.SelectedCandidateId);
        Assert.Equal(new double[] { .6667, .3333 }, score.Probabilities);
        Assert.Equal(.3333, score.ExpectedIndex);
        Assert.Equal(.3333, score.NativeScore);
        Assert.Equal(.33325, score.ExpectedIndexLowerBound, 12);
        Assert.Equal(.33335, score.ExpectedIndexUpperBound, 12);
        Assert.All(new DecisionAnswer[] { binary, choice, score }, a => Assert.Equal(precision, a.Precision));

        foreach (double noncanonical in new[] { Math.BitIncrement(.3333), Math.BitDecrement(.3333), .3333000000009 })
        {
            Assert.Throws<DecisionProtocolException>(() => new BinaryDecisionAnswer("b", noncanonical, precision));
            Assert.Throws<DecisionProtocolException>(() => new ChoiceDecisionAnswer(
                "c", "0", IndexedDistribution([.6667, noncanonical]), precision));
            Assert.Throws<DecisionProtocolException>(() => new ScoreDecisionAnswer(
                "s", [.6667, noncanonical], precision: precision));
            Assert.Throws<DecisionProtocolException>(() => new ScoreDecisionAnswer(
                "s", [.6667, .3333], noncanonical, precision));
        }
    }

    [Theory]
    [InlineData(.250000000002)]
    [InlineData(.249999999998)]
    [InlineData(.25001)]
    [InlineData(.25005)]
    [InlineData(-.0001)]
    [InlineData(1.0001)]
    [InlineData(double.NaN)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(double.PositiveInfinity)]
    public void FourDecimalPrecision_RejectsInvalidValuesOnEveryNumericSurface(double value)
    {
        const DecisionPrecision precision = DecisionPrecision.FourDecimalPlaces;
        Assert.Throws<DecisionProtocolException>(() => new BinaryDecisionAnswer("b", value, precision));
        Assert.Throws<DecisionProtocolException>(() => new ChoiceDecisionAnswer("c", "1", IndexedDistribution([1 - value, value]), precision));
        Assert.Throws<DecisionProtocolException>(() => new ScoreDecisionAnswer("s", [1 - value, value], precision: precision));
        Assert.Throws<DecisionProtocolException>(() => new ScoreDecisionAnswer("s", [.75, .25], value, precision));
    }

    public static IEnumerable<object[]> RoundedDistributions()
    {
        yield return [new double[] { .3333, .3333, .3333 }, .9999, .99995, 1.00005];
        yield return [Enumerable.Repeat(.1429, 7).ToArray(), 3.0009, 2.99985, 3.00015];
        yield return [new double[] { 0, 1 }, 1d, .99995, 1d];
        yield return [new double[] { 1, 0 }, 0d, 0d, .00005];
        yield return [new double[] { 0, .0001, 1 }, 2.0001, 1.99995, 1.99995];
        yield return [new double[] { 0, 0, 1 }, 2d, 1.9999, 2d];
    }

    [Theory]
    [MemberData(nameof(RoundedDistributions))]
    public void RoundedDistributions_UseClippedSimplexBoundsWithoutRenormalizing(
        double[] observed, double expectation, double lower, double upper)
    {
        var original = observed.ToArray();
        var choice = new ChoiceDecisionAnswer("c", "0", IndexedDistribution(observed), DecisionPrecision.FourDecimalPlaces);
        var score = new ScoreDecisionAnswer("s", observed, precision: DecisionPrecision.FourDecimalPlaces);
        observed[0] = .9;
        Assert.Equal(original, choice.Probabilities.Values);
        Assert.Equal(original, score.Probabilities);
        Assert.Equal(expectation, score.ExpectedIndex, 12);
        Assert.Equal(lower, score.ExpectedIndexLowerBound, 12);
        Assert.Equal(upper, score.ExpectedIndexUpperBound, 12);
        Assert.Null(score.NativeScore);
        Assert.Equal("0", choice.SelectedCandidateId);
        Assert.Equal(DecisionPrecision.FourDecimalPlaces, choice.Precision);
        Assert.Equal(DecisionPrecision.FourDecimalPlaces, score.Precision);
        Assert.Throws<NotSupportedException>(() => ((IList<double>)score.Probabilities)[0] = .9);
        Assert.Throws<NotSupportedException>(() => ((IDictionary<string, double>)choice.Probabilities)["0"] = .9);
    }

    [Theory]
    [InlineData(3, .3332)]
    [InlineData(3, .3334)]
    [InlineData(7, .1428)]
    [InlineData(2, 0)]
    [InlineData(2, 1)]
    public void RoundedDistributions_RejectInfeasibleMassAtDifferentCardinalities(int count, double p)
    {
        double[] probabilities = Enumerable.Repeat(p, count).ToArray();
        Assert.Throws<DecisionProtocolException>(() => new ChoiceDecisionAnswer("c", "0", IndexedDistribution(probabilities), DecisionPrecision.FourDecimalPlaces));
        Assert.Throws<DecisionProtocolException>(() => new ScoreDecisionAnswer("s", probabilities, precision: DecisionPrecision.FourDecimalPlaces));
    }

    [Fact]
    public void RoundedChoice_255CandidatesAllowFeasibleDriftWithoutRenormalizing()
    {
        // True uniform 1/255 rounds to .0039. The summed latent interval is
        // [.98175, 1.00725], so observed mass .9945 is feasible despite .0055 drift.
        double[] observed = Enumerable.Repeat(.0039, 255).ToArray();
        var choice = new ChoiceDecisionAnswer("c", "254", IndexedDistribution(observed), DecisionPrecision.FourDecimalPlaces);
        observed[254] = 0;

        Assert.Equal("c", choice.QuestionId);
        Assert.Equal("254", choice.SelectedCandidateId);
        Assert.Equal(DecisionPrecision.FourDecimalPlaces, choice.Precision);
        Assert.Equal(255, choice.Probabilities.Count);
        Assert.All(choice.Probabilities.Values, p => Assert.Equal(.0039, p));
        Assert.Equal(.9945, choice.Probabilities.Values.Sum(), 12);
        Assert.Equal(.0039, choice.Probabilities["254"]);

        // A large cardinality does not excuse genuinely infeasible mass.
        foreach (double impossible in new[] { .0038, .0040 })
        {
            double[] probabilities = Enumerable.Repeat(impossible, 255).ToArray();
            Assert.Throws<DecisionProtocolException>(() => new ChoiceDecisionAnswer(
                "c", "254", IndexedDistribution(probabilities), DecisionPrecision.FourDecimalPlaces));
        }

        Assert.Throws<DecisionProtocolException>(() => new ChoiceDecisionAnswer(
            "c", "254", IndexedDistribution(Enumerable.Repeat(.0039, 255).ToArray())));
    }

    [Fact]
    public void RoundedDistributions_RetainCardinalityAndExactCorrelationGuards()
    {
        const DecisionPrecision precision = DecisionPrecision.FourDecimalPlaces;
        var single = new ChoiceDecisionAnswer("c", " a ", [new(" a ", 1)], precision);
        Assert.Equal(1, Assert.Single(single.Probabilities).Value);
        Assert.Equal(" a ", single.SelectedCandidateId);
        Assert.Throws<DecisionProtocolException>(() => new ChoiceDecisionAnswer("c", "0", [], precision));
        Assert.Throws<DecisionProtocolException>(() => new ScoreDecisionAnswer("s", [], precision: precision));
        Assert.Throws<DecisionProtocolException>(() => new ScoreDecisionAnswer("s", [1], precision: precision));
        Assert.Throws<DecisionProtocolException>(() => new ChoiceDecisionAnswer("c", "a", [new("a", .5), new("a", .5)], precision));
        Assert.Throws<DecisionProtocolException>(() => new ChoiceDecisionAnswer("c", "missing", Distribution(), precision));
        Assert.Throws<DecisionProtocolException>(() => new DecisionResult(Input(Choice()), "m", [single]));
        var shortScore = new ScoreDecisionAnswer("s", [.5, .5], precision: precision);
        Assert.Throws<DecisionProtocolException>(() => new DecisionResult(Input(Score()), "m", [shortScore]));
    }

    [Fact]
    public void RoundedScore_AcceptsIndependentNativeRoundingAndKeepsObservedEstimate()
    {
        double[] probabilities = [.1234, .2344, .6421];
        var score = new ScoreDecisionAnswer("s", probabilities, 1.5187, DecisionPrecision.FourDecimalPlaces);
        probabilities[2] = 0;
        Assert.Equal(new double[] { .1234, .2344, .6421 }, score.Probabilities);
        Assert.Equal(.9999, score.Probabilities.Sum(), 12);
        Assert.Equal(1.5186, score.ExpectedIndex, 12);
        Assert.Equal(1.51865, score.ExpectedIndexLowerBound, 12);
        Assert.Equal(1.51875, score.ExpectedIndexUpperBound, 12);
        Assert.Equal(1.5187, score.NativeScore);
        Assert.Equal(DecisionPrecision.FourDecimalPlaces, score.Precision);
        // High precision must not acquire the provider's quantized tolerance.
        Assert.Throws<DecisionProtocolException>(() => new ScoreDecisionAnswer("s", [.1234, .2344, .6421], 1.5187));
        Assert.Throws<DecisionProtocolException>(() => new ScoreDecisionAnswer("s", [.123449, .234449, .642102], 1.5187));
    }

    [Theory]
    [InlineData(1.5185, false)]
    [InlineData(1.5186, true)]
    [InlineData(1.5187, true)]
    [InlineData(1.5188, true)]
    [InlineData(1.5189, false)]
    public void RoundedScore_RequiresNativeIntervalOverlapIncludingBothTouchingBounds(double native, bool accepted)
    {
        if (!accepted)
        {
            Assert.Throws<DecisionProtocolException>(() => new ScoreDecisionAnswer("s", [.1234, .2344, .6421], native, DecisionPrecision.FourDecimalPlaces));
            return;
        }
        var score = new ScoreDecisionAnswer("s", [.1234, .2344, .6421], native, DecisionPrecision.FourDecimalPlaces);
        Assert.Equal(native, score.NativeScore);
        Assert.Equal(1.5186, score.ExpectedIndex, 12);
        Assert.Equal(1.51865, score.ExpectedIndexLowerBound, 12);
        Assert.Equal(1.51875, score.ExpectedIndexUpperBound, 12);
    }

    [Fact]
    public void RoundedScore_ObservedExpectationCanExceedDomainWithoutClamping()
    {
        var score = new ScoreDecisionAnswer("s", [0, .0001, 1], 2, DecisionPrecision.FourDecimalPlaces);
        Assert.Equal(new double[] { 0, .0001, 1 }, score.Probabilities);
        Assert.Equal(2.0001, score.ExpectedIndex, 12);
        Assert.Equal(1.99995, score.ExpectedIndexLowerBound, 12);
        Assert.Equal(1.99995, score.ExpectedIndexUpperBound, 12);
        Assert.Equal(2, score.NativeScore);
        Assert.Throws<DecisionProtocolException>(() => new ScoreDecisionAnswer("s", [0, .0001, 1], 2.0001, DecisionPrecision.FourDecimalPlaces));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData(" \t")]
    public void BinaryIntent_RequiresBothMeaningfulDescriptionsWhenInstructionIsBlank(string? instruction)
    {
        var question = new BinaryDecisionQuestion(" q ", instruction, "condition met", "condition absent");
        Assert.Equal(" q ", question.Id);
        Assert.Equal(instruction ?? "", question.Instructions);
        Assert.Equal("condition met", question.TrueDescription);
        Assert.Equal("condition absent", question.FalseDescription);
        foreach (string? blank in new string?[] { null, "", " \t" })
        {
            string trueParameter = blank is null ? "instructions" : "trueDescription";
            string falseParameter = blank is null ? "instructions" : "falseDescription";
            Assert.Equal(trueParameter, Assert.Throws<ArgumentException>(() =>
                new BinaryDecisionQuestion("Is the customer asking for a refund?", instruction, blank, "condition absent")).ParamName);
            Assert.Equal(falseParameter, Assert.Throws<ArgumentException>(() =>
                new BinaryDecisionQuestion("Is the customer asking for a refund?", instruction, "condition met", blank)).ParamName);
            Assert.Equal(trueParameter, Assert.Throws<ArgumentException>(() =>
                new BinaryDecisionQuestion("meaningful-looking ID", instruction, blank, blank)).ParamName);
        }
    }

    [Theory]
    [InlineData(null, null, null)]
    [InlineData("", " \t", "trueDescription")]
    [InlineData("true description", null, null)]
    [InlineData(null, "false description", null)]
    [InlineData("true description", "false description", null)]
    [InlineData("", "", "trueDescription")]
    [InlineData(" \t", " \t", "trueDescription")]
    [InlineData("", null, "trueDescription")]
    [InlineData(null, "", "falseDescription")]
    [InlineData(" \t", null, "trueDescription")]
    [InlineData(null, " \t", "falseDescription")]
    [InlineData("", "false description", "trueDescription")]
    [InlineData("true description", "", "falseDescription")]
    [InlineData(" \t", "false description", "trueDescription")]
    [InlineData("true description", " \t", "falseDescription")]
    public void BinaryIntent_ExplicitInstructionPreservesValidDescriptionsAndRejectsBlanks(
        string? trueText, string? falseText, string? invalidParameter)
    {
        if (invalidParameter is not null)
        {
            Assert.Equal(invalidParameter, Assert.Throws<ArgumentException>(() =>
                new BinaryDecisionQuestion("opaque", " Is the condition met? ", trueText, falseText)).ParamName);
            return;
        }

        var question = new BinaryDecisionQuestion("opaque", " Is the condition met? ", trueText, falseText);
        Assert.Equal(" Is the condition met? ", question.Instructions);
        Assert.Equal("opaque", question.Id);
        Assert.Equal(DecisionKind.Binary, question.Kind);
        Assert.Equal(trueText, question.TrueDescription);
        Assert.Equal(falseText, question.FalseDescription);
    }

    [Theory]
    [InlineData(" q ")]
    [InlineData("Q")]
    [InlineData("\u00e9")]
    [InlineData("e\u0301")]
    public void NullInstructions_UseStableGenericChoiceAndScoreTextIndependentOfIds(string id)
    {
        Assert.Equal("Choose one option.", new ChoiceDecisionQuestion(id, null, [new("a", "first")]).Instructions);
        Assert.Equal("Assess the state against the ordered rubric.", new ScoreDecisionQuestion(id, null, ["low", "high"]).Instructions);
        foreach (string literal in new[] { "", " \t", "custom instructions" })
        {
            Assert.Equal(literal, new ChoiceDecisionQuestion(id, literal, [new("a", "first")]).Instructions);
            Assert.Equal(literal, new ScoreDecisionQuestion(id, literal, ["low", "high"]).Instructions);
        }
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData(" \t")]
    public void CandidateAndRubricDescriptions_RejectEveryBlankPosition(string? text)
    {
        Assert.ThrowsAny<ArgumentException>(() => new DecisionCandidate("valid-id", text!));
        Assert.Throws<ArgumentException>(() => new ScoreDecisionQuestion("s", "rate", [text!, "high"]));
        Assert.Throws<ArgumentException>(() => new ScoreDecisionQuestion("s", "rate", ["low", text!]));
    }

    private static IEnumerable<KeyValuePair<string, double>> IndexedDistribution(double[] probabilities) =>
        probabilities.Select((p, i) => KeyValuePair.Create(i.ToString(System.Globalization.CultureInfo.InvariantCulture), p));

    private sealed class DerivedOptions(DecisionGenerationOptions? other) : DecisionGenerationOptions(other)
    {
        public string? Label { get; set; } = (other as DerivedOptions)?.Label;
        public override DecisionGenerationOptions Clone() => new DerivedOptions(this);
    }

    private sealed class UnserializableRaw
    {
        public int Reads;
        public string Value { get { Reads++; throw new InvalidOperationException("Raw data must not be serialized."); } }
    }

    private sealed class WireState(int value)
    {
        public int Value { get; } = value;
    }

    // An empty resolver satisfies System.Text.Json's configuration requirement but offers no
    // reflection fallback: this explicit converter-backed metadata is the only state contract.
    private static JsonTypeInfo<WireState?> WireMetadata(WireStateConverter converter) =>
        JsonMetadataServices.CreateValueInfo<WireState?>(new JsonSerializerOptions
        {
            TypeInfoResolver = JsonTypeInfoResolver.Combine()
        }, converter);

    private sealed class WireStateConverter : JsonConverter<WireState?>
    {
        public int Writes;
        public override bool HandleNull => true;
        public override WireState? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) => throw new NotSupportedException();
        public override void Write(Utf8JsonWriter writer, WireState? value, JsonSerializerOptions options)
        {
            Writes++;
            writer.WriteStartObject();
            writer.WriteString("wire", value is null ? "typed-null" : $"v:{value.Value * 3}");
            writer.WriteEndObject();
        }
    }

    private sealed class RecordingGenerator : IDecisionGenerator
    {
        public object? Service { get; set; }
        public List<(Type Type, object? Key)> ServiceCalls { get; } = [];
        public List<(DecisionInput Input, DecisionGenerationOptions? Options, CancellationToken Token)> GenerationCalls { get; } = [];
        public DecisionResult Result { get; } = new(Input(Binary(), Binary("B")),
            "recorded", [new BinaryDecisionAnswer("b", .6), new BinaryDecisionAnswer("B", .4)]);
        public object? GetService(Type serviceType, object? serviceKey = null)
        {
            ServiceCalls.Add((serviceType, serviceKey));
            return Service;
        }
        public Task<DecisionResult> GenerateAsync(DecisionInput input, DecisionGenerationOptions? options = null, CancellationToken cancellationToken = default)
        {
            GenerationCalls.Add((input, options, cancellationToken));
            return Task.FromResult(Result);
        }
        public void Dispose() { }
    }
}
