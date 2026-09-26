using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Json.Serialization.Metadata;
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
        // Callers own arbitrary raw references; only providers promise their own snapshots.
        var choice = new ChoiceDecisionAnswer("c", "A", probabilities) { RawRepresentation = doc.RootElement.Clone() };
        var score = new ScoreDecisionAnswer("s", ordinal, 1.3) { RawRepresentation = doc.RootElement.Clone() };
        DecisionAnswer[] answers = [score, choice];
        var usage = new UsageDetails
        {
            InputTokenCount = 11, OutputTokenCount = 7, TotalTokenCount = 18,
            AdditionalCounts = new() { ["cached"] = 3 }
        };
        var response = new DecisionResponse(Request(Choice(), Score()), "model", answers, usage) { RawRepresentation = doc.RootElement.Clone() };
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
        Assert.Equal(13, Assert.IsType<JsonElement>(score.RawRepresentation).GetProperty("nested")[0].GetInt32());
        Assert.Equal(13, Assert.IsType<JsonElement>(choice.RawRepresentation).GetProperty("nested")[0].GetInt32());
        Assert.Equal(13, Assert.IsType<JsonElement>(response.RawRepresentation).GetProperty("nested")[0].GetInt32());
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
    public void Metadata_CopiesDistinctSupportedKindsAndDescribesDefaultModel()
    {
        DecisionKind[] kinds = [DecisionKind.Choice, DecisionKind.Choice];
        var metadata = new DecisionClientMetadata("provider", defaultModelId: "Model ", supportedKinds: kinds);
        kinds[0] = DecisionKind.Binary;
        Assert.Equal([DecisionKind.Choice], metadata.SupportedKinds);
        Assert.Equal("provider", metadata.ProviderName);
        Assert.Equal("Model ", metadata.DefaultModelId);
        Assert.Throws<NotSupportedException>(() => ((IList<DecisionKind>)metadata.SupportedKinds!)[0] = DecisionKind.Score);
        // Metadata is descriptive now: fixed-model/blank-model checks live in the provider tests.
        // Unknown versus empty capabilities is tested separately; neither invokes provider policy.
        Assert.Empty(new DecisionClientMetadata("p", defaultModelId: "m", supportedKinds: []).SupportedKinds!);
        Assert.Throws<ArgumentException>(() => new DecisionClientMetadata("p", defaultModelId: "m", supportedKinds: [(DecisionKind)99]));
    }

    [Fact]
    public void Options_DefaultAndNullCopyConstructorsPreserveNullableState()
    {
        DecisionOptions[] options = [new(), new DerivedOptions(null)];
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
        var original = new DecisionOptions
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
        DecisionOptions original = new DerivedOptions(null)
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
        var unknown = new DecisionClientMetadata();
        Assert.Null(unknown.ProviderName);
        Assert.Null(unknown.ProviderUri);
        Assert.Null(unknown.DefaultModelId);
        Assert.Null(unknown.SupportedKinds);
        var uri = new Uri("https://provider.invalid/");
        var none = new DecisionClientMetadata("descriptive", uri, "Model ", []);
        Assert.Equal("descriptive", none.ProviderName);
        Assert.Equal(uri, none.ProviderUri);
        Assert.Equal("Model ", none.DefaultModelId);
        Assert.Empty(none.SupportedKinds!);
        // Descriptive values are not provider policy or per-call validation.
        var descriptive = new DecisionClientMetadata("", defaultModelId: "");
        Assert.Equal("", descriptive.ProviderName);
        Assert.Equal("", descriptive.DefaultModelId);
    }

    [Fact]
    public void RawRepresentation_IsIgnoredByJsonSerializationForResponseAndEveryAnswerKind()
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
        var response = new DecisionResponse(Request(new BinaryDecisionQuestion("b"), Choice(), Score()), "model", answers,
            new() { InputTokenCount = 5 })
        {
            RawRepresentation = unreadable, AdditionalProperties = new() { ["label"] = "response" }
        };
        var root = JsonSerializer.SerializeToElement(response);
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
        var response = new DecisionResponse(Request(new BinaryDecisionQuestion("b"), Choice(), Score()), "model", answers);
        Assert.Null(response.RawRepresentation);
        Assert.Null(response.AdditionalProperties);
        foreach (var answer in answers)
        {
            Assert.Null(answer.RawRepresentation);
            Assert.Null(answer.AdditionalProperties);
            answer.RawRepresentation = raw;
            answer.AdditionalProperties = properties;
        }
        response.RawRepresentation = raw;
        response.AdditionalProperties = properties;
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
        Assert.Same(raw, response.RawRepresentation);
        Assert.Equal(new[] { 13, 42 }, Assert.IsType<List<int>>(response.RawRepresentation));
        Assert.Same(properties, response.AdditionalProperties);
        Assert.Same(nested, response.AdditionalProperties!["nested"]);
        Assert.Equal("visible", response.AdditionalProperties["added"]);
        response.RawRepresentation = "response-replacement";
        response.AdditionalProperties = new() { ["own"] = 2 };
        Assert.Equal("response-replacement", response.RawRepresentation);
        Assert.Equal(2, response.AdditionalProperties["own"]);
        response.RawRepresentation = null;
        response.AdditionalProperties = null;
        Assert.Null(response.RawRepresentation);
        Assert.Null(response.AdditionalProperties);
        Assert.Equal("model", response.ModelId);
        Assert.Equal(["b", "c", "s"], response.Answers.Select(a => a.QuestionId));
        Assert.Equal(.8, ((BinaryDecisionAnswer)answers[0]).TrueProbability);
        Assert.Equal(.75, ((ChoiceDecisionAnswer)answers[1]).Probabilities["A"]);
        Assert.Equal(new[] { .2, .3, .5 }, ((ScoreDecisionAnswer)answers[2]).Probabilities);
        Assert.Equal(1.3, ((ScoreDecisionAnswer)answers[2]).ExpectedIndex, 12);
        Assert.Throws<NotSupportedException>(() => ((IList<DecisionAnswer>)response.Answers)[0] = answers[1]);
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
        using var client = new RecordingClient { Service = service };
        Assert.Same(service, client.GetService<IList<int>>(key));
        Assert.Equal((typeof(IList<int>), key), Assert.Single(client.ServiceCalls));
        Assert.Same(key, client.ServiceCalls[0].Key);
        client.ServiceCalls.Clear();
        Assert.Same(service, client.GetRequiredService<IList<int>>(key));
        Assert.Equal((typeof(IList<int>), key), Assert.Single(client.ServiceCalls));
        Assert.Same(key, client.ServiceCalls[0].Key);
        client.ServiceCalls.Clear();
        Assert.Same(service, client.GetRequiredService(typeof(IList<int>), key));
        Assert.Equal((typeof(IList<int>), key), Assert.Single(client.ServiceCalls));
        Assert.Same(key, client.ServiceCalls[0].Key);
        client.Service = 17;
        Assert.Equal(17, client.GetService<int>(key));
        Assert.Equal(17, client.GetRequiredService<int>(key));
        Assert.Equal(17, client.GetRequiredService(typeof(int), key));
    }

    [Fact]
    public void ServiceExtensions_RejectNullClientOrTypeBeforeDelegating()
    {
        IDecisionClient missing = null!;
        Assert.Equal("client", Assert.Throws<ArgumentNullException>(() => missing.GetService<object>()).ParamName);
        Assert.Equal("client", Assert.Throws<ArgumentNullException>(() => missing.GetRequiredService<object>()).ParamName);
        Assert.Equal("client", Assert.Throws<ArgumentNullException>(() => missing.GetRequiredService(typeof(object))).ParamName);
        using var client = new RecordingClient();
        Assert.Equal("serviceType", Assert.Throws<ArgumentNullException>(() => client.GetRequiredService(null!)).ParamName);
        Assert.Empty(client.ServiceCalls);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ServiceExtensions_HandleMissingAndWrongTypeServices(bool wrongType)
    {
        using var client = new RecordingClient { Service = wrongType ? "wrong-type" : null };
        object key = new();
        Assert.Null(client.GetService<IList<int>>(key));
        Assert.Equal(0, client.GetService<int>(key));
        Assert.Equal($"A service of type '{typeof(IList<int>)}' is not available for the specified key.",
            Assert.Throws<InvalidOperationException>(() => client.GetRequiredService<IList<int>>(key)).Message);
        Assert.Equal($"A service of type '{typeof(IList<int>)}' is not available for the specified key.",
            Assert.Throws<InvalidOperationException>(() => client.GetRequiredService(typeof(IList<int>), key)).Message);
        Assert.Throws<InvalidOperationException>(() => client.GetRequiredService<int>(key));
        Assert.Equal(5, client.ServiceCalls.Count);
        Assert.Equal([typeof(IList<int>), typeof(int), typeof(IList<int>), typeof(IList<int>), typeof(int)],
            client.ServiceCalls.Select(c => c.Type));
        Assert.All(client.ServiceCalls, c => Assert.Same(key, c.Key));
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
        using var client = new RecordingClient();
        DecisionOptions? options = withOptions ? new() { ModelId = "per-call", AdditionalProperties = new() { ["key"] = 13 } } : null;
        DecisionQuestion[] questions = [new BinaryDecisionQuestion("b", "original"), new BinaryDecisionQuestion("B", "second")];
        var originals = questions.ToArray();
        using var cancellation = new CancellationTokenSource();
        var result = await client.GetResponseAsync(nullState ? null : new WireState(7), questions, metadata, options, cancellation.Token);
        questions[0] = new BinaryDecisionQuestion("changed");
        var call = Assert.Single(client.ResponseCalls);
        Assert.Equal(nullState ? """{"wire":"typed-null"}""" : """{"wire":"v:21"}""", call.Request.State.GetRawText());
        Assert.False(call.Request.State.TryGetProperty("Value", out _));
        Assert.Equal(1, converter.Writes);
        Assert.Equal(["b", "B"], call.Request.Questions.Select(q => q.Id));
        Assert.Same(originals[0], call.Request.Questions[0]);
        Assert.Same(originals[1], call.Request.Questions[1]);
        Assert.Same(options, call.Options);
        Assert.Equal(cancellation.Token, call.Token);
        if (withOptions)
        {
            Assert.Equal("per-call", options!.ModelId);
            Assert.Equal(13, options.AdditionalProperties!["key"]);
        }
        Assert.Same(client.Response, result);
        Assert.Equal(.6, Assert.IsType<BinaryDecisionAnswer>(result.Answers[0]).TrueProbability);
    }

    [Fact]
    public async Task TypedStateExtension_PreCancellationDoesNotSerializeOrCallClient()
    {
        var converter = new WireStateConverter();
        var metadata = WireMetadata(converter);
        using var client = new RecordingClient();
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        var error = await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            client.GetResponseAsync(new WireState(7), [new BinaryDecisionQuestion("b")], metadata, cancellationToken: cancellation.Token));
        Assert.Equal(cancellation.Token, error.CancellationToken);
        Assert.Equal(0, converter.Writes);
        Assert.Empty(client.ResponseCalls);
    }

    [Fact]
    public async Task StateExtensions_ValidateInputsWithoutCallingClient()
    {
        var converter = new WireStateConverter();
        var metadata = WireMetadata(converter);
        using var client = new RecordingClient();
        IDecisionClient missing = null!;
        DecisionQuestion[] questions = [new BinaryDecisionQuestion("b")];
        var token = TestContext.Current.CancellationToken;
        Assert.Equal("client", (await Assert.ThrowsAsync<ArgumentNullException>(() => missing.GetResponseAsync(new WireState(7), questions, metadata, cancellationToken: token))).ParamName);
        Assert.Equal("questions", (await Assert.ThrowsAsync<ArgumentNullException>(() => client.GetResponseAsync(new WireState(7), null!, metadata, cancellationToken: token))).ParamName);
        Assert.Equal("stateTypeInfo", (await Assert.ThrowsAsync<ArgumentNullException>(() => client.GetResponseAsync<WireState?>(new(7), questions, null!, cancellationToken: token))).ParamName);
        Assert.Equal("client", (await Assert.ThrowsAsync<ArgumentNullException>(() => missing.GetResponseAsync(Json("null"), questions, cancellationToken: token))).ParamName);
        Assert.Equal("questions", (await Assert.ThrowsAsync<ArgumentNullException>(() => client.GetResponseAsync(Json("null"), null!, cancellationToken: token))).ParamName);
        Assert.Equal("state", (await Assert.ThrowsAsync<ArgumentException>(() => client.GetResponseAsync(default(JsonElement), questions, cancellationToken: token))).ParamName);
        Assert.Equal(0, converter.Writes);
        Assert.Empty(client.ResponseCalls);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    public async Task JsonStateExtension_SnapshotsDocumentAndForwardsExactCallArguments(bool withOptions, bool nullStateAndCanceledToken)
    {
        using var document = JsonDocument.Parse(nullStateAndCanceledToken ? "null" : """{"nested":{"value":42}}""");
        using var client = new RecordingClient();
        DecisionOptions? options = withOptions ? new() { ModelId = "override" } : null;
        DecisionQuestion[] questions = [new BinaryDecisionQuestion("b", "original"), new BinaryDecisionQuestion("B", "second")];
        var originals = questions.ToArray();
        using var cancellation = new CancellationTokenSource();
        // The JSON convenience forwards even a canceled token; the provider owns cancellation.
        if (nullStateAndCanceledToken) cancellation.Cancel();
        var result = await client.GetResponseAsync(document.RootElement, questions, options, cancellation.Token);
        document.Dispose();
        questions[0] = new BinaryDecisionQuestion("changed");
        var call = Assert.Single(client.ResponseCalls);
        if (nullStateAndCanceledToken) Assert.Equal(JsonValueKind.Null, call.Request.State.ValueKind);
        else Assert.Equal(42, call.Request.State.GetProperty("nested").GetProperty("value").GetInt32());
        Assert.Equal(["b", "B"], call.Request.Questions.Select(q => q.Id));
        Assert.Same(originals[0], call.Request.Questions[0]);
        Assert.Same(originals[1], call.Request.Questions[1]);
        Assert.Same(options, call.Options);
        Assert.Equal(cancellation.Token, call.Token);
        Assert.Same(client.Response, result);
        Assert.Equal(.6, Assert.IsType<BinaryDecisionAnswer>(result.Answers[0]).TrueProbability);
    }

    [Fact]
    public void AnswerExtensions_UseExactIdsIncludingCaseSpacesAndUnicode()
    {
        string[] ids = [" q ", "q", "Q", "\u00e9", "e\u0301"];
        var request = Request(ids.Select(id => new BinaryDecisionQuestion(id)).ToArray());
        var answers = ids.Select((id, i) => new BinaryDecisionAnswer(id, (i + 1) / 10d)).ToArray();
        var response = new DecisionResponse(request, "model", answers.Reverse());
        for (int i = 0; i < ids.Length; i++)
        {
            Assert.Same(answers[i], response.GetAnswer(ids[i]));
            Assert.Same(answers[i], response.GetAnswer<BinaryDecisionAnswer>(ids[i]));
            Assert.Equal((i + 1) / 10d, response.GetAnswer<BinaryDecisionAnswer>(ids[i]).TrueProbability);
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
        var response = new DecisionResponse(Request(new BinaryDecisionQuestion(" q "), new BinaryDecisionQuestion("Q"), new BinaryDecisionQuestion("\u00e9")),
            "model", [new BinaryDecisionAnswer(" q ", .2), new BinaryDecisionAnswer("Q", .4), new BinaryDecisionAnswer("\u00e9", .6)]);
        string message = $"No answer exists for question ID '{id}'.";
        Assert.Equal(message, Assert.Throws<KeyNotFoundException>(() => response.GetAnswer(id)).Message);
        Assert.Equal(message, Assert.Throws<KeyNotFoundException>(() => response.GetAnswer<BinaryDecisionAnswer>(id)).Message);
    }

    [Fact]
    public void AnswerExtensions_ReportWrongTypesAndReturnExactMatchingSubtypes()
    {
        var binary = new BinaryDecisionAnswer("b", .8);
        var choice = new ChoiceDecisionAnswer("c", "A", Distribution());
        var score = new ScoreDecisionAnswer("s", [.2, .3, .5], 1.3);
        var response = new DecisionResponse(Request(new BinaryDecisionQuestion("b"), Choice(), Score()), "model", [binary, choice, score]);
        Assert.Same(binary, response.GetAnswer<BinaryDecisionAnswer>("b"));
        Assert.Same(choice, response.GetAnswer<ChoiceDecisionAnswer>("c"));
        Assert.Same(score, response.GetAnswer<ScoreDecisionAnswer>("s"));
        Assert.Same(choice, response.GetAnswer<DecisionAnswer>("c"));
        Assert.Equal("Answer 'b' is not a ChoiceDecisionAnswer.", Assert.Throws<InvalidOperationException>(() => response.GetAnswer<ChoiceDecisionAnswer>("b")).Message);
        Assert.Equal("Answer 'c' is not a ScoreDecisionAnswer.", Assert.Throws<InvalidOperationException>(() => response.GetAnswer<ScoreDecisionAnswer>("c")).Message);
        Assert.Equal("Answer 's' is not a BinaryDecisionAnswer.", Assert.Throws<InvalidOperationException>(() => response.GetAnswer<BinaryDecisionAnswer>("s")).Message);
    }

    [Fact]
    public void AnswerExtensions_RejectNullResponseAndNullOrWhitespaceIds()
    {
        DecisionResponse missing = null!;
        Assert.Equal("response", Assert.Throws<ArgumentNullException>(() => missing.GetAnswer("b")).ParamName);
        Assert.Equal("response", Assert.Throws<ArgumentNullException>(() => missing.GetAnswer<BinaryDecisionAnswer>("b")).ParamName);
        var response = new DecisionResponse(Request(new BinaryDecisionQuestion("b")), "model", [new BinaryDecisionAnswer("b", .6)]);
        Assert.Equal("questionId", Assert.Throws<ArgumentNullException>(() => response.GetAnswer(null!)).ParamName);
        Assert.Equal("questionId", Assert.Throws<ArgumentNullException>(() => response.GetAnswer<BinaryDecisionAnswer>(null!)).ParamName);
        foreach (string id in new[] { "", " \t" })
        {
            Assert.Equal("questionId", Assert.Throws<ArgumentException>(() => response.GetAnswer(id)).ParamName);
            Assert.Equal("questionId", Assert.Throws<ArgumentException>(() => response.GetAnswer<BinaryDecisionAnswer>(id)).ParamName);
        }
    }

    private sealed class DerivedOptions(DecisionOptions? other) : DecisionOptions(other)
    {
        public string? Label { get; set; } = (other as DerivedOptions)?.Label;
        public override DecisionOptions Clone() => new DerivedOptions(this);
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

    private sealed class RecordingClient : IDecisionClient
    {
        public object? Service { get; set; }
        public List<(Type Type, object? Key)> ServiceCalls { get; } = [];
        public List<(DecisionRequest Request, DecisionOptions? Options, CancellationToken Token)> ResponseCalls { get; } = [];
        public DecisionResponse Response { get; } = new(Request(new BinaryDecisionQuestion("b"), new BinaryDecisionQuestion("B")),
            "recorded", [new BinaryDecisionAnswer("b", .6), new BinaryDecisionAnswer("B", .4)]);
        public object? GetService(Type serviceType, object? serviceKey = null)
        {
            ServiceCalls.Add((serviceType, serviceKey));
            return Service;
        }
        public Task<DecisionResponse> GetResponseAsync(DecisionRequest request, DecisionOptions? options = null, CancellationToken cancellationToken = default)
        {
            ResponseCalls.Add((request, options, cancellationToken));
            return Task.FromResult(Response);
        }
        public void Dispose() { }
    }
}
