using System.Numerics.Tensors;
using System.Text.Json;
using DecisionInference.Julia;
using static DecisionInference.Tests.ContractTests;

namespace DecisionInference.Tests;

public sealed class JuliaStagesTests
{
    private static CancellationToken Cancel => TestContext.Current.CancellationToken;

    [Theory]
    [InlineData(false, 1, false)]
    [InlineData(false, 2, true)]
    [InlineData(false, 20, true)]
    [InlineData(false, 21, false)]
    [InlineData(true, 2, true)]
    [InlineData(true, 20, true)]
    [InlineData(true, 21, false)]
    public void Prepare_CardinalityIsProviderLocal(bool score, int count, bool accepted)
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        // Successful core construction proves Choice1 and Score21 are not banned in the contract.
        DecisionQuestion question = score
            ? new ScoreDecisionQuestion("s", "rate", Enumerable.Repeat("a", count))
            : new ChoiceDecisionQuestion("c", "pick", Enumerable.Range(0, count).Select(i => new DecisionCandidate($"id{i}", "a")));
        var request = Request(question);
        if (!accepted)
        {
            Assert.Contains("2-20", Assert.Throws<ArgumentException>(() => prepare.Prepare(request, Cancel)).Message, StringComparison.Ordinal);
            return;
        }
        var batch = prepare.Prepare(request, Cancel);
        Assert.Equal(count, batch.MarkerWidth);
        Assert.Equal(score ? 1L : 0L, batch.QuestionTypes[0]);
        for (int i = 0; i < count; i++)
        {
            Assert.True(batch.MarkerMask[0, i]);
            Assert.Equal(4, batch.InputIds[0, (int)batch.MarkerPositions[0, i]]);
        }
        var response = new DecodeDecisions().Decode(batch, Tensor.Create<float>(new float[count], [1, count]));
        if (score)
        {
            var answer = Assert.IsType<ScoreDecisionAnswer>(Assert.Single(response.Answers));
            Assert.Equal(count, answer.Probabilities.Count);
            Assert.All(answer.Probabilities, p => Assert.Equal(1d / count, p, 14));
            Assert.Equal((count - 1) / 2d, answer.ExpectedIndex, 14);
        }
        else
        {
            var answer = Assert.IsType<ChoiceDecisionAnswer>(Assert.Single(response.Answers));
            Assert.Equal(Enumerable.Range(0, count).Select(i => $"id{i}"), answer.Probabilities.Keys);
            Assert.All(answer.Probabilities.Values, p => Assert.Equal(1d / count, p, 14));
            Assert.Equal("id0", answer.SelectedCandidateId);
        }
    }

    [Fact]
    public void Prepare_CoreScoreOneStillRejectedAndLargerScoreStillConstructs()
    {
        Assert.Contains("at least two", Assert.Throws<ArgumentException>(() => new ScoreDecisionQuestion("s", "", ["a"])).Message, StringComparison.Ordinal);
        Assert.Equal(30, new ScoreDecisionQuestion("s", "", Enumerable.Repeat("a", 30)).Rubric.Count);
        Assert.Single(new ChoiceDecisionQuestion("c", "", [new("id", "a")]).Candidates);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Prepare_BinaryLabelsAndFramingAreExact(bool described)
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        var batch = prepare.Prepare(Request(new BinaryDecisionQuestion("b", "keep instructions",
            described ? "true description" : null, described ? "false description" : null)), Cancel);
        long[] head = fixture.Encode("noul question: keep instructions");
        long[] first = fixture.Encode(described ? " false description" : " false");
        long[] second = fixture.Encode(described ? " true description" : " true");
        long[] expected = [2, .. head, 1, 4, .. first, 4, .. second, 1, .. fixture.Encode("state"), 1];
        Assert.Equal(1, batch.BatchSize);
        Assert.Equal(2, batch.MarkerWidth);
        Assert.Equal((expected.Length + 7) / 8 * 8, batch.SequenceLength);
        JuliaTestFixture.AssertRow(batch, 0, expected, 2, [head.Length + 2, head.Length + first.Length + 3]);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Prepare_BinaryOneSidedDescriptionsAreRejected(bool trueOnly)
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        var question = new BinaryDecisionQuestion("b", "instructions", trueOnly ? "yes" : null, trueOnly ? null : "no");
        Assert.Contains("both descriptions or neither", Assert.Throws<NotSupportedException>(() => prepare.Prepare(Request(question), Cancel)).Message, StringComparison.Ordinal);
        Assert.Equal(2, prepare.Prepare(Request(Binary()), Cancel).MarkerWidth);
    }

    [Fact]
    public void Prepare_IdRenamingLeavesFiveTensorsIdenticalAndRestoresKeys()
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        var request = JuliaTestFixture.Mixed();
        var renamed = new DecisionRequest(request.State,
        [
            new ChoiceDecisionQuestion(" Q <mask> é ", "pick", [new(" e\u0301 <bos> ", "red"), new("É", "blue")]),
            new ScoreDecisionQuestion("q", "rate", ["low", "mid", "high"]),
            new BinaryDecisionQuestion("Q", "decide", "yes", "no")
        ]);
        var first = prepare.Prepare(request, Cancel);
        var second = prepare.Prepare(renamed, Cancel);
        Assert.Equal(first.SequenceLength, second.SequenceLength);
        Assert.Equal(first.MarkerWidth, second.MarkerWidth);
        Assert.Equal(3, second.BatchSize);
        for (int r = 0; r < 3; r++)
        {
            Assert.Equal(JuliaTestFixture.Row(first, r), JuliaTestFixture.Row(second, r));
            Assert.Equal(first.QuestionTypes[r], second.QuestionTypes[r]);
            for (int i = 0; i < first.SequenceLength; i++) Assert.Equal(first.AttentionMask[r, i], second.AttentionMask[r, i]);
            for (int i = 0; i < first.MarkerWidth; i++)
            {
                Assert.Equal(first.MarkerPositions[r, i], second.MarkerPositions[r, i]);
                Assert.Equal(first.MarkerMask[r, i], second.MarkerMask[r, i]);
            }
        }
        var response = new DecodeDecisions().Decode(second, Tensor.Create<float>([0, 1, float.NaN, 0, 0, 0, 0, 1, float.NaN], [3, 3]));
        Assert.Equal([" Q <mask> é ", "q", "Q"], response.Answers.Select(a => a.QuestionId));
        var choice = Assert.IsType<ChoiceDecisionAnswer>(response.Answers[0]);
        Assert.Equal([" e\u0301 <bos> ", "É"], choice.Probabilities.Keys);
        Assert.Equal("É", choice.SelectedCandidateId);
        Assert.Equal(.7310585786300049, choice.Probabilities["É"], 14);
    }

    [Fact]
    public void Prepare_MixedBatchHasExactFiveTensorFramingPaddingAndTypes()
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        var batch = prepare.Prepare(JuliaTestFixture.Mixed(), Cancel);
        string[] heads = ["choice question: pick", "score question: rate", "noul question: decide"];
        string[][] labels = [["red", "blue"], ["low", "mid", "high"], ["no", "yes"]];
        long[] state = fixture.Encode("""{"state": "é"}""");
        int longest = 0;
        for (int row = 0; row < 3; row++)
        {
            List<long> expected = [2, .. fixture.Encode(heads[row]), 1];
            var positions = new List<int>();
            foreach (string label in labels[row])
            {
                positions.Add(expected.Count);
                expected.Add(4);
                expected.AddRange(fixture.Encode(" " + label));
            }
            expected.Add(1);
            expected.AddRange(state);
            expected.Add(1);
            longest = Math.Max(longest, expected.Count);
            JuliaTestFixture.AssertRow(batch, row, expected.ToArray(), row, positions.ToArray());
        }
        Assert.Equal(3, batch.BatchSize);
        Assert.Equal(3, batch.MarkerWidth);
        Assert.Equal((longest + 7) / 8 * 8, batch.SequenceLength);
        Assert.False(batch.MarkerMask[0, 2]);
        Assert.False(batch.MarkerMask[2, 2]);
        Assert.Equal(0, batch.MarkerPositions[0, 2]);
        Assert.Equal(0, batch.MarkerPositions[2, 2]);
    }

    [Theory]
    [InlineData(48, true)]
    [InlineData(49, false)]
    public void Prepare_OptionBudgetCountsTextNotMarker(int tokens, bool accepted)
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        string description = new('a', tokens - 1); // One metaspace token + ASCII characters, no merges.
        Assert.Equal(tokens, fixture.Encode(" " + description).Length);
        var request = Request(new ChoiceDecisionQuestion("c", "", [new("a", description), new("b", "b")]));
        if (!accepted)
        {
            Assert.Contains("option token budget", Assert.Throws<ArgumentException>(() => prepare.Prepare(request, Cancel)).Message, StringComparison.Ordinal);
            return;
        }
        var batch = prepare.Prepare(request, Cancel);
        Assert.Equal(49, batch.MarkerPositions[0, 1] - batch.MarkerPositions[0, 0]); // 48 text + marker.
        Assert.Equal(4, batch.InputIds[0, (int)batch.MarkerPositions[0, 0]]);
        Assert.Equal(fixture.Encode(" " + description),
            JuliaTestFixture.Row(batch, 0).Skip((int)batch.MarkerPositions[0, 0] + 1).Take(48));
    }

    [Fact]
    public void Prepare_HeadMinimumAndOverflowAreLossless()
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        // Five options consuming 239 leave 17 tokens, exactly the shortest score head.
        string a46 = new('a', 46);
        var valid = Request(new ScoreDecisionQuestion("s", "", [a46, a46, a46, a46, new('a', 45)]));
        Assert.Equal(17, fixture.Encode("score question: ").Length);
        var batch = prepare.Prepare(valid, Cancel);
        Assert.Equal(19, batch.MarkerPositions[0, 0]);
        Assert.Equal(5, batch.MarkerWidth);
        Assert.Equal(1, batch.QuestionTypes[0]);
        // 240 option tokens leave the legal minimum 16, but a 17-token head must not be truncated.
        var head16 = Request(new ScoreDecisionQuestion("s", "", Enumerable.Repeat(a46, 5)));
        Assert.Contains("instructions token budget", Assert.Throws<ArgumentException>(() => prepare.Prepare(head16, Cancel)).Message, StringComparison.Ordinal);
        var head15 = Request(new ScoreDecisionQuestion("s", "", [a46, a46, a46, a46, new('a', 47)]));
        Assert.Contains("options exceed the lossless head budget", Assert.Throws<ArgumentException>(() => prepare.Prepare(head15, Cancel)).Message, StringComparison.Ordinal);

        // Two 2-token options + two markers leave 250 tokens for a choice head (base length 18).
        var exactHead = Request(new ChoiceDecisionQuestion("c", new('a', 232), [new("a", "a"), new("b", "b")]));
        var exact = prepare.Prepare(exactHead, Cancel);
        Assert.Equal(252, exact.MarkerPositions[0, 0]);
        Assert.Equal(4, exact.InputIds[0, 252]);
        var overflow = Request(new ChoiceDecisionQuestion("c", new('a', 233), [new("a", "a"), new("b", "b")]));
        Assert.Contains("instructions token budget", Assert.Throws<ArgumentException>(() => prepare.Prepare(overflow, Cancel)).Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(995, 1024)]
    [InlineData(994, 1024)]
    [InlineData(987, 1016)]
    [InlineData(996, 0)]
    [InlineData(1024, 0)]
    public void Prepare_ContextAndPaddingBoundaries(int stateCharacters, int expectedLength)
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        // State prepends metaspace. 18 head + 6 options/markers + 4 framing = 28.
        string text = new('x', stateCharacters);
        Assert.Equal(stateCharacters + 1, fixture.Encode(text).Length);
        var request = new DecisionRequest(Json(JsonSerializer.Serialize(text)),
            [new ChoiceDecisionQuestion("c", "", [new("a", "a"), new("b", "b")])]);
        if (expectedLength == 0)
        {
            Assert.Contains(stateCharacters == 1024 ? "state token budget" : "state exceeds the lossless context",
                Assert.Throws<ArgumentException>(() => prepare.Prepare(request, Cancel)).Message, StringComparison.Ordinal);
            return;
        }
        var batch = prepare.Prepare(request, Cancel);
        Assert.Equal(expectedLength, batch.SequenceLength);
        long[] expected = [2, .. fixture.Encode("choice question: "), 1, 4, .. fixture.Encode(" a"),
            4, .. fixture.Encode(" b"), 1, .. fixture.Encode(text), 1];
        Assert.Equal(stateCharacters + 29, expected.Length);
        JuliaTestFixture.AssertRow(batch, 0, expected, 0, [20, 23]);
    }

    [Theory]
    [InlineData("state")]
    [InlineData("instructions")]
    [InlineData("choice-description")]
    [InlineData("score-description")]
    [InlineData("binary-true")]
    [InlineData("binary-false")]
    public void Prepare_RejectsReservedTextWithoutRewriting(string location)
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        foreach (string token in new[] { "<pad>", "<eos>", "<bos>", "<unk>", "<mask>" })
        {
            string text = "literal " + token + " trailing";
            DecisionQuestion q = location switch
            {
                "instructions" => new ChoiceDecisionQuestion("c", text, [new("a", "a"), new("b", "b")]),
                "choice-description" => new ChoiceDecisionQuestion("c", "", [new("a", text), new("b", "b")]),
                "score-description" => new ScoreDecisionQuestion("s", "", ["ok", text]),
                "binary-true" => new BinaryDecisionQuestion("b", "instructions", text, "no"),
                "binary-false" => new BinaryDecisionQuestion("b", "instructions", "yes", text),
                _ => Choice()
            };
            var request = new DecisionRequest(Json(JsonSerializer.Serialize(location == "state" ? text : "state")), [q]);
            Assert.Contains("Reserved Julia token", Assert.Throws<NotSupportedException>(() => prepare.Prepare(request, Cancel)).Message, StringComparison.Ordinal);
        }
        Assert.Equal(2, prepare.Prepare(Request(Choice()), Cancel).MarkerWidth);
    }

    [Theory]
    [InlineData("""{"\u003cmask>":"value"}""")]
    [InlineData("""{"key":"\u003cmask>"}""")]
    public void Prepare_DecodedJsonKeysAndValuesCannotSmuggleMarkers(string json)
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        var request = new DecisionRequest(Json(json), [Choice()]);
        Assert.Contains("Reserved Julia token in state", Assert.Throws<NotSupportedException>(() => prepare.Prepare(request, Cancel)).Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("""{"z":1,"a":{"β":["é","\t \n",true,null,false]},"space":"  x  "}""",
        """{"z": 1, "a": {"β": ["é", "\t \n", true, null, false]}, "space": "  x  "}""")]
    [InlineData("""{"i":1,"f":1.0,"negativezero":-0.0,"small":1e-5,"big":1e16,"fixed":1e15,"integer":123456789012345678901234567890}""",
        """{"i": 1, "f": 1.0, "negativezero": -0.0, "small": 1e-05, "big": 1e+16, "fixed": 1000000000000000.0, "integer": 123456789012345678901234567890}""")]
    [InlineData("""[{"b":2,"a":1},["雪",1.0,-0.0,0.0001],"\u0001\b\f\r\t\n\"\\"]""",
        """[{"b": 2, "a": 1}, ["雪", 1.0, -0.0, 0.0001], "\u0001\b\f\r\t\n\"\\"]""")]
    [InlineData("\"raw é\\n  text\\t\"", "raw é\n  text\t")]
    public void Prepare_PythonJsonRecipeSurvivesDocumentDisposal(string json, string pythonText)
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        DecisionRequest request;
        using (var document = JsonDocument.Parse(json))
            request = new(document.RootElement, [new ChoiceDecisionQuestion("c", "", [new("a", "a"), new("b", "b")])]);
        var batch = prepare.Prepare(request, Cancel);
        long[] expectedState = fixture.Encode(pythonText);
        // Head, framing and two options occupy 27 tokens before state; this asserts all bytes, not presence.
        long[] expected = [2, .. fixture.Encode("choice question: "), 1, 4, .. fixture.Encode(" a"),
            4, .. fixture.Encode(" b"), 1, .. expectedState, 1];
        JuliaTestFixture.AssertRow(batch, 0, expected, 0, [20, 23]);
        Assert.Equal((expected.Length + 7) / 8 * 8, batch.SequenceLength);
        var response = new DecodeDecisions().Decode(batch, Tensor.Create<float>([0, 1], [1, 2]));
        Assert.Equal("b", Assert.IsType<ChoiceDecisionAnswer>(Assert.Single(response.Answers)).SelectedCandidateId);
    }

    [Theory]
    [InlineData("""{"number":1e309}""", false)]
    [InlineData("""{"a":1,"\u0061":2}""", false)]
    [InlineData("""[{"nested":1,"nested":2}]""", false)]
    [InlineData("null", true)]
    [InlineData("true", true)]
    [InlineData("1", true)]
    [InlineData("-0.0", true)]
    public void Prepare_RejectsUnsupportedJson(string json, bool scalar)
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        var request = new DecisionRequest(Json(json), [Choice()]);
        if (scalar)
            Assert.Contains("text, object or array", Assert.Throws<NotSupportedException>(() => prepare.Prepare(request, Cancel)).Message, StringComparison.Ordinal);
        else
            Assert.Contains(json.Contains("309", StringComparison.Ordinal) ? "finite" : "Duplicate JSON keys",
                Assert.Throws<ArgumentException>(() => prepare.Prepare(request, Cancel)).Message, StringComparison.Ordinal);
        Assert.Equal(2, prepare.Prepare(Request(Choice()), Cancel).MarkerWidth);
    }

    [Fact]
    public void Prepare_RejectsTokenizerMetadataDrift()
    {
        // Each alteration isolates one declared assumption before native tokenizer construction.
        (string Path, string Value)[] changes =
        [
            ("truncation", "{}"), ("padding", "{}"), ("model.type", "\"WordLevel\""),
            ("model.byte_fallback", "false"), ("model.fuse_unk", "false"), ("model.ignore_merges", "true"),
            ("model.unk_token", "\"other\""), ("model.dropout", "0.1"),
            ("model.continuing_subword_prefix", "\"##\""), ("model.end_of_word_suffix", "\"x\""),
            ("normalizer.type", "\"Lowercase\""), ("normalizer.pattern.String", "\"_\""), ("normalizer.content", "\"_\""),
            ("pre_tokenizer.type", "\"Whitespace\""), ("pre_tokenizer.replacement", "\"_\""),
            ("pre_tokenizer.prepend_scheme", "\"never\""), ("pre_tokenizer.split", "false"),
            ("model.vocab.<mask>", "99")
        ];
        foreach (var change in changes)
        {
            using var fixture = new JuliaTestFixture(root =>
            {
                var parts = change.Path.Split('.');
                System.Text.Json.Nodes.JsonNode node = root;
                foreach (string part in parts.SkipLast(1)) node = node[part]!;
                node[parts[^1]] = System.Text.Json.Nodes.JsonNode.Parse(change.Value);
            });
            Assert.Throws<NotSupportedException>(() => fixture.Prepare());
        }
        foreach (string field in new[] { "id", "special" })
        {
            using var fixture = new JuliaTestFixture(root =>
                root["added_tokens"]![4]![field] = field == "id"
                    ? System.Text.Json.Nodes.JsonValue.Create(99)
                    : System.Text.Json.Nodes.JsonValue.Create(false));
            Assert.Contains("framing token metadata", Assert.Throws<NotSupportedException>(() => fixture.Prepare()).Message, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void Prepare_DisposeAndPreCancelLeaveExplicitLifetime()
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        Assert.ThrowsAny<OperationCanceledException>(() => prepare.Prepare(Request(Choice()), cancelled.Token));
        var batch = prepare.Prepare(Request(Choice()), Cancel);
        Assert.Equal(2, batch.MarkerWidth);
        long[] before = JuliaTestFixture.Row(batch, 0);
        prepare.Dispose();
        prepare.Dispose();
        Assert.Equal(before, JuliaTestFixture.Row(batch, 0));
        Assert.Equal("A", Assert.IsType<ChoiceDecisionAnswer>(new DecodeDecisions()
            .Decode(batch, Tensor.Create<float>([0, 1], [1, 2])).Answers[0]).SelectedCandidateId);
        Assert.Throws<ObjectDisposedException>(() => prepare.Prepare(Request(Choice()), Cancel));
    }

    [Fact]
    public void Decode_DoubleProbabilitiesRetainTailsAndZeroBasedScore()
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        var batch = prepare.Prepare(JuliaTestFixture.Mixed(), Cancel);
        var result = new DecodeDecisions().Decode(batch,
            Tensor.Create<float>([0, 5, float.NaN, 0, 1, 2, 0, 5, float.NegativeInfinity], [3, 3]));
        Assert.Equal(JuliaDecisionClient.ModelId, result.ModelId);
        Assert.Null(result.Usage);
        Assert.Equal(["choice", "score", "binary"], result.Answers.Select(a => a.QuestionId));
        Assert.All(result.Answers, a => Assert.Equal(DecisionPrecision.HighPrecision, a.Precision));
        var choice = Assert.IsType<ChoiceDecisionAnswer>(result.Answers[0]);
        Assert.Equal([" a ", "A"], choice.Probabilities.Keys);
        Assert.Equal("A", choice.SelectedCandidateId);
        Assert.Equal(.006692850924284856, choice.Probabilities[" a "], 15);
        Assert.Equal(.9933071490757153, choice.Probabilities["A"], 15);
        Assert.Equal(1, choice.Probabilities.Values.Sum(), 14);
        var score = Assert.IsType<ScoreDecisionAnswer>(result.Answers[1]);
        double[] expected = [.09003057317038046, .24472847105479764, .6652409557748218];
        Assert.Equal(3, score.Probabilities.Count);
        for (int i = 0; i < 3; i++) Assert.Equal(expected[i], score.Probabilities[i], 14);
        Assert.Equal(1, score.Probabilities.Sum(), 14);
        Assert.Equal(1.5752103826044412, score.ExpectedIndex, 14);
        Assert.Null(score.NativeScore);
        Assert.Equal(.9933071490757153, Assert.IsType<BinaryDecisionAnswer>(result.Answers[2]).TrueProbability, 15);
    }

    [Fact]
    public void Decode_ExtremeFloatsAndFirstArgmaxIgnorePadding()
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        var batch = prepare.Prepare(JuliaTestFixture.Mixed(), Cancel);
        var result = new DecodeDecisions().Decode(batch, Tensor.Create<float>(
            [float.MaxValue, float.MaxValue, float.NaN, -float.MaxValue, 0, float.MaxValue, -float.MaxValue, float.MaxValue, float.PositiveInfinity], [3, 3]));
        var choice = Assert.IsType<ChoiceDecisionAnswer>(result.Answers[0]);
        Assert.Equal(" a ", choice.SelectedCandidateId);
        Assert.Equal([" a ", "A"], choice.Probabilities.Keys);
        Assert.Equal(new[] { .5, .5 }, choice.Probabilities.Values);
        var score = Assert.IsType<ScoreDecisionAnswer>(result.Answers[1]);
        Assert.Equal(new[] { 0d, 0d, 1d }, score.Probabilities);
        Assert.Equal(2, score.ExpectedIndex);
        Assert.Equal(1, Assert.IsType<BinaryDecisionAnswer>(result.Answers[2]).TrueProbability);
        Assert.Equal(new double[] { float.MaxValue, float.MaxValue },
            Assert.IsType<JuliaRawAnswer>(choice.RawRepresentation).Logits);
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void Decode_RejectsNonfiniteValidLogits(float value)
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        var batch = prepare.Prepare(JuliaTestFixture.Mixed(), Cancel);
        var decoder = new DecodeDecisions();
        // Last valid column of last row: validates the complete batch, not only its first row.
        Assert.Contains("Nonfinite Julia valid-option", Assert.Throws<DecisionProtocolException>(() =>
            decoder.Decode(batch, Tensor.Create<float>([0, 1, 0, 0, 1, 2, 0, value, 0], [3, 3]))).Message, StringComparison.Ordinal);
        Assert.Equal(.5, Assert.IsType<BinaryDecisionAnswer>(decoder.Decode(batch,
            Tensor.Create<float>(new float[9], [3, 3])).Answers[2]).TrueProbability);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void Decode_RejectsShapeMismatch(int kind)
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        var batch = prepare.Prepare(JuliaTestFixture.Mixed(), Cancel);
        nint[] shape = kind switch { 0 => [9], 1 => [1, 3], _ => [3, 2] };
        int count = (int)shape.Aggregate((nint)1, (a, b) => a * b);
        Assert.Contains("shapes disagree", Assert.Throws<DecisionProtocolException>(() =>
            new DecodeDecisions().Decode(batch, Tensor.Create<float>(new float[count], shape))).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Decode_RawSnapshotsOwnValuesAcrossMutationAndLaterCalls()
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        var batch = prepare.Prepare(Request(Choice()), Cancel);
        var logits = Tensor.Create<float>([.1f, .2f], [1, 2]);
        var decoder = new DecodeDecisions();
        var first = Assert.IsType<ChoiceDecisionAnswer>(decoder.Decode(batch, logits).Answers[0]);
        var raw = Assert.IsType<JuliaRawAnswer>(first.RawRepresentation);
        Assert.Equal(new double[] { .1f, .2f }, raw.Logits); // Widen the actual float values, do not round to decimals.
        // Independently calculated scalar double result from the two exact single-precision operands.
        double expected = 1 / (1 + Math.Exp((double).1f - .2f));
        Assert.Equal(expected, first.Probabilities["A"], 15);
        logits[0, 0] = 50;
        logits[0, 1] = -50;
        var later = Assert.IsType<ChoiceDecisionAnswer>(decoder.Decode(batch, logits).Answers[0]);
        Assert.Equal(" a ", later.SelectedCandidateId);
        Assert.Equal(new double[] { 50, -50 }, Assert.IsType<JuliaRawAnswer>(later.RawRepresentation).Logits);
        Assert.Equal(new double[] { .1f, .2f }, raw.Logits);
        Assert.Equal(expected, first.Probabilities["A"], 15);
        Assert.Equal("A", first.SelectedCandidateId);
        Assert.Throws<NotSupportedException>(() => ((IList<double>)raw.Logits)[0] = 77);
    }
}
