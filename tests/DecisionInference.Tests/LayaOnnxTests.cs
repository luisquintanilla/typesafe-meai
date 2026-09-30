using DecisionInference.Laya;
using Microsoft.ML.OnnxRuntime;
using static DecisionInference.Tests.ContractTests;
using static DecisionInference.Tests.LayaStagesTests;

namespace DecisionInference.Tests;

public sealed class LayaOnnxTests
{
    private static SessionOptions Options() => new() { IntraOpNumThreads = 1, InterOpNumThreads = 1, LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR };

    [Fact]
    public async Task SyntheticOrt_ScoresActualDynamicGraphAndMatchesPublicStageGeneratorForAllKinds()
    {
        using var options = Options();
        using var session = new InferenceSession(SyntheticOnnx.Create(), options);
        using var scorer = new ScoreOnnxDecisionModel(session);
        var prepare = Prepare();
        var batch = prepare.Prepare(Mixed(), TestContext.Current.CancellationToken);
        var outputs = scorer.Score(batch, TestContext.Current.CancellationToken);
        Assert.Equal(new nint[] { 3, 3 }, outputs.Logits.Lengths.ToArray());
        Assert.Equal(new nint[] { 3, 2 }, outputs.ActionProbabilities.Lengths.ToArray());
        for (int row = 0; row < 3; row++)
        {
            for (int i = 0; i < 3; i++) Assert.Equal((float)batch.MarkerPositions[row, i], outputs.Logits[row, i]);
            Assert.Equal(.5f, outputs.ActionProbabilities[row, 0]);
            Assert.Equal(.5f, outputs.ActionProbabilities[row, 1]);
        }
        var decoder = new DecodeDecisions(Profile());
        var direct = decoder.Decode(batch, outputs);
        using var generator = new LayaDecisionGenerator(prepare, scorer, decoder);
        var composed = await generator.GenerateAsync(Mixed(), cancellationToken: TestContext.Current.CancellationToken);
        AssertEquivalent(direct, composed);
        var choice = Assert.IsType<ChoiceDecisionAnswer>(composed.Answers[0]);
        Assert.Equal("opaque-B", choice.SelectedCandidateId);
        Assert.Equal(1d / (1 + Math.Exp(-8)), choice.Probabilities["opaque-B"], 12);
        var binary = Assert.IsType<BinaryDecisionAnswer>(composed.Answers[2]);
        Assert.Equal(1d / (1 + Math.Exp(-10)), binary.TrueProbability, 12);
        var score = Assert.IsType<ScoreDecisionAnswer>(composed.Answers[1]);
        double[] weights = [Math.Exp(-28), Math.Exp(-14), 1]; // Each preceding rubric option is 13 chars plus its marker.
        double denominator = weights.Sum();
        Assert.Equal((weights[1] + 2 * weights[2]) / denominator, score.ExpectedIndex, 12);
        Assert.Equal(LayaProfile.ModelId, composed.ModelId);
        Assert.All(composed.Answers, answer => Assert.Equal(DecisionPrecision.HighPrecision, answer.Precision));
    }

    [Theory]
    [InlineData("input_ids", "type")]
    [InlineData("attention_mask", "type")]
    [InlineData("marker_pos", "type")]
    [InlineData("marker_mask", "type")]
    [InlineData("qtype", "type")]
    [InlineData("input_ids", "rank")]
    [InlineData("attention_mask", "rank")]
    [InlineData("marker_pos", "rank")]
    [InlineData("marker_mask", "rank")]
    [InlineData("qtype", "rank")]
    [InlineData("input_ids", "name")]
    [InlineData("attention_mask", "name")]
    [InlineData("marker_pos", "name")]
    [InlineData("marker_mask", "name")]
    [InlineData("qtype", "name")]
    public void SyntheticOrt_ValidatesEveryInputNameTypeAndRank(string name, string defect)
    {
        var inputs = SyntheticOnnx.Inputs().Select(i => i.Name != name ? i : defect switch
        {
            "type" => i with { ElementType = i.ElementType == 9 ? 7 : 1 },
            "rank" => i with { Shape = i.Shape.Length == 1 ? [-1, -1] : [-1] },
            _ => i with { Name = "wrong_" + i.Name }
        }).ToArray();
        using var options = Options();
        using var session = new InferenceSession(SyntheticOnnx.Create(inputs, "constant"), options);
        var error = Assert.Throws<ArgumentException>(() => new ScoreOnnxDecisionModel(session));
        Assert.Contains(defect == "name" ? "five named inputs" : name, error.Message, StringComparison.Ordinal);
        Assert.False(options.IsClosed);
    }

    [Theory]
    [InlineData("input_ids", 0)]
    [InlineData("input_ids", 1)]
    [InlineData("attention_mask", 0)]
    [InlineData("attention_mask", 1)]
    [InlineData("marker_pos", 0)]
    [InlineData("marker_pos", 1)]
    [InlineData("marker_mask", 0)]
    [InlineData("marker_mask", 1)]
    [InlineData("qtype", 0)]
    public void SyntheticOrt_RejectsEveryMismatchedStaticInputDimensionBeforeRun(string name, int axis)
    {
        var inputs = SyntheticOnnx.Inputs().Select(i =>
        {
            if (i.Name != name) return i;
            var shape = i.Shape.ToArray();
            shape[axis] = 999;
            return i with { Shape = shape };
        }).ToArray();
        using var options = Options();
        using var session = new InferenceSession(SyntheticOnnx.Create(inputs, "constant"), options);
        using var scorer = new ScoreOnnxDecisionModel(session);
        var batch = Prepare().Prepare(Input(Choice()), TestContext.Current.CancellationToken);
        var error = Assert.Throws<ArgumentException>(() => scorer.Score(batch, TestContext.Current.CancellationToken));
        Assert.Contains(name, error.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("transpose-logits")]
    [InlineData("transpose-action")]
    public void SyntheticOrt_RejectsDynamicallyWrongRuntimeOutputShape(string mode)
    {
        using var options = Options();
        using var session = new InferenceSession(SyntheticOnnx.Create(mode: mode), options);
        using var scorer = new ScoreOnnxDecisionModel(session);
        var input = mode == "transpose-logits" ? Input(Choice()) : Mixed();
        var batch = Prepare().Prepare(input, TestContext.Current.CancellationToken);
        Assert.Contains(-1, session.OutputMetadata[mode == "transpose-logits" ? "logits" : "act_probs"].Dimensions);
        var error = Assert.Throws<DecisionProtocolException>(() => scorer.Score(batch, TestContext.Current.CancellationToken));
        Assert.Equal("Unexpected Laya output shape or type.", error.Message);
    }

    [Fact]
    public void SyntheticOrt_AcceptsMatchingStaticDimensionsForAllFiveInputs()
    {
        var batch = Prepare().Prepare(Input(Choice()), TestContext.Current.CancellationToken);
        var inputs = SyntheticOnnx.Inputs().Select(i => i with
        {
            Shape = i.Name == "qtype" ? [1] : i.Name is "marker_pos" or "marker_mask" ? [1, 2] : [1, batch.SequenceLength]
        }).ToArray();
        using var options = Options();
        using var session = new InferenceSession(SyntheticOnnx.Create(inputs), options);
        using var scorer = new ScoreOnnxDecisionModel(session);
        var output = scorer.Score(batch, TestContext.Current.CancellationToken);
        Assert.Equal(new nint[] { 1, 2 }, output.Logits.Lengths.ToArray());
        Assert.Equal((float)batch.MarkerPositions[0, 0], output.Logits[0, 0]);
        Assert.Equal((float)batch.MarkerPositions[0, 1], output.Logits[0, 1]);
        Assert.Equal(.5f, output.ActionProbabilities[0, 0]);
        Assert.Equal(.5f, output.ActionProbabilities[0, 1]);
    }

    [Fact]
    public void SyntheticOrt_ManagedOutputsSurviveSubsequentCallsAndOwnedSessionDisposal()
    {
        using var options = Options();
        using var session = new InferenceSession(SyntheticOnnx.Create(), options);
        var scorer = new ScoreOnnxDecisionModel(session, ownsSession: true);
        var firstBatch = Prepare().Prepare(Input(Choice()), TestContext.Current.CancellationToken);
        var first = scorer.Score(firstBatch, TestContext.Current.CancellationToken);
        var secondBatch = Prepare().Prepare(Input(new BinaryDecisionQuestion("b", "much longer instruction", "yes", "no")), TestContext.Current.CancellationToken);
        var second = scorer.Score(secondBatch, TestContext.Current.CancellationToken);
        float original = first.Logits[0, 0];
        second.Logits[0, 0] = -99;
        scorer.Dispose();
        scorer.Dispose();
        Assert.Equal((float)firstBatch.MarkerPositions[0, 0], original);
        Assert.Equal(original, first.Logits[0, 0]);
        Assert.Equal(.5f, first.ActionProbabilities[0, 1]);
        Assert.Throws<ObjectDisposedException>(() => scorer.Score(firstBatch, TestContext.Current.CancellationToken));
        // ORT 1.23.2 has no public IsDisposed and running a released native handle is unsafe.
        // Inspect only this pinned-package lifetime flag; never execute a disposed session.
        Assert.True(SessionIsDisposed(session));
        Assert.False(options.IsClosed);
    }

    [Fact]
    public async Task SyntheticOrt_BorrowedScorerAndSessionSurviveGeneratorDisposalAndPreCancel()
    {
        using var options = Options();
        using var session = new InferenceSession(SyntheticOnnx.Create(), options);
        using var scorer = new ScoreOnnxDecisionModel(session);
        var batch = Prepare().Prepare(Input(Choice()), TestContext.Current.CancellationToken);
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        Assert.ThrowsAny<OperationCanceledException>(() => scorer.Score(batch, cancellation.Token));
        var generator = new LayaDecisionGenerator(Prepare(), scorer, new(Profile()));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => generator.GenerateAsync(Input(Choice()), cancellationToken: cancellation.Token));
        generator.Dispose();
        var output = scorer.Score(batch, TestContext.Current.CancellationToken);
        Assert.Equal((float)batch.MarkerPositions[0, 1], output.Logits[0, 1]);
        scorer.Dispose();
        Assert.False(SessionIsDisposed(session));
        using var borrowedAgain = new ScoreOnnxDecisionModel(session);
        Assert.Equal(.5f, borrowedAgain.Score(batch, TestContext.Current.CancellationToken).ActionProbabilities[0, 0]);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => generator.GenerateAsync(Input(Choice()), cancellationToken: TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task Generator_PreparesEntireBatchBeforeTouchingNativeScorer()
    {
        using var options = Options();
        using var session = new InferenceSession(SyntheticOnnx.Create(), options);
        using var scorer = new ScoreOnnxDecisionModel(session);
        scorer.Dispose(); // Any attempt to execute even the first question would throw ObjectDisposedException.
        var tokenizer = new CharacterTokenizer();
        using var generator = new LayaDecisionGenerator(Prepare(tokenizer), scorer, new(Profile()));
        var input = Input(Choice(), new BinaryDecisionQuestion("invalid-later", "[MASK]"));
        var error = await Assert.ThrowsAsync<NotSupportedException>(() => generator.GenerateAsync(input, cancellationToken: TestContext.Current.CancellationToken));
        Assert.Contains("Reserved token input", error.Message, StringComparison.Ordinal);
        Assert.Contains("choice question: choose", tokenizer.Calls);
        Assert.DoesNotContain(tokenizer.Calls, text => text.Contains("[MASK]", StringComparison.Ordinal));
    }

    [Fact]
    public async Task Generator_MetadataFixedModelAndOwnedScorerLifetime()
    {
        using var options = Options();
        using var session = new InferenceSession(SyntheticOnnx.Create(), options);
        using var scorer = new ScoreOnnxDecisionModel(session);
        var tokenizer = new CharacterTokenizer();
        var generator = new LayaDecisionGenerator(Prepare(tokenizer), scorer, new(Profile()), ownsScorer: true);
        var metadata = Assert.IsType<DecisionGeneratorMetadata>(generator.GetService(typeof(DecisionGeneratorMetadata)));
        Assert.Equal("laya", metadata.ProviderName);
        Assert.Null(metadata.ProviderUri);
        Assert.Equal(LayaProfile.ModelId, metadata.DefaultModelId);
        Assert.Equal([DecisionKind.Binary, DecisionKind.Choice, DecisionKind.Score], metadata.SupportedKinds);
        Assert.Same(generator, generator.GetService(typeof(IDecisionGenerator)));
        Assert.Null(generator.GetService(typeof(IDecisionGenerator), "key"));
        Assert.Null(generator.GetService(typeof(string)));
        Assert.Equal("serviceType", Assert.Throws<ArgumentNullException>(() => generator.GetService(null!)).ParamName);
        await Assert.ThrowsAsync<NotSupportedException>(() => generator.GenerateAsync(Input(Choice()), new() { ModelId = "other" }, TestContext.Current.CancellationToken));
        Assert.Empty(tokenizer.Calls);
        var callOptions = new DecisionGenerationOptions { ModelId = LayaProfile.ModelId, AdditionalProperties = new() { ["ignored"] = 7 } };
        var answer = await generator.GenerateAsync(Input(Choice()), callOptions, TestContext.Current.CancellationToken);
        var raw = Assert.IsType<LayaRawAnswer>(answer.Answers[0].RawRepresentation);
        var originalLogits = raw.Logits.ToArray();
        var defaultResult = await generator.GenerateAsync(Input(new BinaryDecisionQuestion("b", "different longer instructions")),
            cancellationToken: TestContext.Current.CancellationToken);
        generator.Dispose();
        Assert.Equal("A", Assert.IsType<ChoiceDecisionAnswer>(answer.Answers[0]).SelectedCandidateId);
        Assert.Equal(LayaProfile.ModelId, answer.ModelId);
        Assert.Equal(LayaProfile.ModelId, defaultResult.ModelId);
        Assert.Equal(LayaProfile.ModelId, metadata.DefaultModelId);
        Assert.Equal(LayaProfile.ModelId, callOptions.ModelId);
        Assert.Equal(7, callOptions.AdditionalProperties!["ignored"]);
        Assert.Equal("laya", raw.Provider);
        Assert.Equal(1, raw.Temperature);
        Assert.Equal(originalLogits, raw.Logits);
        Assert.Equal(new[] { .5, .5 }, raw.ActionProbabilities);
        var nextRaw = Assert.IsType<LayaRawAnswer>(defaultResult.Answers[0].RawRepresentation);
        Assert.NotSame(raw.Logits, nextRaw.Logits);
        Assert.NotEqual(raw.Logits[0], nextRaw.Logits[0]);
        Assert.Equal(new[] { .5, .5 }, nextRaw.ActionProbabilities);
        Assert.Throws<ObjectDisposedException>(() => scorer.Score(Prepare().Prepare(Input(Choice()), TestContext.Current.CancellationToken), TestContext.Current.CancellationToken));
    }

    [Theory]
    [InlineData("other")]
    [InlineData("case")]
    [InlineData("")]
    [InlineData(" \t")]
    public async Task Generator_InvalidModelsRejectBeforeTokenizationOrNativeExecution(string model)
    {
        if (model == "case") model = LayaProfile.ModelId.ToUpperInvariant();
        using var options = Options();
        using var session = new InferenceSession(SyntheticOnnx.Create(), options);
        using var scorer = new ScoreOnnxDecisionModel(session);
        scorer.Dispose(); // Reaching native scoring would instead throw ObjectDisposedException.
        var tokenizer = new CharacterTokenizer();
        using var generator = new LayaDecisionGenerator(Prepare(tokenizer), scorer, new(Profile()));
        var callOptions = new DecisionGenerationOptions { ModelId = model };
        if (string.IsNullOrWhiteSpace(model))
            await Assert.ThrowsAsync<ArgumentException>(() => generator.GenerateAsync(Input(Choice()), callOptions, TestContext.Current.CancellationToken));
        else
        {
            var error = await Assert.ThrowsAsync<NotSupportedException>(() => generator.GenerateAsync(Input(Choice()), callOptions, TestContext.Current.CancellationToken));
            Assert.Equal("This local generator cannot switch its loaded model.", error.Message);
        }
        Assert.Empty(tokenizer.Calls);
        Assert.Equal(model, callOptions.ModelId);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Load_UsesLocalSyntheticGraphWithBorrowedOrDefaultSessionOptions(bool borrowed)
    {
        using var file = new TemporaryFixture(SyntheticOnnx.Create());
        using var options = Options();
        var scorer = ScoreOnnxDecisionModel.Load(file.Path, borrowed ? options : null);
        var batch = Prepare().Prepare(Input(Choice()), TestContext.Current.CancellationToken);
        var output = scorer.Score(batch, TestContext.Current.CancellationToken);
        scorer.Dispose();
        Assert.False(options.IsClosed);
        Assert.Equal((float)batch.MarkerPositions[0, 0], output.Logits[0, 0]);
        Assert.Equal(.5f, output.ActionProbabilities[0, 1]);
        Assert.Throws<ObjectDisposedException>(() => scorer.Score(batch, TestContext.Current.CancellationToken));
        using var another = new InferenceSession(SyntheticOnnx.Create(), options);
        Assert.Equal(5, another.InputMetadata.Count);
    }

    private static void AssertEquivalent(DecisionResult expected, DecisionResult actual)
    {
        Assert.Equal(expected.ModelId, actual.ModelId);
        Assert.Equal(expected.Usage!.InputTokenCount, actual.Usage!.InputTokenCount);
        Assert.Equal(expected.Answers.Select(a => a.QuestionId), actual.Answers.Select(a => a.QuestionId));
        for (int i = 0; i < expected.Answers.Count; i++)
        {
            Assert.Equal(expected.Answers[i].Kind, actual.Answers[i].Kind);
            var expectedRaw = Assert.IsType<LayaRawAnswer>(expected.Answers[i].RawRepresentation);
            var actualRaw = Assert.IsType<LayaRawAnswer>(actual.Answers[i].RawRepresentation);
            Assert.Equal(expectedRaw.Provider, actualRaw.Provider);
            Assert.Equal(expectedRaw.Temperature, actualRaw.Temperature);
            Assert.Equal(expectedRaw.Logits, actualRaw.Logits);
            Assert.Equal(expectedRaw.ActionProbabilities, actualRaw.ActionProbabilities);
            switch (expected.Answers[i])
            {
                case BinaryDecisionAnswer binary:
                    Assert.Equal(binary.TrueProbability, Assert.IsType<BinaryDecisionAnswer>(actual.Answers[i]).TrueProbability);
                    break;
                case ChoiceDecisionAnswer choice:
                    var other = Assert.IsType<ChoiceDecisionAnswer>(actual.Answers[i]);
                    Assert.Equal(choice.SelectedCandidateId, other.SelectedCandidateId);
                    Assert.Equal(choice.Probabilities, other.Probabilities);
                    break;
                case ScoreDecisionAnswer score:
                    Assert.Equal(score.Probabilities, Assert.IsType<ScoreDecisionAnswer>(actual.Answers[i]).Probabilities);
                    Assert.Equal(score.ExpectedIndex, ((ScoreDecisionAnswer)actual.Answers[i]).ExpectedIndex);
                    break;
            }

        }
    }

    private static bool SessionIsDisposed(InferenceSession session) =>
        (bool)typeof(InferenceSession).GetField("_disposed", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(session)!;
}
