using System.Numerics.Tensors;
using DecisionInference.Laya;
using Microsoft.ML.OnnxRuntime;
using static DecisionInference.Tests.LayaStagesTests;

namespace DecisionInference.Tests;

public sealed class LayaBatchTests
{
    private static SessionOptions Options() => new() { IntraOpNumThreads = 1, InterOpNumThreads = 1, LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR };
    private static DecisionInput State(int length) => new(System.Text.Json.JsonSerializer.SerializeToElement(new string('a', length)), Mixed().Questions);

    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(16)]
    [InlineData(32)]
    public async Task DynamicGraph_BulkPreservesScalarValuesCorrelationPrecisionAndOwnedOutputs(int count)
    {
        using var options = Options();
        using var session = new InferenceSession(SyntheticOnnx.Create(), options);
        using var scorer = new ScoreOnnxDecisionModel(session);
        using var generator = new LayaDecisionGenerator(Prepare(), scorer, new(Profile()));
        var inputs = Enumerable.Range(0, count).Select(i => State(i % 23 + 1)).ToArray();
        var expected = new List<DecisionResult>();
        foreach (var input in inputs) expected.Add(await generator.GenerateAsync(input, cancellationToken: TestContext.Current.CancellationToken));
        var callOptions = new DecisionGenerationOptions { ModelId = LayaProfile.ModelId, AdditionalProperties = new() { ["retained"] = 7 } };
        var actual = await generator.GenerateBatchAsync(inputs, callOptions, TestContext.Current.CancellationToken);
        Assert.Equal(count, actual.Count);
        for (int i = 0; i < count; i++) Equal(expected[i], actual[i]);
        var retained = Assert.IsType<LayaRawAnswer>(actual[0].Answers[0].RawRepresentation).Logits.ToArray();
        _ = await generator.GenerateBatchAsync([State(40)], cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(retained, Assert.IsType<LayaRawAnswer>(actual[0].Answers[0].RawRepresentation).Logits);
        Assert.Equal(7, callOptions.AdditionalProperties!["retained"]);
        Assert.Equal(LayaProfile.ModelId, callOptions.ModelId);
    }

    [Fact]
    public async Task StaticCombinedRows_ProvesNativeTensorBatchInsteadOfRepeatedScalarCallsAndHandlesFinalShortBatch()
    {
        var graphInputs = SyntheticOnnx.Inputs().Select(i => i with
        {
            Shape = i.Name == "qtype" ? [6] : [6, -1]
        }).ToArray();
        using var options = Options();
        using var session = new InferenceSession(SyntheticOnnx.Create(graphInputs), options);
        using var scorer = new ScoreOnnxDecisionModel(session);
        using var generator = new LayaDecisionGenerator(Prepare(), scorer, new(Profile()));
        var result = await generator.GenerateBatchAsync([State(1), State(15)], cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(2, result.Count);
        Assert.Equal(["opaque [MASK] choice", "score", "binary"], result[0].Answers.Select(a => a.QuestionId));
        await Assert.ThrowsAsync<ArgumentException>(() => generator.GenerateAsync(State(1), cancellationToken: TestContext.Current.CancellationToken));
        await Assert.ThrowsAsync<ArgumentException>(() => generator.GenerateBatchAsync([State(1)], cancellationToken: TestContext.Current.CancellationToken));
        using var dynamicSession = new InferenceSession(SyntheticOnnx.Create(), options);
        using var dynamicScorer = new ScoreOnnxDecisionModel(dynamicSession);
        using var dynamicGenerator = new LayaDecisionGenerator(Prepare(), dynamicScorer, new(Profile()));
        var states = Enumerable.Range(1, 17).Select(State).ToArray();
        var restored = new List<DecisionResult>();
        foreach (var chunk in states.Chunk(16))
            restored.AddRange(await dynamicGenerator.GenerateBatchAsync(chunk, cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal(17, restored.Count);
        for (int i = 0; i < states.Length; i++)
            Equal(await dynamicGenerator.GenerateAsync(states[i], cancellationToken: TestContext.Current.CancellationToken), restored[i]);
    }

    [Fact]
    public void Collator_UsesNonzeroPadPreservesAllMasksTypesPositionsAndOriginalBatches()
    {
        var prepare = new PrepareDecisionInputs(Profile(), new CharacterTokenizer(), new(1, 2, 3, 9, ["[MASK]"]));
        var first = prepare.Prepare(State(1), TestContext.Current.CancellationToken);
        var second = prepare.Prepare(State(31), TestContext.Current.CancellationToken);
        var original = first.Ids.ToArray();
        var combined = LayaBatchCollator.Collate([first, second], 9, TestContext.Current.CancellationToken);
        Assert.Equal(6, combined.Tensors.Rows);
        Assert.Equal(second.SequenceLength, combined.Tensors.Length);
        Assert.Equal(3, combined.Tensors.Width);
        Assert.Equal([0, 3], combined.StateOffsets);
        for (int state = 0; state < 2; state++)
        {
            var input = state == 0 ? first : second;
            for (int q = 0; q < input.BatchSize; q++)
            {
                int target = state * 3 + q;
                Assert.Equal(input.Types[q], combined.Tensors.Types[target]);
                for (int i = 0; i < combined.Tensors.Length; i++)
                {
                    Assert.Equal(i < input.SequenceLength ? input.Ids[q * input.SequenceLength + i] : 9,
                        combined.Tensors.Ids[target * combined.Tensors.Length + i]);
                    Assert.Equal(i < input.SequenceLength ? input.Attention[q * input.SequenceLength + i] : 0,
                        combined.Tensors.Attention[target * combined.Tensors.Length + i]);
                }
                for (int i = 0; i < input.MarkerWidth; i++)
                {
                    Assert.Equal(input.Positions[q * input.MarkerWidth + i], combined.Tensors.Positions[target * combined.Tensors.Width + i]);
                    Assert.Equal(input.Masks[q * input.MarkerWidth + i], combined.Tensors.Masks[target * combined.Tensors.Width + i]);
                }
            }
        }
        Assert.Equal(original, first.Ids);
        Assert.Same(first, combined.Inputs[0]);
    }

    [Fact]
    public void CombinedOutputSlices_ExcludePaddedNonfiniteOptionsAndRejectNonfiniteValidOptions()
    {
        var prepared = Prepare().Prepare(State(3), TestContext.Current.CancellationToken);
        var combined = LayaBatchCollator.Collate([prepared], 0, TestContext.Current.CancellationToken);
        var outputs = new LayaModelOutputs(Tensor.Create<float>([0, 1, float.NaN, 0, 1, 2, 0, 1, float.NaN], [3, 3]),
            Tensor.Create<float>([.5f, .5f, .5f, .5f, .5f, .5f], [3, 2]));
        var slice = combined.Slice(outputs, 0);
        var result = new DecodeDecisions(Profile()).Decode(prepared, slice);
        Assert.Equal(2, Assert.IsType<ChoiceDecisionAnswer>(result.Answers[0]).Probabilities.Count);
        Assert.Equal(3, Assert.IsType<ScoreDecisionAnswer>(result.Answers[1]).Probabilities.Count);
        outputs.Logits[0, 0] = float.PositiveInfinity;
        Assert.Equal(0, slice.Logits[0, 0]);
        Assert.Throws<DecisionProtocolException>(() => new DecodeDecisions(Profile()).Decode(prepared, combined.Slice(outputs, 0)));
    }

    [Fact]
    public void Collator_CheckedBudgetRejectsBeforeCombinedAllocationAndPreparationHasItsOwnBound()
    {
        var hugeShape = new LayaInputBatch(State(1), int.MaxValue, 3, [], [], [], [], [], []);
        var error = Assert.Throws<ArgumentException>(() => LayaBatchCollator.Collate([hugeShape], 0, TestContext.Current.CancellationToken));
        Assert.Contains("64 MiB", error.Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentException>(() => Prepare().PrepareBounded(State(1), 1, TestContext.Current.CancellationToken));
        Assert.Equal(64L * 1024 * 1024, LayaBatchCollator.MaximumManagedTensorBytes);
    }

    [Fact]
    public async Task BulkRejectsChangedDefinitionsModelsEmptyAndReservedInputsBeforeNativeAndRecovers()
    {
        using var options = Options();
        using var session = new InferenceSession(SyntheticOnnx.Create(), options);
        using var scorer = new ScoreOnnxDecisionModel(session);
        var tokenizer = new CharacterTokenizer();
        using var generator = new LayaDecisionGenerator(Prepare(tokenizer), scorer, new(Profile()));
        await Assert.ThrowsAsync<ArgumentException>(() => generator.GenerateBatchAsync([], cancellationToken: TestContext.Current.CancellationToken));
        var changed = new DecisionInput(State(1).State, [
            new ChoiceDecisionQuestion("opaque [MASK] choice", "pick", [new("opaque-A", "changed"), new("opaque-B", "blue")]),
            Mixed().Questions[1], Mixed().Questions[2]]);
        await Assert.ThrowsAsync<ArgumentException>(() => generator.GenerateBatchAsync([State(1), changed], cancellationToken: TestContext.Current.CancellationToken));
        await Assert.ThrowsAsync<NotSupportedException>(() => generator.GenerateBatchAsync([State(1)], new() { ModelId = "other" }, TestContext.Current.CancellationToken));
        Assert.Empty(tokenizer.Calls);
        var reserved = new DecisionInput(System.Text.Json.JsonSerializer.SerializeToElement("[MASK]"), Mixed().Questions);
        await Assert.ThrowsAsync<NotSupportedException>(() => generator.GenerateBatchAsync([State(1), reserved], cancellationToken: TestContext.Current.CancellationToken));
        var result = await generator.GenerateBatchAsync([State(1)], cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(3, Assert.Single(result).Answers.Count);
    }

    [Fact]
    public async Task CancelledWaitInFlightPreparationAndDisposalAreCoordinatedAndGateRecovers()
    {
        using var options = Options();
        using var session = new InferenceSession(SyntheticOnnx.Create(), options);
        using var scorer = new ScoreOnnxDecisionModel(session);
        using var release = new ManualResetEventSlim();
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        int calls = 0;
        var tokenizer = new CharacterTokenizer { Encode = text =>
        {
            if (Interlocked.Increment(ref calls) == 1)
            {
                entered.SetResult();
                if (!release.Wait(TimeSpan.FromSeconds(30), TestContext.Current.CancellationToken)) throw new TimeoutException("Test preparation was not released.");
            }
            return text.Select(c => c + 100).ToArray();
        }};
        var generator = new LayaDecisionGenerator(Prepare(tokenizer), scorer, new(Profile()), ownsScorer: true);
        using var inFlightCancellation = new CancellationTokenSource();
        var first = Task.Run(() => generator.GenerateBatchAsync([State(1)], cancellationToken: inFlightCancellation.Token), TestContext.Current.CancellationToken);
        try
        {
            await entered.Task.WaitAsync(TimeSpan.FromSeconds(30), TestContext.Current.CancellationToken);
            using var waitingCancellation = new CancellationTokenSource();
            var waiting = Task.Run(() => generator.GenerateBatchAsync([State(2)], cancellationToken: waitingCancellation.Token), TestContext.Current.CancellationToken);
            waitingCancellation.Cancel();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => waiting);
            inFlightCancellation.Cancel();
            release.Set();
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => first);
            var recovered = await generator.GenerateBatchAsync([State(2)], cancellationToken: TestContext.Current.CancellationToken);
            Assert.Equal(3, Assert.Single(recovered).Answers.Count);
            generator.Dispose();
            generator.Dispose();
            await Assert.ThrowsAsync<ObjectDisposedException>(() => generator.GenerateBatchAsync([State(1)], cancellationToken: TestContext.Current.CancellationToken));
            Assert.Throws<ObjectDisposedException>(() => scorer.Score(Prepare().Prepare(State(1), TestContext.Current.CancellationToken), TestContext.Current.CancellationToken));
        }
        finally { release.Set(); generator.Dispose(); }
    }

    [Fact]
    public async Task DisposalWaitsForActiveBulkPreparationBeforeReleasingOwnedScorer()
    {
        using var options = Options();
        using var session = new InferenceSession(SyntheticOnnx.Create(), options);
        using var scorer = new ScoreOnnxDecisionModel(session);
        using var release = new ManualResetEventSlim();
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var disposing = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        int calls = 0;
        var tokenizer = new CharacterTokenizer { Encode = text =>
        {
            if (Interlocked.Increment(ref calls) == 1)
            {
                entered.SetResult();
                if (!release.Wait(TimeSpan.FromSeconds(30), TestContext.Current.CancellationToken)) throw new TimeoutException("Test preparation was not released.");
            }
            return text.Select(c => c + 100).ToArray();
        }};
        var generator = new LayaDecisionGenerator(Prepare(tokenizer), scorer, new(Profile()), ownsScorer: true);
        var active = Task.Run(() => generator.GenerateBatchAsync([State(1), State(2)], cancellationToken: TestContext.Current.CancellationToken),
            TestContext.Current.CancellationToken);
        try
        {
            await entered.Task.WaitAsync(TimeSpan.FromSeconds(30), TestContext.Current.CancellationToken);
            var dispose = Task.Run(() => { disposing.SetResult(); generator.Dispose(); }, TestContext.Current.CancellationToken);
            await disposing.Task.WaitAsync(TimeSpan.FromSeconds(30), TestContext.Current.CancellationToken);
            Assert.False(dispose.IsCompleted);
            release.Set();
            Assert.Equal(2, (await active).Count);
            await dispose;
            await Assert.ThrowsAsync<ObjectDisposedException>(() => generator.GenerateBatchAsync([State(1)], cancellationToken: TestContext.Current.CancellationToken));
            Assert.Throws<ObjectDisposedException>(() => scorer.Score(Prepare().Prepare(State(1), TestContext.Current.CancellationToken), TestContext.Current.CancellationToken));
        }
        finally { release.Set(); generator.Dispose(); }
    }

    private static void Equal(DecisionResult expected, DecisionResult actual)
    {
        Assert.Equal(expected.ModelId, actual.ModelId);
        Assert.Equal(expected.Usage!.InputTokenCount, actual.Usage!.InputTokenCount);
        Assert.Equal(expected.Answers.Select(a => a.QuestionId), actual.Answers.Select(a => a.QuestionId));
        for (int i = 0; i < expected.Answers.Count; i++)
        {
            Assert.Equal(expected.Answers[i].Precision, actual.Answers[i].Precision);
            switch (expected.Answers[i])
            {
                case BinaryDecisionAnswer b: Assert.Equal(b.TrueProbability, Assert.IsType<BinaryDecisionAnswer>(actual.Answers[i]).TrueProbability); break;
                case ChoiceDecisionAnswer c:
                    var other = Assert.IsType<ChoiceDecisionAnswer>(actual.Answers[i]);
                    Assert.Equal(c.SelectedCandidateId, other.SelectedCandidateId);
                    Assert.Equal(c.Probabilities, other.Probabilities);
                    break;
                case ScoreDecisionAnswer s:
                    var score = Assert.IsType<ScoreDecisionAnswer>(actual.Answers[i]);
                    Assert.Equal(s.Probabilities, score.Probabilities);
                    Assert.Equal(s.ExpectedIndex, score.ExpectedIndex);
                    Assert.Equal(s.ExpectedIndexLowerBound, score.ExpectedIndexLowerBound);
                    Assert.Equal(s.ExpectedIndexUpperBound, score.ExpectedIndexUpperBound);
                    Assert.Equal(s.NativeScore, score.NativeScore);
                    break;
            }
        }
    }
}
