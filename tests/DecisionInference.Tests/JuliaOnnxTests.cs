using DecisionInference.Julia;
using Microsoft.ML.OnnxRuntime;
using static DecisionInference.Tests.ContractTests;

namespace DecisionInference.Tests;

public sealed class JuliaOnnxTests
{
    private static CancellationToken Cancel => TestContext.Current.CancellationToken;
    private static SessionOptions Options() => new() { IntraOpNumThreads = 1, InterOpNumThreads = 1 };

    [Fact]
    public void Score_ExactFiveInputsAndLogitsOnlyExecute()
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        using var options = Options();
        using var session = new InferenceSession(SyntheticOnnx.CreateJulia(), options);
        using var score = new ScoreOnnxDecisionModel(session);
        Assert.Equal(["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"], session.InputMetadata.Keys);
        Assert.Equal(["logits"], session.OutputMetadata.Keys);
        foreach (var input in session.InputMetadata)
        {
            Assert.Equal(input.Key == "marker_mask" ? typeof(bool) : typeof(long), input.Value.ElementType);
            Assert.Equal(input.Key == "qtype" ? [-1] : new[] { -1, -1 }, input.Value.Dimensions);
        }
        Assert.Equal(typeof(float), session.OutputMetadata["logits"].ElementType);
        Assert.Equal([-1, -1], session.OutputMetadata["logits"].Dimensions);
        var batch = prepare.Prepare(JuliaTestFixture.Mixed(), TestContext.Current.CancellationToken);
        var logits = score.Score(batch, TestContext.Current.CancellationToken);
        Assert.Equal(new nint[] { 3, 3 }, logits.Lengths.ToArray());
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                Assert.Equal((float)batch.MarkerPositions[r, c], logits[r, c]);
        Assert.Equal(0f, logits[0, 2]);
        Assert.Equal(0f, logits[2, 2]);
    }

    [Fact]
    public void Score_WiresAllFiveBuffersToTheirNamedInputs()
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        using var options = Options();
        using var session = new InferenceSession(SyntheticOnnx.CreateJulia(probeInputs: true), options);
        using var score = new ScoreOnnxDecisionModel(session);
        var batch = prepare.Prepare(JuliaTestFixture.Mixed(), Cancel);
        var output = score.Score(batch, Cancel);
        Assert.Equal(new nint[] { 3, 3 }, output.Lengths.ToArray());
        // 10*marker ID(4) + attention(1) + 100*marker mask + 1000*qtype.
        // Padded marker indices address BOS(2), not a mask; its marker mask must be zero.
        float[][] expected = [[141, 141, 21], [1141, 1141, 1141], [2141, 2141, 2021]];
        for (int row = 0; row < 3; row++)
            for (int column = 0; column < 3; column++)
                Assert.Equal(expected[row][column], output[row, column]);
    }

    [Fact]
    public void Score_RejectsWrongInputNamesTypesAndRanks()
    {
        using var options = Options();
        for (int index = 0; index < 5; index++)
        {
            foreach (string change in new[] { "name", "type", "rank" })
            {
                var inputs = SyntheticOnnx.Inputs();
                string name = inputs[index].Name;
                inputs[index] = change switch
                {
                    "name" => inputs[index] with { Name = name.ToUpperInvariant() },
                    "type" => inputs[index] with { ElementType = 1 },
                    _ => inputs[index] with { Shape = index == 4 ? [-1, -1] : [-1] }
                };
                // Constant graph isolates metadata validation from ONNX operator type/shape inference.
                using var session = new InferenceSession(SyntheticOnnx.CreateJulia(inputs, values: [0, 1]), options);
                var error = Assert.Throws<ArgumentException>(() => new ScoreOnnxDecisionModel(session));
                Assert.Contains(change == "name" ? "exactly five named inputs" : $"tensor {name}", error.Message, StringComparison.Ordinal);
            }
        }
        foreach (bool extra in new[] { false, true })
        {
            SyntheticOnnx.Input[] inputs = extra
                ? [.. SyntheticOnnx.Inputs(), new("extra", 7, [-1])]
                : SyntheticOnnx.Inputs().Skip(1).ToArray();
            using var session = new InferenceSession(SyntheticOnnx.CreateJulia(inputs, values: [0, 1]), options);
            Assert.Contains("exactly five named inputs", Assert.Throws<ArgumentException>(() => new ScoreOnnxDecisionModel(session)).Message, StringComparison.Ordinal);
        }
    }

    [Theory]
    [InlineData("name")]
    [InlineData("type")]
    [InlineData("rank")]
    [InlineData("extra-output")]
    public void Score_RejectsWrongOutputMetadata(string change)
    {
        using var options = Options();
        byte[] model = change switch
        {
            "name" => SyntheticOnnx.CreateJulia(outputName: "Logits", values: [0, 1]),
            "type" => SyntheticOnnx.CreateJulia(outputType: 7, values: [0, 1]),
            "rank" => SyntheticOnnx.CreateJulia(outputShape: [2], values: [0, 1]),
            _ => SyntheticOnnx.Create()
        };
        using var session = new InferenceSession(model, options);
        var error = Assert.Throws<ArgumentException>(() => new ScoreOnnxDecisionModel(session));
        Assert.Contains(change is "name" or "extra-output" ? "only the logits output" : "tensor logits", error.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void Score_RejectsIncompatibleStaticDimensionForEveryInput(int inputIndex)
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        using var options = Options();
        var batch = prepare.Prepare(JuliaTestFixture.Mixed(), Cancel);
        long[] matchingShape = inputIndex == 4 ? [3] : [3, inputIndex < 2 ? batch.SequenceLength : 3];
        for (int dimension = 0; dimension < matchingShape.Length; dimension++)
        {
            var inputs = SyntheticOnnx.Inputs();
            long[] wrong = matchingShape.ToArray();
            wrong[dimension]++;
            inputs[inputIndex] = inputs[inputIndex] with { Shape = wrong };
            using var session = new InferenceSession(SyntheticOnnx.CreateJulia(inputs), options);
            using var score = new ScoreOnnxDecisionModel(session);
            Assert.Contains($"static Julia graph input {inputs[inputIndex].Name}",
                Assert.Throws<ArgumentException>(() => score.Score(batch, Cancel)).Message, StringComparison.Ordinal);
        }
        var compatibleInputs = SyntheticOnnx.Inputs();
        compatibleInputs[inputIndex] = compatibleInputs[inputIndex] with { Shape = matchingShape };
        using var compatibleSession = new InferenceSession(SyntheticOnnx.CreateJulia(compatibleInputs), options);
        using var compatible = new ScoreOnnxDecisionModel(compatibleSession);
        Assert.Equal((float)batch.MarkerPositions[1, 2], compatible.Score(batch, Cancel)[1, 2]);
    }

    [Theory]
    [InlineData("rows")]
    [InlineData("width")]
    [InlineData("dynamic-transpose")]
    public void Score_RejectsOutputShapeMismatch(string change)
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        using var options = Options();
        byte[] model = change switch
        {
            "rows" => SyntheticOnnx.CreateJulia(outputShape: [2, 2], values: [0, 1, 2, 3]),
            "width" => SyntheticOnnx.CreateJulia(outputShape: [1, 3], values: [0, 1, 2]),
            _ => SyntheticOnnx.CreateJulia(transpose: true)
        };
        using var session = new InferenceSession(model, options);
        using var score = new ScoreOnnxDecisionModel(session);
        var batch = prepare.Prepare(Request(Choice()), Cancel);
        Assert.Contains("Unexpected Julia logits shape", Assert.Throws<DecisionProtocolException>(() => score.Score(batch, Cancel)).Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void Score_RejectsValidNonfiniteButIgnoresPaddingAndRecovers(float sentinel)
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        using var options = Options();
        using var session = new InferenceSession(SyntheticOnnx.CreateJulia(outputShape: [3, 3],
            values: [0, 1, sentinel, 0, 1, 2, 0, 1, sentinel]), options);
        using var score = new ScoreOnnxDecisionModel(session);
        var mixed = JuliaTestFixture.Mixed();
        var invalid = new DecisionRequest(mixed.State,
            [new ChoiceDecisionQuestion("choice", "pick", [new("a", "a"), new("b", "b"), new("c", "c")]),
             mixed.Questions[1], mixed.Questions[2]]);
        var invalidBatch = prepare.Prepare(invalid, Cancel);
        Assert.Contains("Nonfinite Julia valid-option", Assert.Throws<DecisionProtocolException>(() => score.Score(invalidBatch, Cancel)).Message, StringComparison.Ordinal);
        var validBatch = prepare.Prepare(mixed, Cancel);
        var output = score.Score(validBatch, Cancel);
        Assert.Equal(sentinel, output[0, 2]);
        Assert.Equal(sentinel, output[2, 2]);
        Assert.Equal(2, output[1, 2]);
        var decoded = new DecodeDecisions().Decode(validBatch, output);
        Assert.Equal(.7310585786300049, Assert.IsType<BinaryDecisionAnswer>(decoded.Answers[2]).TrueProbability, 14);
        Assert.Equal("A", Assert.IsType<ChoiceDecisionAnswer>(decoded.Answers[0]).SelectedCandidateId);
        using var lateSession = new InferenceSession(SyntheticOnnx.CreateJulia(outputShape: [3, 3],
            values: [0, 1, 0, 0, 1, 2, 0, sentinel, 0]), options);
        using var lateScore = new ScoreOnnxDecisionModel(lateSession);
        Assert.Contains("Nonfinite Julia valid-option", Assert.Throws<DecisionProtocolException>(() =>
            lateScore.Score(validBatch, Cancel)).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Score_ManagedOutputsSurviveLaterRunsMutationAndOwnedSessionDisposal()
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        using var options = Options();
        using var session = new InferenceSession(SyntheticOnnx.CreateJulia(), options);
        using var score = new ScoreOnnxDecisionModel(session, ownsSession: true);
        var firstBatch = prepare.Prepare(Request(Choice()), Cancel);
        var first = score.Score(firstBatch, Cancel);
        var response = new DecodeDecisions().Decode(firstBatch, first);
        var raw = Assert.IsType<JuliaRawAnswer>(response.Answers[0].RawRepresentation);
        double[] expected = [(float)firstBatch.MarkerPositions[0, 0], (float)firstBatch.MarkerPositions[0, 1]];
        var secondBatch = prepare.Prepare(Request(new BinaryDecisionQuestion("b", "much longer instruction", "yes", "no")), Cancel);
        var second = score.Score(secondBatch, Cancel);
        Assert.NotEqual(first[0, 0], second[0, 0]);
        Assert.Equal((float)secondBatch.MarkerPositions[0, 1], second[0, 1]);
        second[0, 0] = -99;
        score.Dispose();
        score.Dispose();
        Assert.True(SessionIsDisposed(session));
        Assert.False(options.IsClosed);
        for (int i = 0; i < 2; i++) Assert.Equal(expected[i], first[0, i]);
        Assert.Equal(expected, raw.Logits);
        first[0, 0] = -77;
        Assert.Equal(expected, raw.Logits);
        Assert.Throws<ObjectDisposedException>(() => score.Score(firstBatch, Cancel));
    }

    [Theory]
    [InlineData("cardinality")]
    [InlineData("one-sided")]
    [InlineData("marker")]
    [InlineData("head")]
    public async Task Client_PreflightsEntireMixedBatchBeforeIncompatibleScorer(string invalidKind)
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        using var options = Options();
        var inputs = SyntheticOnnx.Inputs();
        inputs[0] = inputs[0] with { Shape = [99, -1] };
        using var session = new InferenceSession(SyntheticOnnx.CreateJulia(inputs), options);
        using var score = new ScoreOnnxDecisionModel(session);
        using var client = new JuliaDecisionClient(prepare, score, new());
        DecisionQuestion late = invalidKind switch
        {
            "cardinality" => new ScoreDecisionQuestion("late", "rate", Enumerable.Repeat("a", 21)),
            "one-sided" => new BinaryDecisionQuestion("late", "b", "yes", null),
            "marker" => new BinaryDecisionQuestion("late", "raw <mask>"),
            _ => new BinaryDecisionQuestion("late", new('a', 300))
        };
        var request = Request(Choice(), Score(), late);
        string expected = invalidKind switch
        {
            "cardinality" => "2-20", "one-sided" => "both descriptions or neither",
            "marker" => "Reserved Julia token in instructions", _ => "instructions token budget"
        };
        Exception error = invalidKind is "one-sided" or "marker"
            ? await Assert.ThrowsAsync<NotSupportedException>(() => client.GetResponseAsync(request, cancellationToken: Cancel))
            : await Assert.ThrowsAsync<ArgumentException>(() => client.GetResponseAsync(request, cancellationToken: Cancel));
        Assert.Contains(expected, error.Message, StringComparison.Ordinal);
        // Positive control: reaching scoring with valid input must hit the deliberately incompatible graph.
        Assert.Contains("static Julia graph input input_ids",
            (await Assert.ThrowsAsync<ArgumentException>(() => client.GetResponseAsync(JuliaTestFixture.Mixed(), cancellationToken: Cancel))).Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Client_BorrowedStagesAndSessionSurviveDisposeAndPreCancel()
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        using var options = Options();
        using var session = new InferenceSession(SyntheticOnnx.CreateJulia(), options);
        using var score = new ScoreOnnxDecisionModel(session);
        using var client = new JuliaDecisionClient(prepare, score, new());
        var batch = prepare.Prepare(Request(Choice()), Cancel);
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        Assert.ThrowsAny<OperationCanceledException>(() => score.Score(batch, cancelled.Token));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => client.GetResponseAsync(Request(Choice()), cancellationToken: cancelled.Token));
        await Assert.ThrowsAsync<NotSupportedException>(() => client.GetResponseAsync(Request(new BinaryDecisionQuestion("bad", "<mask>")), cancellationToken: Cancel));
        var response = await client.GetResponseAsync(Request(Choice()), cancellationToken: Cancel);
        Assert.Equal("A", Assert.IsType<ChoiceDecisionAnswer>(response.Answers[0]).SelectedCandidateId);
        client.Dispose();
        client.Dispose();
        Assert.Equal(2, prepare.Prepare(Request(Choice()), Cancel).MarkerWidth);
        Assert.Equal((float)batch.MarkerPositions[0, 1], score.Score(batch, Cancel)[0, 1]);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => client.GetResponseAsync(Request(Choice()), cancellationToken: Cancel));
        score.Dispose();
        score.Dispose();
        Assert.False(SessionIsDisposed(session));
        using var reused = new ScoreOnnxDecisionModel(session);
        Assert.Equal((float)batch.MarkerPositions[0, 0], reused.Score(batch, Cancel)[0, 0]);
    }

    [Fact]
    public async Task Client_OwnedStagesDisposeAndFixedModelServiceMetadataRemainExact()
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        using var options = Options();
        using var session = new InferenceSession(SyntheticOnnx.CreateJulia(), options);
        using var score = new ScoreOnnxDecisionModel(session, ownsSession: true);
        using var client = new JuliaDecisionClient(prepare, score, new(), ownsStages: true);
        var metadata = Assert.IsType<DecisionClientMetadata>(client.GetService(typeof(DecisionClientMetadata)));
        Assert.Equal("julia", metadata.ProviderName);
        Assert.Equal(JuliaDecisionClient.ModelId, metadata.DefaultModelId);
        Assert.Null(metadata.ProviderUri);
        Assert.Equal([DecisionKind.Binary, DecisionKind.Choice, DecisionKind.Score], metadata.SupportedKinds);
        Assert.Same(client, client.GetService(typeof(IDecisionClient)));
        Assert.Same(client, client.GetService(typeof(JuliaDecisionClient)));
        Assert.Null(client.GetService(typeof(IDecisionClient), "key"));
        Assert.Null(client.GetService(typeof(string)));
        Assert.Equal("serviceType", Assert.Throws<ArgumentNullException>(() => client.GetService(null!)).ParamName);
        foreach (string model in new[] { "other", JuliaDecisionClient.ModelId.ToLowerInvariant(), JuliaDecisionClient.ModelId + " " })
            await Assert.ThrowsAsync<NotSupportedException>(() => client.GetResponseAsync(Request(Choice()), new() { ModelId = model }, Cancel));
        var callOptions = new DecisionOptions { ModelId = JuliaDecisionClient.ModelId, AdditionalProperties = new() { ["ignored"] = 7 } };
        var response = await client.GetResponseAsync(Request(Choice()), callOptions, Cancel);
        var defaultResponse = await client.GetResponseAsync(Request(Binary()), cancellationToken: Cancel);
        Assert.Equal(JuliaDecisionClient.ModelId, response.ModelId);
        Assert.Equal(JuliaDecisionClient.ModelId, defaultResponse.ModelId);
        Assert.Equal(JuliaDecisionClient.ModelId, callOptions.ModelId);
        Assert.Equal(7, callOptions.AdditionalProperties!["ignored"]);
        Assert.Null(response.Usage);
        Assert.Equal("A", Assert.IsType<ChoiceDecisionAnswer>(response.Answers[0]).SelectedCandidateId);
        var batch = prepare.Prepare(Request(Choice()), Cancel);
        client.Dispose();
        client.Dispose();
        Assert.Throws<ObjectDisposedException>(() => prepare.Prepare(Request(Choice()), Cancel));
        Assert.Throws<ObjectDisposedException>(() => score.Score(batch, Cancel));
        Assert.True(SessionIsDisposed(session));
        Assert.Equal("julia", metadata.ProviderName);
    }

    [Fact]
    public async Task Client_RepeatedConcurrentCallsAreDeterministicAndOwnRawSnapshots()
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        using var options = Options();
        using var session = new InferenceSession(SyntheticOnnx.CreateJulia(), options);
        using var score = new ScoreOnnxDecisionModel(session);
        using var client = new JuliaDecisionClient(prepare, score, new());
        var request = JuliaTestFixture.Mixed();
        var batch = prepare.Prepare(request, Cancel);
        // Different requests exercise tokenizer/stage reuse too, not just cached identical inputs.
        var requests = new[] { request, Request(new BinaryDecisionQuestion("other", "longer different instruction", "yes", "no")) };
        var baseline = await client.GetResponseAsync(request, cancellationToken: Cancel);
        Assert.Equal("A", Assert.IsType<ChoiceDecisionAnswer>(baseline.Answers[0]).SelectedCandidateId);
        for (int repetition = 0; repetition < 2; repetition++)
        {
            var responses = await Task.WhenAll(Enumerable.Range(0, 8).Select(i =>
                Task.Run(() => client.GetResponseAsync(requests[i % 2], cancellationToken: Cancel), Cancel)));
            for (int i = 0; i < responses.Length; i++)
            {
                var response = responses[i];
                Assert.Equal(JuliaDecisionClient.ModelId, response.ModelId);
                Assert.Null(response.Usage);
                if (i % 2 != 0)
                {
                    Assert.Equal("other", Assert.Single(response.Answers).QuestionId);
                    Assert.Equal(.9820137900379085, Assert.IsType<BinaryDecisionAnswer>(response.Answers[0]).TrueProbability, 14);
                    Assert.Equal(new double[] { 46, 50 }, Assert.IsType<JuliaRawAnswer>(response.Answers[0].RawRepresentation).Logits);
                    continue;
                }
                Assert.Equal(["choice", "score", "binary"], response.Answers.Select(a => a.QuestionId));
                for (int row = 0; row < 3; row++)
                {
                    var raw = Assert.IsType<JuliaRawAnswer>(response.Answers[row].RawRepresentation);
                    int width = row == 1 ? 3 : 2;
                    Assert.Equal(Enumerable.Range(0, width).Select(c => (double)batch.MarkerPositions[row, c]), raw.Logits);
                    Assert.NotSame(baseline.Answers[row].RawRepresentation, raw);
                    Assert.Equal(DecisionPrecision.HighPrecision, response.Answers[row].Precision);
                }
                Assert.Equal(Assert.IsType<ChoiceDecisionAnswer>(baseline.Answers[0]).Probabilities,
                    Assert.IsType<ChoiceDecisionAnswer>(response.Answers[0]).Probabilities);
                Assert.Equal(Assert.IsType<ScoreDecisionAnswer>(baseline.Answers[1]).ExpectedIndex,
                    Assert.IsType<ScoreDecisionAnswer>(response.Answers[1]).ExpectedIndex);
                Assert.Equal(Assert.IsType<BinaryDecisionAnswer>(baseline.Answers[2]).TrueProbability,
                    Assert.IsType<BinaryDecisionAnswer>(response.Answers[2]).TrueProbability);
            }
        }
    }

    [Fact]
    public async Task Client_PathConstructorLoadsLocalSyntheticModelAndOwnsStages()
    {
        using var fixture = new JuliaTestFixture();
        using var options = Options();
        string modelPath = Path.Combine(Path.GetTempPath(), $"julia-{Guid.NewGuid():N}.onnx");
        File.WriteAllBytes(modelPath, SyntheticOnnx.CreateJulia());
        try
        {
            using var client = new JuliaDecisionClient(modelPath, fixture.Path, options);
            var response = await client.GetResponseAsync(Request(Choice()), cancellationToken: Cancel);
            Assert.Equal("A", Assert.IsType<ChoiceDecisionAnswer>(response.Answers[0]).SelectedCandidateId);
            Assert.Equal(JuliaDecisionClient.ModelId, response.ModelId);
            client.Dispose();
            client.Dispose();
            Assert.False(options.IsClosed);
            await Assert.ThrowsAsync<ObjectDisposedException>(() => client.GetResponseAsync(Request(Choice()), cancellationToken: Cancel));
        }
        finally { File.Delete(modelPath); }
    }

    // ORT 1.23.2 exposes no public IsDisposed. Match the existing suite's pinned lifetime probe;
    // never call Run against a freed handle. No reflection into Julia or its tokenizer.
    private static bool SessionIsDisposed(InferenceSession session) =>
        (bool)typeof(InferenceSession).GetField("_disposed",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(session)!;
}
