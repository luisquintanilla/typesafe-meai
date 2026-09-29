using System.Text.Json;
using Apache.Arrow;
using Apache.Arrow.Types;
using DecisionInference.Arrow;
using static DecisionInference.Tests.DecisionArrowTestSupport;

namespace DecisionInference.Tests;

public sealed class DecisionArrowFidelityTests
{
    [Fact]
    public async Task WriteRead_PreservesOpaqueWhitespaceCaseAndUnicodeIds()
    {
        using var temp = new TempDirectory();
        var identity = new DecisionArrowIdentity(" Provider ", " Model ",
            [KeyValuePair.Create(" Assets/model ", new string('7', 64))], " Prep ", " Decoder ",
            [" q ", "q[0]", "q[2]", "Q[2]"]);
        var contract = new DecisionArrowSchema(
        [
            new BinaryDecisionQuestion(" q ", " Decide "),
            new ChoiceDecisionQuestion("q", " Pick ",
                [new(" a ", " Spaced "), new("A", "Upper"), new("\u00e9", "Composed"), new("e\u0301", "Decomposed")]),
            new ScoreDecisionQuestion("Q", " Rate ", [" Low ", " Mid ", " High "])
        ], DecisionPrecision.HighPrecision, identity);
        var input = new DecisionInput(JsonSerializer.SerializeToElement(new { value = 42 }), contract.Questions);
        var result = new DecisionResult(input, " Model ",
        [
            new ScoreDecisionAnswer("Q", [.25, .5, .25], 1),
            new ChoiceDecisionAnswer("q", " a ",
                [KeyValuePair.Create("e\u0301", .1), KeyValuePair.Create("\u00e9", .1),
                    KeyValuePair.Create("A", .7), KeyValuePair.Create(" a ", .1)]),
            new BinaryDecisionAnswer(" q ", .125)
        ]);
        await DecisionArrowWriter.WriteAsync(temp.Dataset, contract, Rows([new(7, input, result)]), Provenance(),
            cancellationToken: Token);
        using var reader = await DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, [7L], Token);
        using var batch = await reader.ReadNextRecordBatchAsync(Token);
        Assert.Equal(["row_id", " q ", "q", "Q"], batch.Schema.FieldsList.Select(f => f.Name).ToArray());
        Assert.Equal(7L, ((Int64Array)batch.Column(0)).GetValue(0));
        Assert.Equal(.125, ((DoubleArray)batch.Column(1)).GetValue(0));
        var choice = (StructArray)batch.Column(2);
        Assert.Equal(0, ((Int32Array)choice.Fields[0]).GetValue(0)); // Spaced ID, not argmax "A".
        Assert.Equal([.1, .7, .1, .1], ((DoubleArray)((FixedSizeListArray)choice.Fields[1]).Values).Values.ToArray());
        Assert.Equal([" a ", "A", "\u00e9", "e\u0301"],
            Assert.IsType<ChoiceDecisionQuestion>(reader.Contract.Questions[1]).Candidates.Select(c => c.Id).ToArray());
        using var json = JsonDocument.Parse(reader.Contract.CanonicalJson);
        var root = json.RootElement;
        Assert.Equal(" Provider ", root.GetProperty("provider").GetString());
        Assert.Equal(" Model ", root.GetProperty("modelId").GetString());
        Assert.Equal(" Prep ", root.GetProperty("preprocessing").GetString());
        Assert.Equal(" Decoder ", root.GetProperty("decoder").GetString());
        Assert.Equal([" Assets/model "], root.GetProperty("assetHashes").EnumerateObject().Select(p => p.Name).ToArray());
        Assert.Equal(new string('7', 64), root.GetProperty("assetHashes").GetProperty(" Assets/model ").GetString());
        Assert.Equal([" q ", "q[0]", "q[2]", "Q[2]"],
            root.GetProperty("featureProjection").EnumerateArray().Select(e => e.GetString()!).ToArray());
        Assert.Equal([" Low ", " Mid ", " High "], Assert.IsType<ScoreDecisionQuestion>(reader.Contract.Questions[2]).Rubric);
        Assert.Null(await reader.ReadNextRecordBatchAsync(Token));
    }

    [Theory]
    [InlineData(DecisionPrecision.HighPrecision)]
    [InlineData(DecisionPrecision.FourDecimalPlaces)]
    [InlineData(DecisionPrecision.TwoDecimalPlaces)]
    public async Task WriteRead_AcceptsBothInclusiveBinaryProbabilityBoundaries(DecisionPrecision precision)
    {
        using var temp = new TempDirectory();
        var contract = Contract(precision);
        await DecisionArrowWriter.WriteAsync(temp.Dataset, contract,
            Rows([Row(-1, contract, 0), Row(1, contract, 1)]), Provenance(), cancellationToken: Token);

        using var reader = await DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, [1L, -1L], Token);
        using var batch = await reader.ReadNextRecordBatchAsync(Token);
        Assert.Equal([-1L, 1L], ((Int64Array)batch.Column(0)).Values.ToArray());
        Assert.Equal([0.0, 1.0], ((DoubleArray)batch.Column(1)).Values.ToArray());
        Assert.Equal(precision.ToString(), batch.Schema.FieldsList[1].Metadata["decision_arrow.precision"]);
        var choice = (StructArray)batch.Column(2);
        Assert.Equal([2, 2], ((Int32Array)choice.Fields[0]).Values.ToArray());
        Assert.Null(await reader.ReadNextRecordBatchAsync(Token));
    }

    [Theory]
    [InlineData(DecisionPrecision.HighPrecision, .12345678901234567)]
    [InlineData(DecisionPrecision.FourDecimalPlaces, .2718)]
    [InlineData(DecisionPrecision.TwoDecimalPlaces, .27)]
    public async Task WriteRead_PreservesFloat64BitsQuestionOrderAndNonargmaxSelection(DecisionPrecision precision, double probability)
    {
        using var temp = new TempDirectory();
        var contract = Contract(precision);
        var row = Row(-19, contract, probability);
        Assert.Equal(["bin", "pick", "score"], row.Result.Answers.Select(a => a.QuestionId).ToArray());
        var choiceAnswer = Assert.IsType<ChoiceDecisionAnswer>(row.Result.Answers[1]);
        Assert.Equal(["a", "z", "A"], choiceAnswer.Probabilities.Keys.ToArray());
        await DecisionArrowWriter.WriteAsync(temp.Dataset, contract, Rows([row]), Provenance(), cancellationToken: Token);
        using var reader = await DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, [-19L], Token);
        AssertGoldenSchema(reader.Schema, precision);
        using var batch = await reader.ReadNextRecordBatchAsync(Token);
        Assert.Equal(-19L, ((Int64Array)batch.Column(0)).GetValue(0));
        Assert.Equal(BitConverter.DoubleToInt64Bits(probability), BitConverter.DoubleToInt64Bits(((DoubleArray)batch.Column(1)).GetValue(0)!.Value));
        Assert.IsType<DoubleType>(batch.Schema.FieldsList[1].DataType);
        var choice = (StructArray)batch.Column(2);
        Assert.Equal(2, ((Int32Array)choice.Fields[0]).GetValue(0)); // "a", not argmax "A" at index 1.
        var vector = (FixedSizeListArray)choice.Fields[1];
        Assert.Equal([.1, .8, .1], ((DoubleArray)vector.Values).Values.ToArray());
        Assert.Equal("a", Assert.IsType<ChoiceDecisionQuestion>(reader.Contract.Questions[1]).Candidates[2].Id);
        Assert.Null(await reader.ReadNextRecordBatchAsync(Token));
    }

    [Theory]
    [InlineData(DecisionPrecision.HighPrecision, false, 1.0, .999998, 1.000002)]
    [InlineData(DecisionPrecision.HighPrecision, true, 1.0, .999998, 1.000002)]
    [InlineData(DecisionPrecision.FourDecimalPlaces, false, 2.0001, 1.99995, 1.99995)]
    [InlineData(DecisionPrecision.FourDecimalPlaces, true, 2.0001, 1.99995, 1.99995)]
    [InlineData(DecisionPrecision.TwoDecimalPlaces, false, 2.01, 1.995, 1.995)]
    [InlineData(DecisionPrecision.TwoDecimalPlaces, true, 2.01, 1.995, 1.995)]
    public async Task WriteRead_PreservesIndependentScoreExpectationsBoundsAndNativeNull(
        DecisionPrecision precision, bool nullNative, double expected, double lower, double upper)
    {
        using var temp = new TempDirectory();
        var contract = Contract(precision);
        double[] probabilities = precision switch
        {
            DecisionPrecision.HighPrecision => [.25, .5, .25],
            DecisionPrecision.FourDecimalPlaces => [0, .0001, 1],
            _ => [0, .01, 1]
        };
        double? native = nullNative ? null : precision == DecisionPrecision.HighPrecision ? 1 : 2;
        await DecisionArrowWriter.WriteAsync(temp.Dataset, contract, Rows([Row(33, contract, score: probabilities, native: native)]), Provenance(), cancellationToken: Token);
        using var reader = await DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, [33L], Token);
        using var batch = await reader.ReadNextRecordBatchAsync(Token);
        var score = (StructArray)batch.Column(3);
        double[] independent = [expected, lower, upper];
        for (int i = 0; i < 3; i++)
            Assert.Equal(BitConverter.DoubleToInt64Bits(independent[i]), BitConverter.DoubleToInt64Bits(((DoubleArray)score.Fields[i]).GetValue(0)!.Value));
        var nativeArray = (DoubleArray)score.Fields[3];
        Assert.Equal(native, nativeArray.GetValue(0));
        Assert.Equal(nullNative ? 1 : 0, nativeArray.NullCount);
        Assert.Equal(probabilities, ((DoubleArray)((FixedSizeListArray)score.Fields[4]).Values).Values.ToArray());
        if (precision != DecisionPrecision.HighPrecision)
        {
            Assert.True(((DoubleArray)score.Fields[0]).GetValue(0) > 2, "Observed expectation must not be clamped to the rubric.");
            Assert.Equal(lower, upper); // Portable rounding feasibility bounds, not confidence intervals.
        }
        Assert.Equal(precision.ToString(), batch.Schema.FieldsList[3].Metadata["decision_arrow.precision"]);
        Assert.Null(await reader.ReadNextRecordBatchAsync(Token));
    }

    [Theory]
    [InlineData(DecisionPrecision.TwoDecimalPlaces, .33, .99)]
    [InlineData(DecisionPrecision.FourDecimalPlaces, .3333, .9999)]
    public async Task WriteRead_RetainsFeasibleRoundedObservationsWithoutNormalization(DecisionPrecision precision, double p, double expected)
    {
        using var temp = new TempDirectory();
        var contract = Contract(precision);
        var original = Row(17, contract, score: [p, p, p]);
        var result = new DecisionResult(original.Input, "Model",
        [
            original.Result.Answers[2],
            new ChoiceDecisionAnswer("pick", "a", [KeyValuePair.Create("a", p), KeyValuePair.Create("A", p), KeyValuePair.Create("z", p)], precision),
            original.Result.Answers[0]
        ]);
        await DecisionArrowWriter.WriteAsync(temp.Dataset, contract, Rows([new(17, original.Input, result)]), Provenance(), cancellationToken: Token);
        using var reader = await DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, [17L], Token);
        using var batch = await reader.ReadNextRecordBatchAsync(Token);
        var choice = (StructArray)batch.Column(2);
        var score = (StructArray)batch.Column(3);
        foreach (var vector in new[] { (FixedSizeListArray)choice.Fields[1], (FixedSizeListArray)score.Fields[4] })
        {
            var values = ((DoubleArray)vector.Values).Values.ToArray();
            Assert.Equal([p, p, p], values);
            Assert.NotEqual(1.0, values.Sum());
        }
        Assert.Equal(expected, ((DoubleArray)score.Fields[0]).GetValue(0));
        Assert.Null(((DoubleArray)score.Fields[3]).GetValue(0));
        Assert.Equal(2, ((Int32Array)choice.Fields[0]).GetValue(0));
    }

    [Fact]
    public async Task WriteRead_AcceptsSingleCandidateAndOrdinalZero()
    {
        using var temp = new TempDirectory();
        var contract = new DecisionArrowSchema([new ChoiceDecisionQuestion("one", "Choose", [new("A", "Only candidate")])],
            DecisionPrecision.HighPrecision, Identity(["one[0]"]));
        var input = new DecisionInput(JsonSerializer.SerializeToElement(new { source = 7 }), contract.Questions);
        var result = new DecisionResult(input, "Model", [new ChoiceDecisionAnswer("one", "A", [KeyValuePair.Create("A", 1.0)])]);
        await DecisionArrowWriter.WriteAsync(temp.Dataset, contract, Rows([new(7, input, result)]), Provenance(), cancellationToken: Token);
        using var reader = await DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, [7L], Token);
        using var batch = await reader.ReadNextRecordBatchAsync(Token);
        var choice = (StructArray)batch.Column(1);
        Assert.Equal(0, ((Int32Array)choice.Fields[0]).GetValue(0));
        var vector = (FixedSizeListArray)choice.Fields[1];
        Assert.Equal(1, ((FixedSizeListType)vector.Data.DataType).ListSize);
        Assert.Equal(1.0, ((DoubleArray)vector.Values).GetValue(0));
        Assert.Equal("A", Assert.IsType<ChoiceDecisionQuestion>(reader.Contract.Questions[0]).Candidates[0].Id);
    }

    [Fact]
    public async Task FeatureFingerprint_IsIndependentOfPackageVersionsProvenanceRowsAndBatching()
    {
        using var temp = new TempDirectory();
        var contract = Contract();
        var provenance = Provenance();
        string first = Path.Combine(temp.Root, "first"), second = Path.Combine(temp.Root, "second");
        var before = await DecisionArrowWriter.WriteAsync(first, contract, Rows([Row(-7, contract)]), provenance, 1, Token);
        var changed = provenance with
        {
            SourceCommit = "commit-b", Machine = "machine-b", Runtime = "runtime-b", ExecutionMode = "offline-b",
            NativeStateBatchSize = 16, CpuThreads = 4,
            Packages = [provenance.Packages[0] with { Version = "99.0-test", Sha256 = new string('a', 64), Size = 88 }]
        };
        var after = await DecisionArrowWriter.WriteAsync(second, contract, Rows([Row(999, contract), Row(1000, contract)]), changed, 2, Token);
        Assert.Equal(GoldenJson, before.ContractJson);
        Assert.Equal(GoldenJson, after.ContractJson);
        Assert.Equal(GoldenHash, before.FeatureFingerprint);
        Assert.Equal(GoldenHash, after.FeatureFingerprint);
        Assert.NotEqual(before.DataSha256, after.DataSha256);
        Assert.Equal(1, before.RowCount);
        Assert.Equal(2, after.RowCount);
        Assert.Equal("23-test-a", ReadManifest(ManifestPath(first)).Provenance.Packages[0].Version);
        Assert.Equal("99.0-test", ReadManifest(ManifestPath(second)).Provenance.Packages[0].Version);
        using var reader = await DecisionArrowDatasetReader.OpenAsync(ManifestPath(second), contract, [1000L, 999L], Token);
        using var batch = await reader.ReadNextRecordBatchAsync(Token);
        Assert.Equal([999L, 1000L], ((Int64Array)batch.Column(0)).Values.ToArray());
        Assert.Equal(2, reader.Manifest.OutputBatchSize);
    }
}
