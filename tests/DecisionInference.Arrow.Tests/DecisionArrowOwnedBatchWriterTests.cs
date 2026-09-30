using System.Runtime.CompilerServices;
using Apache.Arrow;
using DecisionInference.Arrow;
using static DecisionInference.Tests.DecisionArrowTestSupport;

namespace DecisionInference.Tests;

public sealed class DecisionArrowOwnedBatchWriterTests
{
    [Theory]
    [InlineData(DecisionPrecision.HighPrecision)]
    [InlineData(DecisionPrecision.FourDecimalPlaces)]
    [InlineData(DecisionPrecision.TwoDecimalPlaces)]
    public async Task WriteBatches_SlicedNestedOffsetsPreserveEveryBitAndDisposeTransferredBatches(DecisionPrecision precision)
    {
        using var temp = new TempDirectory();
        var contract = Contract(precision);
        double lower = precision switch { DecisionPrecision.HighPrecision => .999998, DecisionPrecision.FourDecimalPlaces => .9999, _ => .99 };
        double upper = precision switch { DecisionPrecision.HighPrecision => 1.000002, DecisionPrecision.FourDecimalPlaces => 1.0001, _ => 1.01 };
        var rows = new[] { -7L, 91, 0, long.MaxValue }.Select(id => new RawRow
        {
            Id = id, Binary = .25, Lower = lower, Upper = upper, Native = null
        }).ToArray();
        using var parent = BuildRawBatch(contract.Schema, rows);
        using var slice = parent.SliceShared(1, 2);
        var transferred = new TrackingBatch(slice.Clone());
        var manifest = await DecisionArrowWriter.WriteBatchesAsync(temp.Dataset, contract, Batches([transferred], Token), Provenance(), 2, Token);
        Assert.Equal(1, transferred.DisposeCount);
        Assert.Equal(2, manifest.RowCount);
        AssertIntegrity(temp.Dataset, manifest);
        using var reader = await DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, [0L, 91L], Token);
        using var actual = await reader.ReadNextRecordBatchAsync(Token);
        Assert.Equal([91L, 0L], ((Int64Array)actual.Column(0)).Values.ToArray());
        Assert.Equal([.25, .25], ((DoubleArray)actual.Column(1)).Values.ToArray());
        var choice = (StructArray)actual.Column(2);
        Assert.Equal([2, 2], ((Int32Array)choice.Fields[0]).Values.ToArray());
        var vector = (FixedSizeListArray)choice.Fields[1];
        Assert.Equal([.1, .8, .1, .1, .8, .1], ((DoubleArray)vector.Values).Values.ToArray());
        var score = (StructArray)actual.Column(3);
        Assert.Equal([1d, 1d], ((DoubleArray)score.Fields[0]).Values.ToArray());
        Assert.Equal([lower, lower], ((DoubleArray)score.Fields[1]).Values.ToArray());
        Assert.Equal([upper, upper], ((DoubleArray)score.Fields[2]).Values.ToArray());
        Assert.Equal(2, score.Fields[3].NullCount);
        Assert.Null(await reader.ReadNextRecordBatchAsync(Token));
    }

    [Theory]
    [InlineData("selected")]
    [InlineData("probability")]
    [InlineData("native")]
    [InlineData("bound")]
    [InlineData("null")]
    [InlineData("schema")]
    [InlineData("batch-bound")]
    public async Task WriteBatches_InvalidBatchDisposesAndCannotPublishComplete(string fault)
    {
        using var temp = new TempDirectory();
        var contract = Contract();
        var row = fault switch
        {
            "selected" => new RawRow { Selected = 3 },
            "probability" => new RawRow { Choice = [.1, .8, .2] },
            "native" => new RawRow { Native = double.NaN },
            "bound" => new RawRow { Upper = 1.1 },
            "null" => new RawRow { ScoreVectorNull = true },
            _ => new RawRow()
        };
        var schema = fault == "schema" ? Contract(DecisionPrecision.TwoDecimalPlaces).Schema : contract.Schema;
        var batch = new TrackingBatch(BuildRawBatch(schema, fault == "batch-bound" ? [row, row with { Id = 42 }] : [row]));
        var error = await Record.ExceptionAsync(() => DecisionArrowWriter.WriteBatchesAsync(temp.Dataset, contract, Batches([batch], Token), Provenance(), 1, Token));
        Assert.NotNull(error);
        Assert.Equal(1, batch.DisposeCount);
        DecisionArrowWriterTests.AssertFailureRecord(temp.Dataset, "failed", 0, error);
        Assert.False(File.Exists(ManifestPath(temp.Dataset)));
    }

    [Fact]
    public async Task WriteBatches_DuplicateAcrossBatchesDisposesBothAndPreservesOnlyAcceptedCount()
    {
        using var temp = new TempDirectory();
        var contract = Contract();
        var first = new TrackingBatch(BuildRawBatch(contract.Schema, [new RawRow { Id = -7 }]));
        var duplicate = new TrackingBatch(BuildRawBatch(contract.Schema, [new RawRow { Id = -7 }]));
        var error = await Assert.ThrowsAsync<DecisionProtocolException>(() =>
            DecisionArrowWriter.WriteBatchesAsync(temp.Dataset, contract, Batches([first, duplicate], Token), Provenance(), 1, Token));
        Assert.Equal("Duplicate source row ID -7.", error.Message);
        Assert.Equal(1, first.DisposeCount);
        Assert.Equal(1, duplicate.DisposeCount);
        DecisionArrowWriterTests.AssertFailureRecord(temp.Dataset, "failed", 1, error);
    }

    [Fact]
    public async Task WriteBatches_CancellationAtTransferDisposesBatchAndEnumeratorWithoutManifest()
    {
        using var temp = new TempDirectory();
        var contract = Contract();
        var batch = new TrackingBatch(BuildRawBatch(contract.Schema, [new RawRow()]));
        using var cancellation = new CancellationTokenSource();
        bool disposed = false;
        async IAsyncEnumerable<RecordBatch> Produce()
        {
            await Task.CompletedTask;
            try { cancellation.Cancel(); yield return batch; }
            finally { disposed = true; }
        }
        var error = await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            DecisionArrowWriter.WriteBatchesAsync(temp.Dataset, contract, Produce(), Provenance(), 1, cancellation.Token));
        Assert.Equal(cancellation.Token, error.CancellationToken);
        Assert.True(disposed);
        Assert.Equal(1, batch.DisposeCount);
        DecisionArrowWriterTests.AssertFailureRecord(temp.Dataset, "cancelled", 0, error);
    }

    [Fact]
    public async Task WriteBatches_ProducerThrowAndNullBatchAreExplicitFailures()
    {
        using var temp = new TempDirectory();
        var contract = Contract();
        var batch = new TrackingBatch(BuildRawBatch(contract.Schema, [new RawRow()]));
        var sentinel = new InvalidOperationException("owned producer failed");
        bool disposed = false;
        async IAsyncEnumerable<RecordBatch> Produce()
        {
            await Task.CompletedTask;
            try { yield return batch; throw sentinel; }
            finally { disposed = true; }
        }
        var error = await Record.ExceptionAsync(() =>
            DecisionArrowWriter.WriteBatchesAsync(temp.Dataset, contract, Produce(), Provenance(), 1, Token));
        Assert.Same(sentinel, error);
        Assert.True(disposed);
        Assert.Equal(1, batch.DisposeCount);
        DecisionArrowWriterTests.AssertFailureRecord(temp.Dataset, "failed", 1, sentinel);
        using var nullTemp = new TempDirectory();
        var nullError = await Assert.ThrowsAsync<ArgumentNullException>(() =>
            DecisionArrowWriter.WriteBatchesAsync(nullTemp.Dataset, contract, Batches([null!], Token), Provenance(), 1, Token));
        Assert.Equal("batch", nullError.ParamName);
        DecisionArrowWriterTests.AssertFailureRecord(nullTemp.Dataset, "failed", 0, nullError);
    }

    [Fact]
    public async Task WriteBatches_PrepublicationFailureDisposesAndCannotPublishComplete()
    {
        using var temp = new TempDirectory();
        var contract = Contract();
        var batch = new TrackingBatch(BuildRawBatch(contract.Schema, [new RawRow()]));
        var sentinel = new InvalidOperationException("final expected-ID/runtime check failed");
        bool callback = false;
        var error = await Record.ExceptionAsync(() => DecisionArrowWriter.WriteBatchesBeforePublicationAsync(
            temp.Dataset, contract, Batches([batch], Token), Provenance(), (manifest, token) =>
            {
                callback = true;
                Assert.Equal("complete", manifest.Status);
                Assert.False(File.Exists(ManifestPath(temp.Dataset)));
                Assert.True(File.Exists(Path.Combine(temp.Dataset, "decisions.arrow")));
                throw sentinel;
            }, Token));
        Assert.Same(sentinel, error);
        Assert.True(callback);
        Assert.Equal(1, batch.DisposeCount);
        Assert.False(File.Exists(ManifestPath(temp.Dataset)));
        DecisionArrowWriterTests.AssertFailureRecord(temp.Dataset, "failed", 1, sentinel);
    }

    [Fact]
    public void TypedAppend_LastQuestionFailureIsAtomicAndBuilderRecovers()
    {
        var contract = Contract();
        var builder = new DecisionArrowBatchBuilder(contract, 2);
        builder.Append(Row(7, contract));
        var valid = Row(9, contract);
        var questions = contract.Questions.ToArray();
        questions[^1] = new BinaryDecisionQuestion("score", "different kind");
        var input = new DecisionInput(valid.Input.State, questions);
        var result = new DecisionResult(input, contract.Identity.ModelId,
            [valid.Result.Answers[0], valid.Result.Answers[1], new BinaryDecisionAnswer("score", .5)]);
        Assert.Equal("Answer ID or kind mismatch.", Assert.Throws<DecisionProtocolException>(() =>
            builder.Append(new(8, valid.Input, result))).Message);
        Assert.Equal(1, builder.Count);
        builder.Append(valid);
        using var batch = builder.Build();
        Assert.Equal([7L, 9L], ((Int64Array)batch.Column(0)).Values.ToArray());
        Assert.All(batch.Arrays, column => Assert.Equal(2, column.Length));
        var choice = (StructArray)batch.Column(2);
        Assert.Equal([2, 2], ((Int32Array)choice.Fields[0]).Values.ToArray());
        Assert.Throws<InvalidOperationException>(() => builder.Append(valid));
    }

    [Fact]
    public void PublicSchemaMutationDoesNotChangeCompiledWriterContract()
    {
        var contract = Contract();
        if (contract.Schema.Metadata is IDictionary<string, string> metadata) metadata["decision_arrow.contract"] = "changed";
        Assert.Equal(GoldenJson, contract.Schema.Metadata["decision_arrow.contract"]);
        var builder = new DecisionArrowBatchBuilder(contract, 1);
        builder.Append(Row(7, contract));
        using var batch = builder.Build();
        Assert.Equal(GoldenJson, batch.Schema.Metadata["decision_arrow.contract"]);
        DecisionArrowDatasetReader.ValidateBatch(contract, batch);
    }

    [Fact]
    public void NumericFrame_LateBoundsFailureIsAtomicAndSelectedNonArgmaxNativeNullRemainValid()
    {
        var contract = Contract();
        var builder = new DecisionArrowBatchBuilder(contract, 1);
        int[] selected = [-1, 2, -1], offsets = [0, 2, 5, 8];
        double[] probabilities = [.3, .7, .1, .8, .1, .25, .5, .25];
        double[] scores = [0, 0, 0, 0, 0, 0, 1, .999998, 1.01];
        Assert.Equal("Score expectation or portable rounding bounds mismatch.", Assert.Throws<DecisionProtocolException>(() =>
            builder.AppendHighPrecisionFrame(91, selected, offsets, probabilities, scores)).Message);
        Assert.Equal(0, builder.Count);
        scores[^1] = 1.000002;
        builder.AppendHighPrecisionFrame(91, selected, offsets, probabilities, scores);
        using var batch = builder.Build();
        DecisionArrowDatasetReader.ValidateBatch(contract, batch);
        Assert.Equal(91, ((Int64Array)batch.Column(0)).GetValue(0));
        Assert.Equal(.7, ((DoubleArray)batch.Column(1)).GetValue(0));
        var choice = (StructArray)batch.Column(2);
        Assert.Equal(2, ((Int32Array)choice.Fields[0]).GetValue(0));
        Assert.Equal([.1, .8, .1], ((DoubleArray)((FixedSizeListArray)choice.Fields[1]).Values).Values.ToArray());
        Assert.True(((StructArray)batch.Column(3)).Fields[3].IsNull(0));
    }

    private sealed class TrackingBatch(RecordBatch batch) : RecordBatch(batch.Schema, batch.Arrays, batch.Length)
    {
        internal int DisposeCount { get; private set; }
        protected override void Dispose(bool disposing)
        {
            DisposeCount++;
            base.Dispose(disposing);
        }
    }

    private static async IAsyncEnumerable<RecordBatch> Batches(IEnumerable<RecordBatch> batches,
        [EnumeratorCancellation] CancellationToken token)
    {
        await Task.CompletedTask;
        foreach (var batch in batches) { token.ThrowIfCancellationRequested(); yield return batch; }
    }
}
