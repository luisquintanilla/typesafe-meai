using Apache.Arrow;
using Apache.Arrow.Ipc;
using DecisionInference.Arrow;
using static DecisionInference.Tests.DecisionArrowTestSupport;

namespace DecisionInference.Tests;

public sealed class DecisionArrowDatasetReaderTests
{
    [Theory]
    [InlineData("contract")]
    [InlineData("expectedRowIds")]
    [InlineData("duplicate")]
    [InlineData("enumeration")]
    public async Task OpenAsync_RejectsInvalidArgumentsBeforeReadingManifest(string fault)
    {
        using var temp = new TempDirectory();
        string nonexistent = ManifestPath(temp.Dataset);
        var sentinel = new InvalidOperationException("expected ID enumeration failed");
        IEnumerable<long> ThrowingIds()
        {
            yield return 7;
            throw sentinel;
        }
        if (fault == "enumeration")
        {
            var error = await Record.ExceptionAsync(() => DecisionArrowDatasetReader.OpenAsync(nonexistent, Contract(), ThrowingIds(), Token));
            Assert.Same(sentinel, error);
        }
        else if (fault == "duplicate")
        {
            var error = await Assert.ThrowsAsync<ArgumentException>(() => DecisionArrowDatasetReader.OpenAsync(nonexistent, Contract(), [7L, 19L, 7L], Token));
            Assert.Equal("expectedRowIds", error.ParamName);
            Assert.Contains("Duplicate expected row ID 7.", error.Message);
        }
        else
        {
            var error = await Assert.ThrowsAsync<ArgumentNullException>(() => DecisionArrowDatasetReader.OpenAsync(nonexistent,
                fault == "contract" ? null! : Contract(), fault == "expectedRowIds" ? null! : [], Token));
            Assert.Equal(fault == "contract" ? "expectedContract" : "expectedRowIds", error.ParamName);
        }
        Assert.False(File.Exists(nonexistent));
    }

    [Fact]
    public async Task ReadNext_ReplaysAllBatchesInProducerOrderAndCallerOwnsTheirLifetime()
    {
        using var temp = new TempDirectory();
        var contract = Contract();
        long[] ids = [long.MinValue, -7, 0, 19, long.MaxValue];
        await DecisionArrowWriter.WriteAsync(temp.Dataset, contract, Rows(ids.Select(id => Row(id, contract))), Provenance(), 2, Token);
        var reader = await DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, ids.Reverse(), Token);
        IArrowArrayStream stream = reader;
        var retained = new List<RecordBatch>();
        try
        {
            Assert.Equal(GoldenJson, reader.Contract.CanonicalJson);
            Assert.Equal(GoldenHash, reader.Manifest.FeatureFingerprint);
            AssertGoldenSchema(stream.Schema, DecisionPrecision.HighPrecision);
            RecordBatch? batch;
            while ((batch = await stream.ReadNextRecordBatchAsync(Token)) is not null) retained.Add(batch);
            Assert.Null(await stream.ReadNextRecordBatchAsync(Token));
            Assert.Equal([2, 2, 1], retained.Select(b => b.Length).ToArray());
            Assert.Equal(ids, retained.SelectMany(b => ((Int64Array)b.Column(0)).Values.ToArray()).ToArray());
            Assert.Throws<IOException>(() =>
            {
                using var locked = new FileStream(DataPath(temp.Dataset), FileMode.Open, FileAccess.ReadWrite, FileShare.None);
            });
            stream.Dispose();
            // Advancing and disposing the stream must not invalidate any caller-owned batch.
            Assert.Equal(long.MinValue, ((Int64Array)retained[0].Column(0)).GetValue(0));
            Assert.Equal(long.MaxValue, ((Int64Array)retained[2].Column(0)).GetValue(0));
            foreach (var held in retained)
            {
                Assert.Equal(.12345678901234567, ((DoubleArray)held.Column(1)).GetValue(0));
                Assert.Equal(2, ((Int32Array)((StructArray)held.Column(2)).Fields[0]).GetValue(0));
            }
            AssertReleased(DataPath(temp.Dataset));
        }
        finally
        {
            reader.Dispose();
            foreach (var batch in retained) batch.Dispose();
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(2)]
    public async Task OpenAsync_ManifestPathRelativeDataAndBothOverloadsSupportEmptyAndNonempty(int count)
    {
        using var temp = new TempDirectory();
        var contract = Contract();
        var ids = Enumerable.Range(0, count).Select(i => -10L - i).ToArray();
        await DecisionArrowWriter.WriteAsync(temp.Dataset, contract, Rows(ids.Select(id => Row(id, contract))), Provenance(), cancellationToken: Token);
        using (var withIds = await DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, ids.Reverse(), Token))
        {
            Assert.Equal(count, withIds.Manifest.RowCount);
            using var batch = await withIds.ReadNextRecordBatchAsync(Token);
            if (count == 0) Assert.Null(batch);
            else Assert.Equal(ids, ((Int64Array)batch.Column(0)).Values.ToArray());
            Assert.Null(await withIds.ReadNextRecordBatchAsync(Token));
        }
        using (var withoutIds = await DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, Token))
        {
            using var batch = await withoutIds.ReadNextRecordBatchAsync(Token);
            if (count == 0) Assert.Null(batch);
            else Assert.Equal(2, batch.Length);
            Assert.Null(await withoutIds.ReadNextRecordBatchAsync(Token));
        }
        AssertReleased(DataPath(temp.Dataset));
    }

    [Fact]
    public async Task OpenAsync_PreCancelledTokenDoesNotReturnReaderOrLeakFileHandle()
    {
        using var temp = new TempDirectory();
        var contract = Contract();
        await DecisionArrowWriter.WriteAsync(temp.Dataset, contract, Rows([Row(7, contract)]), Provenance(), cancellationToken: Token);
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        var error = await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, [7L], cancellation.Token));
        Assert.Equal(cancellation.Token, error.CancellationToken);
        AssertReleased(DataPath(temp.Dataset));
        using var manifest = new FileStream(ManifestPath(temp.Dataset), FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        Assert.True(manifest.Length > 0);
    }
}
