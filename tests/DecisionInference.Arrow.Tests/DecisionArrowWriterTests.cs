using System.Runtime.CompilerServices;
using System.Text.Json;
using Apache.Arrow;
using Apache.Arrow.Ipc;
using DecisionInference.Arrow;
using static DecisionInference.Tests.DecisionArrowTestSupport;

namespace DecisionInference.Tests;

public sealed class DecisionArrowWriterTests
{
    [Theory]
    [InlineData(0, 256, new int[] { })]
    [InlineData(1, 256, new[] { 1 })]
    [InlineData(256, 256, new[] { 256 })]
    [InlineData(257, 256, new[] { 256, 1 })]
    [InlineData(775, 256, new[] { 256, 256, 256, 7 })]
    [InlineData(5, 2, new[] { 2, 2, 1 })]
    [InlineData(3, 1, new[] { 1, 1, 1 })]
    [InlineData(1, 65536, new[] { 1 })]
    public async Task WriteAsync_EmitsBoundedBatchesAndFinalizedManifest(int count, int bound, int[] lengths)
    {
        using var temp = new TempDirectory();
        var contract = Contract();
        var ids = Enumerable.Range(0, count).Select(i => 1000L - i * 3).ToArray();
        var provenance = Provenance();
        var manifest = bound == 256
            ? await DecisionArrowWriter.WriteAsync(temp.Dataset, contract, Rows(ids.Select(id => Row(id, contract))), provenance, cancellationToken: Token)
            : await DecisionArrowWriter.WriteAsync(temp.Dataset, contract, Rows(ids.Select(id => Row(id, contract))), provenance, bound, Token);
        Assert.Equal(256, DecisionArrowWriter.DefaultBatchSize);
        Assert.Equal("complete", manifest.Status);
        Assert.Null(manifest.Error);
        Assert.Equal(count, manifest.RowCount);
        Assert.Equal(bound, manifest.OutputBatchSize);
        Assert.Equal("decisions.arrow", manifest.DataFile);
        Assert.Equal(GoldenJson, manifest.ContractJson);
        Assert.Equal(GoldenHash, manifest.FeatureFingerprint);
        Assert.Equal(provenance.Packages, manifest.Provenance.Packages);
        AssertIntegrity(temp.Dataset, manifest);
        Assert.Equal(JsonSerializer.Serialize(manifest, DecisionArrowManifest.JsonOptions),
            File.ReadAllText(ManifestPath(temp.Dataset)));
        Assert.Equal(GoldenJson, File.ReadAllText(Path.Combine(temp.Dataset, "contract.json")));
        Assert.Equal(["contract.json", "decisions.arrow", "manifest.json"],
            Directory.GetFiles(temp.Dataset).Select(p => Path.GetFileName(p)!).Order(StringComparer.Ordinal).ToArray());
        using var file = File.OpenRead(DataPath(temp.Dataset));
        using var ipc = new ArrowStreamReader(file);
        var observedLengths = new List<int>();
        var observedIds = new List<long>();
        RecordBatch? batch;
        while ((batch = await ipc.ReadNextRecordBatchAsync(Token)) is not null)
        {
            using (batch)
            {
                observedLengths.Add(batch.Length);
                for (int i = 0; i < batch.Length; i++)
                {
                    observedIds.Add(((Int64Array)batch.Column(0)).GetValue(i)!.Value);
                    Assert.Equal(BitConverter.DoubleToInt64Bits(.12345678901234567),
                        BitConverter.DoubleToInt64Bits(((DoubleArray)batch.Column(1)).GetValue(i)!.Value));
                }
            }
        }
        Assert.Equal(lengths, observedLengths);
        Assert.Equal(ids, observedIds);
        AssertGoldenSchema(ipc.Schema, DecisionPrecision.HighPrecision);
    }

    [Theory]
    [InlineData("directory")]
    [InlineData("contract")]
    [InlineData("rows")]
    [InlineData("provenance")]
    [InlineData("batchSize")]
    public async Task WriteAsync_RejectsArgumentsBeforePublication(string parameter)
    {
        using var temp = new TempDirectory();
        var contract = Contract();
        object?[] invalids = parameter switch
        {
            "directory" => [null, "", " \t"],
            "batchSize" => [0, -1, 65537],
            _ => [null]
        };
        foreach (var invalid in invalids)
        {
            var error = await Assert.ThrowsAnyAsync<ArgumentException>(() => DecisionArrowWriter.WriteAsync(
                parameter == "directory" ? (string)invalid! : temp.Dataset,
                parameter == "contract" ? null! : contract,
                parameter == "rows" ? null! : Rows([]),
                parameter == "provenance" ? null! : Provenance(),
                parameter == "batchSize" ? (int)invalid! : 2, Token));
            Assert.Equal(parameter, error.ParamName);
            Assert.False(Directory.Exists(temp.Dataset));
        }
    }

    [Theory]
    [InlineData("file")]
    [InlineData("directory")]
    [InlineData("completed")]
    public async Task WriteAsync_ProtectsAllExistingArtifactsByteForByte(string existing)
    {
        using var temp = new TempDirectory();
        Directory.CreateDirectory(temp.Dataset);
        var contract = Contract();
        if (existing == "completed")
            await DecisionArrowWriter.WriteAsync(temp.Dataset, contract, Rows([Row(-7, contract)]), Provenance(), cancellationToken: Token);
        else if (existing == "directory")
        {
            Directory.CreateDirectory(Path.Combine(temp.Dataset, "nested"));
            File.WriteAllBytes(Path.Combine(temp.Dataset, "nested", "sentinel"), [0, 255, 17]);
        }
        else File.WriteAllBytes(Path.Combine(temp.Dataset, "sentinel"), [0, 255, 17]);
        var paths = Directory.GetFileSystemEntries(temp.Dataset, "*", SearchOption.AllDirectories)
            .Select(p => Path.GetRelativePath(temp.Dataset, p)).Order(StringComparer.Ordinal).ToArray();
        var files = Directory.GetFiles(temp.Dataset, "*", SearchOption.AllDirectories)
            .ToDictionary(p => p, File.ReadAllBytes, StringComparer.Ordinal);
        var error = await Assert.ThrowsAsync<IOException>(() =>
            DecisionArrowWriter.WriteAsync(temp.Dataset, contract, Rows([Row(19, contract)]), Provenance(), cancellationToken: Token));
        Assert.Contains("existing artifacts are never replaced", error.Message);
        Assert.Equal(paths, Directory.GetFileSystemEntries(temp.Dataset, "*", SearchOption.AllDirectories)
            .Select(p => Path.GetRelativePath(temp.Dataset, p)).Order(StringComparer.Ordinal).ToArray());
        foreach (var file in files) Assert.Equal(file.Value, File.ReadAllBytes(file.Key));
    }

    [Fact]
    public async Task WriteAsync_OnlyPartialArtifactsExistThroughoutProducerLifetime()
    {
        using var temp = new TempDirectory();
        Directory.CreateDirectory(temp.Dataset); // Existing empty directory is permitted.
        var contract = Contract();
        int checkpoints = 0;
        bool disposed = false;
        async IAsyncEnumerable<DecisionArrowRow> Produce()
        {
            await Task.CompletedTask;
            try
            {
                for (int i = 0; i < 3; i++)
                {
                    CheckPartial();
                    yield return Row(i, contract);
                }
                CheckPartial(); // Exhaustion is still before completion publication.
            }
            finally { disposed = true; }
        }
        void CheckPartial()
        {
            var partial = ReadManifest(Path.Combine(temp.Dataset, "run.partial.json"));
            Assert.Equal("partial", partial.Status);
            Assert.Equal(0, partial.RowCount);
            Assert.Null(partial.DataSha256);
            Assert.Equal("decisions.arrow.partial", partial.DataFile);
            Assert.True(File.Exists(Path.Combine(temp.Dataset, "decisions.arrow.partial")));
            Assert.Equal(GoldenJson, File.ReadAllText(Path.Combine(temp.Dataset, "contract.json")));
            Assert.False(File.Exists(ManifestPath(temp.Dataset)));
            Assert.False(File.Exists(DataPath(temp.Dataset)));
            checkpoints++;
        }
        var manifest = await DecisionArrowWriter.WriteAsync(temp.Dataset, contract, Produce(), Provenance(), 2, Token);
        Assert.Equal(4, checkpoints);
        Assert.True(disposed);
        Assert.Equal(3, manifest.RowCount);
        AssertIntegrity(temp.Dataset, manifest);
        Assert.False(File.Exists(Path.Combine(temp.Dataset, "run.partial.json")));
        Assert.Empty(Directory.GetFiles(temp.Dataset, "*.tmp"));
    }

    [Theory]
    [InlineData(0, false)]
    [InlineData(2, false)]
    [InlineData(0, true)]
    [InlineData(2, true)]
    public async Task WriteAsync_ProducerErrorsPreserveInstanceAcceptedCountAndDisposal(int accepted, bool cancelled)
    {
        using var temp = new TempDirectory();
        var contract = Contract();
        Exception sentinel = cancelled ? new OperationCanceledException("producer stopped") : new InvalidOperationException("producer failed");
        bool disposed = false;
        async IAsyncEnumerable<DecisionArrowRow> Produce()
        {
            await Task.CompletedTask;
            try
            {
                for (int i = 0; i < accepted; i++) yield return Row(i, contract);
                throw sentinel;
            }
            finally { disposed = true; }
        }
        var error = await Record.ExceptionAsync(() => DecisionArrowWriter.WriteAsync(temp.Dataset, contract, Produce(), Provenance(), 1, Token));
        Assert.Same(sentinel, error);
        Assert.True(disposed);
        AssertFailureRecord(temp.Dataset, cancelled ? "cancelled" : "failed", accepted, sentinel);
        if (accepted == 0)
        {
            // The producer threw on its very first MoveNext; a schema must already be present.
            using var file = File.OpenRead(Path.Combine(temp.Dataset, "decisions.arrow.partial"));
            using var ipc = new ArrowStreamReader(file);
            AssertGoldenSchema(ipc.Schema, DecisionPrecision.HighPrecision);
        }
    }

    [Fact]
    public async Task WriteAsync_DeterministicTokenCancellationRetainsProducerException()
    {
        using var temp = new TempDirectory();
        using var cancellation = new CancellationTokenSource();
        var contract = Contract();
        OperationCanceledException? sentinel = null;
        bool disposed = false;
        async IAsyncEnumerable<DecisionArrowRow> Produce([EnumeratorCancellation] CancellationToken token = default)
        {
            await Task.CompletedTask;
            try
            {
                yield return Row(17, contract);
                cancellation.Cancel();
                try { token.ThrowIfCancellationRequested(); }
                catch (OperationCanceledException error) { sentinel = error; throw; }
            }
            finally { disposed = true; }
        }
        var actual = await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            DecisionArrowWriter.WriteAsync(temp.Dataset, contract, Produce(cancellation.Token), Provenance(), 1, cancellation.Token));
        Assert.Same(sentinel, actual);
        Assert.Equal(cancellation.Token, actual.CancellationToken);
        Assert.True(disposed);
        AssertFailureRecord(temp.Dataset, "cancelled", 1, actual);
    }

    [Fact]
    public async Task WriteAsync_PreCancelledTokenRecordsCancellationWithoutCompletedManifest()
    {
        using var temp = new TempDirectory();
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        var error = await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            DecisionArrowWriter.WriteAsync(temp.Dataset, Contract(), Rows([]), Provenance(), cancellationToken: cancellation.Token));
        AssertFailureRecord(temp.Dataset, "cancelled", 0, error);
        Assert.Equal(cancellation.Token, error.CancellationToken);
    }

    [Fact]
    public async Task WriteAsync_RecordingFailureKeepsOriginalAsFirstAggregateInner()
    {
        using var temp = new TempDirectory();
        var sentinel = new InvalidOperationException("original producer error");
        async IAsyncEnumerable<DecisionArrowRow> Produce()
        {
            await Task.CompletedTask;
            File.Delete(Path.Combine(temp.Dataset, "run.partial.json"));
            Directory.CreateDirectory(Path.Combine(temp.Dataset, "run.partial.json"));
            if (Directory.Exists(temp.Dataset)) throw sentinel;
            yield break;
        }
        var error = await Assert.ThrowsAsync<AggregateException>(() =>
            DecisionArrowWriter.WriteAsync(temp.Dataset, Contract(), Produce(), Provenance(), cancellationToken: Token));
        Assert.Equal(2, error.InnerExceptions.Count);
        Assert.Same(sentinel, error.InnerExceptions[0]);
        Assert.True(error.InnerExceptions[1] is IOException or UnauthorizedAccessException);
        Assert.False(File.Exists(ManifestPath(temp.Dataset)));
        Assert.True(Directory.Exists(Path.Combine(temp.Dataset, "run.partial.json")));
    }

    [Fact]
    public async Task WriteAsync_CleanupFailureMustNotPublishCompletedManifest()
    {
        using var temp = new TempDirectory();
        FileStream? partialLock = null;
        async IAsyncEnumerable<DecisionArrowRow> Produce()
        {
            await Task.CompletedTask;
            partialLock = new FileStream(Path.Combine(temp.Dataset, "run.partial.json"), FileMode.Open, FileAccess.Read, FileShare.Read);
            yield break;
        }
        try
        {
            var error = await Assert.ThrowsAsync<AggregateException>(() =>
                DecisionArrowWriter.WriteAsync(temp.Dataset, Contract(), Produce(), Provenance(), cancellationToken: Token));
            Assert.Equal(2, error.InnerExceptions.Count);
            Assert.True(error.InnerExceptions[0] is IOException or UnauthorizedAccessException);
            Assert.False(File.Exists(ManifestPath(temp.Dataset)), "Cleanup failure must not leave a success-shaped completed dataset.");
            Assert.Equal("partial", ReadManifest(Path.Combine(temp.Dataset, "run.partial.json")).Status);
        }
        finally { partialLock?.Dispose(); }
    }

    [Fact]
    public async Task WriteAsync_MergesNonnegativeMeasurementsWithoutMutatingCaller()
    {
        using var temp = new TempDirectory();
        var caller = new Dictionary<string, double> { ["callerObservation"] = 42.5 };
        var provenance = Provenance() with { Measurements = caller };
        var result = await DecisionArrowWriter.WriteAsync(temp.Dataset, Contract(), Rows([]), provenance, cancellationToken: Token);
        Assert.Single(caller);
        Assert.Equal(42.5, caller["callerObservation"]);
        Assert.Same(caller, provenance.Measurements);
        Assert.NotSame(caller, result.Provenance.Measurements);
        Assert.Equal(["callerObservation", "exportBeforeManifestPublicationMilliseconds", "exportProcessAllocatedBytes"],
            result.Provenance.Measurements!.Keys.Order(StringComparer.Ordinal).ToArray());
        Assert.Equal(42.5, result.Provenance.Measurements["callerObservation"]);
        foreach (string key in new[] { "exportBeforeManifestPublicationMilliseconds", "exportProcessAllocatedBytes" })
        {
            Assert.True(double.IsFinite(result.Provenance.Measurements[key]));
            Assert.True(result.Provenance.Measurements[key] >= 0);
        }
        Assert.Equal(GoldenHash, result.FeatureFingerprint);
    }

    [Theory]
    [InlineData("exportBeforeManifestPublicationMilliseconds")]
    [InlineData("exportProcessAllocatedBytes")]
    public async Task WriteAsync_RejectsReservedMeasurementNamesBeforeAnyPublication(string name)
    {
        using var temp = new TempDirectory();
        var caller = new Dictionary<string, double> { [name] = 42.5 };
        var provenance = Provenance() with { Measurements = caller };
        var error = await Assert.ThrowsAsync<ArgumentException>(() =>
            DecisionArrowWriter.WriteAsync(temp.Dataset, Contract(), Rows([]), provenance, cancellationToken: Token));

        Assert.Equal("provenance", error.ParamName);
        Assert.Contains("Export measurement names are reserved", error.Message);
        Assert.False(Directory.Exists(temp.Dataset));
        Assert.Single(caller);
        Assert.Equal(42.5, caller[name]);
    }

    internal static void AssertFailureRecord(string directory, string status, long count, Exception error)
    {
        var record = ReadManifest(Path.Combine(directory, "run.partial.json"));
        Assert.Equal(status, record.Status);
        Assert.Equal(count, record.RowCount);
        Assert.Equal($"{error.GetType().Name}: {error.Message}", record.Error);
        Assert.Null(record.DataSha256);
        Assert.Null(record.DataSize);
        Assert.False(File.Exists(ManifestPath(directory)));
    }
}
