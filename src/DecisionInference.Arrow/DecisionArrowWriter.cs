using System.Security.Cryptography;
using System.Text.Json;
using System.Diagnostics;
using System.Collections.ObjectModel;
using System.Runtime.CompilerServices;
using Apache.Arrow;
using Apache.Arrow.Ipc;

namespace DecisionInference.Arrow;

public static class DecisionArrowWriter
{
    public const int DefaultBatchSize = 256;

    /// <summary>Creates a new dataset. Failure retains partial files. Completion manifest is published last.</summary>
    public static async Task<DecisionArrowManifest> WriteAsync(string directory, DecisionArrowSchema contract,
        IAsyncEnumerable<DecisionArrowRow> rows, DecisionArrowProvenance provenance,
        int batchSize = DefaultBatchSize, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        ArgumentNullException.ThrowIfNull(contract);
        ArgumentNullException.ThrowIfNull(rows);
        return await WriteCoreAsync(directory, contract, provenance, batchSize, cancellationToken,
            (ids, progress, token) => TypedBatches(contract, rows, batchSize, ids, progress, token));
    }

    /// <summary>
    /// Takes ownership of every yielded standard RecordBatch. Each batch is validated and disposed after writing,
    /// including cancellation or failure. Producers must not mutate or dispose a transferred batch concurrently.
    /// </summary>
    public static async Task<DecisionArrowManifest> WriteBatchesAsync(string directory, DecisionArrowSchema contract,
        IAsyncEnumerable<RecordBatch> batches, DecisionArrowProvenance provenance,
        int batchSize = DefaultBatchSize, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(batches);
        return await WriteCoreAsync(directory, contract, provenance, batchSize, cancellationToken,
            (ids, progress, token) => ValidatedBatches(contract, batches, batchSize, ids, progress, token));
    }

    internal static Task<DecisionArrowManifest> WriteBatchesBeforePublicationAsync(string directory, DecisionArrowSchema contract,
        IAsyncEnumerable<RecordBatch> batches, DecisionArrowProvenance provenance,
        Func<DecisionArrowManifest, CancellationToken, Task> beforePublication, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(batches);
        ArgumentNullException.ThrowIfNull(beforePublication);
        return WriteCoreAsync(directory, contract, provenance, DefaultBatchSize, cancellationToken,
            (ids, progress, token) => ValidatedBatches(contract, batches, DefaultBatchSize, ids, progress, token), beforePublication);
    }

    private sealed class Progress { internal long Rows; }

    private static async IAsyncEnumerable<RecordBatch> TypedBatches(DecisionArrowSchema contract,
        IAsyncEnumerable<DecisionArrowRow> rows, int batchSize, HashSet<long> ids, Progress progress,
        [EnumeratorCancellation] CancellationToken token)
    {
        var builder = new DecisionArrowBatchBuilder(contract, batchSize);
        await foreach (var row in rows.WithCancellation(token))
        {
            token.ThrowIfCancellationRequested();
            ArgumentNullException.ThrowIfNull(row);
            if (!ids.Add(row.RowId)) throw new DecisionProtocolException($"Duplicate source row ID {row.RowId}.");
            builder.Append(row);
            progress.Rows = checked(progress.Rows + 1);
            if (builder.Count == batchSize)
            {
                yield return builder.Build();
                builder = new(contract, batchSize);
            }
        }
        if (builder.Count > 0) yield return builder.Build();
    }

    private static async IAsyncEnumerable<RecordBatch> ValidatedBatches(DecisionArrowSchema contract,
        IAsyncEnumerable<RecordBatch> batches, int batchSize, HashSet<long> ids, Progress progress,
        [EnumeratorCancellation] CancellationToken token)
    {
        // Check bounds even when the input is empty, before asking the producer for a batch.
        foreach (int width in contract.Compiled.Widths) _ = checked(batchSize * width * sizeof(double));
        await foreach (var batch in batches.WithCancellation(token))
        {
            bool transferred = false;
            try
            {
                ArgumentNullException.ThrowIfNull(batch);
                token.ThrowIfCancellationRequested();
                if (batch.Length < 0 || batch.Length > batchSize)
                    throw new InvalidDataException("Batch exceeds the declared output bound.");
                DecisionArrowDatasetReader.ValidateSchema(contract.Compiled.Schema, batch.Schema);
                DecisionArrowDatasetReader.ValidateBatch(contract, batch);
                var rowIds = (Int64Array)batch.Column(0);
                for (int i = 0; i < batch.Length; i++)
                    if (!ids.Add(rowIds.Values[i]))
                        throw new DecisionProtocolException($"Duplicate source row ID {rowIds.Values[i]}.");
                progress.Rows = checked(progress.Rows + batch.Length);
                transferred = true;
            }
            finally
            {
                if (!transferred) batch?.Dispose();
            }
            yield return batch;
        }
    }

    private static async Task<DecisionArrowManifest> WriteCoreAsync(string directory, DecisionArrowSchema contract,
        DecisionArrowProvenance provenance, int batchSize, CancellationToken cancellationToken,
        Func<HashSet<long>, Progress, CancellationToken, IAsyncEnumerable<RecordBatch>> batches,
        Func<DecisionArrowManifest, CancellationToken, Task>? beforePublication = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        ArgumentNullException.ThrowIfNull(contract);
        ArgumentNullException.ThrowIfNull(provenance);
        if (batchSize is < 1 or > 65536) throw new ArgumentOutOfRangeException(nameof(batchSize));
        if (provenance.Measurements?.ContainsKey("exportBeforeManifestPublicationMilliseconds") == true ||
            provenance.Measurements?.ContainsKey("exportProcessAllocatedBytes") == true)
            throw new ArgumentException("Export measurement names are reserved for the writer.", nameof(provenance));
        Directory.CreateDirectory(directory);
        if (Directory.EnumerateFileSystemEntries(directory).Any())
            throw new IOException("A dataset output directory must be empty; existing artifacts are never replaced.");
        string partial = Path.Combine(directory, "decisions.arrow.partial");
        var progress = new Progress();
        var manifest = new DecisionArrowManifest
        {
            Status = "partial", ContractJson = contract.CanonicalJson, FeatureFingerprint = contract.FeatureFingerprint,
            DataFile = Path.GetFileName(partial), RowCount = 0, OutputBatchSize = batchSize, Provenance = provenance
        };
        var timer = Stopwatch.StartNew();
        long allocated = GC.GetTotalAllocatedBytes(precise: true);
        try
        {
            await PublishAsync(Path.Combine(directory, "run.partial.json"), manifest, CancellationToken.None);
            await File.WriteAllTextAsync(Path.Combine(directory, "contract.json"), contract.CanonicalJson, cancellationToken);
            await using (var file = new FileStream(partial, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                using var writer = new ArrowStreamWriter(file, contract.Compiled.Schema, leaveOpen: true);
                await writer.WriteStartAsync(cancellationToken);
                var ids = new HashSet<long>();
                await foreach (var batch in batches(ids, progress, cancellationToken).WithCancellation(cancellationToken))
                {
                    using (batch)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        await writer.WriteRecordBatchAsync(batch, cancellationToken);
                    }
                }
                await writer.WriteEndAsync(cancellationToken);
                await file.FlushAsync(cancellationToken);
                file.Flush(flushToDisk: true);
            }
            string hash = await FileHashAsync(partial, cancellationToken);
            cancellationToken.ThrowIfCancellationRequested();
            string data = Path.Combine(directory, "decisions.arrow");
            File.Move(partial, data);
            var measurements = provenance.Measurements?.ToDictionary(p => p.Key, p => p.Value, StringComparer.Ordinal)
                ?? new Dictionary<string, double>(StringComparer.Ordinal);
            measurements.Add("exportBeforeManifestPublicationMilliseconds", timer.Elapsed.TotalMilliseconds);
            measurements.Add("exportProcessAllocatedBytes", GC.GetTotalAllocatedBytes(precise: true) - allocated);
            manifest = manifest with
            {
                Status = "complete", DataFile = Path.GetFileName(data), DataSha256 = hash,
                DataSize = new FileInfo(data).Length, RowCount = progress.Rows,
                Provenance = provenance with { Measurements = new ReadOnlyDictionary<string, double>(measurements) }
            };
            if (beforePublication is not null) await beforePublication(manifest, cancellationToken);
            File.Delete(Path.Combine(directory, "run.partial.json"));
            await PublishAsync(Path.Combine(directory, "manifest.json"), manifest, cancellationToken);
            return manifest;
        }
        catch (Exception error)
        {
            manifest = manifest with
            {
                Status = error is OperationCanceledException ? "cancelled" : "failed", RowCount = progress.Rows,
                Error = $"{error.GetType().Name}: {error.Message}", DataSha256 = null, DataSize = null
            };
            // This record is never a completed manifest; recording failure must not hide the original error.
            try { await PublishAsync(Path.Combine(directory, "run.partial.json"), manifest, CancellationToken.None); }
            catch (Exception recordingError) { throw new AggregateException("Dataset failed and its failure record could not be written.", error, recordingError); }
            throw;
        }
    }

    internal static async Task<string> FileHashAsync(string path, CancellationToken cancellationToken)
    {
        await using var file = File.OpenRead(path);
        return Convert.ToHexStringLower(await SHA256.HashDataAsync(file, cancellationToken));
    }

    private static async Task PublishAsync(string path, DecisionArrowManifest manifest, CancellationToken cancellationToken)
    {
        string temporary = path + ".tmp";
        await using (var file = new FileStream(temporary, FileMode.Create, FileAccess.Write, FileShare.None))
        {
            await JsonSerializer.SerializeAsync(file, manifest, DecisionArrowManifest.JsonOptions, cancellationToken);
            await file.FlushAsync(cancellationToken);
            file.Flush(flushToDisk: true);
        }
        cancellationToken.ThrowIfCancellationRequested();
        File.Move(temporary, path, overwrite: true);
    }
}
