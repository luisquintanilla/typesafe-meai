using System.Security.Cryptography;
using System.Text.Json;
using System.Diagnostics;
using System.Collections.ObjectModel;
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
        ArgumentNullException.ThrowIfNull(provenance);
        if (batchSize is < 1 or > 65536) throw new ArgumentOutOfRangeException(nameof(batchSize));
        if (provenance.Measurements?.ContainsKey("exportBeforeManifestPublicationMilliseconds") == true ||
            provenance.Measurements?.ContainsKey("exportProcessAllocatedBytes") == true)
            throw new ArgumentException("Export measurement names are reserved for the writer.", nameof(provenance));
        Directory.CreateDirectory(directory);
        if (Directory.EnumerateFileSystemEntries(directory).Any())
            throw new IOException("A dataset output directory must be empty; existing artifacts are never replaced.");
        string partial = Path.Combine(directory, "decisions.arrow.partial");
        long count = 0;
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
                using var writer = new ArrowStreamWriter(file, contract.Schema, leaveOpen: true);
                await writer.WriteStartAsync(cancellationToken);
                var ids = new HashSet<long>();
                var builder = new DecisionArrowBatchBuilder(contract, batchSize);
                await foreach (var row in rows.WithCancellation(cancellationToken))
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    ArgumentNullException.ThrowIfNull(row);
                    if (!ids.Add(row.RowId)) throw new DecisionProtocolException($"Duplicate source row ID {row.RowId}.");
                    builder.Append(row);
                    count = checked(count + 1);
                    if (builder.Count == batchSize)
                    {
                        using var batch = builder.Build();
                        await writer.WriteRecordBatchAsync(batch, cancellationToken);
                        builder = new(contract, batchSize);
                    }
                }
                if (builder.Count > 0)
                {
                    using var batch = builder.Build();
                    await writer.WriteRecordBatchAsync(batch, cancellationToken);
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
                DataSize = new FileInfo(data).Length, RowCount = count,
                Provenance = provenance with { Measurements = new ReadOnlyDictionary<string, double>(measurements) }
            };
            File.Delete(Path.Combine(directory, "run.partial.json"));
            await PublishAsync(Path.Combine(directory, "manifest.json"), manifest, cancellationToken);
            return manifest;
        }
        catch (Exception error)
        {
            manifest = manifest with
            {
                Status = error is OperationCanceledException ? "cancelled" : "failed", RowCount = count,
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
