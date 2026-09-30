using DecisionInference.Arrow;
using Apache.Arrow;
using System.Runtime.CompilerServices;

if (args.Length == 4 && args[0] == "roundtrip")
{
    var trusted = DecisionArrowSchema.FromCanonicalJson(await File.ReadAllTextAsync(args[2]));
    long[] ids = Enumerable.Range(1, 257).Select(i => (long)i).ToArray();
    using var input = await DecisionArrowDatasetReader.OpenAsync(args[1], trusted, ids);
    var output = await DecisionArrowWriter.WriteBatchesAsync(args[3], trusted, OwnedBatches(input),
        input.Manifest.Provenance with { Measurements = null });
    using var verified = await DecisionArrowDatasetReader.OpenAsync(Path.Combine(args[3], "manifest.json"), trusted, ids);
    if (output.RowCount != 257 || output.DataSha256 != input.Manifest.DataSha256)
        throw new InvalidDataException("Owned standard-batch package roundtrip changed bytes/count.");
    Console.WriteLine($"Package-only owned standard RecordBatch ABI/byte roundtrip verified: 257 rows, {trusted.Precision}.");
    return;
}

if (args.Length is not (2 or 3)) throw new ArgumentException("Pass completed manifest, trusted expected contract and optional expected contiguous source-row count.");
int expectedCount = args.Length == 3 ? int.Parse(args[2], System.Globalization.CultureInfo.InvariantCulture) : 257;
ArgumentOutOfRangeException.ThrowIfNegative(expectedCount);
var contract = DecisionArrowSchema.FromCanonicalJson(await File.ReadAllTextAsync(args[1]));
using var reader = await DecisionArrowDatasetReader.OpenAsync(args[0], contract, Enumerable.Range(1, expectedCount).Select(i => (long)i));
long count = 0;
RecordBatch? batch;
while ((batch = await reader.ReadNextRecordBatchAsync()) is not null)
{
    using (batch)
    {
        var ids = (Int64Array)batch.Column(0);
        for (int i = 0; i < batch.Length; i++)
            if (ids.GetValue(i) != count + i + 1) throw new InvalidDataException("Source ID mismatch.");
        count += batch.Length;
    }
}
if (count != expectedCount) throw new InvalidDataException("Source count mismatch.");
Console.WriteLine($"Package-only standard Arrow IPC interoperability verified: {count} rows, {contract.FeatureFingerprint}.");

static async IAsyncEnumerable<RecordBatch> OwnedBatches(DecisionArrowDatasetReader reader,
    [EnumeratorCancellation] CancellationToken token = default)
{
    while (await reader.ReadNextRecordBatchAsync(token) is { } batch) yield return batch;
}
