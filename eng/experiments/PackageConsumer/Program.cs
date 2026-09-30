using DecisionInference.Arrow;
using Apache.Arrow;

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
