using DecisionInference.Arrow;
using Apache.Arrow;

if (args.Length != 2) throw new ArgumentException("Pass completed fixture manifest and trusted expected contract.");
var contract = DecisionArrowSchema.FromCanonicalJson(await File.ReadAllTextAsync(args[1]));
using var reader = await DecisionArrowDatasetReader.OpenAsync(args[0], contract, Enumerable.Range(1, 257).Select(i => (long)i));
long count = 0;
RecordBatch? batch;
while ((batch = await reader.ReadNextRecordBatchAsync()) is not null)
{
    using (batch)
    {
        var ids = (Int64Array)batch.Column(0);
        for (int i = 0; i < batch.Length; i++)
            if (ids.GetValue(i) != count + i + 1) throw new InvalidDataException("Fixture ID mismatch.");
        count += batch.Length;
    }
}
if (count != 257) throw new InvalidDataException("Fixture count mismatch.");
Console.WriteLine($"Package-only standard Arrow IPC interoperability verified: {count} rows, {contract.FeatureFingerprint}.");
