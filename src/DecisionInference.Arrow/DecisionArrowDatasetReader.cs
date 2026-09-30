using System.Security.Cryptography;
using System.Text.Json;
using Apache.Arrow;
using Apache.Arrow.Ipc;
using Apache.Arrow.Types;

namespace DecisionInference.Arrow;

/// <summary>Validates the entire dataset before exposure. Returned batches belong to the caller.</summary>
public sealed class DecisionArrowDatasetReader : IArrowArrayStream
{
    private readonly FileStream _file;
    private readonly ArrowStreamReader _reader;
    public DecisionArrowManifest Manifest { get; }
    public DecisionArrowSchema Contract { get; }
    public Schema Schema => Contract.Schema;

    private DecisionArrowDatasetReader(FileStream file, ArrowStreamReader reader, DecisionArrowManifest manifest, DecisionArrowSchema contract)
        => (_file, _reader, Manifest, Contract) = (file, reader, manifest, contract);

    public static async Task<DecisionArrowDatasetReader> OpenAsync(string manifestPath, DecisionArrowSchema expectedContract,
        IEnumerable<long> expectedRowIds, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(expectedRowIds);
        var expected = new HashSet<long>();
        foreach (long id in expectedRowIds)
            if (!expected.Add(id)) throw new ArgumentException($"Duplicate expected row ID {id}.", nameof(expectedRowIds));
        return await OpenCoreAsync(manifestPath, expectedContract, expected, cancellationToken);
    }

    public static Task<DecisionArrowDatasetReader> OpenAsync(string manifestPath, DecisionArrowSchema expectedContract,
        CancellationToken cancellationToken = default) => OpenCoreAsync(manifestPath, expectedContract, null, cancellationToken);

    private static async Task<DecisionArrowDatasetReader> OpenCoreAsync(string manifestPath, DecisionArrowSchema expectedContract,
        HashSet<long>? expected, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(expectedContract);
        await using var manifestFile = File.OpenRead(manifestPath);
        var manifest = await JsonSerializer.DeserializeAsync<DecisionArrowManifest>(manifestFile,
            DecisionArrowManifest.JsonOptions, cancellationToken) ?? throw new InvalidDataException("Missing manifest.");
        if (manifest.SchemaVersion != DecisionArrowSchema.Version || manifest.Status != "complete" || manifest.Error is not null ||
            manifest.RowCount < 0 || manifest.OutputBatchSize is < 1 or > 65536 ||
            manifest.DataSize is null || !DecisionArrowIdentity.IsHash(manifest.DataSha256))
            throw new InvalidDataException("Dataset is incomplete or has an invalid manifest.");
        var contract = DecisionArrowSchema.FromCanonicalJson(manifest.ContractJson);
        if (contract.FeatureFingerprint != manifest.FeatureFingerprint ||
            contract.CanonicalJson != expectedContract.CanonicalJson)
            throw new InvalidDataException("Feature contract, questions or model identity mismatch.");
        if (string.IsNullOrWhiteSpace(manifest.DataFile) || Path.GetFileName(manifest.DataFile) != manifest.DataFile ||
            manifest.DataFile != "decisions.arrow")
            throw new InvalidDataException("Invalid completed data filename.");
        string path = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(manifestPath))!, manifest.DataFile);
        var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        ArrowStreamReader? reader = null;
        try
        {
            if (file.Length != manifest.DataSize) throw new InvalidDataException("Dataset size mismatch.");
            string hash = Convert.ToHexStringLower(await SHA256.HashDataAsync(file, cancellationToken));
            if (hash != manifest.DataSha256) throw new InvalidDataException("Dataset hash mismatch.");
            if (file.Length < 8) throw new InvalidDataException("Missing IPC completion marker.");
            file.Position = file.Length - 8;
            byte[] end = new byte[8];
            await file.ReadExactlyAsync(end, cancellationToken);
            if (!end.AsSpan().SequenceEqual(new byte[] { 255, 255, 255, 255, 0, 0, 0, 0 }))
                throw new InvalidDataException("Missing IPC completion marker.");
            file.Position = 0;
            reader = new ArrowStreamReader(file, leaveOpen: true);
            long count = 0;
            var seen = new HashSet<long>();
            RecordBatch? batch;
            while ((batch = await reader.ReadNextRecordBatchAsync(cancellationToken)) is not null)
            {
                using (batch)
                {
                    ValidateSchema(contract.Compiled.Schema, batch.Schema);
                    if (batch.Length > manifest.OutputBatchSize) throw new InvalidDataException("Batch exceeds the declared output bound.");
                    ValidateBatch(contract, batch);
                    var ids = (Int64Array)batch.Column(0);
                    for (int i = 0; i < batch.Length; i++)
                    {
                        long id = ids.GetValue(i)!.Value;
                        if (!seen.Add(id)) throw new InvalidDataException($"Duplicate source row ID {id}.");
                        if (expected is not null && !expected.Contains(id)) throw new InvalidDataException($"Unexpected source row ID {id}.");
                    }
                    count = checked(count + batch.Length);
                }
            }
            // The pinned FileStream reader consumes exact message buffers; Position is the logical consumed offset.
            if (file.Position != file.Length)
                throw new InvalidDataException("Trailing data or concatenated IPC streams after the completion marker.");
            ValidateSchema(contract.Compiled.Schema, reader.Schema);
            if (count != manifest.RowCount || (expected is not null && !seen.SetEquals(expected)))
                throw new InvalidDataException("Dataset row count or source row IDs mismatch.");
            reader.Dispose();
            file.Position = 0;
            reader = new ArrowStreamReader(file, leaveOpen: true);
            // Reading the schema during validation has not consumed a batch on this newly owned reader.
            return new(file, reader, manifest, contract);
        }
        catch
        {
            reader?.Dispose();
            await file.DisposeAsync();
            throw;
        }
    }

    public ValueTask<RecordBatch> ReadNextRecordBatchAsync(CancellationToken cancellationToken = default) =>
        _reader.ReadNextRecordBatchAsync(cancellationToken);

    public void Dispose()
    {
        _reader.Dispose();
        _file.Dispose();
    }

    internal static void ValidateSchema(Schema expected, Schema actual)
    {
        if (!MetadataEqual(expected.Metadata, actual.Metadata) || expected.FieldsList.Count != actual.FieldsList.Count)
            throw new InvalidDataException("Arrow schema metadata or field count mismatch.");
        for (int i = 0; i < expected.FieldsList.Count; i++) ValidateField(expected.FieldsList[i], actual.FieldsList[i]);
    }

    private static void ValidateField(Field expected, Field actual)
    {
        if (expected.Name != actual.Name || expected.IsNullable != actual.IsNullable ||
            expected.DataType.TypeId != actual.DataType.TypeId || !MetadataEqual(expected.Metadata, actual.Metadata))
            throw new InvalidDataException($"Arrow field contract mismatch: {expected.Name}.");
        switch (expected.DataType)
        {
            case StructType s:
                var other = (StructType)actual.DataType;
                if (s.Fields.Count != other.Fields.Count) throw new InvalidDataException("Struct field count mismatch.");
                for (int i = 0; i < s.Fields.Count; i++) ValidateField(s.Fields[i], other.Fields[i]);
                break;
            case FixedSizeListType l:
                var list = (FixedSizeListType)actual.DataType;
                if (l.ListSize != list.ListSize) throw new InvalidDataException("Probability vector width mismatch.");
                ValidateField(l.ValueField, list.ValueField);
                break;
        }
    }

    private static bool MetadataEqual(IReadOnlyDictionary<string, string>? first, IReadOnlyDictionary<string, string>? second) =>
        (first?.Count ?? 0) == (second?.Count ?? 0) &&
        (first is null || first.All(p => second is not null && second.TryGetValue(p.Key, out string? value) && value == p.Value));

    internal static void ValidateBatch(DecisionArrowSchema contract, RecordBatch batch)
    {
        if (batch.ColumnCount != contract.Questions.Count + 1) throw new InvalidDataException("Column count mismatch.");
        for (int column = 0; column < batch.ColumnCount; column++)
        {
            var array = batch.Column(column);
            if (array.Length != batch.Length || array.NullCount != 0) throw new InvalidDataException("Missing non-null observations.");
        }
        for (int q = 0; q < contract.Questions.Count; q++)
        {
            var question = contract.Questions[q];
            if (question is BinaryDecisionQuestion)
            {
                var binary = (DoubleArray)batch.Column(q + 1);
                for (int row = 0; row < batch.Length; row++)
                    DecisionObservationValidation.Binary(Required(binary, row), contract.Precision);
                continue;
            }
            var structure = (StructArray)batch.Column(q + 1);
            var fields = structure.Fields;
            int fieldCount = question.Kind == DecisionKind.Choice ? 2 : 5;
            if (fields.Count != fieldCount) throw new InvalidDataException("Struct field count mismatch.");
            foreach (var field in fields)
                if (field.Length != batch.Length) throw new InvalidDataException("Nested field length mismatch.");
            var vector = (FixedSizeListArray)fields[^1];
            int width = ((FixedSizeListType)vector.Data.DataType).ListSize;
            if (width != contract.Compiled.Widths[q]) throw new InvalidDataException("Probability vector width mismatch.");
            var values = (DoubleArray)vector.Values;
            var selectedArray = question.Kind == DecisionKind.Choice ? (Int32Array)fields[0] : null;
            var expectedArray = question.Kind == DecisionKind.Score ? (DoubleArray)fields[0] : null;
            var lowerArray = question.Kind == DecisionKind.Score ? (DoubleArray)fields[1] : null;
            var upperArray = question.Kind == DecisionKind.Score ? (DoubleArray)fields[2] : null;
            var nativeArray = question.Kind == DecisionKind.Score ? (DoubleArray)fields[3] : null;
            for (int row = 0; row < batch.Length; row++)
            {
                if (vector.IsNull(row)) throw new InvalidDataException("Null probability vector.");
                int start = checked((row + vector.Offset) * width);
                if (start < 0 || start > values.Length - width) throw new InvalidDataException("Probability vector offset outside values.");
                var probabilities = values.Values.Slice(start, width);
                for (int i = 0; i < width; i++) _ = Required(values, checked(start + i));
                if (question is ChoiceDecisionQuestion)
                {
                    int selected = selectedArray!.GetValue(row) ?? throw new InvalidDataException("Missing selected index.");
                    if (selected < 0 || selected >= width) throw new InvalidDataException("Selected index outside candidate domain.");
                    DecisionObservationValidation.Distribution(probabilities, contract.Precision);
                }
                else
                {
                    var native = nativeArray!.GetValue(row);
                    var answer = DecisionObservationValidation.Score(probabilities, native, contract.Precision);
                    if (Required(expectedArray!, row) != answer.Expected ||
                        Required(lowerArray!, row) != answer.Lower ||
                        Required(upperArray!, row) != answer.Upper)
                        throw new InvalidDataException("Score expectation or portable rounding bounds mismatch.");
                }
            }
        }
    }

    private static double Required(DoubleArray array, int index) =>
        array.GetValue(index) is { } value && double.IsFinite(value) ? value : throw new InvalidDataException("Missing or nonfinite observation.");
}
