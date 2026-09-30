using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Apache.Arrow;
using Apache.Arrow.Ipc;
using Apache.Arrow.Types;
using DecisionInference.Arrow;

namespace DecisionInference.Tests;

internal static class DecisionArrowTestSupport
{
    internal const string BinaryJson = """{"id":"bin","kind":"binary","instructions":" Decide ","trueDescription":null,"falseDescription":null}""";
    internal const string ChoiceJson = """{"id":"pick","kind":"choice","instructions":" Pick ","candidates":[{"id":"z","description":"Zed"},{"id":"A","description":"Upper"},{"id":"a","description":"Lower"}]}""";
    internal const string ScoreJson = """{"id":"score","kind":"score","instructions":" Rate ","rubric":["low","middle","high"]}""";
    internal const string GoldenJson = """{"schemaVersion":1,"numericRepresentation":"float64","precision":"HighPrecision","questions":[{"id":"bin","kind":"binary","instructions":" Decide ","trueDescription":null,"falseDescription":null},{"id":"pick","kind":"choice","instructions":" Pick ","candidates":[{"id":"z","description":"Zed"},{"id":"A","description":"Upper"},{"id":"a","description":"Lower"}]},{"id":"score","kind":"score","instructions":" Rate ","rubric":["low","middle","high"]}],"provider":" provider ","modelId":"Model","assetHashes":{"A":"0000000000000000000000000000000000000000000000000000000000000000","z":"1111111111111111111111111111111111111111111111111111111111111111"},"preprocessing":" prep ","decoder":" decoder ","featureProjection":["bin","pick[2]","score[0]"]}""";
    internal const string GoldenHash = "51c5dcca8275938a792407753bf835fe95949f43a99a1281a649e9e4effebb12";
    internal static readonly byte[] Eos = [255, 255, 255, 255, 0, 0, 0, 0];

    internal static DecisionQuestion[] Questions() =>
    [
        new BinaryDecisionQuestion("bin", " Decide "),
        new ChoiceDecisionQuestion("pick", " Pick ",
            [new("z", "Zed"), new("A", "Upper"), new("a", "Lower")]),
        new ScoreDecisionQuestion("score", " Rate ", ["low", "middle", "high"])
    ];

    internal static DecisionArrowIdentity Identity(IEnumerable<string>? projection = null) =>
        new(" provider ", "Model", [KeyValuePair.Create("z", new string('1', 64)), KeyValuePair.Create("A", new string('0', 64))],
            " prep ", " decoder ", projection ?? ["bin", "pick[2]", "score[0]"]);

    internal static DecisionArrowSchema Contract(DecisionPrecision precision = DecisionPrecision.HighPrecision) =>
        new(Questions(), precision, Identity());

    internal static DecisionArrowProvenance Provenance() => new()
    {
        SourceRepository = "offline/repository", SourceCommit = "commit-a",
        InputSha256 = new string('2', 64), QuestionsSha256 = new string('3', 64),
        Packages = [new("Apache.Arrow", "23-test-a", new string('4', 64), 12345)],
        Machine = "test-machine", Runtime = ".NET 10", ExecutionMode = "offline",
        NativeStateBatchSize = 8, CpuThreads = 2,
        ModelInputShapes = new Dictionary<string, int[]> { ["input_ids"] = [1, 17] },
        AuxiliaryAssetHashes = new Dictionary<string, string> { ["vocabulary"] = new string('6', 64) },
        Measurements = new Dictionary<string, double> { ["callerObservation"] = 42.5 }
    };

    internal static DecisionArrowRow Row(long id, DecisionArrowSchema contract, double? binary = null,
        double[]? score = null, double? native = null)
    {
        double probability = binary ?? contract.Precision switch
        {
            DecisionPrecision.HighPrecision => 0.12345678901234567,
            DecisionPrecision.FourDecimalPlaces => 0.2718,
            _ => 0.27
        };
        score ??= [0.25, 0.5, 0.25];
        var input = new DecisionInput(JsonSerializer.SerializeToElement(new { id, label = "offline" }), contract.Questions);
        // Both enumerations intentionally differ from the contract's order.
        var result = new DecisionResult(input, contract.Identity.ModelId,
        [
            new ScoreDecisionAnswer("score", score, native, contract.Precision),
            new ChoiceDecisionAnswer("pick", "a",
                [KeyValuePair.Create("a", .1), KeyValuePair.Create("z", .1), KeyValuePair.Create("A", .8)], contract.Precision),
            new BinaryDecisionAnswer("bin", probability, contract.Precision)
        ]);
        return new(id, input, result);
    }

    internal static CancellationToken Token => TestContext.Current.CancellationToken;

    internal static IAsyncEnumerable<DecisionArrowRow> Rows(IEnumerable<DecisionArrowRow> rows) =>
        ProduceRows(rows, Token);

    private static async IAsyncEnumerable<DecisionArrowRow> ProduceRows(IEnumerable<DecisionArrowRow> rows,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        foreach (var row in rows)
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return row;
        }
    }

    internal static string ManifestPath(string directory) => Path.Combine(directory, "manifest.json");
    internal static string DataPath(string directory) => Path.Combine(directory, "decisions.arrow");
    internal static string Hash(byte[] bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));

    internal static DecisionArrowManifest ReadManifest(string path) =>
        JsonSerializer.Deserialize<DecisionArrowManifest>(File.ReadAllText(path), DecisionArrowManifest.JsonOptions)!;

    internal static void SaveManifest(string directory, DecisionArrowManifest manifest) =>
        File.WriteAllText(ManifestPath(directory), JsonSerializer.Serialize(manifest, DecisionArrowManifest.JsonOptions));

    internal static void AssertIntegrity(string directory, DecisionArrowManifest manifest)
    {
        byte[] bytes = File.ReadAllBytes(DataPath(directory));
        Assert.Equal(bytes.LongLength, manifest.DataSize);
        Assert.Equal(Hash(bytes), manifest.DataSha256);
        Assert.Equal(Eos, bytes[^8..]);
    }

    internal static void AssertReleased(string path)
    {
        using (var handle = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None))
            Assert.True(handle.Length > 0, "The test must exercise a real data artifact.");
        File.Delete(path);
        Assert.False(File.Exists(path));
    }

    internal static void AssertMetadata(IReadOnlyDictionary<string, string>? actual, params (string Key, string Value)[] expected)
    {
        Assert.Equal(expected.Length, actual?.Count ?? 0);
        foreach (var (key, value) in expected) Assert.Equal(value, actual![key]);
    }

    internal static void AssertGoldenSchema(Schema schema, DecisionPrecision precision)
    {
        string canonical = GoldenJson.Replace("\"HighPrecision\"", $"\"{precision}\"", StringComparison.Ordinal);
        string fingerprint = precision == DecisionPrecision.HighPrecision ? GoldenHash : Hash(Encoding.UTF8.GetBytes(canonical));
        AssertMetadata(schema.Metadata, ("decision_arrow.version", "1"), ("decision_arrow.contract", canonical),
            ("decision_arrow.feature_fingerprint", fingerprint));
        Assert.Equal(["row_id", "bin", "pick", "score"], schema.FieldsList.Select(f => f.Name).ToArray());
        Field ids = schema.FieldsList[0];
        Assert.IsType<Int64Type>(ids.DataType);
        Assert.False(ids.IsNullable);
        AssertMetadata(ids.Metadata);
        for (int i = 1; i < 4; i++)
        {
            var field = schema.FieldsList[i];
            Assert.False(field.IsNullable);
            AssertMetadata(field.Metadata,
                ("decision_arrow.definition", new[] { BinaryJson, ChoiceJson, ScoreJson }[i - 1]),
                ("decision_arrow.precision", precision.ToString()));
        }
        Assert.IsType<DoubleType>(schema.FieldsList[1].DataType);
        var choice = Assert.IsType<StructType>(schema.FieldsList[2].DataType);
        Assert.Equal(["selected_index", "probabilities"], choice.Fields.Select(f => f.Name).ToArray());
        Assert.IsType<Int32Type>(choice.Fields[0].DataType);
        Assert.False(choice.Fields[0].IsNullable);
        AssertMetadata(choice.Fields[0].Metadata);
        AssertList(choice.Fields[1]);
        var score = Assert.IsType<StructType>(schema.FieldsList[3].DataType);
        Assert.Equal(["expected_index", "expected_index_lower_bound", "expected_index_upper_bound", "native_score", "probabilities"],
            score.Fields.Select(f => f.Name).ToArray());
        for (int i = 0; i < 4; i++)
        {
            Assert.IsType<DoubleType>(score.Fields[i].DataType);
            Assert.Equal(i == 3, score.Fields[i].IsNullable);
            AssertMetadata(score.Fields[i].Metadata);
        }
        AssertList(score.Fields[4]);

        static void AssertList(Field field)
        {
            Assert.False(field.IsNullable);
            AssertMetadata(field.Metadata);
            var list = Assert.IsType<FixedSizeListType>(field.DataType);
            Assert.Equal(3, list.ListSize);
            Assert.Equal("item", list.ValueField.Name);
            Assert.IsType<DoubleType>(list.ValueField.DataType);
            Assert.False(list.ValueField.IsNullable);
            AssertMetadata(list.ValueField.Metadata);
        }
    }

    // Values are specified independently of the validated abstraction constructors.
    internal sealed record RawRow
    {
        internal long? Id { get; init; } = 41;
        internal double? Binary { get; init; } = .125;
        internal int? Selected { get; init; } = 2;
        internal double?[] Choice { get; init; } = [.1, .8, .1];
        internal double?[] Score { get; init; } = [.25, .5, .25];
        internal double? Expected { get; init; } = 1;
        internal double? Lower { get; init; } = .999998;
        internal double? Upper { get; init; } = 1.000002;
        internal double? Native { get; init; } = 1;
        internal bool ChoiceNull { get; init; }
        internal bool ScoreNull { get; init; }
        internal bool ChoiceVectorNull { get; init; }
        internal bool ScoreVectorNull { get; init; }
        internal bool ExtraBinaryStorage { get; init; }
    }

    internal static async Task<DecisionArrowManifest> WriteRawAsync(string directory, DecisionArrowSchema contract,
        IReadOnlyList<RawRow[]> batches, Schema? schema = null, int bound = 256)
    {
        Directory.CreateDirectory(directory);
        schema ??= contract.Schema;
        await using (var file = new FileStream(DataPath(directory), FileMode.Create, FileAccess.Write, FileShare.None))
        {
            using var writer = new ArrowStreamWriter(file, schema, leaveOpen: true);
            await writer.WriteStartAsync();
            foreach (var rows in batches)
            {
                using var batch = BuildRawBatch(schema, rows);
                await writer.WriteRecordBatchAsync(batch);
            }
            await writer.WriteEndAsync();
        }
        byte[] data = File.ReadAllBytes(DataPath(directory));
        var manifest = new DecisionArrowManifest
        {
            Status = "complete", ContractJson = contract.CanonicalJson, FeatureFingerprint = contract.FeatureFingerprint,
            DataFile = "decisions.arrow", DataSize = data.LongLength, DataSha256 = Hash(data),
            RowCount = batches.Sum(b => (long)b.Length), OutputBatchSize = bound, Provenance = Provenance()
        };
        SaveManifest(directory, manifest);
        AssertIntegrity(directory, manifest);
        return manifest;
    }

    internal static RecordBatch BuildRawBatch(Schema schema, RawRow[] rows)
    {
        var ids = new Int64Array.Builder();
        var selected = new Int32Array.Builder();
        foreach (var row in rows)
        {
            if (row.Id is { } id) ids.Append(id); else ids.AppendNull();
            if (row.Selected is { } index) selected.Append(index); else selected.AppendNull();
        }
        var choiceType = (StructType)schema.FieldsList[2].DataType;
        var scoreType = (StructType)schema.FieldsList[3].DataType;
        var choice = new StructArray(choiceType, rows.Length,
            [selected.Build(), Vector((FixedSizeListType)choiceType.Fields[1].DataType, rows, false)],
            Bitmap(rows.Select(r => !r.ChoiceNull)), rows.Count(r => r.ChoiceNull));
        var score = new StructArray(scoreType, rows.Length,
            [Doubles(rows.Select(r => r.Expected)), Doubles(rows.Select(r => r.Lower)), Doubles(rows.Select(r => r.Upper)),
                Doubles(rows.Select(r => r.Native)), Vector((FixedSizeListType)scoreType.Fields[4].DataType, rows, true)],
            Bitmap(rows.Select(r => !r.ScoreNull)), rows.Count(r => r.ScoreNull));
        var binaryValues = rows.Select(r => r.Binary);
        if (rows.Any(r => r.ExtraBinaryStorage)) binaryValues = binaryValues.Append(.125);
        return new(schema, [ids.Build(), Doubles(binaryValues), choice, score], rows.Length);
    }

    private static DoubleArray Doubles(IEnumerable<double?> values)
    {
        var builder = new DoubleArray.Builder();
        foreach (var value in values)
            if (value is { } number) builder.Append(number); else builder.AppendNull();
        return builder.Build();
    }

    private static ArrowBuffer Bitmap(IEnumerable<bool> values)
    {
        var builder = new ArrowBuffer.BitmapBuilder();
        foreach (bool value in values) builder.Append(value);
        return builder.Build();
    }

    private static FixedSizeListArray Vector(FixedSizeListType type, RawRow[] rows, bool score)
    {
        // Use the public constructor to keep the parent validity independent of the child bitmap.
        return new(type, rows.Length, Doubles(rows.SelectMany(r => score ? r.Score : r.Choice)),
            Bitmap(rows.Select(r => score ? !r.ScoreVectorNull : !r.ChoiceVectorNull)),
            rows.Count(r => score ? r.ScoreVectorNull : r.ChoiceVectorNull));
    }

    internal sealed class TempDirectory : IDisposable
    {
        internal string Root { get; } = Path.Combine(Path.GetTempPath(), "DecisionArrowTests-" + Guid.NewGuid().ToString("N"));
        internal string Dataset => Path.Combine(Root, "dataset");
        internal TempDirectory() => Directory.CreateDirectory(Root);
        public void Dispose()
        {
            if (Directory.Exists(Root)) Directory.Delete(Root, recursive: true);
        }
    }
}
