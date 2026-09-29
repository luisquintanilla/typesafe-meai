using System.Text.Json;
using System.Text.Json.Nodes;
using Apache.Arrow;
using Apache.Arrow.Ipc;
using Apache.Arrow.Types;
using DecisionInference.Arrow;
using static DecisionInference.Tests.DecisionArrowTestSupport;

namespace DecisionInference.Tests;

public sealed class DecisionArrowDatasetReaderCorruptionTests
{
    [Fact]
    public async Task OpenAsync_RejectsTrailingAndConcatenatedStreamsEvenWithRefreshedIntegrityAndNoExpectedIds()
    {
        using var temp = new TempDirectory();
        var contract = Contract();
        var valid = await WriteRawAsync(temp.Dataset, contract, [[new RawRow()]]);
        byte[] original = File.ReadAllBytes(DataPath(temp.Dataset));
        foreach (byte[] bytes in new[]
        {
            original.Concat(original).ToArray(),
            original.Concat(new byte[] { 1, 2, 3 }).Concat(Eos).ToArray()
        })
        {
            File.WriteAllBytes(DataPath(temp.Dataset), bytes);
            var manifest = valid with { DataSha256 = Hash(bytes), DataSize = bytes.LongLength };
            SaveManifest(temp.Dataset, manifest);
            AssertIntegrity(temp.Dataset, manifest);
            var error = await Assert.ThrowsAsync<InvalidDataException>(() =>
                DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, Token));
            Assert.Equal("Trailing data or concatenated IPC streams after the completion marker.", error.Message);
            AssertReleased(DataPath(temp.Dataset));
        }
    }

    [Fact]
    public async Task RawIpcFixture_MissingColumnsCannotReachReaderViaPublicWriter()
    {
        // Explicit fixture-feasibility evidence, NOT a claim of reaching the reader's column-count guard.
        var schema = Contract().Schema;
        using var bytes = new MemoryStream();
        using var writer = new ArrowStreamWriter(bytes, schema, leaveOpen: true);
        await writer.WriteStartAsync(Token);
        using var incomplete = new RecordBatch(schema, [new Int64Array.Builder().Append(41).Build()], 1);
        var error = await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => writer.WriteRecordBatchAsync(incomplete, Token));
        Assert.Equal("index", error.ParamName);
        Assert.Equal(4, schema.FieldsList.Count);
        Assert.Equal(1, incomplete.ColumnCount);
        Assert.True(bytes.Length > 0); // Schema was written; the missing-array failure is in batch serialization.
    }

    [Theory]
    [InlineData("completion")]
    [InlineData("counts")]
    [InlineData("integrity-fields")]
    public async Task OpenAsync_RejectsInvalidCompletionEnvelope(string group)
    {
        using var temp = new TempDirectory();
        var contract = Contract();
        var valid = await WriteRawAsync(temp.Dataset, contract, [[new RawRow()]]);
        DecisionArrowManifest[] invalid = group switch
        {
            "completion" =>
            [
                valid with { SchemaVersion = 2 }, valid with { Status = "partial" }, valid with { Status = "failed" },
                valid with { Status = "cancelled" }, valid with { Status = "Complete" }, valid with { Error = "" }
            ],
            "counts" => [valid with { RowCount = -1 }, valid with { OutputBatchSize = 0 }, valid with { OutputBatchSize = 65537 }],
            _ =>
            [
                valid with { DataSize = null }, valid with { DataSha256 = null }, valid with { DataSha256 = "bad" },
                valid with { DataSha256 = new string('g', 64) }, valid with { DataSha256 = new string('A', 64) }
            ]
        };
        foreach (var manifest in invalid)
        {
            SaveManifest(temp.Dataset, manifest);
            var error = await Assert.ThrowsAsync<InvalidDataException>(() =>
                DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, [41L], Token));
            Assert.Equal("Dataset is incomplete or has an invalid manifest.", error.Message);
            Assert.Equal(valid.DataSha256, Hash(File.ReadAllBytes(DataPath(temp.Dataset))));
        }
        AssertReleased(DataPath(temp.Dataset));
    }

    [Fact]
    public async Task OpenAsync_RejectsMalformedMissingAndNullManifestJsonWithNativeCategories()
    {
        using var temp = new TempDirectory();
        var contract = Contract();
        var manifest = await WriteRawAsync(temp.Dataset, contract, []);
        var node = JsonNode.Parse(JsonSerializer.Serialize(manifest, DecisionArrowManifest.JsonOptions))!.AsObject();
        var unknown = node.DeepClone().AsObject();
        unknown["intruder"] = true;
        var missing = node.DeepClone().AsObject();
        missing.Remove("provenance");
        foreach (var json in new[] { "{", unknown.ToJsonString(), missing.ToJsonString() })
        {
            File.WriteAllText(ManifestPath(temp.Dataset), json);
            await Assert.ThrowsAsync<JsonException>(() => DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, [], Token));
            Assert.True(File.Exists(DataPath(temp.Dataset)));
        }
        File.WriteAllText(ManifestPath(temp.Dataset), "null");
        Assert.Equal("Missing manifest.", (await Assert.ThrowsAsync<InvalidDataException>(() =>
            DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, [], Token))).Message);
        File.Delete(ManifestPath(temp.Dataset));
        await Assert.ThrowsAsync<FileNotFoundException>(() => DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, [], Token));
        AssertReleased(DataPath(temp.Dataset));
    }

    [Fact]
    public async Task OpenAsync_RejectsPathBearingAndIncorrectDataFilename()
    {
        using var temp = new TempDirectory();
        var contract = Contract();
        var valid = await WriteRawAsync(temp.Dataset, contract, [[new RawRow()]]);
        foreach (string? filename in new string?[]
        {
            null, "", " ", "other.arrow", "Decisions.arrow", "../decisions.arrow", "nested/decisions.arrow",
            @"nested\decisions.arrow", DataPath(temp.Dataset)
        })
        {
            SaveManifest(temp.Dataset, valid with { DataFile = filename! });
            var error = await Assert.ThrowsAsync<InvalidDataException>(() =>
                DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, [41L], Token));
            Assert.Equal("Invalid completed data filename.", error.Message);
            Assert.Equal(valid.DataSize, new FileInfo(DataPath(temp.Dataset)).Length);
        }
        AssertReleased(DataPath(temp.Dataset));
    }

    [Theory]
    [InlineData("size", "Dataset size mismatch.")]
    [InlineData("hash", "Dataset hash mismatch.")]
    [InlineData("short", "Missing IPC completion marker.")]
    [InlineData("marker", "Missing IPC completion marker.")]
    [InlineData("missing", "")]
    public async Task OpenAsync_RejectsIndependentIntegrityAndCompletionMarkerFaults(string fault, string expectedMessage)
    {
        using var temp = new TempDirectory();
        var contract = Contract();
        var manifest = await WriteRawAsync(temp.Dataset, contract, [[new RawRow()]]);
        byte[] original = File.ReadAllBytes(DataPath(temp.Dataset));
        if (fault == "missing")
        {
            File.Delete(DataPath(temp.Dataset));
            var missing = await Assert.ThrowsAsync<FileNotFoundException>(() =>
                DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, [41L], Token));
            Assert.Equal(DataPath(temp.Dataset), missing.FileName);
            Assert.Equal(manifest.DataSize, original.LongLength);
            return;
        }
        if (fault == "size") manifest = manifest with { DataSize = original.LongLength + 1 };
        else if (fault == "hash")
        {
            byte[] changed = (byte[])original.Clone();
            changed[0] ^= 1;
            File.WriteAllBytes(DataPath(temp.Dataset), changed);
            Assert.Equal(original.Length, changed.Length);
            Assert.NotEqual(manifest.DataSha256, Hash(changed));
        }
        else
        {
            byte[] changed = fault == "short" ? [1, 2, 3] : original[..^8];
            File.WriteAllBytes(DataPath(temp.Dataset), changed);
            manifest = manifest with { DataSize = changed.LongLength, DataSha256 = Hash(changed) };
            Assert.Equal(Hash(File.ReadAllBytes(DataPath(temp.Dataset))), manifest.DataSha256);
            Assert.Equal(new FileInfo(DataPath(temp.Dataset)).Length, manifest.DataSize);
        }
        SaveManifest(temp.Dataset, manifest);
        var error = await Assert.ThrowsAsync<InvalidDataException>(() =>
            DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, [41L], Token));
        Assert.Equal(expectedMessage, error.Message);
        AssertReleased(DataPath(temp.Dataset));
    }

    [Theory]
    [InlineData("duplicate")]
    [InlineData("unexpected")]
    [InlineData("rowcount")]
    [InlineData("expected-set")]
    [InlineData("batch-bound")]
    public async Task OpenAsync_RejectsIdsCountsAndBatchBoundBeforeReturningReader(string fault)
    {
        using var temp = new TempDirectory();
        var contract = Contract();
        RawRow[][] batches = fault switch
        {
            "duplicate" => [[new RawRow()], [new RawRow()]],
            "batch-bound" => [[new RawRow { Id = 40 }, new RawRow()]],
            "expected-set" => [],
            _ => [[new RawRow { Id = 40 }], [new RawRow()]]
        };
        var valid = await WriteRawAsync(temp.Dataset, contract, batches, bound: 1);
        long[] expectedIds = fault switch
        {
            "duplicate" => [41],
            "unexpected" => [40, 99],
            "expected-set" => [41],
            _ => [40, 41]
        };
        if (fault == "rowcount") SaveManifest(temp.Dataset, valid with { RowCount = 1 });
        string message = fault switch
        {
            "duplicate" => "Duplicate source row ID 41.",
            "unexpected" => "Unexpected source row ID 41.",
            "batch-bound" => "Batch exceeds the declared output bound.",
            _ => "Dataset row count or source row IDs mismatch."
        };
        var error = await Assert.ThrowsAsync<InvalidDataException>(() =>
            DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, expectedIds, Token));
        Assert.Equal(message, error.Message);
        AssertIntegrity(temp.Dataset, valid);
        if (fault == "rowcount")
        {
            SaveManifest(temp.Dataset, valid with { RowCount = 3 });
            Assert.Equal(message, (await Assert.ThrowsAsync<InvalidDataException>(() =>
                DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, expectedIds, Token))).Message);
            SaveManifest(temp.Dataset, valid);
            Assert.Equal(message, (await Assert.ThrowsAsync<InvalidDataException>(() =>
                DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, [40L, 41L, 99L], Token))).Message);
        }
        if (fault == "duplicate")
        {
            await WriteRawAsync(temp.Dataset, contract, [[new RawRow(), new RawRow()]], bound: 2);
            Assert.Equal(message, (await Assert.ThrowsAsync<InvalidDataException>(() =>
                DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, [41L], Token))).Message);
        }
        AssertReleased(DataPath(temp.Dataset));
    }

    [Theory]
    [InlineData("coherent-semantic-variants")]
    [InlineData("manifest-contract")]
    public async Task OpenAsync_RejectsFeatureContractsEvenWithIndependentCorrectIpcIntegrity(string fault)
    {
        using var temp = new TempDirectory();
        var expected = Contract();
        if (fault == "coherent-semantic-variants")
        {
            foreach (string variant in DecisionArrowSchemaTests.SemanticVariants.Select(row => row.Data))
            {
                var changed = DecisionArrowSchemaTests.Variant(variant);
                var manifest = await WriteRawAsync(temp.Dataset, changed, []);
                Assert.NotEqual(GoldenHash, manifest.FeatureFingerprint);
                AssertIntegrity(temp.Dataset, manifest);
                var error = await Assert.ThrowsAsync<InvalidDataException>(() =>
                    DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), expected, [], Token));
                Assert.Equal("Feature contract, questions or model identity mismatch.", error.Message);
            }
        }
        else
        {
            var valid = await WriteRawAsync(temp.Dataset, expected, []);
            var variants = new[]
            {
                (valid with { FeatureFingerprint = new string('0', 64) }, "Feature contract, questions or model identity mismatch."),
                (valid with { ContractJson = " " + GoldenJson }, "Contract JSON is not canonical."),
                (valid with { ContractJson = GoldenJson.Replace("\"float64\"", "\"float32\"", StringComparison.Ordinal) },
                    "Unsupported decision Arrow schema version or numeric representation.")
            };
            foreach (var (manifest, message) in variants)
            {
                SaveManifest(temp.Dataset, manifest);
                var error = await Assert.ThrowsAsync<InvalidDataException>(() =>
                    DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), expected, [], Token));
                Assert.Equal(message, error.Message);
                AssertIntegrity(temp.Dataset, valid);
            }
        }
        AssertReleased(DataPath(temp.Dataset));
    }

    [Theory]
    [InlineData("schema-metadata")]
    [InlineData("question-metadata")]
    [InlineData("row-field")]
    [InlineData("field-count-order")]
    [InlineData("binary-role")]
    [InlineData("choice-children")]
    [InlineData("score-children")]
    [InlineData("choice-list")]
    [InlineData("score-list")]
    [InlineData("item-field")]
    [InlineData("score-item-field")]
    public async Task OpenAsync_RejectsRealIpcStructuralContractsIncludingSchemaOnlyStreams(string group)
    {
        using var temp = new TempDirectory();
        var contract = Contract();
        foreach (var (schema, message) in SchemaFaults(contract.Schema, group))
        {
            // A schema-only stream is valid Arrow IPC and must still undergo recursive validation.
            var manifest = await WriteRawAsync(temp.Dataset, contract, [], schema);
            Assert.Equal(0, manifest.RowCount);
            AssertIntegrity(temp.Dataset, manifest);
            var error = await Assert.ThrowsAsync<InvalidDataException>(() =>
                DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, [], Token));
            Assert.Equal(message, error.Message);
            if (group is "schema-metadata" or "question-metadata" or "row-field")
            {
                // These mutations preserve data types; additionally prove rejection on actual batches.
                if (schema.FieldsList[0].DataType is not Apache.Arrow.Types.Int64Type) continue;
                manifest = await WriteRawAsync(temp.Dataset, contract, [[new RawRow()]], schema);
                AssertIntegrity(temp.Dataset, manifest);
                error = await Assert.ThrowsAsync<InvalidDataException>(() =>
                    DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, [41L], Token));
                Assert.Equal(message, error.Message);
            }
        }
        AssertReleased(DataPath(temp.Dataset));
    }

    private static IEnumerable<(Schema Schema, string Message)> SchemaFaults(Schema original, string group)
    {
        const string schemaError = "Arrow schema metadata or field count mismatch.";
        var fields = original.FieldsList.ToArray();
        Schema Replace(int index, Field field)
        {
            var changed = (Field[])fields.Clone();
            changed[index] = field;
            return new(changed, original.Metadata);
        }
        Field Copy(Field field, string? name = null, IArrowType? type = null, bool? nullable = null,
            IReadOnlyDictionary<string, string>? metadata = null) =>
            new(name ?? field.Name, type ?? field.DataType, nullable ?? field.IsNullable, metadata ?? field.Metadata);
        Dictionary<string, string> Extra(IReadOnlyDictionary<string, string>? metadata, string key, string value) =>
            new(metadata ?? new Dictionary<string, string>(), StringComparer.Ordinal) { [key] = value };
        string FieldError(Field field) => $"Arrow field contract mismatch: {field.Name}.";
        if (group == "schema-metadata")
        {
            var missing = new Dictionary<string, string>(original.Metadata);
            missing.Remove("decision_arrow.version");
            foreach (var metadata in new[]
            {
                missing, Extra(original.Metadata, "decision_arrow.contract", "{}"),
                Extra(original.Metadata, "decision_arrow.feature_fingerprint", new string('0', 64)),
                Extra(original.Metadata, "extra", "ignored?"), Extra(original.Metadata, "jevaro.version", "legacy")
            })
                yield return (new(fields, metadata), schemaError);
        }
        else if (group == "question-metadata")
        {
            for (int i = 1; i < 4; i++)
            {
                yield return (Replace(i, Copy(fields[i], metadata: new Dictionary<string, string>())), FieldError(fields[i]));
                yield return (Replace(i, Copy(fields[i], metadata: Extra(fields[i].Metadata, "decision_arrow.precision", "TwoDecimalPlaces"))), FieldError(fields[i]));
                yield return (Replace(i, Copy(fields[i], metadata: Extra(fields[i].Metadata, "decision_arrow.definition", "{}"))), FieldError(fields[i]));
                yield return (Replace(i, Copy(fields[i], metadata: Extra(fields[i].Metadata, "extra", "value"))), FieldError(fields[i]));
            }
        }
        else if (group == "row-field")
        {
            foreach (var row in new[]
            {
                Copy(fields[0], name: "ROW_ID"), Copy(fields[0], type: Int32Type.Default),
                Copy(fields[0], nullable: true), Copy(fields[0], metadata: Extra(fields[0].Metadata, "extra", "value"))
            })
                yield return (Replace(0, row), FieldError(fields[0]));
        }
        else if (group == "field-count-order")
        {
            yield return (new(fields[..^1], original.Metadata), schemaError);
            yield return (new(fields.Concat([new Field("extra", DoubleType.Default, false)]), original.Metadata), schemaError);
            yield return (new([fields[0], fields[2], fields[1], fields[3]], original.Metadata), FieldError(fields[1]));
        }
        else if (group == "binary-role")
        {
            yield return (Replace(1, Copy(fields[1], type: FloatType.Default)), FieldError(fields[1]));
            yield return (Replace(1, Copy(fields[1], nullable: true)), FieldError(fields[1]));
            yield return (Replace(2, Copy(fields[2], nullable: true)), FieldError(fields[2]));
            yield return (Replace(3, Copy(fields[3], name: "Score")), FieldError(fields[3]));
        }
        else
        {
            bool choice = group is "choice-children" or "choice-list" or "item-field";
            int index = choice ? 2 : 3;
            var parent = fields[index];
            var children = ((StructType)parent.DataType).Fields.ToArray();
            int vectorIndex = children.Length - 1;
            Schema Children(Field[] changed) => Replace(index, Copy(parent, type: new StructType(changed)));
            if (group.EndsWith("children", StringComparison.Ordinal))
            {
                yield return (Children(children[..^1]), "Struct field count mismatch.");
                var swapped = (Field[])children.Clone();
                (swapped[0], swapped[1]) = (swapped[1], swapped[0]);
                yield return (Children(swapped), FieldError(children[0]));
                for (int i = 0; i < vectorIndex; i++)
                {
                    var changed = (Field[])children.Clone();
                    changed[i] = Copy(children[i], nullable: !children[i].IsNullable);
                    yield return (Children(changed), FieldError(children[i]));
                }
                var wrongType = (Field[])children.Clone();
                wrongType[0] = Copy(children[0], type: choice ? DoubleType.Default : Int32Type.Default);
                yield return (Children(wrongType), FieldError(children[0]));
                var childMeta = (Field[])children.Clone();
                childMeta[0] = Copy(children[0], metadata: new Dictionary<string, string> { ["extra"] = "value" });
                yield return (Children(childMeta), FieldError(children[0]));
            }
            else
            {
                var list = (FixedSizeListType)children[vectorIndex].DataType;
                bool itemField = group.EndsWith("item-field", StringComparison.Ordinal);
                var replacementLists = itemField
                    ? new[]
                    {
                        new FixedSizeListType(new Field("element", DoubleType.Default, false), 3),
                        new FixedSizeListType(new Field("item", FloatType.Default, false), 3),
                        new FixedSizeListType(new Field("item", DoubleType.Default, true), 3),
                        new FixedSizeListType(new Field("item", DoubleType.Default, false, new Dictionary<string, string> { ["extra"] = "value" }), 3)
                    }
                    : new[] { new FixedSizeListType(list.ValueField, 2) };
                foreach (var replacement in replacementLists)
                {
                    var changed = (Field[])children.Clone();
                    changed[vectorIndex] = Copy(children[vectorIndex], type: replacement);
                    yield return (Children(changed), itemField ? "Arrow field contract mismatch: item." : "Probability vector width mismatch.");
                }
                var nullable = (Field[])children.Clone();
                nullable[vectorIndex] = Copy(children[vectorIndex], nullable: true);
                yield return (Children(nullable), "Arrow field contract mismatch: probabilities.");
            }
        }
    }

    [Theory]
    [InlineData("top-null", DecisionPrecision.HighPrecision)]
    [InlineData("array-length", DecisionPrecision.HighPrecision)]
    [InlineData("binary-nonfinite", DecisionPrecision.HighPrecision)]
    [InlineData("binary-domain", DecisionPrecision.HighPrecision)]
    [InlineData("binary-grid", DecisionPrecision.TwoDecimalPlaces)]
    [InlineData("binary-grid", DecisionPrecision.FourDecimalPlaces)]
    [InlineData("choice-selected", DecisionPrecision.HighPrecision)]
    [InlineData("choice-vector", DecisionPrecision.HighPrecision)]
    [InlineData("choice-nonfinite", DecisionPrecision.HighPrecision)]
    [InlineData("choice-domain-sum", DecisionPrecision.HighPrecision)]
    [InlineData("choice-rounded", DecisionPrecision.TwoDecimalPlaces)]
    [InlineData("choice-rounded", DecisionPrecision.FourDecimalPlaces)]
    [InlineData("score-fields", DecisionPrecision.HighPrecision)]
    [InlineData("score-vector", DecisionPrecision.HighPrecision)]
    [InlineData("score-native", DecisionPrecision.HighPrecision)]
    [InlineData("score-rounded", DecisionPrecision.TwoDecimalPlaces)]
    [InlineData("score-rounded", DecisionPrecision.FourDecimalPlaces)]
    public async Task OpenAsync_EagerlyRejectsLaterBatchNumericalFaultsWithRefreshedIntegrity(string group, DecisionPrecision precision)
    {
        using var temp = new TempDirectory();
        var contract = Contract(precision);
        var baseline = new RawRow
        {
            Binary = precision == DecisionPrecision.TwoDecimalPlaces ? .27 : precision == DecisionPrecision.FourDecimalPlaces ? .2718 : .125,
            Lower = precision == DecisionPrecision.HighPrecision ? .999998 : precision == DecisionPrecision.TwoDecimalPlaces ? .99 : .9999,
            Upper = precision == DecisionPrecision.HighPrecision ? 1.000002 : precision == DecisionPrecision.TwoDecimalPlaces ? 1.01 : 1.0001
        };
        // Establish that this exact precision-specific baseline reaches and passes every validator.
        await WriteRawAsync(temp.Dataset, contract, [[baseline]], bound: 1);
        using (var control = await DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, [41L], Token))
        {
            using var batch = await control.ReadNextRecordBatchAsync(Token);
            Assert.Equal(baseline.Binary, ((DoubleArray)batch.Column(1)).GetValue(0));
            Assert.Equal(2, ((Int32Array)((StructArray)batch.Column(2)).Fields[0]).GetValue(0));
            Assert.Equal(1.0, ((DoubleArray)((StructArray)batch.Column(3)).Fields[0]).GetValue(0));
            Assert.Null(await control.ReadNextRecordBatchAsync(Token));
        }
        foreach (var (row, exceptionType, message) in NumericFaults(baseline, group, precision))
        {
            // Content faults are in batch TWO (or THREE for score fields), never in the valid prefix.
            RawRow[][] batches = group == "score-fields"
                ? [[baseline with { Id = 39 }], [baseline with { Id = 40 }], [row]]
                : [[baseline with { Id = 40 }], [row]];
            long[] expectedIds = group == "score-fields" ? [39L, 40L, 41L] : [40L, 41L];
            var manifest = await WriteRawAsync(temp.Dataset, contract, batches, bound: 1);
            Assert.Equal(expectedIds.Length, manifest.RowCount);
            AssertIntegrity(temp.Dataset, manifest);
            var error = await Record.ExceptionAsync(() =>
                DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, expectedIds, Token));
            Assert.NotNull(error);
            Assert.Equal(exceptionType, error.GetType());
            Assert.Equal(message, error.Message);
            // Failed eager validation must release its internally opened IPC handle.
            using var released = new FileStream(DataPath(temp.Dataset), FileMode.Open, FileAccess.ReadWrite, FileShare.None);
            Assert.Equal(manifest.DataSize, released.Length);
        }
        AssertReleased(DataPath(temp.Dataset));
    }

    private static IEnumerable<(RawRow Row, Type Type, string Message)> NumericFaults(RawRow row, string group, DecisionPrecision precision)
    {
        const string observation = "Missing or nonfinite observation.";
        const string probability = "Probability must be finite and in [0,1].";
        const string scoreFields = "Score expectation or portable rounding bounds mismatch.";
        const string infeasible = "Rounded observations cannot represent a normalized distribution.";
        string grid = precision == DecisionPrecision.TwoDecimalPlaces
            ? "Value does not lie on the declared two-decimal grid." : "Value does not lie on the declared four-decimal grid.";
        var invalid = typeof(InvalidDataException);
        var protocol = typeof(DecisionProtocolException);
        switch (group)
        {
            case "array-length":
                yield return (row with { ExtraBinaryStorage = true }, invalid, "Missing non-null observations.");
                break;
            case "top-null":
                foreach (var nullRow in new[] { row with { Id = null }, row with { Binary = null }, row with { ChoiceNull = true }, row with { ScoreNull = true } })
                    yield return (nullRow, invalid, "Missing non-null observations.");
                break;
            case "binary-nonfinite":
                foreach (double p in new[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity })
                    yield return (row with { Binary = p }, invalid, observation);
                break;
            case "binary-domain":
                foreach (double p in new[] { -.01, 1.01 })
                    yield return (row with { Binary = p }, protocol, probability);
                break;
            case "binary-grid":
                yield return (row with { Binary = precision == DecisionPrecision.TwoDecimalPlaces ? .271 : .27181 }, protocol, grid);
                break;
            case "choice-selected":
                yield return (row with { Selected = null }, invalid, "Missing selected index.");
                yield return (row with { Selected = -1 }, invalid, "Selected index outside candidate domain.");
                yield return (row with { Selected = 3 }, invalid, "Selected index outside candidate domain.");
                break;
            case "choice-vector":
                yield return (row with { ChoiceVectorNull = true }, invalid, "Null probability vector.");
                yield return (row with { Choice = [.1, null, .1] }, invalid, observation);
                break;
            case "choice-nonfinite":
                foreach (double p in new[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity })
                    yield return (row with { Choice = [.1, p, .1] }, invalid, observation);
                break;
            case "choice-domain-sum":
                yield return (row with { Choice = [-.1, .8, .3] }, protocol, probability);
                yield return (row with { Choice = [.1, 1.1, .1] }, protocol, probability);
                yield return (row with { Choice = [.5, .500002, 0] }, protocol, "Distribution sum outside absolute tolerance 1e-6.");
                break;
            case "choice-rounded":
                yield return (row with { Choice = precision == DecisionPrecision.TwoDecimalPlaces ? [.333, .333, .334] : [.33333, .33333, .33334] }, protocol, grid);
                yield return (row with { Choice = precision == DecisionPrecision.TwoDecimalPlaces ? [.49, .49, 0] : [.4999, .4999, 0] }, protocol, infeasible);
                break;
            case "score-fields":
                foreach (var corrupt in new[]
                {
                    row with { Expected = null }, row with { Expected = double.NaN },
                    row with { Lower = null }, row with { Lower = double.PositiveInfinity },
                    row with { Upper = null }, row with { Upper = double.NegativeInfinity }
                }) yield return (corrupt, invalid, observation);
                foreach (var corrupt in new[]
                {
                    row with { Expected = double.BitIncrement(1.0) },
                    row with { Lower = .999 }, row with { Upper = 1.001 }
                }) yield return (corrupt, invalid, scoreFields);
                break;
            case "score-vector":
                yield return (row with { ScoreVectorNull = true }, invalid, "Null probability vector.");
                yield return (row with { Score = [.25, null, .25] }, invalid, observation);
                foreach (double p in new[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity })
                    yield return (row with { Score = [.25, p, .25] }, invalid, observation);
                yield return (row with { Score = [-.1, .8, .3] }, protocol, probability);
                yield return (row with { Score = [.5, .500002, 0], Native = null }, protocol, "Distribution sum outside absolute tolerance 1e-6.");
                break;
            case "score-native":
                foreach (double n in new[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity, -.01, 2.01, 1.2 })
                    yield return (row with { Native = n }, protocol, "Native score disagrees with the distribution's expected index.");
                break;
            case "score-rounded":
                yield return (row with { Score = precision == DecisionPrecision.TwoDecimalPlaces ? [.333, .333, .334] : [.33333, .33333, .33334], Native = null }, protocol, grid);
                yield return (row with { Score = precision == DecisionPrecision.TwoDecimalPlaces ? [.49, .49, 0] : [.4999, .4999, 0], Native = null }, protocol, infeasible);
                yield return (row with { Native = precision == DecisionPrecision.TwoDecimalPlaces ? 1.001 : 1.00001 }, protocol, grid);
                yield return (row with { Native = 1.5 }, protocol, "Native score is incompatible with rounded probability observations.");
                foreach (double n in new[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity, -.01, 2.01 })
                    yield return (row with { Native = n }, protocol, "Rounded value must be finite and inside its domain.");
                break;
            default: throw new ArgumentOutOfRangeException(nameof(group));
        }
    }

    [Fact]
    public async Task OpenAsync_AcceptsMetadataInsertionOrderAndEmptyUnownedMetadata()
    {
        using var temp = new TempDirectory();
        var contract = Contract();
        var schema = contract.Schema;
        var fields = schema.FieldsList.Select(f => new Field(f.Name, f.DataType, f.IsNullable,
            f.Metadata is null ? new Dictionary<string, string>() : f.Metadata.Reverse().ToDictionary(p => p.Key, p => p.Value))).ToArray();
        var reordered = new Schema(fields, schema.Metadata.Reverse().ToDictionary(p => p.Key, p => p.Value));
        await WriteRawAsync(temp.Dataset, contract, [[new RawRow()]], reordered);
        using (var reader = await DecisionArrowDatasetReader.OpenAsync(ManifestPath(temp.Dataset), contract, [41L], Token))
        {
            using var batch = await reader.ReadNextRecordBatchAsync(Token);
            Assert.Equal(41L, ((Int64Array)batch.Column(0)).GetValue(0));
            Assert.Equal(.125, ((DoubleArray)batch.Column(1)).GetValue(0));
            Assert.Equal(2, ((Int32Array)((StructArray)batch.Column(2)).Fields[0]).GetValue(0));
            Assert.Null(await reader.ReadNextRecordBatchAsync(Token));
        }
        AssertReleased(DataPath(temp.Dataset));
    }
}
