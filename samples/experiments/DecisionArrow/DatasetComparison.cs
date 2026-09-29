using System.Text.Json;
using Apache.Arrow;
using DecisionInference;
using DecisionInference.Arrow;

internal static class DatasetComparison
{
    internal static async Task RunAsync(string scalarPath, string nativePath, string contractPath, string outputPath, CancellationToken token)
    {
        if (File.Exists(outputPath) || File.Exists(outputPath + ".partial")) throw new IOException("Comparison output already exists.");
        var contract = DecisionArrowSchema.FromCanonicalJson(await File.ReadAllTextAsync(contractPath, token));
        long rows = 0;
        double maximum = 0;
        await using var partial = new FileStream(outputPath + ".partial", FileMode.CreateNew, FileAccess.ReadWrite, FileShare.Read);
        await JsonSerializer.SerializeAsync(partial, new { status = "running" }, DecisionArrowManifest.JsonOptions, token);
        try
        {
            using var scalar = await DecisionArrowDatasetReader.OpenAsync(scalarPath, contract, token);
            using var native = await DecisionArrowDatasetReader.OpenAsync(nativePath, contract, token);
            if (scalar.Manifest.Provenance.InputSha256 != native.Manifest.Provenance.InputSha256 ||
                scalar.Manifest.Provenance.QuestionsSha256 != native.Manifest.Provenance.QuestionsSha256 ||
                scalar.Manifest.RowCount != native.Manifest.RowCount)
                throw new InvalidDataException("Scalar/native source, questions or row-count mismatch.");
            using var first = new Cursor(scalar);
            using var second = new Cursor(native);
            while (await first.MoveNextAsync(token))
            {
                if (!await second.MoveNextAsync(token) || first.Id != second.Id)
                    throw new InvalidDataException($"Scalar/native source correlation mismatch after {rows} rows.");
                for (int column = 1; column < scalar.Schema.FieldsList.Count; column++)
                    Compare(first.Batch.Column(column), first.Index, second.Batch.Column(column), second.Index,
                        first.Id, scalar.Schema.FieldsList[column].Name, ref maximum);
                rows++;
            }
            if (await second.MoveNextAsync(token)) throw new InvalidDataException("Extra native rows.");
            partial.SetLength(0);
            partial.Position = 0;
            await JsonSerializer.SerializeAsync(partial, new
            {
                status = "passed", scope = "completed-portable-observation-parity",
                rows, maximumAbsoluteDifference = maximum, tolerance = DecisionAnswer.DistributionTolerance,
                featureFingerprint = contract.FeatureFingerprint,
                scalarExecutionMode = scalar.Manifest.Provenance.ExecutionMode,
                nativeExecutionMode = native.Manifest.Provenance.ExecutionMode,
                scalarDataSha256 = scalar.Manifest.DataSha256, nativeDataSha256 = native.Manifest.DataSha256,
                inputSha256 = scalar.Manifest.Provenance.InputSha256,
                questionsSha256 = scalar.Manifest.Provenance.QuestionsSha256
            }, DecisionArrowManifest.JsonOptions, token);
            await partial.FlushAsync(token);
            partial.Flush(flushToDisk: true);
            token.ThrowIfCancellationRequested();
            await partial.DisposeAsync();
            File.Move(outputPath + ".partial", outputPath);
        }
        catch (Exception error)
        {
            try
            {
                await partial.DisposeAsync();
                await File.WriteAllTextAsync(outputPath + ".partial", JsonSerializer.Serialize(new
                {
                    status = error is OperationCanceledException ? "cancelled" : "failed",
                    comparedRows = rows, maximumAbsoluteDifference = maximum, errorType = error.GetType().Name, error = error.Message
                }, DecisionArrowManifest.JsonOptions), CancellationToken.None);
            }
            catch (Exception recordingError) { throw new AggregateException("Comparison failed and its partial record could not be written.", error, recordingError); }
            throw;
        }
        Console.WriteLine($"Completed dataset parity passed for {rows} source rows; max absolute difference={maximum:R}.");
    }

    private static void Compare(IArrowArray first, int a, IArrowArray second, int b, long id, string field, ref double maximum)
    {
        switch (first, second)
        {
            case (DoubleArray x, DoubleArray y):
                double? left = x.GetValue(a), right = y.GetValue(b);
                if (left is null && right is null) return;
                if (left is not { } leftValue || right is not { } rightValue)
                    throw new DecisionProtocolException($"Native-score null mismatch: row {id}, field {field}.");
                double difference = Math.Abs(leftValue - rightValue);
                if (double.IsFinite(difference)) maximum = Math.Max(maximum, difference);
                if (!double.IsFinite(difference) || difference > DecisionAnswer.DistributionTolerance)
                    throw new DecisionProtocolException($"Parity exceeds existing tolerance: row {id}, field {field}, scalar={leftValue:R}, native={rightValue:R}, difference={difference:R}.");
                break;
            case (Int32Array x, Int32Array y):
                if (x.GetValue(a) != y.GetValue(b)) throw new DecisionProtocolException($"Selected-index mismatch: row {id}, field {field}.");
                break;
            case (StructArray x, StructArray y):
                for (int i = 0; i < x.Fields.Count; i++)
                    Compare(x.Fields[i], a, y.Fields[i], b, id,
                        field + "." + ((Apache.Arrow.Types.StructType)x.Data.DataType).Fields[i].Name, ref maximum);
                break;
            case (FixedSizeListArray x, FixedSizeListArray y):
                int width = ((Apache.Arrow.Types.FixedSizeListType)x.Data.DataType).ListSize;
                for (int i = 0; i < width; i++)
                    Compare(x.Values, checked((a + x.Offset) * width + i), y.Values, checked((b + y.Offset) * width + i),
                        id, $"{field}[{i}]", ref maximum);
                break;
            default: throw new InvalidDataException("Unexpected portable observation type.");
        }
    }

    private sealed class Cursor(DecisionArrowDatasetReader reader) : IDisposable
    {
        private RecordBatch? _batch;
        internal RecordBatch Batch => _batch ?? throw new InvalidOperationException("Cursor is not positioned.");
        internal int Index { get; private set; } = -1;
        internal long Id => ((Int64Array)Batch.Column(0)).GetValue(Index)!.Value;

        internal async Task<bool> MoveNextAsync(CancellationToken token)
        {
            Index++;
            while (_batch is null || Index >= _batch.Length)
            {
                _batch?.Dispose();
                _batch = await reader.ReadNextRecordBatchAsync(token);
                Index = 0;
                if (_batch is null) return false;
            }
            return true;
        }

        public void Dispose() => _batch?.Dispose();
    }
}
