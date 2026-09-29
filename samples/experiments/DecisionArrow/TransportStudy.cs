using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Apache.Arrow;
using DecisionInference;
using DecisionInference.Arrow;

internal static class TransportStudy
{
    public static async Task RunAsync(string directory, int count, DecisionArrowSchema contract, DecisionArrowProvenance provenance,
        Func<IAsyncEnumerable<DecisionArrowRow>> rows, CancellationToken token)
    {
        if (count < 0) throw new ArgumentOutOfRangeException(nameof(count));
        if (Directory.Exists(directory) && Directory.EnumerateFileSystemEntries(directory).Any()) throw new IOException("Study output must be empty.");
        Directory.CreateDirectory(directory);
        var samples = new List<object>();
        for (int repeat = 0; repeat < 3; repeat++)
        {
            string arrowDirectory = Path.Combine(directory, $"arrow-{repeat}");
            string jsonPath = Path.Combine(directory, $"equivalent-{repeat}.json");
            var arrowEncode = await MeasureAsync(() => DecisionArrowWriter.WriteAsync(arrowDirectory, contract, rows(), provenance, cancellationToken: token));
            var jsonEncode = await MeasureAsync(() => WriteJsonAsync(jsonPath, contract, rows(), token));
            var arrowDecode = await MeasureAsync(async () =>
            {
                using var reader = await DecisionArrowDatasetReader.OpenAsync(Path.Combine(arrowDirectory, "manifest.json"), contract,
                    Enumerable.Range(1, count).Select(i => (long)i), token);
                var generated = rows().GetAsyncEnumerator(token);
                await using (generated)
                {
                    RecordBatch? batch;
                    while ((batch = await reader.ReadNextRecordBatchAsync(token)) is not null)
                    {
                        using (batch)
                            for (int row = 0; row < batch.Length; row++)
                            {
                                if (!await generated.MoveNextAsync()) throw new InvalidDataException("Unexpected Arrow row.");
                                var expected = generated.Current;
                                if (((Int64Array)batch.Column(0)).GetValue(row) != expected.RowId) throw new InvalidDataException("Arrow ID mismatch.");
                                VerifyArrowRow(batch, row, contract, expected.Result);
                            }
                    }
                    if (await generated.MoveNextAsync()) throw new InvalidDataException("Missing Arrow row.");
                }
            });
            var jsonDecode = await MeasureAsync(async () =>
            {
                await using var file = File.OpenRead(jsonPath);
                using var document = await JsonDocument.ParseAsync(file, cancellationToken: token);
                if (document.RootElement.GetProperty("contractJson").GetString() != contract.CanonicalJson) throw new InvalidDataException("JSON metadata mismatch.");
                var observations = document.RootElement.GetProperty("rows").EnumerateArray();
                await foreach (var expected in rows().WithCancellation(token))
                {
                    if (!observations.MoveNext()) throw new InvalidDataException("Missing JSON row.");
                    if (observations.Current.GetRawText() != RowJson(expected, contract)) throw new InvalidDataException("JSON portable content mismatch.");
                }
                if (observations.MoveNext()) throw new InvalidDataException("Unexpected JSON row.");
            });
            var arrowSummary = await MeasureAsync(async () =>
            {
                using var reader = await DecisionArrowDatasetReader.OpenAsync(Path.Combine(arrowDirectory, "manifest.json"), contract, token);
                double sum = 0;
                RecordBatch? batch;
                while ((batch = await reader.ReadNextRecordBatchAsync(token)) is not null)
                {
                    using (batch)
                        for (int row = 0; row < batch.Length; row++)
                            for (int q = 0; q < contract.Questions.Count; q++)
                            {
                                if (batch.Column(q + 1) is DoubleArray binary) sum += binary.GetValue(row)!.Value;
                                else
                                {
                                    var vector = (FixedSizeListArray)((StructArray)batch.Column(q + 1)).Fields[^1];
                                    var values = (DoubleArray)vector.Values;
                                    int width = ((Apache.Arrow.Types.FixedSizeListType)vector.Data.DataType).ListSize;
                                    for (int i = 0; i < width; i++) sum += values.GetValue((row + vector.Offset) * width + i)!.Value;
                                }
                            }
                }
                GC.KeepAlive(sum);
            });
            var jsonSummary = await MeasureAsync(async () =>
            {
                await using var file = File.OpenRead(jsonPath);
                using var document = await JsonDocument.ParseAsync(file, cancellationToken: token);
                double sum = 0;
                foreach (var row in document.RootElement.GetProperty("rows").EnumerateArray())
                    foreach (var q in contract.Questions)
                    {
                        var value = row.GetProperty(q.Id);
                        if (q is BinaryDecisionQuestion) sum += value.GetDouble();
                        else foreach (var p in value.GetProperty("probabilities").EnumerateArray()) sum += p.GetDouble();
                    }
                GC.KeepAlive(sum);
            });
            samples.Add(new
            {
                repeat, temperature = repeat == 0 ? "first-in-process" : "warm",
                arrowEncode, jsonEncode, arrowDecodeAndValidation = arrowDecode, jsonDecodeAndValidation = jsonDecode,
                arrowSummaryIncludingValidation = arrowSummary, jsonSummaryIncludingParse = jsonSummary,
                arrowIpcBytes = new FileInfo(Path.Combine(arrowDirectory, "decisions.arrow")).Length,
                arrowCompletionEnvelopeBytes = new FileInfo(Path.Combine(arrowDirectory, "manifest.json")).Length,
                equivalentJsonBytes = new FileInfo(jsonPath).Length,
                sharedContractUtf8Bytes = System.Text.Encoding.UTF8.GetByteCount(contract.CanonicalJson),
                processCumulativePeakWorkingSetBytes = Process.GetCurrentProcess().PeakWorkingSet64
            });
        }
        await File.WriteAllTextAsync(Path.Combine(directory, "transport-measurements.json"), JsonSerializer.Serialize(new
        {
            scope = "synthetic-equivalent-portable-content-no-inference",
            rows = count, featureFingerprint = contract.FeatureFingerprint, samples,
            limitations = "First-in-process is not a fresh-process cold benchmark. Arrow encode includes completion/hash; JSON is a single envelope. Allocations are process-wide deltas, peak memory cumulative. Decode includes exact content checks and row generation. Summary includes reader validation/JSON parse; not kernel-only throughput."
        }, DecisionArrowManifest.JsonOptions), token);
    }

    private static async Task<object> MeasureAsync(Func<Task> action)
    {
        long before = GC.GetTotalAllocatedBytes(precise: true);
        var timer = Stopwatch.StartNew();
        await action();
        timer.Stop();
        return new { milliseconds = timer.Elapsed.TotalMilliseconds, allocatedBytes = GC.GetTotalAllocatedBytes(precise: true) - before };
    }

    private static async Task WriteJsonAsync(string path, DecisionArrowSchema contract, IAsyncEnumerable<DecisionArrowRow> rows, CancellationToken token)
    {
        await using var file = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None);
        using var writer = new Utf8JsonWriter(file);
        writer.WriteStartObject();
        writer.WriteString("contractJson", contract.CanonicalJson);
        writer.WriteStartArray("rows");
        await foreach (var row in rows.WithCancellation(token))
        {
            WriteRow(writer, row, contract);
            await writer.FlushAsync(token);
        }
        writer.WriteEndArray();
        writer.WriteEndObject();
        await writer.FlushAsync(token);
        file.Flush(flushToDisk: true);
    }

    private static string RowJson(DecisionArrowRow row, DecisionArrowSchema contract)
    {
        using var file = new MemoryStream();
        using (var writer = new Utf8JsonWriter(file)) WriteRow(writer, row, contract);
        return System.Text.Encoding.UTF8.GetString(file.ToArray());
    }

    private static void WriteRow(Utf8JsonWriter writer, DecisionArrowRow row, DecisionArrowSchema contract)
    {
        writer.WriteStartObject();
        writer.WriteNumber("row_id", row.RowId);
        for (int q = 0; q < contract.Questions.Count; q++)
        {
            var answer = row.Result.Answers[q];
            if (answer is BinaryDecisionAnswer binary) writer.WriteNumber(answer.QuestionId, binary.TrueProbability);
            else
            {
                writer.WriteStartObject(answer.QuestionId);
                IEnumerable<double> probabilities;
                if (answer is ChoiceDecisionAnswer choice)
                {
                    var candidates = ((ChoiceDecisionQuestion)contract.Questions[q]).Candidates;
                    writer.WriteNumber("selected_index", candidates.Select((c, i) => (c, i)).Single(c => c.c.Id == choice.SelectedCandidateId).i);
                    probabilities = candidates.Select(c => choice.Probabilities[c.Id]);
                }
                else
                {
                    var score = (ScoreDecisionAnswer)answer;
                    writer.WriteNumber("expected_index", score.ExpectedIndex);
                    writer.WriteNumber("expected_index_lower_bound", score.ExpectedIndexLowerBound);
                    writer.WriteNumber("expected_index_upper_bound", score.ExpectedIndexUpperBound);
                    if (score.NativeScore is { } native) writer.WriteNumber("native_score", native); else writer.WriteNull("native_score");
                    probabilities = score.Probabilities;
                }
                writer.WriteStartArray("probabilities");
                foreach (var p in probabilities) writer.WriteNumberValue(p);
                writer.WriteEndArray();
                writer.WriteEndObject();
            }
        }
        writer.WriteEndObject();
    }

    private static void VerifyArrowRow(RecordBatch batch, int row, DecisionArrowSchema contract, DecisionResult result)
    {
        for (int q = 0; q < contract.Questions.Count; q++)
        {
            var answer = result.Answers[q];
            if (answer is BinaryDecisionAnswer binary)
            {
                if (((DoubleArray)batch.Column(q + 1)).GetValue(row) != binary.TrueProbability) throw new InvalidDataException("Binary content differs.");
                continue;
            }
            var structure = (StructArray)batch.Column(q + 1);
            IEnumerable<double> expected;
            if (answer is ChoiceDecisionAnswer choice)
            {
                var candidates = ((ChoiceDecisionQuestion)contract.Questions[q]).Candidates;
                int index = ((Int32Array)structure.Fields[0]).GetValue(row)!.Value;
                if (candidates[index].Id != choice.SelectedCandidateId) throw new InvalidDataException("Selected ID differs.");
                expected = candidates.Select(c => choice.Probabilities[c.Id]);
            }
            else
            {
                var score = (ScoreDecisionAnswer)answer;
                double?[] values = [score.ExpectedIndex, score.ExpectedIndexLowerBound, score.ExpectedIndexUpperBound, score.NativeScore];
                for (int i = 0; i < values.Length; i++)
                    if (((DoubleArray)structure.Fields[i]).GetValue(row) != values[i]) throw new InvalidDataException("Score content differs.");
                expected = score.Probabilities;
            }
            var vector = (FixedSizeListArray)structure.Fields[^1];
            var probabilities = (DoubleArray)vector.Values;
            var expectedValues = expected.ToArray();
            for (int i = 0; i < expectedValues.Length; i++)
                if (probabilities.GetValue((row + vector.Offset) * expectedValues.Length + i) != expectedValues[i])
                    throw new InvalidDataException("Vector content differs.");
        }
    }
}
