using System.Text.Json;
using Apache.Arrow;
using DecisionInference;
using DecisionInference.Arrow;
using DecisionInference.Julia;

internal static class JuliaControlComparison
{
    internal static async Task CompareAsync(string referenceManifest, string referenceContract, string candidateManifest,
        string candidateContract, string statesPath, string reportPath, CancellationToken token)
    {
        if (File.Exists(reportPath) || File.Exists(reportPath + ".partial")) throw new IOException("Comparison receipts are immutable.");
        var expected = DecisionArrowSchema.FromCanonicalJson(await File.ReadAllTextAsync(referenceContract, token));
        var actual = DecisionArrowSchema.FromCanonicalJson(await File.ReadAllTextAsync(candidateContract, token));
        var a = expected.Identity;
        var b = actual.Identity;
        if (a.Provider != "julia" || a.ModelId != JuliaDecisionGenerator.ModelId ||
            expected.Precision != DecisionPrecision.HighPrecision || actual.Precision != DecisionPrecision.HighPrecision ||
            new DecisionArrowSchema(actual.Questions, actual.Precision,
                new(b.Provider, b.ModelId, b.AssetHashes, b.Preprocessing, a.Decoder, b.FeatureProjection)).CanonicalJson != expected.CanonicalJson)
            throw new InvalidDataException("Julia controls must have identical semantics except explicitly declared arithmetic identity.");
        var ids = new List<long>();
        await foreach (var state in JuliaExperiment.States(statesPath, expected.Questions, token))
        {
            if (ids.Count == 128) throw new InvalidDataException("Control comparison is bounded to at most 128 externally selected states.");
            ids.Add(state.Id);
        }
        if (ids.Count == 0) throw new InvalidDataException("Control comparison cannot be empty.");
        string inputHash = await JuliaAssetIdentity.HashFileAsync(statesPath, token);
        double maximum = 0;
        long? failedId = null;
        int? failedCoordinate = null;
        double? failedReferenceValue = null;
        double? failedCandidateValue = null;
        double? failedDifference = null;
        try
        {
            var left = await ReadAsync(referenceManifest, expected, ids, inputHash, token);
            var right = await ReadAsync(candidateManifest, actual, ids, inputHash, token);
            for (int row = 0; row < ids.Count; row++)
            {
                failedId = ids[row];
                failedCoordinate = null;
                failedReferenceValue = failedCandidateValue = failedDifference = null;
                if (!left[row].Selected.SequenceEqual(right[row].Selected))
                    throw new DecisionProtocolException($"Selected ordinal differs at original ID {ids[row]}.");
                for (int coordinate = 0; coordinate < left[row].Values.Length; coordinate++)
                {
                    failedCoordinate = coordinate;
                    failedReferenceValue = left[row].Values[coordinate];
                    failedCandidateValue = right[row].Values[coordinate];
                    double difference = Math.Abs(failedReferenceValue.Value - failedCandidateValue.Value);
                    failedDifference = double.IsFinite(difference) ? difference : null;
                    if (double.IsFinite(difference)) maximum = Math.Max(maximum, difference);
                    if (!double.IsFinite(difference) || difference > DecisionAnswer.DistributionTolerance)
                        throw new DecisionProtocolException($"Julia numeric parity failed at original ID {ids[row]}, coordinate {coordinate}, difference={difference:R}.");
                }
            }
            await SaveAsync(reportPath, new
            {
                status = "PASS", rows = ids.Count, ids, inputHash,
                referenceFingerprint = expected.FeatureFingerprint, candidateFingerprint = actual.FeatureFingerprint,
                referenceManifestSha256 = await JuliaAssetIdentity.HashFileAsync(referenceManifest, token),
                candidateManifestSha256 = await JuliaAssetIdentity.HashFileAsync(candidateManifest, token),
                exactIdsSelectionPrecisionNativeNull = true, maximumAbsoluteDifference = maximum,
                tolerance = DecisionAnswer.DistributionTolerance
            }, token);
        }
        catch (Exception error)
        {
            try
            {
                await SaveAsync(reportPath + ".partial", new
                {
                    status = "FAILED", failedId, failedCoordinate, failedReferenceValue, failedCandidateValue,
                    failedDifference, error = error.Message, maximumAbsoluteDifference = maximum,
                    tolerance = DecisionAnswer.DistributionTolerance
                }, CancellationToken.None);
            }
            catch (Exception recordingError)
            {
                throw new AggregateException("Julia comparison failed and its failure receipt could not be recorded.", error, recordingError);
            }
            throw;
        }
    }

    private sealed record Row(int[] Selected, double[] Values);

    private static async Task<List<Row>> ReadAsync(string path, DecisionArrowSchema contract, List<long> ids,
        string inputHash, CancellationToken token)
    {
        using var reader = await DecisionArrowDatasetReader.OpenAsync(path, contract, ids, token);
        if (reader.Manifest.Provenance.InputSha256 != inputHash) throw new InvalidDataException("Control source bytes differ.");
        var rows = new List<Row>(ids.Count);
        while (await reader.ReadNextRecordBatchAsync(token) is { } batch)
        {
            using (batch)
                for (int row = 0; row < batch.Length; row++)
                {
                    if (((Int64Array)batch.Column(0)).GetValue(row) != ids[rows.Count])
                        throw new InvalidDataException("Original control ID order changed.");
                    var selected = new List<int>();
                    var values = new List<double>();
                    for (int question = 0; question < contract.Questions.Count; question++)
                        switch (contract.Questions[question])
                        {
                            case BinaryDecisionQuestion:
                                values.Add(((DoubleArray)batch.Column(question + 1)).GetValue(row)!.Value);
                                break;
                            case ChoiceDecisionQuestion choice:
                                var c = (StructArray)batch.Column(question + 1);
                                selected.Add(((Int32Array)c.Fields[0]).GetValue(row)!.Value);
                                AddVector((FixedSizeListArray)c.Fields[1], row, choice.Candidates.Count, values);
                                break;
                            case ScoreDecisionQuestion score:
                                var s = (StructArray)batch.Column(question + 1);
                                if (!s.Fields[3].IsNull(row)) throw new InvalidDataException("Julia native score must remain absent.");
                                for (int field = 0; field < 3; field++) values.Add(((DoubleArray)s.Fields[field]).GetValue(row)!.Value);
                                AddVector((FixedSizeListArray)s.Fields[4], row, score.Rubric.Count, values);
                                break;
                            default: throw new InvalidDataException("Unknown Julia question.");
                        }
                    rows.Add(new(selected.ToArray(), values.ToArray()));
                }
        }
        return rows;
    }

    private static void AddVector(FixedSizeListArray vector, int row, int width, List<double> values)
    {
        var span = ((DoubleArray)vector.Values).Values.Slice(checked((row + vector.Offset) * width), width);
        foreach (double value in span) values.Add(value);
    }

    private static async Task SaveAsync(string path, object value, CancellationToken token)
    {
        await using var file = new FileStream(path, FileMode.CreateNew);
        await JsonSerializer.SerializeAsync(file, value, DecisionArrowManifest.JsonOptions, token);
    }
}
