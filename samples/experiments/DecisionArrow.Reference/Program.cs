using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using DecisionInference;
using DecisionInference.Arrow;
using DecisionInference.Julia;
using Microsoft.ML.OnnxRuntime;

return await OriginalJuliaReference.RunAsync(args);

internal static class OriginalJuliaReference
{
    internal const string OriginalCommit = "6dda59d19a60fdad2fc14152e3774d1d5ad9b345";
    private static readonly JsonSerializerOptions Json = new() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase, WriteIndented = true };

    public static async Task<int> RunAsync(string[] args)
    {
        if (args.Length == 0 || args[0] == "--help")
        {
            Console.WriteLine("""
                Independent original Julia oracle. Freeze its complete published directory BEFORE shared source changes.
                preflight <tokenizer.json> <questions.v1.json> <states.jsonl> <new-output-directory>
                control <assets-directory> <canonical-contract.json> <label-free-states.jsonl> <new-output-directory>
                oracle <numeric-cases.jsonl> <new-results.jsonl>
                Preflight never opens an ONNX session or scores. Control is explicit CPU ORT 1.23.2, intra=4/inter=1.
                Failures are nonzero. No truncation, label inputs, exclusions or completed partial datasets.
                """);
            return 0;
        }
        using var cancellation = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) => { e.Cancel = true; cancellation.Cancel(); };
        try
        {
            switch (args[0])
            {
                case "preflight" when args.Length == 5:
                    return await PreflightAsync(args[1], args[2], args[3], args[4], cancellation.Token);
                case "control" when args.Length == 5:
                    await ControlAsync(args[1], args[2], args[3], args[4], cancellation.Token);
                    return 0;
                case "oracle" when args.Length == 3:
                    Oracle(args[1], args[2]);
                    return 0;
                default: throw new ArgumentException("Invalid original-reference command. Run --help.");
            }
        }
        catch (Exception error)
        {
            Console.Error.WriteLine($"{error.GetType().Name}: {error.Message}");
            return error is OperationCanceledException ? 2 : 1;
        }
    }

    private static async Task<int> PreflightAsync(string tokenizer, string questionsPath, string statesPath, string directory, CancellationToken token)
    {
        NewDirectory(directory);
        using var questionsDocument = JsonDocument.Parse(await File.ReadAllTextAsync(questionsPath, token));
        if (questionsDocument.RootElement.GetProperty("schemaVersion").GetInt32() != 1)
            throw new InvalidDataException("Unsupported questions version.");
        var questions = DecisionArrowSchema.ParseQuestions(questionsDocument.RootElement.GetProperty("questions"));
        using var prepare = new PrepareDecisionInputs(tokenizer);
        var failures = new List<object>();
        var shapes = new SortedDictionary<int, int>();
        var ids = new List<long>();
        await using var tensorFile = new StreamWriter(new FileStream(Path.Combine(directory, "prepared.jsonl"), FileMode.CreateNew));
        await foreach (var (id, input) in States(statesPath, questions, token))
        {
            ids.Add(id);
            try
            {
                var batch = prepare.Prepare(input, token);
                shapes[batch.SequenceLength] = shapes.GetValueOrDefault(batch.SequenceLength) + 1;
                await tensorFile.WriteLineAsync(JsonSerializer.Serialize(new
                {
                    rowId = id, batch.BatchSize, batch.SequenceLength, batch.MarkerWidth,
                    tensorSha256 = TensorHash(batch)
                }));
            }
            catch (Exception error) when (error is ArgumentException or NotSupportedException or DecisionProtocolException)
            {
                // Provider errors contain category/budget information, never caller text.
                failures.Add(new { rowId = id, category = error.GetType().Name, error = error.Message });
            }
        }
        await tensorFile.FlushAsync(token);
        await tensorFile.DisposeAsync();
        bool fullIds = ids.Count == 5574 && ids.SequenceEqual(Enumerable.Range(1, 5574).Select(i => (long)i));
        var receipt = new
        {
            schemaVersion = 1, status = failures.Count == 0 && fullIds ? "PASS" : "BLOCKED",
            scope = "original-julia-label-free-strict-preparation-no-scoring",
            originalSourceCommit = OriginalCommit, runtime = RuntimeInformation.FrameworkDescription,
            inputs = new { tokenizerSha256 = HashFile(tokenizer), questionsSha256 = HashFile(questionsPath), statesSha256 = HashFile(statesPath) },
            totalRows = ids.Count, requiredIds1Through5574 = fullIds, preparedRows = ids.Count - failures.Count,
            failures, shapeCounts = shapes, preparedSha256 = HashFile(Path.Combine(directory, "prepared.jsonl")),
            policy = "strict1024/head256/option48/min16/pad8; no truncation/exclusions/labels/split metadata"
        };
        await File.WriteAllTextAsync(Path.Combine(directory, "preflight.receipt.json"), JsonSerializer.Serialize(receipt, Json), token);
        Console.WriteLine($"Original strict preflight: rows={ids.Count}; failures={failures.Count}; exactFullIds={fullIds}.");
        return failures.Count == 0 && fullIds ? 0 : 1;
    }

    private static async Task ControlAsync(string assets, string contractPath, string statesPath, string directory, CancellationToken token)
    {
        var contract = DecisionArrowSchema.FromCanonicalJson(await File.ReadAllTextAsync(contractPath, token));
        if (contract.Identity.Provider != "julia" || contract.Identity.ModelId != JuliaDecisionGenerator.ModelId ||
            contract.Precision != DecisionPrecision.HighPrecision)
            throw new InvalidDataException("Control requires the producer's trusted canonical Julia HighPrecision contract.");
        using var prepare = new PrepareDecisionInputs(Path.Combine(assets, "tokenizer.json"));
        using var options = new SessionOptions { IntraOpNumThreads = 4, InterOpNumThreads = 1 };
        using var score = ScoreOnnxDecisionModel.Load(Path.Combine(assets, "model.onnx"), options);
        var decode = new DecodeDecisions();
        var ids = new List<long>();
        var tensors = new List<object>();
        async IAsyncEnumerable<DecisionArrowRow> Rows([EnumeratorCancellation] CancellationToken ct)
        {
            await foreach (var (id, input) in States(statesPath, contract.Questions, ct))
            {
                var batch = prepare.Prepare(input, ct);
                tensors.Add(new { rowId = id, tensorSha256 = TensorHash(batch), batch.SequenceLength, batch.MarkerWidth });
                var result = decode.Decode(batch, score.Score(batch, ct));
                ids.Add(id);
                yield return new(id, input, result);
            }
        }
        var provenance = new DecisionArrowProvenance
        {
            SourceRepository = "luisquintanilla/typesafe-meai", SourceCommit = OriginalCommit,
            InputSha256 = HashFile(statesPath), QuestionsSha256 = HashFile(contractPath), Packages = [],
            Machine = Environment.MachineName, Runtime = RuntimeInformation.FrameworkDescription,
            ExecutionMode = "original-scalar-cpu-control", CpuThreads = 4, NativeStateBatchSize = 1
        };
        await DecisionArrowWriter.WriteAsync(directory, contract, Rows(token), provenance, cancellationToken: token);
        using var reader = await DecisionArrowDatasetReader.OpenAsync(Path.Combine(directory, "manifest.json"), contract, ids, token);
        await File.WriteAllTextAsync(Path.Combine(directory, "prepared-control.json"), JsonSerializer.Serialize(tensors, Json), token);
        Console.WriteLine($"Original CPU control complete: rows={reader.Manifest.RowCount}.");
    }

    internal static async IAsyncEnumerable<(long Id, DecisionInput Input)> States(string path, IReadOnlyList<DecisionQuestion> questions,
        [EnumeratorCancellation] CancellationToken token)
    {
        using var file = File.OpenText(path);
        var ids = new HashSet<long>();
        string? line;
        while ((line = await file.ReadLineAsync(token)) is not null)
        {
            using var document = JsonDocument.Parse(line);
            var root = document.RootElement;
            var fields = root.EnumerateObject().ToArray();
            if (fields.Length != 2 || fields.Count(p => p.Name == "rowId") != 1 || fields.Count(p => p.Name == "state") != 1)
                throw new InvalidDataException("Label-free state records must contain only rowId/state.");
            long id = root.GetProperty("rowId").GetInt64();
            if (!ids.Add(id)) throw new InvalidDataException($"Duplicate source ID {id}.");
            var state = root.GetProperty("state");
            if (state.ValueKind != JsonValueKind.String) throw new InvalidDataException($"Source ID {id} must retain a string state.");
            yield return (id, new(state, questions));
        }
    }

    internal static string TensorHash(JuliaInputBatch batch)
    {
        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(batch.BatchSize);
            writer.Write(batch.SequenceLength);
            writer.Write(batch.MarkerWidth);
            for (int row = 0; row < batch.BatchSize; row++)
            {
                for (int i = 0; i < batch.SequenceLength; i++) writer.Write(batch.InputIds[row, i]);
                for (int i = 0; i < batch.SequenceLength; i++) writer.Write(batch.AttentionMask[row, i]);
                for (int i = 0; i < batch.MarkerWidth; i++) writer.Write(batch.MarkerPositions[row, i]);
                for (int i = 0; i < batch.MarkerWidth; i++) writer.Write(batch.MarkerMask[row, i]);
                writer.Write(batch.QuestionTypes[row]);
            }
        }
        return Convert.ToHexStringLower(SHA256.HashData(stream.GetBuffer().AsSpan(0, checked((int)stream.Length))));
    }

    private static void Oracle(string input, string output)
    {
        using var writer = new StreamWriter(new FileStream(output, FileMode.CreateNew));
        foreach (string line in File.ReadLines(input))
        {
            using var document = JsonDocument.Parse(line);
            var root = document.RootElement;
            var precision = Enum.Parse<DecisionPrecision>(root.GetProperty("precision").GetString()!, ignoreCase: false);
            double[] values = root.GetProperty("values").EnumerateArray().Select(v =>
                double.Parse(v.GetString()!, CultureInfo.InvariantCulture)).ToArray();
            double? native = root.TryGetProperty("native", out var n) && n.ValueKind != JsonValueKind.Null
                ? double.Parse(n.GetString()!, CultureInfo.InvariantCulture) : null;
            object result;
            try
            {
                DecisionAnswer answer = root.GetProperty("kind").GetString() switch
                {
                    "binary" => new BinaryDecisionAnswer("q", values.Single(), precision),
                    "choice" => new ChoiceDecisionAnswer("q", root.GetProperty("selected").GetInt32().ToString(CultureInfo.InvariantCulture),
                        values.Select((v, i) => KeyValuePair.Create(i.ToString(CultureInfo.InvariantCulture), v)), precision),
                    "score" => new ScoreDecisionAnswer("q", values, native, precision),
                    _ => throw new InvalidDataException("Unknown oracle kind.")
                };
                long[] bits = answer switch
                {
                    BinaryDecisionAnswer b => [BitConverter.DoubleToInt64Bits(b.TrueProbability)],
                    ChoiceDecisionAnswer c => c.Probabilities.Values.Select(BitConverter.DoubleToInt64Bits).ToArray(),
                    ScoreDecisionAnswer s => [.. s.Probabilities.Select(BitConverter.DoubleToInt64Bits),
                        BitConverter.DoubleToInt64Bits(s.ExpectedIndex), BitConverter.DoubleToInt64Bits(s.ExpectedIndexLowerBound),
                        BitConverter.DoubleToInt64Bits(s.ExpectedIndexUpperBound)],
                    _ => throw new InvalidOperationException()
                };
                result = new { accepted = true, bits };
            }
            catch (Exception error) when (error is DecisionProtocolException or ArgumentException or OverflowException)
            {
                result = new { accepted = false, category = error.GetType().Name, error = error.Message };
            }
            writer.WriteLine(JsonSerializer.Serialize(result));
        }
    }

    private static string HashFile(string path)
    {
        using var file = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(file));
    }

    private static void NewDirectory(string path)
    {
        if (Directory.Exists(path) && Directory.EnumerateFileSystemEntries(path).Any())
            throw new IOException("Reference output must be new/empty; artifacts are never replaced.");
        Directory.CreateDirectory(path);
    }
}
