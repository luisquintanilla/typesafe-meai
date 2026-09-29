using System.Diagnostics;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Apache.Arrow;
using DecisionInference;
using DecisionInference.Arrow;
using DecisionInference.Laya;
using Microsoft.ML.OnnxRuntime;

return await Experiment.RunAsync(args);

internal static class Experiment
{
    public static async Task<int> RunAsync(string[] args)
    {
        if (args.Length == 0 || args[0] is "help" or "--help")
        {
            Console.WriteLine("""
                Offline and explicit local-only experiment; no acquisition or hosted calls.
                fixture <new-output-directory> <row-count> <precision> [questions.v1.json]
                transport <new-output-directory> <row-count> <precision> [questions.v1.json]
                summarize <manifest.json> <expected-contract.json>
                preflight <assets-directory> <questions.v1.json> <states.jsonl> <new-output-directory>
                export <assets-directory> <questions.v1.json> <states.jsonl> <new-output-directory> <source-commit> <package-receipt.json> [scalar|native] [state-batch-size] [cpu-threads]
                parity <assets-directory> <questions.v1.json> <synthetic-states.jsonl> <new-output-directory> <cpu-threads>
                Precision: HighPrecision, FourDecimalPlaces, TwoDecimalPlaces.
                Export defaults: scalar, native state batch size 16, CPU ORT 1.23.2, one owned session, one CPU thread.
                Real export always preflights every record first. Any failure stops publication.
                """);
            return 0;
        }
        using var cancellation = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) => { e.Cancel = true; cancellation.Cancel(); };
        try
        {
            switch (args[0])
            {
                case "fixture" or "transport" when args.Length is 4 or 5:
                    await FixtureAsync(args[1], int.Parse(args[2], CultureInfo.InvariantCulture),
                        Enum.Parse<DecisionPrecision>(args[3], ignoreCase: false), args.Length == 5 ? args[4] : null, cancellation.Token,
                        transport: args[0] == "transport");
                    break;
                case "summarize" when args.Length == 3:
                    await SummarizeAsync(args[1], DecisionArrowSchema.FromCanonicalJson(await File.ReadAllTextAsync(args[2], cancellation.Token)), cancellation.Token);
                    break;
                case "preflight" when args.Length == 5:
                    await PreflightAsync(args[1], args[2], args[3], args[4], cancellation.Token);
                    break;
                case "export" when args.Length is 7 or 10:
                    await ExportAsync(args, cancellation.Token);
                    break;
                case "parity" when args.Length == 6:
                    await ParityAsync(args, cancellation.Token);
                    break;
                default: throw new ArgumentException("Invalid command or arguments. Run --help.");
            }
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine($"{error.GetType().Name}: {error.Message}");
            return error is OperationCanceledException ? 2 : 1;
        }
    }

    private static (IReadOnlyList<DecisionQuestion> Questions, string[] Projection) Questions(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var root = document.RootElement;
        if (root.GetProperty("schemaVersion").GetInt32() != 1) throw new InvalidDataException("Unsupported questions version.");
        return (DecisionArrowSchema.ParseQuestions(root.GetProperty("questions")),
            root.GetProperty("featureProjection").EnumerateArray().Select(p => p.GetString()!).ToArray());
    }

    private static async Task FixtureAsync(string output, int count, DecisionPrecision precision, string? questionsPath, CancellationToken token,
        bool transport = false)
    {
        if (count < 0) throw new ArgumentOutOfRangeException(nameof(count));
        (IReadOnlyList<DecisionQuestion> questions, string[] projection) = questionsPath is null
            ? (new DecisionQuestion[] {
                new BinaryDecisionQuestion("binary", "Synthetic binary observation."),
                new ChoiceDecisionQuestion("choice", "Synthetic choice.", [new("first", "First option."), new("second", "Second option.")]),
                new ScoreDecisionQuestion("score", "Synthetic ordinal observation.", ["none", "mild", "urgent"]) },
                ["binary", "choice[0]", "choice[1]", "score[0]", "score[1]", "score[2]"])
            : Questions(questionsPath);
        var identity = new DecisionArrowIdentity("synthetic", "fixture-v1",
            [KeyValuePair.Create("fixture", new string('0', 64))], "synthetic-no-preprocessing-v1", "portable-fixture-v1", projection);
        var contract = new DecisionArrowSchema(questions, precision, identity);
        var provenance = new DecisionArrowProvenance
        {
            SourceRepository = "luisquintanilla/typesafe-meai",
            SourceCommit = typeof(Experiment).Assembly.GetCustomAttributes(typeof(System.Reflection.AssemblyInformationalVersionAttribute), false)
                .Cast<System.Reflection.AssemblyInformationalVersionAttribute>().Single().InformationalVersion.Split('+')[^1],
            InputSha256 = Hash(System.Text.Encoding.UTF8.GetBytes($"fixture-v1:{count}")),
            QuestionsSha256 = questionsPath is null ? Hash(System.Text.Encoding.UTF8.GetBytes(contract.CanonicalJson)) : await FileHashAsync(questionsPath, token),
            Packages = [], Machine = Environment.MachineName, Runtime = RuntimeInformation.FrameworkDescription,
            ExecutionMode = "synthetic"
        };
        if (transport)
        {
            await TransportStudy.RunAsync(output, count, contract, provenance, () => FixtureRows(count, questions, precision, token), token);
            Console.WriteLine($"Synthetic equivalent-content transport study complete: rows={count}.");
            return;
        }
        await DecisionArrowWriter.WriteAsync(output, contract, FixtureRows(count, questions, precision, token), provenance, cancellationToken: token);
        using var reader = await DecisionArrowDatasetReader.OpenAsync(Path.Combine(output, "manifest.json"), contract,
            Enumerable.Range(1, count).Select(i => (long)i), token);
        Console.WriteLine($"Synthetic fixture complete: rows={reader.Manifest.RowCount}; fingerprint={contract.FeatureFingerprint}");
    }

    private static async IAsyncEnumerable<DecisionArrowRow> FixtureRows(int count, IReadOnlyList<DecisionQuestion> questions,
        DecisionPrecision precision, [EnumeratorCancellation] CancellationToken token)
    {
        for (int row = 0; row < count; row++)
        {
            token.ThrowIfCancellationRequested();
            var input = new DecisionInput(JsonSerializer.SerializeToElement("synthetic"), questions);
            var answers = questions.Select(q => q switch
            {
                BinaryDecisionQuestion => (DecisionAnswer)new BinaryDecisionAnswer(q.Id, .42, precision),
                ChoiceDecisionQuestion c => new ChoiceDecisionAnswer(q.Id, c.Candidates[0].Id,
                    c.Candidates.Select((v, i) => KeyValuePair.Create(v.Id, i == c.Candidates.Count - 1 ? 1d : 0d)), precision),
                ScoreDecisionQuestion s => new ScoreDecisionAnswer(q.Id, ScoreValues(s.Rubric.Count, precision), precision: precision),
                _ => throw new NotSupportedException()
            });
            yield return new(row + 1L, input, new(input, "fixture-v1", answers));
        }
        await Task.CompletedTask;
    }

    private static double[] ScoreValues(int width, DecisionPrecision precision)
    {
        var values = new double[width];
        values[^1] = 1;
        values[^2] = precision switch
        {
            DecisionPrecision.FourDecimalPlaces => .0001,
            DecisionPrecision.TwoDecimalPlaces => .01,
            _ => 0
        };
        return values;
    }

    private static async Task SummarizeAsync(string manifest, DecisionArrowSchema contract, CancellationToken token)
    {
        using var reader = await DecisionArrowDatasetReader.OpenAsync(manifest, contract, token);
        var sums = contract.Questions.Select(q => new double[q switch
        {
            ChoiceDecisionQuestion c => c.Candidates.Count, ScoreDecisionQuestion s => s.Rubric.Count, _ => 1
        }]).ToArray();
        long count = 0;
        RecordBatch? batch;
        while ((batch = await reader.ReadNextRecordBatchAsync(token)) is not null)
        {
            using (batch)
            {
                for (int q = 0; q < sums.Length; q++)
                {
                    if (batch.Column(q + 1) is DoubleArray binary)
                        for (int row = 0; row < batch.Length; row++) sums[q][0] += binary.GetValue(row)!.Value;
                    else
                    {
                        var vector = (FixedSizeListArray)((StructArray)batch.Column(q + 1)).Fields[^1];
                        var values = (DoubleArray)vector.Values;
                        for (int row = 0; row < batch.Length; row++)
                            for (int i = 0; i < sums[q].Length; i++) sums[q][i] += values.GetValue((row + vector.Offset) * sums[q].Length + i)!.Value;
                    }
                }
                count += batch.Length;
            }
        }
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            rows = count, featureFingerprint = contract.FeatureFingerprint,
            means = sums.Select((values, q) => new { question = contract.Questions[q].Id,
                values = values.Select(v => count == 0 ? (double?)null : v / count) })
        }, DecisionArrowManifest.JsonOptions));
    }

    private static readonly IReadOnlyDictionary<string, string> ExpectedAssets = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["laya.onnx"] = "a874eb254b58b0fcb1e7ad56fbb188c29d64e08c9a46b689433e1f52c66dba1e",
        ["laya.onnx.data"] = "487746363a8da57bcadb4345352997d22a0fb90d70aa22c6856668d023242aba",
        [@"tokenizer\tokenizer.json"] = "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30",
        ["laya_config.json"] = "5049005dc6ae3ca5e82cc7d85c421357d5c543817300c8e8c5281ddbc69bb561"
    };

    private static async Task<IReadOnlyDictionary<string, string>> VerifyAssetsAsync(string directory, CancellationToken token)
    {
        var hashes = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var expected in ExpectedAssets)
        {
            string path = Path.Combine(directory, expected.Key);
            string actual = await FileHashAsync(path, token);
            if (actual != expected.Value) throw new InvalidDataException($"Pinned Laya asset hash mismatch: {expected.Key}.");
            hashes.Add(expected.Key, actual);
        }
        return hashes;
    }

    private static async Task<IReadOnlyDictionary<string, string>> AuxiliaryAssetsAsync(string directory, CancellationToken token)
    {
        const string name = @"tokenizer\tokenizer_config.json";
        var hashes = new Dictionary<string, string>(StringComparer.Ordinal);
        string path = Path.Combine(directory, name);
        if (File.Exists(path))
        {
            string actual = await FileHashAsync(path, token);
            if (actual != "50044de60daaa73df97d262e15a40d4faf0160e7d742df64b377877a1320dd12")
                throw new InvalidDataException("Pinned auxiliary tokenizer_config.json hash mismatch.");
            hashes.Add(name, actual);
        }
        return hashes;
    }

    private static async IAsyncEnumerable<(long Id, DecisionInput Input)> States(string path, IReadOnlyList<DecisionQuestion> questions,
        [EnumeratorCancellation] CancellationToken token)
    {
        using var reader = File.OpenText(path);
        long line = 0;
        var ids = new HashSet<long>();
        string? text;
        while ((text = await reader.ReadLineAsync(token)) is not null)
        {
            line++;
            using var doc = JsonDocument.Parse(text);
            var fields = doc.RootElement.EnumerateObject().Select(p => p.Name).ToArray();
            if (fields.Length != 2 || fields.Count(p => p == "rowId") != 1 || fields.Count(p => p == "state") != 1)
                throw new InvalidDataException($"Label-free state record {line} must contain only rowId/state.");
            long id = doc.RootElement.GetProperty("rowId").GetInt64();
            if (!ids.Add(id)) throw new InvalidDataException($"Duplicate rowId {id} at line {line}.");
            var state = doc.RootElement.GetProperty("state");
            if (state.ValueKind != JsonValueKind.String) throw new InvalidDataException($"SMS state {id} must preserve an original string.");
            yield return (id, new(state, questions));
        }
    }

    private static async Task<(string InputHash, string QuestionsHash)> PreflightAsync(string assets, string questionsPath, string statesPath, string output, CancellationToken token)
    {
        if (Directory.Exists(output) && Directory.EnumerateFileSystemEntries(output).Any()) throw new IOException("Preflight output must be empty.");
        Directory.CreateDirectory(output);
        try
        {
        var hashes = await VerifyAssetsAsync(assets, token);
        var (questions, _) = Questions(questionsPath);
        var profile = LayaProfile.Load(Path.Combine(assets, "laya_config.json"));
        var (tokenizer, metadata) = LayaTokenizer.Load(Path.Combine(assets, @"tokenizer\tokenizer.json"));
        var prepare = new PrepareDecisionInputs(profile, tokenizer, metadata);
        long count = 0;
        var failures = new List<object>();
        await foreach (var state in States(statesPath, questions, token))
        {
            count++;
            try { _ = prepare.Prepare(state.Input, token); }
            catch (Exception e) when (e is ArgumentException or NotSupportedException)
            {
                failures.Add(new { rowId = state.Id, errorType = e.GetType().Name, budget = e.Message });
            }
        }
        string inputHash = await FileHashAsync(statesPath, token);
        string questionsHash = await FileHashAsync(questionsPath, token);
        await File.WriteAllTextAsync(Path.Combine(output, "preflight.partial.json"), JsonSerializer.Serialize(new
        {
            status = failures.Count == 0 ? "preflight-passed-not-scored" : "failed",
            rows = count, failureCount = failures.Count, failures, inputSha256 = inputHash,
            questionsSha256 = questionsHash, assetHashes = hashes,
            auxiliaryAssetHashes = await AuxiliaryAssetsAsync(assets, token),
            maxLength = profile.MaxLength, headMaxLength = profile.HeadMaxLength, optionTokenBudget = 49,
            minimumInstructionBudget = 16, truncation = false
        }, DecisionArrowManifest.JsonOptions), token);
        if (failures.Count > 0) throw new InvalidDataException($"Strict preflight failed for {failures.Count}/{count} records; IDs/budgets are in preflight.partial.json. No inference occurred.");
        Console.WriteLine($"Strict preflight passed for {count} records. No inference occurred.");
        return (inputHash, questionsHash);
        }
        catch (Exception error)
        {
            try
            {
                await File.WriteAllTextAsync(Path.Combine(output, "run.partial.json"), JsonSerializer.Serialize(new
                {
                    status = error is OperationCanceledException ? "cancelled" : "failed",
                    errorType = error.GetType().Name, error = error.Message, inferenceOccurred = false
                }, DecisionArrowManifest.JsonOptions), CancellationToken.None);
            }
            catch (Exception recordingError) { throw new AggregateException("Preflight failed and its failure record could not be written.", error, recordingError); }
            throw;
        }
    }

    private static async Task ExportAsync(string[] args, CancellationToken token)
    {
        string assets = args[1], questionsPath = args[2], statesPath = args[3], output = args[4], commit = args[5];
        if (commit.Length != 40 || !commit.All(Uri.IsHexDigit)) throw new ArgumentException("Explicit clean source commit SHA is required.");
        string compiledCommit = typeof(Experiment).Assembly.GetCustomAttributes(typeof(System.Reflection.AssemblyInformationalVersionAttribute), false)
            .Cast<System.Reflection.AssemblyInformationalVersionAttribute>().Single().InformationalVersion.Split('+')[^1];
        if (compiledCommit != commit) throw new InvalidDataException("Source commit differs from the compiled exporter.");
        string mode = args.Length == 10 ? args[7] : "scalar";
        int batchSize = args.Length == 10 ? int.Parse(args[8], CultureInfo.InvariantCulture) : 16;
        int threads = args.Length == 10 ? int.Parse(args[9], CultureInfo.InvariantCulture) : 1;
        if (mode is not ("scalar" or "native") || batchSize < 1 || threads < 1) throw new ArgumentException("Explicit scalar/native mode and positive batch/CPU thread counts are required.");
        string preflightOutput = output + ".preflight";
        var validated = await PreflightAsync(assets, questionsPath, statesPath, preflightOutput, token);
        var hashes = await VerifyAssetsAsync(assets, token);
        var (questions, projection) = Questions(questionsPath);
        var contract = new DecisionArrowSchema(questions, DecisionPrecision.HighPrecision, new("laya", LayaProfile.ModelId,
            hashes, "laya-strict-render-framing-nfc-bytelevel-v1", "laya-valid-options-profile-temperature-softmax-v1", projection));
        using var receipt = JsonDocument.Parse(await File.ReadAllTextAsync(args[6], token));
        if (receipt.RootElement.GetProperty("status").GetString() != "READY" ||
            !receipt.RootElement.GetProperty("source").GetProperty("cleanTree").GetBoolean())
            throw new InvalidDataException("A clean immutable producer package receipt is required.");
        var packageEntries = receipt.RootElement.GetProperty("packages").EnumerateArray().ToArray().AsEnumerable();
        if (receipt.RootElement.TryGetProperty("dependencyReceipts", out var dependencies))
            packageEntries = packageEntries.Concat(dependencies.EnumerateArray().SelectMany(r => r.GetProperty("packages").EnumerateArray()));
        var packages = packageEntries.Select(p => new DecisionArrowPackageArtifact(
            p.GetProperty("id").GetString()!, p.GetProperty("version").GetString()!, p.GetProperty("sha256").GetString()!,
            p.GetProperty("sizeBytes").GetInt64())).ToArray();
        string inputHash = await FileHashAsync(statesPath, token);
        if (inputHash != validated.InputHash || await FileHashAsync(questionsPath, token) != validated.QuestionsHash)
            throw new InvalidDataException("Frozen inputs changed after preflight.");
        var measurements = new Dictionary<string, double>(StringComparer.Ordinal);
        var load = Stopwatch.StartNew();
        using var generator = LoadGenerator(assets, threads, out var shapes);
        load.Stop();
        measurements["loadMilliseconds"] = load.Elapsed.TotalMilliseconds;
        var provenance = new DecisionArrowProvenance
        {
            SourceRepository = "luisquintanilla/typesafe-meai", SourceCommit = commit, InputSha256 = inputHash,
            QuestionsSha256 = await FileHashAsync(questionsPath, token), Packages = packages,
            Machine = Environment.MachineName, Runtime = RuntimeInformation.FrameworkDescription,
            ExecutionMode = mode + "-cpu", CpuThreads = threads, NativeStateBatchSize = mode == "native" ? batchSize : 1,
            Measurements = measurements, ModelInputShapes = shapes,
            AuxiliaryAssetHashes = await AuxiliaryAssetsAsync(assets, token)
        };
        var elapsed = Stopwatch.StartNew();
        var manifest = await DecisionArrowWriter.WriteAsync(output, contract, InferenceRows(generator, questions, statesPath, inputHash,
            measurements, mode == "native", batchSize, output, token),
            provenance, cancellationToken: token);
        elapsed.Stop();
        Console.WriteLine($"{mode} export complete: rows={manifest.RowCount}; fingerprint={contract.FeatureFingerprint}; writerWallMilliseconds={elapsed.Elapsed.TotalMilliseconds:R}");
    }

    private static async IAsyncEnumerable<DecisionArrowRow> InferenceRows(LayaDecisionGenerator generator,
        IReadOnlyList<DecisionQuestion> questions, string path, string inputHash, IDictionary<string, double> measurements,
        bool native, int batchSize, string output,
        [EnumeratorCancellation] CancellationToken token)
    {
        var timer = Stopwatch.StartNew();
        long count = 0, calls = 0;
        var pending = new List<(long Id, DecisionInput Input)>(native ? batchSize : 1);
        double preparation = 0, scoring = 0, decoding = 0;
        long maximumTensorBytes = 0;
        await using var raw = new StreamWriter(new FileStream(Path.Combine(output, "batch-measurements.jsonl"), FileMode.CreateNew, FileAccess.Write, FileShare.Read));
        async Task RecordAsync()
        {
            var current = generator.LastExecutionMetrics ?? throw new InvalidOperationException("Successful local inference has no execution metrics.");
            preparation += current.PreparationMilliseconds;
            scoring += current.ScoringMilliseconds;
            decoding += current.DecodingMilliseconds;
            maximumTensorBytes = Math.Max(maximumTensorBytes, current.ManagedTensorBytes);
            await raw.WriteLineAsync(JsonSerializer.Serialize(new { sourceIds = pending.Select(p => p.Id).ToArray(), metrics = current }).AsMemory(), token);
            await raw.FlushAsync(token);
        }
        await foreach (var row in States(path, questions, token))
        {
            pending.Add(row);
            if (pending.Count != (native ? batchSize : 1)) continue;
            var results = await ScoreAsync(generator, pending, native, token);
            await RecordAsync();
            calls++;
            foreach (var result in results)
            {
                if (count == 0) measurements["firstResultMilliseconds"] = timer.Elapsed.TotalMilliseconds;
                count++;
                yield return result;
            }
            pending.Clear();
        }
        if (pending.Count > 0)
        {
            var results = await ScoreAsync(generator, pending, native, token);
            await RecordAsync();
            calls++;
            foreach (var result in results)
            {
                if (count == 0) measurements["firstResultMilliseconds"] = timer.Elapsed.TotalMilliseconds;
                count++;
                yield return result;
            }
        }
        if (await FileHashAsync(path, token) != inputHash) throw new InvalidDataException("States changed during export.");
        measurements["inferenceAndAppendMilliseconds"] = timer.Elapsed.TotalMilliseconds;
        measurements["onnxCalls"] = calls;
        measurements["preparationMilliseconds"] = preparation;
        measurements["scoringMilliseconds"] = scoring;
        measurements["decodingMilliseconds"] = decoding;
        measurements["maximumManagedTensorBytes"] = maximumTensorBytes;
        measurements["processPeakWorkingSetBytes"] = Process.GetCurrentProcess().PeakWorkingSet64;
    }

    private static async Task<IReadOnlyList<DecisionArrowRow>> ScoreAsync(LayaDecisionGenerator generator,
        IReadOnlyList<(long Id, DecisionInput Input)> states, bool native, CancellationToken token)
    {
        IReadOnlyList<DecisionResult> results = native
            ? await generator.GenerateBatchAsync(states.Select(s => s.Input).ToArray(), cancellationToken: token)
            : new[] { await generator.GenerateAsync(states[0].Input, cancellationToken: token) };
        return states.Select((s, i) => new DecisionArrowRow(s.Id, s.Input, results[i])).ToArray();
    }

    private static LayaDecisionGenerator LoadGenerator(string assets, int threads, out Dictionary<string, int[]> shapes)
    {
        var profile = LayaProfile.Load(Path.Combine(assets, "laya_config.json"));
        var (tokenizer, metadata) = LayaTokenizer.Load(Path.Combine(assets, @"tokenizer\tokenizer.json"));
        using var options = new SessionOptions { IntraOpNumThreads = threads, InterOpNumThreads = 1 };
        var session = new InferenceSession(Path.Combine(assets, "laya.onnx"), options);
        try
        {
            shapes = session.InputMetadata.ToDictionary(p => p.Key, p => p.Value.Dimensions.ToArray(), StringComparer.Ordinal);
            var scorer = new ScoreOnnxDecisionModel(session, ownsSession: true);
            return new(new(profile, tokenizer, metadata), scorer, new(profile), ownsScorer: true);
        }
        catch { session.Dispose(); throw; }
    }

    private static async Task ParityAsync(string[] args, CancellationToken token)
    {
        string assets = args[1], questionsPath = args[2], statesPath = args[3], output = args[4];
        int threads = int.Parse(args[5], CultureInfo.InvariantCulture);
        if (threads < 1) throw new ArgumentOutOfRangeException(nameof(threads));
        await PreflightAsync(assets, questionsPath, statesPath, output + ".preflight", token);
        if (Directory.Exists(output) && Directory.EnumerateFileSystemEntries(output).Any()) throw new IOException("Parity output must be empty.");
        Directory.CreateDirectory(output);
        var (questions, _) = Questions(questionsPath);
        var states = new List<(long Id, DecisionInput Input)>();
        await foreach (var row in States(statesPath, questions, token))
        {
            if (states.Count == 33) throw new ArgumentException("Parity is a bounded synthetic reference (at most 33 states), not a reduced SMS corpus study.");
            states.Add(row);
        }
        if (states.Count == 0) throw new ArgumentException("Parity requires explicit synthetic states.");
        var measurements = new List<object>();
        try
        {
            var load = Stopwatch.StartNew();
            using var generator = LoadGenerator(assets, threads, out var shapes);
            load.Stop();
            var first = Stopwatch.StartNew();
            _ = await generator.GenerateAsync(states[0].Input, cancellationToken: token);
            first.Stop();
            var scalar = new List<DecisionResult>();
            var timer = Stopwatch.StartNew();
            foreach (var state in states) scalar.Add(await generator.GenerateAsync(state.Input, cancellationToken: token));
            timer.Stop();
            measurements.Add(new { mode = "scalar", stateBatchSize = 1, milliseconds = timer.Elapsed.TotalMilliseconds,
                onnxCalls = states.Count, rows = states.Count });
            foreach (int size in new[] { 1, 4, 16, 32 })
            {
                _ = await generator.GenerateBatchAsync(states.Take(Math.Min(size, states.Count)).Select(s => s.Input).ToArray(), cancellationToken: token);
                var results = new List<DecisionResult>();
                timer.Restart();
                foreach (var batch in states.Chunk(size))
                    results.AddRange(await generator.GenerateBatchAsync(batch.Select(s => s.Input).ToArray(), cancellationToken: token));
                timer.Stop();
                double maxDifference = 0;
                for (int i = 0; i < scalar.Count; i++)
                    maxDifference = Math.Max(maxDifference, Compare(scalar[i], results[i], states[i].Id));
                measurements.Add(new { mode = "native", stateBatchSize = size, milliseconds = timer.Elapsed.TotalMilliseconds,
                    onnxCalls = (states.Count + size - 1) / size, rows = results.Count, maximumAbsoluteDifference = maxDifference });
            }
            await File.WriteAllTextAsync(Path.Combine(output, "parity.json"), JsonSerializer.Serialize(new
            {
                status = "passed", scope = "retained-real-model-synthetic-state-reference-not-SMS-study",
                questionsSha256 = await FileHashAsync(questionsPath, token), statesSha256 = await FileHashAsync(statesPath, token),
                loadMilliseconds = load.Elapsed.TotalMilliseconds, firstResultMilliseconds = first.Elapsed.TotalMilliseconds,
                modelInputShapes = shapes, cpuThreads = threads, tolerance = DecisionAnswer.DistributionTolerance, measurements
            }, DecisionArrowManifest.JsonOptions), token);
            Console.WriteLine($"Real retained-model synthetic parity passed: {states.Count} states, native sweep 1/4/16/32.");
        }
        catch (Exception error)
        {
            await File.WriteAllTextAsync(Path.Combine(output, "parity.partial.json"), JsonSerializer.Serialize(new
            {
                status = error is OperationCanceledException ? "cancelled" : "failed", errorType = error.GetType().Name,
                error = error.Message, measurements
            }, DecisionArrowManifest.JsonOptions), CancellationToken.None);
            throw;
        }
    }

    private static double Compare(DecisionResult expected, DecisionResult actual, long id)
    {
        if (expected.ModelId != actual.ModelId || expected.Answers.Count != actual.Answers.Count) throw new DecisionProtocolException($"Parity identity mismatch at row {id}.");
        double maximum = 0;
        void Numeric(double a, double b)
        {
            double difference = Math.Abs(a - b);
            if (!double.IsFinite(difference) || difference > DecisionAnswer.DistributionTolerance)
                throw new DecisionProtocolException($"Scalar/native parity exceeded existing tolerance at source row {id}: difference={difference:R}.");
            maximum = Math.Max(maximum, difference);
        }
        for (int i = 0; i < expected.Answers.Count; i++)
        {
            var a = expected.Answers[i];
            var b = actual.Answers[i];
            if (a.QuestionId != b.QuestionId || a.Kind != b.Kind || a.Precision != b.Precision) throw new DecisionProtocolException($"Parity question/precision mismatch at row {id}.");
            switch (a, b)
            {
                case (BinaryDecisionAnswer x, BinaryDecisionAnswer y): Numeric(x.TrueProbability, y.TrueProbability); break;
                case (ChoiceDecisionAnswer x, ChoiceDecisionAnswer y):
                    if (x.SelectedCandidateId != y.SelectedCandidateId) throw new DecisionProtocolException($"Selected ID parity mismatch at row {id}.");
                    foreach (var p in x.Probabilities) Numeric(p.Value, y.Probabilities[p.Key]);
                    break;
                case (ScoreDecisionAnswer x, ScoreDecisionAnswer y):
                    for (int j = 0; j < x.Probabilities.Count; j++) Numeric(x.Probabilities[j], y.Probabilities[j]);
                    Numeric(x.ExpectedIndex, y.ExpectedIndex);
                    Numeric(x.ExpectedIndexLowerBound, y.ExpectedIndexLowerBound);
                    Numeric(x.ExpectedIndexUpperBound, y.ExpectedIndexUpperBound);
                    if (x.NativeScore.HasValue != y.NativeScore.HasValue) throw new DecisionProtocolException($"Native score null parity mismatch at row {id}.");
                    if (x.NativeScore is { } native) Numeric(native, y.NativeScore!.Value);
                    break;
                default: throw new DecisionProtocolException($"Parity answer type mismatch at row {id}.");
            }
        }
        return maximum;
    }

    private static string Hash(byte[] bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));
    private static async Task<string> FileHashAsync(string path, CancellationToken token)
    {
        await using var file = File.OpenRead(path);
        return Convert.ToHexStringLower(await SHA256.HashDataAsync(file, token));
    }
}
