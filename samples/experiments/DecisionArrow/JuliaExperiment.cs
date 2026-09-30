using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json;
using DecisionInference;
using DecisionInference.Arrow;
using DecisionInference.Julia;
using Microsoft.ML.OnnxRuntime;

internal static class JuliaExperiment
{
    internal static async Task<int> RunAsync(string[] args)
    {
        if (args.Length == 0 || args[0] is "--help" or "julia-help")
        {
            Console.WriteLine("""
                Julia commands are explicit, local-only, scalar state1/output256; no implicit acquisition/fallback.
                julia-contract <frozen-questions.json> <explicit-profile.json> <new-directory>
                julia-fixture <frozen-questions.json> <explicit-profile.json> <new-directory>
                julia-preflight <assets-directory> <frozen-questions.json> <all5574-label-free-states.jsonl> <new-directory>
                julia-control <assets-directory> <frozen-questions.json> <label-free-controls.jsonl> <canonical-contract.json> <explicit-profile.json> <new-directory> [diagnostic]
                julia-export <assets-directory> <frozen-questions.json> <all5574-label-free-states.jsonl> <canonical-contract.json> <selected-profile.json> <new-directory>
                julia-compare <reference-manifest> <reference-contract> <candidate-manifest> <candidate-contract> <label-free-controls> <new-report.json>
                CPU host: DecisionArrow; CUDA host: DecisionArrow.Cuda, only ORT Gpu1.23.2.
                Profile uses exact scalar-cpu or scalar-cuda, intra4/inter1, explicit NVIDIA index/UUID, TF32off.
                Diagnostic placement is separate from timing. Full export requires SELECTED profile and qualification receipt.
                """);
            return 0;
        }
        using var cancellation = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) => { e.Cancel = true; cancellation.Cancel(); };
        try
        {
            switch (args[0])
            {
                case "julia-contract" or "julia-fixture" when args.Length == 4:
                    await ContractAsync(args[1], args[2], args[3], args[0] == "julia-fixture", cancellation.Token);
                    break;
                case "julia-preflight" when args.Length == 5:
                    await PreflightAsync(args[1], args[2], args[3], args[4], full: true, cancellation.Token);
                    break;
                case "julia-control" when args.Length is 7 or 8:
                    if (args.Length == 8 && args[7] != "diagnostic") throw new ArgumentException("Only explicit diagnostic placement is supported.");
                    await ExportAsync(args, full: false, diagnostic: args.Length == 8, cancellation.Token);
                    break;
                case "julia-export" when args.Length == 7:
                    await ExportAsync(args, full: true, diagnostic: false, cancellation.Token);
                    break;
                case "julia-compare" when args.Length == 7:
                    await JuliaControlComparison.CompareAsync(args[1], args[2], args[3], args[4], args[5], args[6], cancellation.Token);
                    break;
                default: throw new ArgumentException("Invalid Julia command. Run julia-help.");
            }
            return 0;
        }
        catch (Exception error)
        {
            if (args.Length >= 7 && args[0] is "julia-control" or "julia-export")
            {
                try
                {
                    await using var failed = new FileStream(args[6] + ".execution.partial.json", FileMode.CreateNew);
                    await JsonSerializer.SerializeAsync(failed, new
                    {
                        status = error is OperationCanceledException ? "cancelled" : "failed",
                        errorType = error.GetType().Name, error = error.Message, cpuFallbackOccurred = false
                    }, DecisionArrowManifest.JsonOptions, CancellationToken.None);
                }
                catch (Exception recordingError)
                {
                    throw new AggregateException("Julia execution failed and its failure receipt could not be recorded.", error, recordingError);
                }
            }
            Console.Error.WriteLine($"{error.GetType().Name}: {error.Message}");
            return error is OperationCanceledException ? 2 : 1;
        }
    }

    private static async Task<(IReadOnlyList<DecisionQuestion> Questions, string[] Projection)> QuestionsAsync(string path, CancellationToken token)
    {
        if (await JuliaAssetIdentity.HashFileAsync(path, token) != JuliaAssetIdentity.QuestionsHash)
            throw new InvalidDataException("Original frozen question bytes changed.");
        using var document = JsonDocument.Parse(await File.ReadAllTextAsync(path, token));
        if (document.RootElement.GetProperty("schemaVersion").GetInt32() != 1) throw new InvalidDataException("Unsupported questions version.");
        return (DecisionArrowSchema.ParseQuestions(document.RootElement.GetProperty("questions")),
            document.RootElement.GetProperty("featureProjection").EnumerateArray().Select(v => v.GetString()!).ToArray());
    }

    private static DecisionArrowSchema Contract(IReadOnlyList<DecisionQuestion> questions, string[] projection, JuliaSessionProfile profile,
        DecisionPrecision precision = DecisionPrecision.HighPrecision) => new(questions, precision,
        new("julia", JuliaDecisionGenerator.ModelId, JuliaAssetIdentity.Hashes, JuliaAssetIdentity.Preprocessing,
            JuliaAssetIdentity.Decoder + ";" + profile.ArithmeticIdentity, projection));

    private static async Task ContractAsync(string questionsPath, string profilePath, string directory, bool fixture, CancellationToken token)
    {
        NewDirectory(directory);
        var (questions, projection) = await QuestionsAsync(questionsPath, token);
        var profile = await JuliaSessionProfile.ReadAsync(profilePath, token);
        var contract = Contract(questions, projection, profile);
        await File.WriteAllTextAsync(Path.Combine(directory, "julia-contract.json"), contract.CanonicalJson, token);
        await File.WriteAllTextAsync(Path.Combine(directory, "profile.json"), JsonSerializer.Serialize(profile, DecisionArrowManifest.JsonOptions), token);
        if (fixture)
            foreach (var precision in Enum.GetValues<DecisionPrecision>())
            {
                var fixtureContract = Contract(questions, projection, profile, precision);
                string output = Path.Combine(directory, precision.ToString());
                var ids = Enumerable.Range(1, 257).Select(i => (long)i).ToArray();
                await DecisionArrowWriter.WriteAsync(output, fixtureContract, FixtureRows(ids, fixtureContract, token),
                    Provenance("julia-offline-fixture-not-real", new string('0', 64)), cancellationToken: token);
                using var reader = await DecisionArrowDatasetReader.OpenAsync(Path.Combine(output, "manifest.json"), fixtureContract, ids, token);
            }
        Console.WriteLine($"Julia canonical identity: {contract.FeatureFingerprint}; fixture={fixture}, inference=false.");
    }

    private static async IAsyncEnumerable<DecisionArrowRow> FixtureRows(long[] ids, DecisionArrowSchema contract,
        [EnumeratorCancellation] CancellationToken token)
    {
        await Task.CompletedTask;
        foreach (long id in ids)
        {
            token.ThrowIfCancellationRequested();
            var input = new DecisionInput(JsonSerializer.SerializeToElement("fictional-julia-offline-fixture"), contract.Questions);
            var answers = contract.Questions.Select(q => q switch
            {
                BinaryDecisionQuestion => (DecisionAnswer)new BinaryDecisionAnswer(q.Id, .42, contract.Precision),
                ChoiceDecisionQuestion c => new ChoiceDecisionAnswer(q.Id, c.Candidates[0].Id,
                    c.Candidates.Select((v, i) => KeyValuePair.Create(v.Id, i == c.Candidates.Count - 1 ? 1d : 0d)), contract.Precision),
                ScoreDecisionQuestion s => new ScoreDecisionAnswer(q.Id,
                    Enumerable.Range(0, s.Rubric.Count).Select(i => i == 0 ? .25 : i == s.Rubric.Count - 1 ? .75 : 0), precision: contract.Precision),
                _ => throw new NotSupportedException()
            });
            yield return new(id, input, new(input, contract.Identity.ModelId, answers));
        }
    }

    private static async Task PreflightAsync(string assets, string questionsPath, string statesPath, string directory, bool full, CancellationToken token)
    {
        NewDirectory(directory);
        await JuliaAssetIdentity.VerifyAsync(assets, token);
        var (questions, _) = await QuestionsAsync(questionsPath, token);
        if (full && await JuliaAssetIdentity.HashFileAsync(statesPath, token) != JuliaAssetIdentity.StatesHash)
            throw new InvalidDataException("Original frozen full state bytes changed.");
        using var prepare = new PrepareDecisionInputs(Path.Combine(assets, "tokenizer.json"));
        var failures = new List<object>();
        var shapes = new SortedDictionary<int, int>();
        var ids = new List<long>();
        string preparedPath = Path.Combine(directory, "prepared.jsonl");
        await using var prepared = new StreamWriter(new FileStream(preparedPath, FileMode.CreateNew));
        await foreach (var state in States(statesPath, questions, token))
        {
            ids.Add(state.Id);
            try
            {
                var batch = prepare.Prepare(state.Input, token);
                shapes[batch.SequenceLength] = shapes.GetValueOrDefault(batch.SequenceLength) + 1;
                await prepared.WriteLineAsync(JsonSerializer.Serialize(new
                {
                    rowId = state.Id, batch.BatchSize, batch.SequenceLength, batch.MarkerWidth,
                    tensorSha256 = JuliaPreparedIdentity.Hash(batch)
                }).AsMemory(), token);
            }
            catch (Exception error) when (error is ArgumentException or NotSupportedException or DecisionProtocolException)
            {
                failures.Add(new { rowId = state.Id, category = error.GetType().Name, error = error.Message });
            }
        }
        await prepared.FlushAsync(token);
        await prepared.DisposeAsync();
        bool exact = !full || ids.SequenceEqual(Enumerable.Range(1, 5574).Select(i => (long)i));
        await File.WriteAllTextAsync(Path.Combine(directory, "preflight.receipt.json"), JsonSerializer.Serialize(new
        {
            status = failures.Count == 0 && exact ? "PASS" : "BLOCKED", rows = ids.Count, failures, full, exactIds = exact,
            statesSha256 = await JuliaAssetIdentity.HashFileAsync(statesPath, token),
            questionsSha256 = JuliaAssetIdentity.QuestionsHash, assetHashes = JuliaAssetIdentity.Hashes,
            shapes, preparedSha256 = await JuliaAssetIdentity.HashFileAsync(preparedPath, token),
            scoringOccurred = false, truncation = false
        }, DecisionArrowManifest.JsonOptions), token);
        if (failures.Count > 0 || !exact) throw new InvalidDataException($"Julia strict preparation blocked: failures={failures.Count}; exactIds={exact}.");
    }

    private static async Task ExportAsync(string[] args, bool full, bool diagnostic, CancellationToken token)
    {
        string assets = args[1], questionsPath = args[2], statesPath = args[3], contractPath = args[4], profilePath = args[5], directory = args[6];
        var (questions, projection) = await QuestionsAsync(questionsPath, token);
        var profile = await JuliaSessionProfile.ReadAsync(profilePath, token);
        var contract = DecisionArrowSchema.FromCanonicalJson(await File.ReadAllTextAsync(contractPath, token));
        if (contract.CanonicalJson != Contract(questions, projection, profile).CanonicalJson)
            throw new InvalidDataException("Trusted Julia contract disagrees with the explicit arithmetic/device profile.");
        if (full && (profile.Status != "SELECTED" || !DecisionArrowIdentityHash(profile.QualificationReceiptSha256)))
            throw new InvalidDataException("Full export requires the frozen accepted qualification receipt/profile.");
        if (full)
        {
            if (profile.QualificationReceiptPath is null ||
                await JuliaAssetIdentity.HashFileAsync(profile.QualificationReceiptPath, token) != profile.QualificationReceiptSha256)
                throw new InvalidDataException("Selected qualification receipt is missing or its bytes changed.");
            using var qualification = JsonDocument.Parse(await File.ReadAllTextAsync(profile.QualificationReceiptPath, token));
            if (qualification.RootElement.GetProperty("status").GetString() != "GO-READY" ||
                qualification.RootElement.GetProperty("featureFingerprint").GetString() != contract.FeatureFingerprint)
                throw new InvalidDataException("Qualification did not accept this exact feature identity.");
        }
        if (Directory.Exists(directory)) throw new IOException("Julia output directory must be new.");
        await PreflightAsync(assets, questionsPath, statesPath, directory + ".preflight", full, token);
        string inputHash = await JuliaAssetIdentity.HashFileAsync(statesPath, token);
        if (full && inputHash != JuliaAssetIdentity.StatesHash)
            throw new InvalidDataException("Frozen full states changed after preflight.");
        var expectedIds = new List<long>();
        await foreach (var state in States(statesPath, questions, token)) expectedIds.Add(state.Id);
        var packages = await JuliaRuntimePackages.ReadAsync(profile, token);
        // Native initialization cannot silently substitute CPU under a CUDA identity.
        profile.SelectNativeDirectories();
        using var options = new SessionOptions();
        profile.Configure(options, diagnostic ? directory + ".placement" : null);
        var timer = Stopwatch.StartNew();
        using var session = new InferenceSession(Path.Combine(assets, "model.onnx"), options);
        using var prepare = new PrepareDecisionInputs(Path.Combine(assets, "tokenizer.json"));
        using var score = new ScoreOnnxDecisionModel(session);
        using var generator = new JuliaDecisionGenerator(prepare, score, new DecodeDecisions());
        var loadMilliseconds = timer.Elapsed.TotalMilliseconds;
        timer.Restart();
        object outputMemory;
        await using (var first = States(statesPath, questions, token).GetAsyncEnumerator(token))
        {
            if (!await first.MoveNextAsync()) throw new InvalidDataException("Real Julia controls/export cannot be empty.");
            outputMemory = JuliaHostOutputProbe.Check(session, prepare.Prepare(first.Current.Input, token), token);
            var warmup = new JuliaArrowExporter(contract, generator);
            using var warmed = await warmup.WarmupAsync(first.Current, token);
        }
        double warmupMilliseconds = timer.Elapsed.TotalMilliseconds;
        var exporter = new JuliaArrowExporter(contract, generator);
        var graph = new
        {
            inputs = session.InputMetadata.ToDictionary(p => p.Key, p => new { element = p.Value.ElementType.FullName, dimensions = p.Value.Dimensions }),
            outputs = session.OutputMetadata.ToDictionary(p => p.Key, p => new { element = p.Value.ElementType.FullName, dimensions = p.Value.Dimensions })
        };
        timer.Restart();
        double extractionAndValidationMilliseconds = 0;
        var manifest = await DecisionArrowWriter.WriteBatchesBeforePublicationAsync(directory, contract,
            exporter.Batches(States(statesPath, questions, token), token), Provenance(profile.Mode, inputHash, packages),
            async (pending, ct) =>
        {
            string validationPath = Path.Combine(directory, "completion.validation.pending.json");
            await File.WriteAllTextAsync(validationPath, JsonSerializer.Serialize(pending, DecisionArrowManifest.JsonOptions), ct);
            try
            {
                using var validated = await DecisionArrowDatasetReader.OpenAsync(validationPath, contract, expectedIds, ct);
                if (await JuliaAssetIdentity.HashFileAsync(statesPath, ct) != inputHash)
                    throw new InvalidDataException("States changed during extraction.");
                if (await JuliaAssetIdentity.HashFileAsync(questionsPath, ct) != JuliaAssetIdentity.QuestionsHash)
                    throw new InvalidDataException("Questions changed during extraction.");
                extractionAndValidationMilliseconds = timer.Elapsed.TotalMilliseconds;
                await JuliaAssetIdentity.VerifyAsync(assets, ct);
                object? placement = null;
                if (diagnostic)
                {
                    string trace = session.EndProfiling();
                    using var document = JsonDocument.Parse(await File.ReadAllTextAsync(trace, ct));
                    var kernels = document.RootElement.EnumerateArray().Where(e => e.TryGetProperty("cat", out var cat) && cat.GetString() == "Node" &&
                        e.TryGetProperty("args", out var a) && a.TryGetProperty("provider", out _))
                        .Select(e => new { name = e.GetProperty("name").GetString(), provider = e.GetProperty("args").GetProperty("provider").GetString(),
                            op = e.GetProperty("args").TryGetProperty("op_name", out var op) ? op.GetString() : null }).ToArray();
                    bool scoringCuda = kernels.Any(k => k.provider == "CUDAExecutionProvider" && k.op is "MatMul" or "Gemm" or "Attention" or "FusedMatMul");
                    placement = new { trace, traceSha256 = await JuliaAssetIdentity.HashFileAsync(trace, ct), kernels,
                        label = kernels.Any(k => k.provider == "CPUExecutionProvider") ? "HYBRID" : "CUDA",
                        actualCudaScoring = scoringCuda };
                    if (profile.Mode == "scalar-cuda" && !scoringCuda)
                        throw new InvalidDataException("Diagnostic trace does not prove actual Julia CUDA scoring kernels.");
                }
                await File.WriteAllTextAsync(Path.Combine(directory, "runtime.receipt.json"), JsonSerializer.Serialize(new
                {
                    schemaVersion = 1, status = full ? "FULL-EXPORT-VERIFIED" : "COMPLETE-CONTROL-NOT-QUALIFIED", full, diagnostic,
                    profile, graph, runtime = await profile.LoadedModulesAsync(ct), placement,
                    packages,
                    loadMilliseconds, warmupMilliseconds, warmupStateCount = 1, warmupOnnxCalls = 2, outputMemory,
                    extractionAndValidationMilliseconds,
                    contract.FeatureFingerprint, pending.RowCount, inputHash,
                    scoringRecipe = "stateBatch1/outputBatch256/FP32graph/DOUBLEsoftmax/nativeScoreNULL/nohiddenfallback",
                    precisionTolerance = DecisionAnswer.DistributionTolerance,
                    timing = "prep/synchronizedRun+transfers/decode/builder/IPC/full validation; excludes load and runtime receipt/module hashing"
                }, DecisionArrowManifest.JsonOptions), ct);
            }
            finally { File.Delete(validationPath); }
        }, token);
        Console.WriteLine($"Julia {profile.Mode} complete: rows={manifest.RowCount}; fingerprint={contract.FeatureFingerprint}; extractionAndValidationMilliseconds={extractionAndValidationMilliseconds:R}.");
    }

    private static DecisionArrowProvenance Provenance(string mode, string inputHash,
        IReadOnlyList<DecisionArrowPackageArtifact>? packages = null) => new()
    {
        SourceRepository = "luisquintanilla/typesafe-meai",
        SourceCommit = typeof(JuliaExperiment).Assembly.GetCustomAttributes(typeof(System.Reflection.AssemblyInformationalVersionAttribute), false)
            .Cast<System.Reflection.AssemblyInformationalVersionAttribute>().Single().InformationalVersion.Split('+')[^1],
        InputSha256 = inputHash, QuestionsSha256 = JuliaAssetIdentity.QuestionsHash, Packages = packages ?? [],
        Machine = Environment.MachineName, Runtime = RuntimeInformation.FrameworkDescription,
        ExecutionMode = mode, CpuThreads = 4, NativeStateBatchSize = 1
    };

    private static bool DecisionArrowIdentityHash(string? value) =>
        value is { Length: 64 } && value.All(c => c is >= '0' and <= '9' or >= 'a' and <= 'f');

    internal static async IAsyncEnumerable<(long Id, DecisionInput Input)> States(string path, IReadOnlyList<DecisionQuestion> questions,
        [EnumeratorCancellation] CancellationToken token)
    {
        using var file = File.OpenText(path);
        var seen = new HashSet<long>();
        string? line;
        while ((line = await file.ReadLineAsync(token)) is not null)
        {
            using var doc = JsonDocument.Parse(line);
            var fields = doc.RootElement.EnumerateObject().ToArray();
            if (fields.Length != 2 || fields.Count(p => p.Name == "rowId") != 1 || fields.Count(p => p.Name == "state") != 1)
                throw new InvalidDataException("Only label-free rowId/state fields are accepted.");
            long id = doc.RootElement.GetProperty("rowId").GetInt64();
            if (!seen.Add(id)) throw new InvalidDataException($"Duplicate original ID {id}.");
            var state = doc.RootElement.GetProperty("state");
            if (state.ValueKind != JsonValueKind.String) throw new InvalidDataException($"Original ID {id} must preserve its string state.");
            yield return (id, new(state, questions));
        }
    }

    private static void NewDirectory(string directory)
    {
        if (Directory.Exists(directory)) throw new IOException("Julia artifact directory must be new.");
        Directory.CreateDirectory(directory);
    }
}
