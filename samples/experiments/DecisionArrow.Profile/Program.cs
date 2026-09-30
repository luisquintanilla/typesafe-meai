using System.Diagnostics;
using System.Linq.Expressions;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Apache.Arrow;
using DecisionInference;
using DecisionInference.Arrow;

return await ProducerProfile.RunAsync(args);

internal static class ProducerProfile
{
    private static readonly Type BuilderType = typeof(DecisionArrowWriter).Assembly.GetType("DecisionInference.Arrow.DecisionArrowBatchBuilder", throwOnError: true)!;
    private static readonly Func<DecisionArrowSchema, int, object> NewBuilder = CompileConstructor();
    private static readonly Action<object, DecisionArrowRow> Append = CompileMethod<DecisionArrowRow>("Append");
    private static readonly Func<object, RecordBatch> Build = CompileBuild();
    private static readonly Action<DecisionArrowSchema, RecordBatch> Validate =
        typeof(DecisionArrowDatasetReader).GetMethod("ValidateBatch", BindingFlags.Static | BindingFlags.NonPublic)!
            .CreateDelegate<Action<DecisionArrowSchema, RecordBatch>>();

    public static async Task<int> RunAsync(string[] args)
    {
        if (args.Length == 0 || args[0] == "--help")
        {
            Console.WriteLine("profile <frozen-questions.json> <new-output-directory> <repetitions:1..7> [existing-Laya-manifest trusted-contract]");
            Console.WriteLine("Offline only. Reflection binds ORIGINAL private scopes once before warmup, identically for both executables.");
            return 0;
        }
        try
        {
            if (args.Length is not (4 or 6) || args[0] != "profile") throw new ArgumentException("Invalid profile arguments.");
            int repetitions = int.Parse(args[3], System.Globalization.CultureInfo.InvariantCulture);
            if (repetitions is < 1 or > 7) throw new ArgumentOutOfRangeException(nameof(repetitions));
            if (Directory.Exists(args[2])) throw new IOException("Profile output must be new.");
            Directory.CreateDirectory(args[2]);
            using var qdoc = JsonDocument.Parse(File.ReadAllText(args[1]));
            var questions = DecisionArrowSchema.ParseQuestions(qdoc.RootElement.GetProperty("questions"));
            var projection = qdoc.RootElement.GetProperty("featureProjection").EnumerateArray().Select(v => v.GetString()!).ToArray();
            var identity = new DecisionArrowIdentity("synthetic", "profile-v1", [KeyValuePair.Create("fixture", new string('0', 64))],
                "no-tokenization", "portable-profile-v1", projection);
            var samples = new List<object>();
            foreach (var precision in Enum.GetValues<DecisionPrecision>())
                foreach (int count in new[] { 0, 1, 257, 65536 })
                {
                    var contract = new DecisionArrowSchema(questions, precision, identity);
                    var rows = Enumerable.Range(1, count).Select(id => Row(id, contract)).ToArray();
                    var batches = MakeBatches(rows, contract);
                    try
                    {
                        for (int repeat = -1; repeat < repetitions; repeat++)
                        {
                            var builders = Enumerable.Range(0, (count + 255) / 256).Select(_ => NewBuilder(contract, 256)).ToArray();
                            var append = Measure(() =>
                            {
                                for (int i = 0; i < count; i++) Append(builders[i / 256], rows[i]);
                            });
                            // Build outside append scope. Dispose every owning batch.
                            foreach (var builder in builders) using (Build(builder)) { }
                            var validation = Measure(() => { foreach (var batch in batches) Validate(contract, batch); });
                            var schema = Measure(() =>
                            {
                                for (int i = 0; i < 100; i++) GC.KeepAlive(new DecisionArrowSchema(questions, precision, identity).Schema);
                            });
                            if (repeat >= 0) samples.Add(new { precision = precision.ToString(), count, repeat, append, validation, schemaCompile100 = schema });
                        }
                        string dataset = Path.Combine(args[2], $"{precision}-{count}");
                        var provenance = Provenance(HashFile(args[1]));
                        var encode = await MeasureAsync(async () =>
                            await DecisionArrowWriter.WriteAsync(dataset, contract, Rows(rows), provenance));
                        var open = await MeasureAsync(async () =>
                        {
                            using var reader = await DecisionArrowDatasetReader.OpenAsync(Path.Combine(dataset, "manifest.json"), contract, rows.Select(r => r.RowId));
                        });
                        var replay = await MeasureAsync(async () =>
                        {
                            using var reader = await DecisionArrowDatasetReader.OpenAsync(Path.Combine(dataset, "manifest.json"), contract, rows.Select(r => r.RowId));
                            long seen = 0;
                            while (await reader.ReadNextRecordBatchAsync() is { } batch)
                            {
                                using (batch) seen += batch.Length;
                            }
                            if (seen != count) throw new InvalidDataException("Replay count mismatch.");
                        });
                        samples.Add(new { precision = precision.ToString(), count, repeat = "IO-first-in-process-not-cold",
                            ipcEncodeComplete = encode, completedOpen = open, validatedReplayIncludingOpen = replay,
                            ipcBytes = new FileInfo(Path.Combine(dataset, "decisions.arrow")).Length,
                            ipcSha256 = HashFile(Path.Combine(dataset, "decisions.arrow")), contract.FeatureFingerprint });
                    }
                    finally { foreach (var batch in batches) batch.Dispose(); }
                }
            if (args.Length == 6)
            {
                var trusted = DecisionArrowSchema.FromCanonicalJson(File.ReadAllText(args[5]));
                for (int repeat = -1; repeat < repetitions; repeat++)
                {
                    var open = await MeasureAsync(async () => { using var reader = await DecisionArrowDatasetReader.OpenAsync(args[4], trusted); });
                    if (repeat >= 0) samples.Add(new { scope = "retained-Laya-completed-open-no-inference", repeat, open });
                }
            }
            var assembly = typeof(DecisionArrowWriter).Assembly;
            var receipt = new
            {
                schemaVersion = 1, status = "MEASURED", repetitions, samples,
                source = assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()!.InformationalVersion,
                arrowAssemblySha256 = HashFile(assembly.Location), abstractionsAssemblySha256 = HashFile(typeof(DecisionAnswer).Assembly.Location),
                inputQuestionsSha256 = HashFile(args[1]), runtime = RuntimeInformation.FrameworkDescription,
                settings = "batch256; sequential unchanged ordered fixture values/IDs; warmup excluded; isolated append and math-validation; no scoring",
                limitations = "Initial original profile, not optimization acceptance. IO measurements separate single first-in-process samples, not cold-cache claims. No shared-machine cache flushing. AB/BA paired clean-process acceptance is separate."
            };
            await File.WriteAllTextAsync(Path.Combine(args[2], "profile.receipt.json"),
                JsonSerializer.Serialize(receipt, DecisionArrowManifest.JsonOptions));
            Console.WriteLine($"Measured offline producer matrix: {samples.Count} samples.");
            return 0;
        }
        catch (Exception error)
        {
            Console.Error.WriteLine($"{error.GetType().Name}: {error.Message}");
            return 1;
        }
    }

    private static DecisionArrowRow Row(long id, DecisionArrowSchema contract)
    {
        var input = new DecisionInput(JsonSerializer.SerializeToElement("synthetic-profile-v1"), contract.Questions);
        var answers = new List<DecisionAnswer>();
        foreach (var q in contract.Questions)
        {
            switch (q)
            {
                case BinaryDecisionQuestion: answers.Add(new BinaryDecisionAnswer(q.Id, .42, contract.Precision)); break;
                case ChoiceDecisionQuestion c:
                    answers.Add(new ChoiceDecisionAnswer(q.Id, c.Candidates[0].Id, c.Candidates.Select((v, i) =>
                        KeyValuePair.Create(v.Id, i == c.Candidates.Count - 1 ? 1d : 0d)), contract.Precision));
                    break;
                case ScoreDecisionQuestion s:
                    var values = new double[s.Rubric.Count];
                    values[0] = .25; values[^1] = .75;
                    answers.Add(new ScoreDecisionAnswer(q.Id, values, precision: contract.Precision));
                    break;
            }
        }
        return new(id, input, new(input, contract.Identity.ModelId, answers));
    }

    private static RecordBatch[] MakeBatches(DecisionArrowRow[] rows, DecisionArrowSchema contract)
    {
        var result = new List<RecordBatch>();
        try
        {
            for (int start = 0; start < rows.Length; start += 256)
            {
                object builder = NewBuilder(contract, 256);
                for (int i = start; i < Math.Min(rows.Length, start + 256); i++) Append(builder, rows[i]);
                result.Add(Build(builder));
            }
            return result.ToArray();
        }
        catch { foreach (var batch in result) batch.Dispose(); throw; }
    }

    private static async IAsyncEnumerable<DecisionArrowRow> Rows(DecisionArrowRow[] rows,
        [EnumeratorCancellation] CancellationToken token = default)
    {
        await Task.CompletedTask;
        foreach (var row in rows) { token.ThrowIfCancellationRequested(); yield return row; }
    }

    private static object Measure(Action action)
    {
        int[] collections = [GC.CollectionCount(0), GC.CollectionCount(1), GC.CollectionCount(2)];
        using var process = Process.GetCurrentProcess();
        var cpu = process.TotalProcessorTime;
        var timer = new Stopwatch();
        long allocation = GC.GetAllocatedBytesForCurrentThread();
        timer.Start();
        action();
        timer.Stop();
        long bytes = GC.GetAllocatedBytesForCurrentThread() - allocation;
        process.Refresh();
        return new { milliseconds = timer.Elapsed.TotalMilliseconds, allocatedBytes = bytes,
            cpuMilliseconds = (process.TotalProcessorTime - cpu).TotalMilliseconds,
            gc0 = GC.CollectionCount(0) - collections[0], gc1 = GC.CollectionCount(1) - collections[1], gc2 = GC.CollectionCount(2) - collections[2],
            currentWorkingSetBytes = process.WorkingSet64, peakWorkingSetBytes = process.PeakWorkingSet64 };
    }

    private static async Task<object> MeasureAsync(Func<Task> action)
    {
        long allocation = GC.GetTotalAllocatedBytes(precise: true);
        using var process = Process.GetCurrentProcess();
        var cpu = process.TotalProcessorTime;
        var timer = Stopwatch.StartNew();
        await action();
        timer.Stop();
        process.Refresh();
        return new { milliseconds = timer.Elapsed.TotalMilliseconds, processAllocatedBytes = GC.GetTotalAllocatedBytes(precise: true) - allocation,
            cpuMilliseconds = (process.TotalProcessorTime - cpu).TotalMilliseconds,
            currentWorkingSetBytes = process.WorkingSet64, peakWorkingSetBytes = process.PeakWorkingSet64 };
    }

    private static DecisionArrowProvenance Provenance(string hash) => new()
    {
        SourceRepository = "luisquintanilla/typesafe-meai", SourceCommit = "original-or-candidate-profile",
        InputSha256 = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes("synthetic-profile-v1:IDs1..N:binary.42:choiceLast1:score.25/.75"))),
        QuestionsSha256 = hash, Packages = [], Machine = Environment.MachineName,
        Runtime = RuntimeInformation.FrameworkDescription, ExecutionMode = "synthetic-profile"
    };

    private static string HashFile(string path)
    {
        using var file = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(file));
    }

    private static Func<DecisionArrowSchema, int, object> CompileConstructor()
    {
        var contract = Expression.Parameter(typeof(DecisionArrowSchema));
        var capacity = Expression.Parameter(typeof(int));
        return Expression.Lambda<Func<DecisionArrowSchema, int, object>>(Expression.Convert(
            Expression.New(BuilderType.GetConstructor([typeof(DecisionArrowSchema), typeof(int)])!, contract, capacity), typeof(object)), contract, capacity).Compile();
    }

    private static Action<object, T> CompileMethod<T>(string name)
    {
        var builder = Expression.Parameter(typeof(object));
        var value = Expression.Parameter(typeof(T));
        return Expression.Lambda<Action<object, T>>(Expression.Call(Expression.Convert(builder, BuilderType),
            BuilderType.GetMethod(name)!, value), builder, value).Compile();
    }

    private static Func<object, RecordBatch> CompileBuild()
    {
        var builder = Expression.Parameter(typeof(object));
        return Expression.Lambda<Func<object, RecordBatch>>(Expression.Call(Expression.Convert(builder, BuilderType),
            BuilderType.GetMethod("Build")!), builder).Compile();
    }
}
