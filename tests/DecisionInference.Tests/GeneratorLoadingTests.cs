using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Json;
using DecisionInference.Julia;
using DecisionInference.Laya;
using Microsoft.ML.OnnxRuntime;
using static DecisionInference.Tests.ContractTests;

namespace DecisionInference.Tests;

public sealed class GeneratorLoadingTests
{
    private static CancellationToken Cancel => TestContext.Current.CancellationToken;

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Options_AreRequiredInitOnlyAndDefaultToUnconfiguredSession(bool julia)
    {
        Type type = julia ? typeof(JuliaDecisionGeneratorOptions) : typeof(LayaDecisionGeneratorOptions);
        string[] required = julia ? ["ModelPath", "TokenizerJsonPath"] : ["ModelPath", "TokenizerJsonPath", "ProfilePath"];
        Assert.Equal(required.Order(), type.GetProperties().Where(p => p.IsDefined(typeof(RequiredMemberAttribute))).Select(p => p.Name).Order());
        Assert.All(type.GetProperties(), p => Assert.Contains(typeof(IsExternalInit), p.SetMethod!.ReturnParameter.GetRequiredCustomModifiers()));
        object options = julia
            ? new JuliaDecisionGeneratorOptions { ModelPath = "model", TokenizerJsonPath = "tokenizer" }
            : new LayaDecisionGeneratorOptions { ModelPath = "model", TokenizerJsonPath = "tokenizer", ProfilePath = "profile" };
        Assert.Null(type.GetProperty("ConfigureSession")!.GetValue(options));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Load_NullOptionsAreRejected(bool julia)
    {
        var error = Assert.Throws<ArgumentNullException>(() =>
        {
            using var generator = julia ? (IDecisionGenerator)JuliaDecisionGenerator.Load(null!) : LayaDecisionGenerator.Load(null!);
        });
        Assert.Equal("options", error.ParamName);
    }

    [Theory]
    [InlineData(true, null)]
    [InlineData(false, null)]
    [InlineData(true, "")]
    [InlineData(false, "")]
    [InlineData(true, " \t")]
    [InlineData(false, " \t")]
    public void Directory_InvalidNamesFailBeforeCallback(bool julia, string? name)
    {
        int calls = 0;
        var error = Assert.ThrowsAny<ArgumentException>(() => LoadDirectory(julia, name!, _ => calls++));
        Assert.Equal("modelDirectory", error.ParamName);
        Assert.Equal(0, calls);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Directory_MissingOrFilePathFailsBeforeCallback(bool julia)
    {
        using var files = new Artifacts(julia);
        int calls = 0;
        Assert.Throws<DirectoryNotFoundException>(() => LoadDirectory(julia, Path.Combine(files.Directory, "missing"), _ => calls++));
        Assert.Throws<DirectoryNotFoundException>(() => LoadDirectory(julia, files.Model, _ => calls++));
        Assert.Equal(0, calls);
    }

    public static IEnumerable<object?[]> InvalidPaths()
    {
        foreach (bool julia in new[] { true, false })
            foreach (string property in julia ? new[] { "ModelPath", "TokenizerJsonPath" } : ["ModelPath", "TokenizerJsonPath", "ProfilePath"])
                foreach (string? value in new[] { null, "", " \t", "missing" })
                    yield return [julia, property, value];
    }

    [Theory]
    [MemberData(nameof(InvalidPaths))]
    public void Load_InvalidPathsFailBeforeCallback(bool julia, string property, string? value)
    {
        using var files = new Artifacts(julia);
        int calls = 0;
        string? path = value == "missing" ? Path.Combine(files.Directory, "absent") : value;
        string model = property == "ModelPath" ? path! : files.Model;
        string tokenizer = property == "TokenizerJsonPath" ? path! : files.Tokenizer;
        string profile = property == "ProfilePath" ? path! : files.Profile;
        void Act()
        {
            using var generator = julia
                ? (IDecisionGenerator)JuliaDecisionGenerator.Load(new() { ModelPath = model, TokenizerJsonPath = tokenizer, ConfigureSession = _ => calls++ })
                : LayaDecisionGenerator.Load(new() { ModelPath = model, TokenizerJsonPath = tokenizer, ProfilePath = profile, ConfigureSession = _ => calls++ });
        }
        if (value == "missing") Assert.Equal(path, Assert.Throws<FileNotFoundException>(Act).FileName);
        else Assert.Equal(property, Assert.ThrowsAny<ArgumentException>(Act).ParamName);
        Assert.Equal(0, calls);
    }

    [Theory]
    [InlineData(true, "model")]
    [InlineData(true, "tokenizer")]
    [InlineData(false, "model")]
    [InlineData(false, "tokenizer")]
    [InlineData(false, "profile")]
    public void Directory_RequiresExactFilenamesWithoutSearchingOrSubstitution(bool julia, string artifact)
    {
        using var files = new Artifacts(julia);
        string path = artifact switch { "model" => files.Model, "tokenizer" => files.Tokenizer, _ => files.Profile };
        File.Move(path, Path.Combine(files.Directory, "alternative-" + Path.GetFileName(path)));
        int calls = 0;
        Assert.Equal(path, Assert.Throws<FileNotFoundException>(() => LoadDirectory(julia, files.Directory, _ => calls++)).FileName);
        Assert.Equal(0, calls);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task Load_AndDirectoryMatchPathConstructorAndOwnResources(bool julia)
    {
        using var files = new Artifacts(julia);
        SessionOptions? configured = null;
        int calls = 0;
        string optimized = Path.Combine(files.Directory, "optimized.onnx");
        void Configure(SessionOptions options)
        {
            configured = options;
            calls++;
            Assert.False(options.IsClosed);
            using var defaults = new SessionOptions();
            Assert.Equal(defaults.IntraOpNumThreads, options.IntraOpNumThreads);
            Assert.Equal(defaults.InterOpNumThreads, options.InterOpNumThreads);
            Assert.Equal(defaults.GraphOptimizationLevel, options.GraphOptimizationLevel);
            options.IntraOpNumThreads = 2;
            options.InterOpNumThreads = 1;
            options.OptimizedModelFilePath = optimized;
        }
        using var loaded = files.Load(Configure);
        Assert.Equal(1, calls);
        Assert.True(configured!.IsClosed);
        Assert.True(File.Exists(optimized)); // ORT used the callback's actual options.
        using var directory = LoadDirectory(julia, files.Directory);
        using var borrowed = new SessionOptions { IntraOpNumThreads = 2, InterOpNumThreads = 1 };
        using var direct = files.Construct(borrowed);
        var input = Input(Choice("choice"));
        var expected = (await direct.GenerateAsync(input, cancellationToken: Cancel)).GetAnswer<ChoiceDecisionAnswer>("choice");
        foreach (var generator in new[] { loaded, directory })
        {
            var actual = (await generator.GenerateAsync(input, cancellationToken: Cancel)).GetAnswer<ChoiceDecisionAnswer>("choice");
            Assert.Equal(expected.SelectedCandidateId, actual.SelectedCandidateId);
            Assert.Equal(expected.Probabilities, actual.Probabilities);
            var scorer = PrivateField<object>(generator, "_score");
            var runner = PrivateField<object>(scorer, "_runner");
            // Observe the existing ORT lifetime probe without ever running a freed native handle.
            var session = (InferenceSession)runner.GetType().GetFields(BindingFlags.Instance | BindingFlags.NonPublic)
                .Single(f => f.FieldType == typeof(InferenceSession)).GetValue(runner)!;
            object prepare = PrivateField<object>(generator, "_prepare");
            Assert.False(PrivateField<bool>(session, "_disposed"));
            generator.Dispose();
            generator.Dispose();
            Assert.True(PrivateField<bool>(session, "_disposed"));
            if (prepare is DecisionInference.Julia.PrepareDecisionInputs nativePrepare)
                Assert.Throws<ObjectDisposedException>(() => nativePrepare.Prepare(input, Cancel));
            await Assert.ThrowsAsync<ObjectDisposedException>(() => generator.GenerateAsync(input, cancellationToken: Cancel));
        }
        direct.Dispose();
        Assert.False(borrowed.IsClosed);
        using var stillUsable = new InferenceSession(files.Model, borrowed);
        Assert.Contains("logits", stillUsable.OutputMetadata.Keys);
        Assert.Equal(1, calls);
    }

    [Theory]
    [InlineData(true, true)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(false, false)]
    public void Callback_ThrowsSameExceptionOnceAndDisposesTemporaryOptions(bool julia, bool directory)
    {
        using var files = new Artifacts(julia);
        SessionOptions? captured = null;
        int calls = 0;
        var expected = new InvalidOperationException("configuration failed");
        void Configure(SessionOptions options) { captured = options; calls++; throw expected; }
        var actual = Assert.Throws<InvalidOperationException>(() =>
        {
            using var generator = directory ? LoadDirectory(julia, files.Directory, Configure) : files.Load(Configure);
        });
        Assert.Same(expected, actual);
        Assert.Equal(1, calls);
        Assert.True(captured!.IsClosed);
    }

    [Theory]
    [InlineData(true, "tokenizer")]
    [InlineData(false, "tokenizer")]
    [InlineData(false, "profile")]
    [InlineData(true, "model")]
    [InlineData(false, "model")]
    [InlineData(true, "metadata")]
    [InlineData(false, "metadata")]
    [InlineData(true, "sidecar")]
    [InlineData(false, "sidecar")]
    public async Task ConstructionFailure_DisposesOptionsPropagatesAndAllowsRecovery(bool julia, string failure)
    {
        using var files = new Artifacts(julia);
        string damaged = failure switch { "tokenizer" => files.Tokenizer, "profile" => files.Profile, _ => files.Model };
        byte[] original = File.ReadAllBytes(damaged);
        if (failure == "metadata")
            File.WriteAllBytes(damaged, julia ? SyntheticOnnx.Create() : SyntheticOnnx.CreateJulia());
        else if (failure == "sidecar") File.WriteAllBytes(damaged, SyntheticOnnx.CreateExternalData(julia));
        else File.WriteAllText(damaged, "invalid artifact");
        SessionOptions? captured = null;
        void Configure(SessionOptions options) { captured = options; options.IntraOpNumThreads = 1; options.InterOpNumThreads = 1; }
        void Act() { using var generator = files.Load(Configure); }
        if (failure is "tokenizer" or "profile") Assert.ThrowsAny<JsonException>(Act);
        else if (failure == "metadata") Assert.Throws<ArgumentException>(Act);
        else
        {
            var error = Assert.Throws<OnnxRuntimeException>(Act);
            if (failure == "sidecar") Assert.Contains("weights.bin", error.Message, StringComparison.Ordinal);
        }
        Assert.True(captured!.IsClosed);
        File.WriteAllBytes(damaged, original);
        using var recovered = files.Load(Configure);
        Assert.True(captured.IsClosed);
        var answer = (await recovered.GenerateAsync(Input(Choice("choice")), cancellationToken: Cancel)).GetAnswer<ChoiceDecisionAnswer>("choice");
        Assert.Equal(2, answer.Probabilities.Count);
        Assert.Equal(1, answer.Probabilities.Values.Sum(), 10);
    }

    private static T PrivateField<T>(object owner, string name) =>
        (T)owner.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(owner)!;

    private static IDecisionGenerator LoadDirectory(bool julia, string path, Action<SessionOptions>? configure = null) =>
        julia ? JuliaDecisionGenerator.LoadFromDirectory(path, configure) : LayaDecisionGenerator.LoadFromDirectory(path, configure);

    private sealed class Artifacts : IDisposable
    {
        private readonly bool _julia;
        internal string Directory { get; } = Path.Combine(Path.GetTempPath(), $"decision-loading-{Guid.NewGuid():N}");
        internal string Model => Path.Combine(Directory, _julia ? "model.onnx" : "laya.onnx");
        internal string Tokenizer => _julia ? Path.Combine(Directory, "tokenizer.json") : Path.Combine(Directory, "tokenizer", "tokenizer.json");
        internal string Profile => Path.Combine(Directory, "laya_config.json");

        internal Artifacts(bool julia)
        {
            _julia = julia;
            System.IO.Directory.CreateDirectory(Path.GetDirectoryName(Tokenizer)!);
            File.WriteAllBytes(Model, julia ? SyntheticOnnx.CreateJulia() : SyntheticOnnx.Create());
            if (julia)
            {
                using var fixture = new JuliaTestFixture();
                File.Copy(fixture.Path, Tokenizer);
            }
            else
            {
                string[] framing = ["[PAD]", "[CLS]", "[SEP]", "[MASK]"];
                var vocab = framing.Concat(Enumerable.Range(33, 94).Select(c => ((char)c).ToString())).Append("\u0120")
                    .Select((text, id) => KeyValuePair.Create(text, id)).ToDictionary();
                File.WriteAllText(Tokenizer, JsonSerializer.Serialize(new
                {
                    normalizer = new { type = "NFC" },
                    pre_tokenizer = new { type = "ByteLevel", add_prefix_space = false, trim_offsets = true, use_regex = true },
                    added_tokens = framing.Select((content, id) => new { content, id, special = true, lstrip = false, rstrip = false, single_word = false }),
                    model = new
                    {
                        type = "BPE", byte_fallback = false, ignore_merges = false, fuse_unk = false,
                        dropout = (object?)null, unk_token = (object?)null, continuing_subword_prefix = (object?)null,
                        end_of_word_suffix = (object?)null, vocab, merges = Array.Empty<string>()
                    }
                }));
                File.WriteAllText(Profile, """{"max_len":512,"head_max_len":256,"temperature":[1,1,1],"temperature_by_options":{}}""");
            }
        }

        internal IDecisionGenerator Load(Action<SessionOptions>? configure) => _julia
            ? JuliaDecisionGenerator.Load(new() { ModelPath = Model, TokenizerJsonPath = Tokenizer, ConfigureSession = configure })
            : LayaDecisionGenerator.Load(new() { ModelPath = Model, TokenizerJsonPath = Tokenizer, ProfilePath = Profile, ConfigureSession = configure });

        internal IDecisionGenerator Construct(SessionOptions options) => _julia
            ? new JuliaDecisionGenerator(Model, Tokenizer, options)
            : new LayaDecisionGenerator(Model, Tokenizer, Profile, options);

        public void Dispose() => System.IO.Directory.Delete(Directory, recursive: true);
    }
}
