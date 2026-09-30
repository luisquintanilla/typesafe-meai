using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using DecisionInference.Arrow;
using Microsoft.ML.OnnxRuntime;

internal sealed record JuliaSessionProfile
{
    public int SchemaVersion { get; init; } = 1;
    public required string Mode { get; init; }
    public string Status { get; init; } = "CANDIDATE";
    public int IntraThreads { get; init; } = 4;
    public int InterThreads { get; init; } = 1;
    public int? DeviceIndex { get; init; }
    public string? DeviceUuid { get; init; }
    public string? QualificationReceiptSha256 { get; init; }
    public string? QualificationReceiptPath { get; init; }
    public string? PackageReceiptPath { get; init; }
    public IReadOnlyList<string> NativeSearchDirectories { get; init; } = [];

    internal string ArithmeticIdentity => Mode switch
    {
        "scalar-cpu" => "ort1.23.2-cpu-fp32-intra4-inter1-default-sync-v1",
        "scalar-cuda" => $"ort1.23.2-cuda-fp32-tf32off-intra4-inter1-device{DeviceIndex}-{DeviceUuid}-default-sync-cuda-cpu-hybrid-v1",
        _ => throw new InvalidDataException("Unknown Julia mode.")
    };

    internal static async Task<JuliaSessionProfile> ReadAsync(string path, CancellationToken token)
    {
        var profile = JsonSerializer.Deserialize<JuliaSessionProfile>(await File.ReadAllTextAsync(path, token), DecisionArrowManifest.JsonOptions)
            ?? throw new InvalidDataException("Missing explicit Julia profile.");
        if (profile.SchemaVersion != 1 || profile.Mode is not ("scalar-cpu" or "scalar-cuda") ||
            profile.IntraThreads != 4 || profile.InterThreads != 1 || profile.Status is not ("CANDIDATE" or "SELECTED"))
            throw new InvalidDataException("Julia profile requires exact approved scalar mode and intra4/inter1.");
        if (profile.Mode == "scalar-cpu" && (profile.DeviceIndex is not null || profile.DeviceUuid is not null))
            throw new InvalidDataException("CPU profile cannot declare a CUDA device.");
        if (profile.Mode == "scalar-cuda" && (profile.DeviceIndex is null or < 0 || profile.DeviceUuid is null ||
            !profile.DeviceUuid.StartsWith("GPU-", StringComparison.Ordinal)))
            throw new InvalidDataException("CUDA requires an explicit NVIDIA index and UUID.");
        if (profile.Mode == "scalar-cpu" && profile.NativeSearchDirectories.Count != 0)
            throw new InvalidDataException("CPU profile cannot select CUDA DLL directories.");
        return profile;
    }

    internal void SelectNativeDirectories()
    {
        if (Mode != "scalar-cuda") return;
        if (NativeSearchDirectories.Count == 0 || NativeSearchDirectories.Any(p => !Path.IsPathFullyQualified(p) || !Directory.Exists(p)))
            throw new InvalidDataException("CUDA requires explicit existing process-local DLL directories.");
        // This changes only this isolated executable's process, never user/system PATH.
        Environment.SetEnvironmentVariable("PATH", string.Join(Path.PathSeparator, NativeSearchDirectories) +
            Path.PathSeparator + Environment.GetEnvironmentVariable("PATH"));
    }

    internal void Configure(SessionOptions options, string? diagnosticPrefix)
    {
        options.IntraOpNumThreads = IntraThreads;
        options.InterOpNumThreads = InterThreads;
        if (diagnosticPrefix is not null)
        {
            options.ProfileOutputPathPrefix = diagnosticPrefix;
            options.EnableProfiling = true;
        }
        var runtime = Assembly.GetEntryAssembly()!.GetReferencedAssemblies().SingleOrDefault(a => a.Name == "Microsoft.ML.OnnxRuntime");
        if (runtime is not null && runtime.Version?.Major != 1) throw new InvalidDataException("Unexpected managed ORT version.");
        bool cudaHost = Assembly.GetEntryAssembly()!.GetName().Name == "DecisionArrow.Cuda";
        if ((Mode == "scalar-cuda") != cudaHost)
            throw new InvalidDataException("Run CPU and CUDA in their separately restored hosts; native package substitution is forbidden.");
        if (OrtEnv.Instance().GetVersionString() != "1.23.2")
            throw new InvalidDataException("Actually loaded native ORT must be exactly 1.23.2.");
        if (cudaHost)
        {
            VerifyNvidiaDevice();
            using var cuda = new OrtCUDAProviderOptions();
            cuda.UpdateOptions(new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["device_id"] = DeviceIndex!.Value.ToString(System.Globalization.CultureInfo.InvariantCulture),
                ["use_tf32"] = "0",
                ["do_copy_in_default_stream"] = "1"
            });
            options.AppendExecutionProvider_CUDA(cuda);
        }
    }

    private void VerifyNvidiaDevice()
    {
        using var process = Process.Start(new ProcessStartInfo("nvidia-smi")
        {
            UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true,
            Arguments = $"--id={DeviceIndex} --query-gpu=uuid --format=csv,noheader"
        }) ?? throw new InvalidOperationException("Cannot query explicit NVIDIA device.");
        string output = process.StandardOutput.ReadToEnd().Trim();
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();
        if (process.ExitCode != 0 || output != DeviceUuid)
            throw new InvalidDataException($"NVIDIA device index/UUID mismatch or query failure: {error}");
    }

    internal async Task<object> LoadedModulesAsync(CancellationToken token)
    {
        using var process = Process.GetCurrentProcess();
        var modules = new List<object>();
        foreach (ProcessModule module in process.Modules)
        {
            string name = Path.GetFileName(module.FileName);
            if (!name.Contains("onnxruntime", StringComparison.OrdinalIgnoreCase) &&
                !name.StartsWith("cud", StringComparison.OrdinalIgnoreCase) &&
                !name.StartsWith("cublas", StringComparison.OrdinalIgnoreCase) &&
                !name.StartsWith("nv", StringComparison.OrdinalIgnoreCase)) continue;
            modules.Add(new { path = module.FileName, version = module.FileVersionInfo.FileVersion,
                sha256 = await JuliaAssetIdentity.HashFileAsync(module.FileName, token) });
        }
        return new { mode = Mode, arithmetic = ArithmeticIdentity, DeviceIndex, DeviceUuid, tf32 = false,
            nativeOrt = OrtEnv.Instance().GetVersionString(), NativeSearchDirectories,
            providerOptions = Mode == "scalar-cuda" ? new Dictionary<string, string>
            {
                ["device_id"] = DeviceIndex!.Value.ToString(System.Globalization.CultureInfo.InvariantCulture),
                ["use_tf32"] = "0", ["do_copy_in_default_stream"] = "1"
            } : null,
            runtime = RuntimeInformation.FrameworkDescription, managedOrt = typeof(InferenceSession).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion,
            modules, hostWorkingSetBytes = process.WorkingSet64, hostPeakWorkingSetBytes = process.PeakWorkingSet64,
            nativeOutputOwnership = "original owned100B state snapshots retained; synchronous Run/copy; no borrowed GPU pointers",
            transfers = "Run includes synchronization and transfers; separate transfer duration not measured" };
    }
}
