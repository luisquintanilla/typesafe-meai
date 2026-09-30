using System.Reflection;
using System.Text.Json;
using DecisionInference.Arrow;

internal static class JuliaRuntimePackages
{
    internal static async Task<IReadOnlyList<DecisionArrowPackageArtifact>> ReadAsync(JuliaSessionProfile profile, CancellationToken token)
    {
        string host = Assembly.GetEntryAssembly()!.GetName().Name!;
        string assetsPath = Path.Combine(Environment.CurrentDirectory, "samples", "experiments", host, "obj", "project.assets.json");
        using var assets = JsonDocument.Parse(await File.ReadAllTextAsync(assetsPath, token));
        string cache = assets.RootElement.GetProperty("packageFolders").EnumerateObject().Single().Name;
        string config = Path.Combine(Path.GetDirectoryName(cache.TrimEnd(Path.DirectorySeparatorChar))!, "NuGet.Experiment.config");
        if (!File.Exists(config)) throw new InvalidDataException("Runtime requires its explicitly mapped isolated package cache.");
        var packages = new List<DecisionArrowPackageArtifact>();
        foreach (var library in assets.RootElement.GetProperty("libraries").EnumerateObject())
        {
            if (library.Value.GetProperty("type").GetString() != "package") continue;
            int separator = library.Name.LastIndexOf('/');
            string id = library.Name[..separator], version = library.Name[(separator + 1)..];
            string folder = Path.Combine(cache, library.Value.GetProperty("path").GetString()!.Replace('/', Path.DirectorySeparatorChar));
            using var metadata = JsonDocument.Parse(await File.ReadAllTextAsync(Path.Combine(folder, ".nupkg.metadata"), token));
            string? source = metadata.RootElement.GetProperty("source").GetString();
            if (id is not ("Apache.Arrow" or "Apache.Arrow.Scalars") && source != "https://www.nuget.org/api/v2/")
                throw new InvalidDataException($"Runtime public dependency escaped the verified source: {id}.");
            string file = Path.Combine(folder, $"{id.ToLowerInvariant()}.{version.ToLowerInvariant()}.nupkg");
            packages.Add(new(id, version, await JuliaAssetIdentity.HashFileAsync(file, token), new FileInfo(file).Length));
        }
        bool cuda = profile.Mode == "scalar-cuda";
        if (packages.Any(p => p.Id == "Apache.Arrow.Compute") ||
            !packages.Any(p => p.Id == "Apache.Arrow" && p.Version == "23.0.0-exp.decisions.1.g6aafc634d65c" &&
                p.Sha256 == "a8674375a36a580b4f5e0146030c0e95c55d60c72e89296bf5b68049216c4fd4") ||
            !packages.Any(p => p.Id == "Microsoft.ML.OnnxRuntime.Managed" && p.Version == "1.23.2") ||
            packages.Any(p => p.Id.StartsWith("Microsoft.ML.OnnxRuntime", StringComparison.Ordinal) && p.Version != "1.23.2") ||
            (cuda && (!packages.Any(p => p.Id == "Microsoft.ML.OnnxRuntime.Gpu") || packages.Any(p => p.Id == "Microsoft.ML.OnnxRuntime"))) ||
            (!cuda && (!packages.Any(p => p.Id == "Microsoft.ML.OnnxRuntime") || packages.Any(p => p.Id.StartsWith("Microsoft.ML.OnnxRuntime.Gpu", StringComparison.Ordinal)))))
            throw new InvalidDataException("Runtime closure violates the exact isolated CPU/CUDA/portable pins.");
        if (profile.PackageReceiptPath is null) throw new InvalidDataException("Explicit immutable portable package receipt is required before scoring.");
        using var receipt = JsonDocument.Parse(await File.ReadAllTextAsync(profile.PackageReceiptPath, token));
        var root = receipt.RootElement;
        if (root.GetProperty("status").GetString() != "READY" || root.GetProperty("kind").GetString() != "decision-adapter" ||
            !root.GetProperty("source").GetProperty("cleanTree").GetBoolean() || !root.GetProperty("packaging").GetProperty("immutable").GetBoolean())
            throw new InvalidDataException("Portable producer receipt is not clean/immutable.");
        string sourceCommit = root.GetProperty("source").GetProperty("commit").GetString()!;
        if (sourceCommit is not { Length: 40 } || sourceCommit.Any(c => c is not (>= '0' and <= '9' or >= 'a' and <= 'f')))
            throw new InvalidDataException("Portable source commit must be exact lowercase SHA-1.");
        string versionPin = $"0.1.0-exp.decisions.1.g{sourceCommit[..12]}";
        var artifacts = root.GetProperty("packages").EnumerateArray().ToArray();
        if (artifacts.Length != 2 || artifacts.Select(p => p.GetProperty("id").GetString()).Distinct(StringComparer.Ordinal).Count() != 2)
            throw new InvalidDataException("Both exact portable packages are required.");
        foreach (var item in artifacts)
        {
            string id = item.GetProperty("id").GetString()!, version = item.GetProperty("version").GetString()!,
                file = item.GetProperty("file").GetString()!, hash = item.GetProperty("sha256").GetString()!;
            long size = item.GetProperty("sizeBytes").GetInt64();
            if (id is not ("DecisionInference.Arrow" or "DecisionInference.Abstractions") || version != versionPin || Path.GetFileName(file) != file)
                throw new InvalidDataException("Unexpected portable package artifact.");
            string path = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(profile.PackageReceiptPath))!, file);
            if (new FileInfo(path).Length != size || await JuliaAssetIdentity.HashFileAsync(path, token) != hash)
                throw new InvalidDataException("Portable package receipt bytes changed.");
            packages.Add(new(id, version, hash, size));
        }
        return packages.AsReadOnly();
    }
}
