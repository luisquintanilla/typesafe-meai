using System.Text.Json;
using System.Text.Json.Serialization;

namespace DecisionInference.Arrow;

public sealed record DecisionArrowPackageArtifact(string Id, string Version, string Sha256, long Size);

public sealed record DecisionArrowProvenance
{
    public required string SourceRepository { get; init; }
    public required string SourceCommit { get; init; }
    public required string InputSha256 { get; init; }
    public required string QuestionsSha256 { get; init; }
    public required IReadOnlyList<DecisionArrowPackageArtifact> Packages { get; init; }
    public required string Machine { get; init; }
    public required string Runtime { get; init; }
    public required string ExecutionMode { get; init; }
    public int? NativeStateBatchSize { get; init; }
    public int? CpuThreads { get; init; }
    public IReadOnlyDictionary<string, int[]>? ModelInputShapes { get; init; }
    public IReadOnlyDictionary<string, string>? AuxiliaryAssetHashes { get; init; }
    public IReadOnlyDictionary<string, double>? Measurements { get; init; }
}

public sealed record DecisionArrowManifest
{
    public int SchemaVersion { get; init; } = DecisionArrowSchema.Version;
    public required string Status { get; init; }
    public required string ContractJson { get; init; }
    public required string FeatureFingerprint { get; init; }
    public required string DataFile { get; init; }
    public string? DataSha256 { get; init; }
    public long? DataSize { get; init; }
    public required long RowCount { get; init; }
    public required int OutputBatchSize { get; init; }
    public required DecisionArrowProvenance Provenance { get; init; }
    public string? Error { get; init; }

    public static JsonSerializerOptions JsonOptions { get; } = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true,
        UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow
    };
}
