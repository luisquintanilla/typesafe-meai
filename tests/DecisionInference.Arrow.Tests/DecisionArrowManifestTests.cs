using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using DecisionInference.Arrow;
using static DecisionInference.Tests.DecisionArrowTestSupport;

namespace DecisionInference.Tests;

public sealed class DecisionArrowManifestTests
{
    internal static DecisionArrowManifest Envelope(string status = "complete") => new()
    {
        Status = status, ContractJson = GoldenJson, FeatureFingerprint = GoldenHash, DataFile = "decisions.arrow",
        DataSha256 = new string('5', 64), DataSize = 999, RowCount = 3, OutputBatchSize = 2,
        Provenance = Provenance()
    };

    [Fact]
    public void JsonOptions_ExposeExactWireEnvelopeAndAllProvenanceValues()
    {
        var manifest = Envelope();
        string json = JsonSerializer.Serialize(manifest, DecisionArrowManifest.JsonOptions);
        Assert.Contains("\n  \"schemaVersion\": 1,", json);
        Assert.Equal(JsonUnmappedMemberHandling.Disallow, DecisionArrowManifest.JsonOptions.UnmappedMemberHandling);
        using var document = JsonDocument.Parse(json);
        var root = document.RootElement;
        Assert.Equal(["schemaVersion", "status", "contractJson", "featureFingerprint", "dataFile", "dataSha256",
            "dataSize", "rowCount", "outputBatchSize", "provenance", "error"], root.EnumerateObject().Select(p => p.Name).ToArray());
        Assert.Equal(1, root.GetProperty("schemaVersion").GetInt32());
        Assert.Equal("complete", root.GetProperty("status").GetString());
        Assert.Equal(GoldenJson, root.GetProperty("contractJson").GetString());
        Assert.Equal(GoldenHash, root.GetProperty("featureFingerprint").GetString());
        Assert.Equal("decisions.arrow", root.GetProperty("dataFile").GetString());
        Assert.Equal(new string('5', 64), root.GetProperty("dataSha256").GetString());
        Assert.Equal(999, root.GetProperty("dataSize").GetInt64());
        Assert.Equal(3, root.GetProperty("rowCount").GetInt64());
        Assert.Equal(2, root.GetProperty("outputBatchSize").GetInt32());
        Assert.Equal(JsonValueKind.Null, root.GetProperty("error").ValueKind);
        var provenance = root.GetProperty("provenance");
        Assert.Equal(["sourceRepository", "sourceCommit", "inputSha256", "questionsSha256", "packages", "machine",
            "runtime", "executionMode", "nativeStateBatchSize", "cpuThreads", "modelInputShapes", "auxiliaryAssetHashes", "measurements"],
            provenance.EnumerateObject().Select(p => p.Name).ToArray());
        Assert.Equal("offline/repository", provenance.GetProperty("sourceRepository").GetString());
        Assert.Equal("commit-a", provenance.GetProperty("sourceCommit").GetString());
        Assert.Equal(new string('2', 64), provenance.GetProperty("inputSha256").GetString());
        Assert.Equal(new string('3', 64), provenance.GetProperty("questionsSha256").GetString());
        Assert.Equal("test-machine", provenance.GetProperty("machine").GetString());
        Assert.Equal(".NET 10", provenance.GetProperty("runtime").GetString());
        Assert.Equal("offline", provenance.GetProperty("executionMode").GetString());
        Assert.Equal(8, provenance.GetProperty("nativeStateBatchSize").GetInt32());
        Assert.Equal(2, provenance.GetProperty("cpuThreads").GetInt32());
        Assert.Equal([1, 17], provenance.GetProperty("modelInputShapes").GetProperty("input_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray());
        Assert.Equal(new string('6', 64), provenance.GetProperty("auxiliaryAssetHashes").GetProperty("vocabulary").GetString());
        Assert.Equal(42.5, provenance.GetProperty("measurements").GetProperty("callerObservation").GetDouble());
        var package = provenance.GetProperty("packages")[0];
        Assert.Equal(["id", "version", "sha256", "size"], package.EnumerateObject().Select(p => p.Name).ToArray());
        Assert.Equal("Apache.Arrow", package.GetProperty("id").GetString());
        Assert.Equal("23-test-a", package.GetProperty("version").GetString());
        Assert.Equal(new string('4', 64), package.GetProperty("sha256").GetString());
        Assert.Equal(12345, package.GetProperty("size").GetInt64());
        foreach (string status in new[] { "partial", "failed", "cancelled" })
        {
            using var partial = JsonDocument.Parse(JsonSerializer.Serialize(manifest with
            {
                Status = status, DataSha256 = null, DataSize = null, Error = "Sentinel: detail",
                Provenance = manifest.Provenance with
                {
                    NativeStateBatchSize = null, CpuThreads = null, Measurements = null,
                    ModelInputShapes = null, AuxiliaryAssetHashes = null
                }
            }, DecisionArrowManifest.JsonOptions));
            Assert.Equal(status, partial.RootElement.GetProperty("status").GetString());
            Assert.Equal(JsonValueKind.Null, partial.RootElement.GetProperty("dataSha256").ValueKind);
            Assert.Equal(JsonValueKind.Null, partial.RootElement.GetProperty("dataSize").ValueKind);
            Assert.Equal("Sentinel: detail", partial.RootElement.GetProperty("error").GetString());
            Assert.Equal(JsonValueKind.Null, partial.RootElement.GetProperty("provenance").GetProperty("cpuThreads").ValueKind);
            Assert.Equal(JsonValueKind.Null, partial.RootElement.GetProperty("provenance").GetProperty("auxiliaryAssetHashes").ValueKind);
        }
    }

    [Fact]
    public void WithCopies_KeepOriginalRecordsAndProvenanceIndependentOfFeatureIdentity()
    {
        var original = Envelope();
        var package = original.Provenance.Packages[0];
        Assert.Equal(new DecisionArrowPackageArtifact("Apache.Arrow", "23-test-a", new string('4', 64), 12345), package);
        var changed = original with
        {
            OutputBatchSize = 17, RowCount = 12,
            Provenance = original.Provenance with
            {
                SourceRepository = "other/repo", SourceCommit = "other-commit", InputSha256 = new string('a', 64),
                QuestionsSha256 = new string('b', 64), Machine = "other-host", Runtime = "other-runtime",
                ExecutionMode = "portable", NativeStateBatchSize = 1, CpuThreads = 4,
                Packages = [package with { Id = "Other", Version = "99", Sha256 = new string('c', 64), Size = 91 }],
                Measurements = new Dictionary<string, double> { ["other"] = 7 }
            }
        };
        Assert.Equal("23-test-a", package.Version);
        Assert.Equal("commit-a", original.Provenance.SourceCommit);
        Assert.Equal(2, original.OutputBatchSize);
        Assert.Equal(3, original.RowCount);
        Assert.NotEqual(original.Provenance, changed.Provenance);
        Assert.Equal(GoldenHash, DecisionArrowSchema.FromCanonicalJson(changed.ContractJson).FeatureFingerprint);
        Assert.Equal(GoldenJson, changed.ContractJson);
        Assert.Equal("99", changed.Provenance.Packages[0].Version);
        Assert.Equal(42.5, original.Provenance.Measurements!["callerObservation"]);
    }

    [Theory]
    [InlineData("unknown-root")]
    [InlineData("unknown-provenance")]
    [InlineData("required-manifest")]
    [InlineData("required-provenance")]
    public void JsonOptions_RejectUnknownAndMissingRequiredMembers(string fault)
    {
        var original = JsonNode.Parse(JsonSerializer.Serialize(Envelope(), DecisionArrowManifest.JsonOptions))!;
        string[] members = fault switch
        {
            "required-manifest" => ["status", "contractJson", "featureFingerprint", "dataFile", "rowCount", "outputBatchSize", "provenance"],
            "required-provenance" => ["sourceRepository", "sourceCommit", "inputSha256", "questionsSha256", "packages", "machine", "runtime", "executionMode"],
            _ => ["intruder"]
        };
        foreach (string member in members)
        {
            var root = original.DeepClone().AsObject();
            var target = fault.Contains("provenance", StringComparison.Ordinal) ? root["provenance"]!.AsObject() : root;
            if (fault.StartsWith("required", StringComparison.Ordinal)) Assert.True(target.Remove(member)); else target[member] = true;
            var error = Assert.Throws<JsonException>(() =>
                JsonSerializer.Deserialize<DecisionArrowManifest>(root.ToJsonString(), DecisionArrowManifest.JsonOptions));
            Assert.Contains(member, error.Message);
            Assert.Equal(GoldenHash, original["featureFingerprint"]!.GetValue<string>());
        }
    }
}
