using DecisionInference.Arrow;
using static DecisionInference.Tests.DecisionArrowTestSupport;

namespace DecisionInference.Tests;

public sealed class DecisionArrowIdentityTests
{
    [Fact]
    public void Constructor_PreservesExactStringsAndSnapshotsOrdinalCollections()
    {
        var assets = new List<KeyValuePair<string, string>>
        {
            new("z", new string('1', 64)), new("a", new string('2', 64)), new("A", new string('0', 64))
        };
        var projection = new List<string> { "bin", "pick[2]", "score[0]" };
        var identity = new DecisionArrowIdentity(" provider ", "Model", assets, " prep ", " decoder ", projection);
        assets.Clear();
        projection[0] = "changed";
        Assert.Equal(" provider ", identity.Provider);
        Assert.Equal("Model", identity.ModelId);
        Assert.Equal(" prep ", identity.Preprocessing);
        Assert.Equal(" decoder ", identity.Decoder);
        Assert.Equal(["A", "a", "z"], identity.AssetHashes.Keys.ToArray());
        Assert.Equal(new string('2', 64), identity.AssetHashes["a"]);
        Assert.Equal(["bin", "pick[2]", "score[0]"], identity.FeatureProjection);
        var reverse = new DecisionArrowIdentity(identity.Provider, identity.ModelId, identity.AssetHashes.Reverse(),
            identity.Preprocessing, identity.Decoder, identity.FeatureProjection);
        Assert.Equal(new DecisionArrowSchema(Questions(), DecisionPrecision.HighPrecision, identity).CanonicalJson,
            new DecisionArrowSchema(Questions(), DecisionPrecision.HighPrecision, reverse).CanonicalJson);
        Assert.Equal(new DecisionArrowSchema(Questions(), DecisionPrecision.HighPrecision, identity).FeatureFingerprint,
            new DecisionArrowSchema(Questions(), DecisionPrecision.HighPrecision, reverse).FeatureFingerprint);
    }

    [Theory]
    [InlineData("provider")]
    [InlineData("modelId")]
    [InlineData("preprocessing")]
    [InlineData("decoder")]
    public void Constructor_RejectsBlankAndNullSemanticStrings(string parameter)
    {
        foreach (string? bad in new string?[] { null, "", " \t" })
        {
            var error = Assert.ThrowsAny<ArgumentException>(() => new DecisionArrowIdentity(
                parameter == "provider" ? bad! : "p", parameter == "modelId" ? bad! : "m",
                [KeyValuePair.Create("asset", new string('0', 64))],
                parameter == "preprocessing" ? bad! : "prep", parameter == "decoder" ? bad! : "decode", []));
            Assert.Equal(parameter, error.ParamName);
            Assert.Equal(bad is null, error is ArgumentNullException);
        }
    }

    [Theory]
    [InlineData("null-assets")]
    [InlineData("null-projection")]
    [InlineData("empty-assets")]
    [InlineData("bad-hashes")]
    [InlineData("duplicate-assets")]
    [InlineData("blank-asset-key")]
    [InlineData("bad-projection")]
    public void Constructor_RejectsInvalidAssetsAndProjection(string fault)
    {
        if (fault == "bad-hashes")
        {
            foreach (string? hash in new string?[] { null, "", new('0', 63), new('0', 65), new('g', 64), new('A', 64) })
            {
                var error = Assert.Throws<ArgumentException>(() =>
                    new DecisionArrowIdentity("p", "m", [KeyValuePair.Create("asset", hash!)], "prep", "decode", []));
                Assert.Equal("Asset hashes must be lowercase SHA-256.", error.Message);
            }
            return;
        }
        var valid = new[] { KeyValuePair.Create("asset", new string('0', 64)) };
        IEnumerable<KeyValuePair<string, string>> assets = fault switch
        {
            "null-assets" => null!,
            "empty-assets" => [],
            "duplicate-assets" => [valid[0], valid[0]],
            "blank-asset-key" => [KeyValuePair.Create(" ", new string('0', 64))],
            _ => valid
        };
        IEnumerable<string> projection = fault switch
        {
            "null-projection" => null!,
            "bad-projection" => ["bin", "bin"],
            _ => []
        };
        var exception = Assert.ThrowsAny<ArgumentException>(() =>
            new DecisionArrowIdentity("p", "m", assets, "prep", "decode", projection));
        if (fault.StartsWith("null-", StringComparison.Ordinal))
        {
            Assert.IsType<ArgumentNullException>(exception);
            Assert.Equal(fault == "null-assets" ? "assetHashes" : "featureProjection", exception.ParamName);
        }
        else Assert.IsNotType<ArgumentNullException>(exception);
        if (fault == "empty-assets") Assert.Equal("Explicit asset identities are required.", exception.Message);
        if (fault == "blank-asset-key") Assert.Equal("item.Key", exception.ParamName);
        if (fault == "duplicate-assets") Assert.Contains("same key", exception.Message, StringComparison.OrdinalIgnoreCase);
        if (fault == "bad-projection")
        {
            Assert.Equal("Feature projection must contain unique, explicit coordinates.", exception.Message);
            foreach (string? coordinate in new string?[] { null, "", " " })
                Assert.Throws<ArgumentException>(() =>
                    new DecisionArrowIdentity("p", "m", valid, "prep", "decode", [coordinate!]));
        }
    }
}
