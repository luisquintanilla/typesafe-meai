using System.Text;
using System.Text.Json;
using DecisionInference.Arrow;
using static DecisionInference.Tests.DecisionArrowTestSupport;

namespace DecisionInference.Tests;

public sealed class DecisionArrowSchemaTests
{
    [Fact]
    public void CanonicalJsonAndFingerprint_MatchIndependentGolden()
    {
        var contract = Contract();
        Assert.Equal(GoldenJson, contract.CanonicalJson);
        Assert.Equal(746, Encoding.UTF8.GetByteCount(GoldenJson));
        Assert.Equal(GoldenHash, contract.FeatureFingerprint);
        Assert.Equal(GoldenHash, Hash(Encoding.UTF8.GetBytes(GoldenJson)));
        Assert.Matches("^[0-9a-f]{64}$", contract.FeatureFingerprint);
        Assert.Equal(1, DecisionArrowSchema.Version);
        Assert.Equal("decision_arrow.", DecisionArrowSchema.MetadataPrefix);
        using var document = JsonDocument.Parse(contract.CanonicalJson);
        Assert.Equal(["schemaVersion", "numericRepresentation", "precision", "questions", "provider", "modelId",
            "assetHashes", "preprocessing", "decoder", "featureProjection"],
            document.RootElement.EnumerateObject().Select(p => p.Name).ToArray());
    }

    [Theory]
    [InlineData(DecisionPrecision.HighPrecision)]
    [InlineData(DecisionPrecision.FourDecimalPlaces)]
    [InlineData(DecisionPrecision.TwoDecimalPlaces)]
    public void Schema_HasExactPhysicalStructureAndMetadataAndParsesCanonicalContract(DecisionPrecision precision)
    {
        var contract = Contract(precision);
        var first = contract.Schema;
        var second = contract.Schema;
        Assert.NotSame(first, second);
        AssertGoldenSchema(first, precision);
        AssertGoldenSchema(second, precision);
        string canonical = GoldenJson.Replace("\"HighPrecision\"", $"\"{precision}\"", StringComparison.Ordinal);
        var parsed = DecisionArrowSchema.FromCanonicalJson(canonical);
        Assert.Equal(canonical, parsed.CanonicalJson);
        Assert.Equal(contract.FeatureFingerprint, parsed.FeatureFingerprint);
        Assert.Equal(["bin", "pick", "score"], parsed.Questions.Select(q => q.Id).ToArray());
        Assert.Equal(["z", "A", "a"], Assert.IsType<ChoiceDecisionQuestion>(parsed.Questions[1]).Candidates.Select(c => c.Id).ToArray());
        Assert.Equal(["low", "middle", "high"], Assert.IsType<ScoreDecisionQuestion>(parsed.Questions[2]).Rubric);
        Assert.Equal(["bin", "pick[2]", "score[0]"], parsed.Identity.FeatureProjection);
        Assert.Equal(" provider ", parsed.Identity.Provider);
        Assert.Equal("Model", parsed.Identity.ModelId);
        AssertGoldenSchema(parsed.Schema, precision);
    }

    [Fact]
    public void CanonicalJson_PreservesEscapingDescriptionsAndCollectionSnapshots()
    {
        const string instruction = " Quote \" and slash \\ newline\n café ";
        var candidates = new List<DecisionCandidate> { new("A", " Upper "), new("a", instruction) };
        var rubric = new List<string> { " Low ", instruction };
        var questions = new List<DecisionQuestion>
        {
            new BinaryDecisionQuestion("Q", instruction, " YES ", "no"),
            new ChoiceDecisionQuestion("q", " Choose ", candidates),
            new ScoreDecisionQuestion("score", " Rate ", rubric)
        };
        var identity = Identity(["Q", "q[1]", "score[0]"]);
        var contract = new DecisionArrowSchema(questions, DecisionPrecision.HighPrecision, identity);
        candidates.Clear();
        rubric[0] = "changed";
        questions.Reverse();
        Assert.Same(identity, contract.Identity);
        Assert.Equal(DecisionPrecision.HighPrecision, contract.Precision);
        Assert.Equal(["Q", "q", "score"], contract.Questions.Select(q => q.Id).ToArray());
        using var document = JsonDocument.Parse(contract.CanonicalJson);
        var qs = document.RootElement.GetProperty("questions");
        Assert.Equal(instruction, qs[0].GetProperty("instructions").GetString());
        Assert.Equal(" YES ", qs[0].GetProperty("trueDescription").GetString());
        Assert.Equal("no", qs[0].GetProperty("falseDescription").GetString());
        Assert.Equal(["id", "kind", "instructions", "trueDescription", "falseDescription"],
            qs[0].EnumerateObject().Select(p => p.Name).ToArray());
        Assert.Equal(["id", "description"], qs[1].GetProperty("candidates")[1].EnumerateObject().Select(p => p.Name).ToArray());
        Assert.Equal(instruction, qs[1].GetProperty("candidates")[1].GetProperty("description").GetString());
        Assert.Equal(" Low ", qs[2].GetProperty("rubric")[0].GetString());
        var parsed = DecisionArrowSchema.ParseQuestions(qs);
        Assert.Equal(instruction, parsed[0].Instructions);
        Assert.Equal(" YES ", Assert.IsType<BinaryDecisionQuestion>(parsed[0]).TrueDescription);
        Assert.Equal(["A", "a"], Assert.IsType<ChoiceDecisionQuestion>(parsed[1]).Candidates.Select(c => c.Id).ToArray());
        Assert.Equal([" Low ", instruction], Assert.IsType<ScoreDecisionQuestion>(parsed[2]).Rubric);
    }

    public static TheoryData<string> SemanticVariants => new()
    {
        "question-id", "kind", "question-order", "instructions", "true-description", "false-description",
        "candidate-id", "candidate-order", "candidate-description", "rubric-order", "rubric-content",
        "provider", "model", "asset-key", "asset-hash", "preprocessing", "decoder",
        "four-precision", "two-precision", "projection-order", "projection-subset", "projection-empty"
    };

    internal static DecisionArrowSchema Variant(string variant)
    {
        var questions = Questions();
        var old = Identity();
        var projection = old.FeatureProjection.ToArray();
        var assets = old.AssetHashes.ToArray();
        string provider = old.Provider, model = old.ModelId, preprocessing = old.Preprocessing, decoder = old.Decoder;
        var precision = DecisionPrecision.HighPrecision;
        switch (variant)
        {
            case "question-id": questions[0] = new BinaryDecisionQuestion("BIN", " Decide "); projection[0] = "BIN"; break;
            case "kind": questions[0] = new ScoreDecisionQuestion("bin", " Decide ", ["no", "yes"]); projection[0] = "bin[0]"; break;
            case "question-order": (questions[0], questions[1]) = (questions[1], questions[0]); break;
            case "instructions": questions[0] = new BinaryDecisionQuestion("bin", " decide "); break;
            case "true-description": questions[0] = new BinaryDecisionQuestion("bin", " Decide ", " yes "); break;
            case "false-description": questions[0] = new BinaryDecisionQuestion("bin", " Decide ", null, "NO"); break;
            case "candidate-id": questions[1] = new ChoiceDecisionQuestion("pick", " Pick ", [new("Z", "Zed"), new("A", "Upper"), new("a", "Lower")]); break;
            case "candidate-order": questions[1] = new ChoiceDecisionQuestion("pick", " Pick ", [new("a", "Lower"), new("A", "Upper"), new("z", "Zed")]); break;
            case "candidate-description": questions[1] = new ChoiceDecisionQuestion("pick", " Pick ", [new("z", "zed"), new("A", "Upper"), new("a", "Lower")]); break;
            case "rubric-order": questions[2] = new ScoreDecisionQuestion("score", " Rate ", ["high", "middle", "low"]); break;
            case "rubric-content": questions[2] = new ScoreDecisionQuestion("score", " Rate ", ["Low", "middle", "high"]); break;
            case "provider": provider = " Provider "; break;
            case "model": model = "Model "; break;
            case "asset-key": assets[0] = new("a", assets[0].Value); break;
            case "asset-hash": assets[0] = new(assets[0].Key, new string('f', 64)); break;
            case "preprocessing": preprocessing = "prep"; break;
            case "decoder": decoder = " Decoder "; break;
            case "four-precision": precision = DecisionPrecision.FourDecimalPlaces; break;
            case "two-precision": precision = DecisionPrecision.TwoDecimalPlaces; break;
            case "projection-order": projection = projection.Reverse().ToArray(); break;
            case "projection-subset": projection = ["pick[1]"]; break;
            case "projection-empty": projection = []; break;
            default: throw new ArgumentOutOfRangeException(nameof(variant));
        }
        return new(questions, precision, new(provider, model, assets, preprocessing, decoder, projection));
    }

    [Theory]
    [MemberData(nameof(SemanticVariants))]
    public void FeatureFingerprint_ChangesForSemanticVariants(string variant)
    {
        var changed = Variant(variant);
        Assert.NotEqual(GoldenJson, changed.CanonicalJson);
        Assert.NotEqual(GoldenHash, changed.FeatureFingerprint);
        Assert.Equal(Hash(Encoding.UTF8.GetBytes(changed.CanonicalJson)), changed.FeatureFingerprint);
        Assert.Equal(changed.CanonicalJson, DecisionArrowSchema.FromCanonicalJson(changed.CanonicalJson).CanonicalJson);
    }

    [Theory]
    [InlineData("null-questions")]
    [InlineData("null-identity")]
    [InlineData("precision")]
    [InlineData("invalid-question-set")]
    [InlineData("projection")]
    public void Constructor_RejectsInvalidContractsAndOrdinalIdentities(string fault)
    {
        if (fault == "invalid-question-set")
        {
            foreach (DecisionQuestion[] questions in new DecisionQuestion[][]
            {
                [], [null!], [new BinaryDecisionQuestion("row_id", "reserved")],
                [new BinaryDecisionQuestion("bin", "one"), new BinaryDecisionQuestion("bin", "two")]
            })
            {
                var error = Assert.Throws<ArgumentException>(() => new DecisionArrowSchema(questions, DecisionPrecision.HighPrecision, Identity([])));
                Assert.Equal("Questions must be nonempty, unique and must not use reserved row_id.", error.Message);
            }
            var valid = new DecisionArrowSchema([new BinaryDecisionQuestion("ROW_ID", "Upper"), new BinaryDecisionQuestion("q", "lower"),
                new BinaryDecisionQuestion("Q", "upper")], DecisionPrecision.HighPrecision, Identity(["ROW_ID", "q", "Q"]));
            Assert.Equal(["row_id", "ROW_ID", "q", "Q"], valid.Schema.FieldsList.Select(f => f.Name).ToArray());
            Assert.Throws<ArgumentException>(() => new ChoiceDecisionQuestion("pick", "Pick", [new("A", "one"), new("A", "two")]));
            return;
        }
        if (fault == "projection")
        {
            foreach (string coordinate in new[] { "BIN", "pick[3]", "pick[-1]", "score[3]", "score" })
            {
                var error = Assert.Throws<ArgumentException>(() => new DecisionArrowSchema(Questions(), DecisionPrecision.HighPrecision, Identity([coordinate])));
                Assert.Equal("Feature projection refers to an unknown observation.", error.Message);
            }
            Assert.Empty(new DecisionArrowSchema(Questions(), DecisionPrecision.HighPrecision, Identity([])).Identity.FeatureProjection);
            return;
        }
        var exception = Assert.ThrowsAny<ArgumentException>(() => new DecisionArrowSchema(
            fault == "null-questions" ? null! : Questions(), fault == "precision" ? (DecisionPrecision)99 : DecisionPrecision.HighPrecision,
            fault == "null-identity" ? null! : Identity()));
        Assert.Equal(fault == "precision" ? "precision" : fault == "null-questions" ? "questions" : "identity", exception.ParamName);
        Assert.Equal(fault == "precision", exception is ArgumentOutOfRangeException);
    }

    [Theory]
    [InlineData("noncanonical")]
    [InlineData("unsupported")]
    [InlineData("malformed")]
    [InlineData("bad-questions")]
    public void Parsers_RejectInvalidCanonicalAndQuestionJson(string fault)
    {
        if (fault == "noncanonical")
        {
            foreach (string json in new[]
            {
                " " + GoldenJson, GoldenJson.Replace("\"trueDescription\":null,", "", StringComparison.Ordinal),
                GoldenJson[..^1] + ",\"unknown\":1}", GoldenJson.Replace("\"Model\"", "\"\\u004dodel\"", StringComparison.Ordinal),
                GoldenJson.Replace("\"schemaVersion\":1,\"numericRepresentation\":\"float64\"", "\"numericRepresentation\":\"float64\",\"schemaVersion\":1", StringComparison.Ordinal)
            })
                Assert.Equal("Contract JSON is not canonical.", Assert.Throws<InvalidDataException>(() => DecisionArrowSchema.FromCanonicalJson(json)).Message);
        }
        else if (fault == "unsupported")
        {
            foreach (string json in new[] { GoldenJson.Replace("\"schemaVersion\":1", "\"schemaVersion\":2", StringComparison.Ordinal),
                GoldenJson.Replace("\"float64\"", "\"float32\"", StringComparison.Ordinal) })
                Assert.Contains("Unsupported decision Arrow schema", Assert.Throws<InvalidDataException>(() => DecisionArrowSchema.FromCanonicalJson(json)).Message);
            Assert.Throws<ArgumentException>(() => DecisionArrowSchema.FromCanonicalJson(GoldenJson.Replace("HighPrecision", "highPrecision", StringComparison.Ordinal)));
        }
        else if (fault == "malformed")
        {
            var error = Assert.ThrowsAny<JsonException>(() => DecisionArrowSchema.FromCanonicalJson("{"));
            Assert.Equal(1, error.BytePositionInLine);
            Assert.Throws<KeyNotFoundException>(() => DecisionArrowSchema.FromCanonicalJson("{}"));
            Assert.Throws<InvalidOperationException>(() => DecisionArrowSchema.FromCanonicalJson("[]"));
        }
        else
        {
            using var unknown = JsonDocument.Parse("""[{"id":"q","kind":"future","instructions":"Decide"}]""");
            Assert.Equal("Unsupported question kind.", Assert.Throws<InvalidDataException>(() => DecisionArrowSchema.ParseQuestions(unknown.RootElement)).Message);
            using var missing = JsonDocument.Parse("""[{"id":"q","kind":"choice","instructions":"Pick"}]""");
            Assert.Throws<KeyNotFoundException>(() => DecisionArrowSchema.ParseQuestions(missing.RootElement));
            using var wrong = JsonDocument.Parse("""[{"id":3,"kind":"binary","instructions":"Decide"}]""");
            Assert.Throws<InvalidOperationException>(() => DecisionArrowSchema.ParseQuestions(wrong.RootElement));
        }
    }
}
