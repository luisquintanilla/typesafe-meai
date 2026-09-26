using System.Text.Json;
using Microsoft.Extensions.AI;
using TypeSafe.MEAI.Tests.Fixtures;
using Xunit;

namespace TypeSafe.MEAI.Tests.Protocol;

[Trait("Stage", "A")]
public sealed class NativeQuestionContractTests
{
    [Fact]
    public void Questions_SerializeVariantsEntriesAndOmission()
    {
        var questions = NativeFixture.Questions();
        var noul = questions[0].ToJson();
        Assert.Equal("noul", noul.GetProperty("type").GetString());
        Assert.Equal("Follow up?", noul.GetProperty("instructions").GetString());
        Assert.Equal("needs assistance", noul.GetProperty("criteria").GetProperty("true").GetProperty("when").GetString());
        Assert.Equal(JsonValueKind.Null, noul.GetProperty("criteria").GetProperty("false").ValueKind);
        var choice = questions[1].ToJson();
        Assert.Equal("choice", choice.GetProperty("type").GetString());
        Assert.Equal("Billing", choice.GetProperty("criteria").GetProperty("billing").GetString());
        Assert.Equal("Technical", choice.GetProperty("criteria").GetProperty("technical").GetProperty("description").GetString());
        Assert.Equal("Other", choice.GetProperty("criteria").GetProperty("general")[1].GetString());
        var score = questions[2].ToJson();
        Assert.Equal("score", score.GetProperty("type").GetString());
        Assert.Equal("poor", score.GetProperty("criteria")[0].GetString());
        Assert.Equal("adequate", score.GetProperty("criteria")[1].GetProperty("level").GetString());
        Assert.Equal(JsonValueKind.Null, score.GetProperty("criteria")[2].ValueKind);
        var omitted = new NoulQuestion("omitted").ToJson();
        Assert.False(omitted.TryGetProperty("instructions", out _));
        Assert.False(omitted.TryGetProperty("criteria", out _));
        var explicitNull = new NoulQuestion("null", NativeFixture.Json("null"), NativeFixture.Json("null")).ToJson();
        Assert.Equal(JsonValueKind.Null, explicitNull.GetProperty("instructions").ValueKind);
        Assert.Equal(JsonValueKind.Null, explicitNull.GetProperty("criteria").ValueKind);
    }

    [Theory]
    [InlineData(0, false)]
    [InlineData(1, true)]
    [InlineData(2, true)]
    [InlineData(255, true)]
    [InlineData(256, false)]
    public void Choice_EnforcesOptionBounds(int count, bool valid)
    {
        var criteria = Enumerable.Range(0, count).Select(i => KeyValuePair.Create($"label-{i}", TypeSafeJson.Entry($"description-{i}")));
        if (!valid)
        {
            var error = Assert.Throws<ArgumentException>(() => new ChoiceQuestion("route", default, criteria));
            Assert.Contains("1 through 255", error.Message);
            return;
        }
        var json = new ChoiceQuestion("route", default, criteria).ToJson();
        Assert.Equal(count, json.GetProperty("criteria").EnumerateObject().Count());
        Assert.Equal($"description-{count - 1}", json.GetProperty("criteria").GetProperty($"label-{count - 1}").GetString());
    }

    [Theory]
    [InlineData(0, false)]
    [InlineData(1, false)]
    [InlineData(2, true)]
    [InlineData(10, true)]
    [InlineData(11, false)]
    public void Score_EnforcesLevelBoundsAndOrder(int count, bool valid)
    {
        var criteria = Enumerable.Range(0, count).Select(i => TypeSafeJson.Entry($"level-{i}"));
        if (!valid)
        {
            var error = Assert.Throws<ArgumentException>(() => new ScoreQuestion("quality", default, criteria));
            Assert.Contains("2 through 10", error.Message);
            return;
        }
        var json = new ScoreQuestion("quality", default, criteria).ToJson();
        Assert.Equal(count, json.GetProperty("criteria").GetArrayLength());
        Assert.Equal(Enumerable.Range(0, count).Select(i => $"level-{i}"), json.GetProperty("criteria").EnumerateArray().Select(v => v.GetString()));
    }

    [Fact]
    public void Questions_RejectDuplicateIdsAndLabelsButKeepCaseDistinct()
    {
        Assert.Throws<ArgumentException>(() => new ChatOptions().WithTypeSafeQuestions([]));
        Assert.Throws<ArgumentException>(() => new ChatOptions().WithTypeSafeQuestions([new NoulQuestion("id"), new NoulQuestion("id")]));
        Assert.Throws<ArgumentException>(() => new ChoiceQuestion("id", default,
            [KeyValuePair.Create("a", TypeSafeJson.Entry("one")), KeyValuePair.Create("a", TypeSafeJson.Entry("two"))]));
        var options = new ChatOptions().WithTypeSafeQuestions([new NoulQuestion("id"), new NoulQuestion("ID")]);
        var payload = Assert.IsType<JsonElement>(options.AdditionalProperties![NativeFixture.QuestionsKey]);
        Assert.Equal(["id", "ID"], payload.EnumerateObject().Select(p => p.Name));
    }

    [Theory]
    [InlineData("")]
    [InlineData(" ")]
    [InlineData(null)]
    public void Questions_RejectInvalidIds(string? id)
    {
        Assert.ThrowsAny<ArgumentException>(() => new NoulQuestion(id!));
        Assert.ThrowsAny<ArgumentException>(() => new ChoiceQuestion(id!, default, [KeyValuePair.Create("a", TypeSafeJson.Entry("a"))]));
        Assert.ThrowsAny<ArgumentException>(() => new ScoreQuestion(id!, default, [TypeSafeJson.Entry("a"), TypeSafeJson.Entry("b")]));
    }

    [Theory]
    [InlineData("true")]
    [InlineData("17")]
    public void Entries_RejectUnsupportedScalars(string json)
    {
        var entry = NativeFixture.Json(json);
        Assert.Throws<ArgumentException>(() => TypeSafeState.FromJson(entry));
        Assert.Throws<ArgumentException>(() => new NoulQuestion("q", entry));
        Assert.Throws<ArgumentException>(() => new ChoiceQuestion("q", default, [KeyValuePair.Create("a", entry)]));
        Assert.Throws<ArgumentException>(() => new ScoreQuestion("q", default, [entry, TypeSafeJson.Entry("b")]));
    }

    [Fact]
    public void StateAndQuestions_OwnDisposedJsonAndCollectionSnapshots()
    {
        using var document = JsonDocument.Parse("""{"state":{"ticket":17},"instructions":["judge"],"level":{"name":"first"}}""");
        var state = TypeSafeState.FromJson(document.RootElement.GetProperty("state"));
        var levels = new List<JsonElement> { document.RootElement.GetProperty("level"), NativeFixture.Json("null") };
        var question = new ScoreQuestion("quality", document.RootElement.GetProperty("instructions"), levels);
        var descriptions = new Dictionary<string, JsonElement> { ["a"] = document.RootElement.GetProperty("level") };
        var choice = new ChoiceQuestion("route", default, descriptions);
        document.Dispose();
        levels.Clear();
        descriptions["a"] = TypeSafeJson.Entry("changed");
        var data = Assert.IsType<DataContent>(Assert.Single(state.ToMessage().Contents));
        using var stateDocument = JsonDocument.Parse(data.Data);
        Assert.Equal(17, stateDocument.RootElement.GetProperty("ticket").GetInt32());
        Assert.Equal("judge", question.ToJson().GetProperty("instructions")[0].GetString());
        Assert.Equal("first", question.ToJson().GetProperty("criteria")[0].GetProperty("name").GetString());
        Assert.Equal(JsonValueKind.Null, question.ToJson().GetProperty("criteria")[1].ValueKind);
        Assert.Equal("first", choice.ToJson().GetProperty("criteria").GetProperty("a").GetProperty("name").GetString());
    }
}
