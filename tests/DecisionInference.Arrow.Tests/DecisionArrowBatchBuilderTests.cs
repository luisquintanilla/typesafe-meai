using System.Text.Json;
using DecisionInference.Arrow;
using static DecisionInference.Tests.DecisionArrowTestSupport;

namespace DecisionInference.Tests;

public sealed class DecisionArrowBatchBuilderTests
{
    [Theory]
    [InlineData("model", "Model identity changed within the dataset.")]
    [InlineData("questions", "Question definitions changed within the dataset.")]
    [InlineData("order", "Question definitions changed within the dataset.")]
    [InlineData("association-count", "Missing, extra, duplicate or null answers.")]
    [InlineData("association-kind", "Answer ID or kind mismatch.")]
    [InlineData("association-candidates", "Incomplete or unexpected candidate distribution.")]
    [InlineData("association-score-width", "Incomplete or unexpected rubric distribution.")]
    [InlineData("precision", "Precision changed within the dataset.")]
    public async Task WriteAsync_RevalidatesImmutableRowAssociationAndDatasetContract(string fault, string message)
    {
        using var temp = new TempDirectory();
        var contract = Contract();
        var valid = Row(91, contract);
        var input = valid.Input;
        DecisionResult result;
        if (fault is "questions" or "order")
        {
            var changed = DecisionArrowSchemaTests.Variant(fault == "questions" ? "instructions" : "question-order");
            valid = Row(91, changed);
            input = valid.Input;
            result = valid.Result;
        }
        else if (fault == "model") result = new(input, "model", valid.Result.Answers);
        else if (fault == "precision") result = Row(91, Contract(DecisionPrecision.TwoDecimalPlaces)).Result;
        else
        {
            var questions = Questions().ToList();
            var answers = valid.Result.Answers.ToList();
            switch (fault)
            {
                case "association-count": questions.RemoveAt(2); answers.RemoveAt(2); break;
                case "association-kind":
                    questions[0] = new ChoiceDecisionQuestion("bin", "Pick", [new("only", "Only")]);
                    answers[0] = new ChoiceDecisionAnswer("bin", "only", [KeyValuePair.Create("only", 1.0)]);
                    break;
                case "association-candidates":
                    questions[1] = new ChoiceDecisionQuestion("pick", "Pick", [new("other", "Other")]);
                    answers[1] = new ChoiceDecisionAnswer("pick", "other", [KeyValuePair.Create("other", 1.0)]);
                    break;
                case "association-score-width":
                    questions[2] = new ScoreDecisionQuestion("score", "Rate", ["low", "high"]);
                    answers[2] = new ScoreDecisionAnswer("score", [.5, .5]);
                    break;
            }
            var otherInput = new DecisionInput(JsonSerializer.SerializeToElement(19), questions);
            result = new(otherInput, "Model", answers);
        }
        var row = new DecisionArrowRow(91, input, result);
        Assert.Same(input, row.Input); // Row constructor intentionally defers association checks.
        Assert.Same(result, row.Result);
        var error = await Assert.ThrowsAsync<DecisionProtocolException>(() =>
            DecisionArrowWriter.WriteAsync(temp.Dataset, contract, Rows([row]), Provenance(), 2, Token));
        Assert.Equal(message, error.Message);
        DecisionArrowWriterTests.AssertFailureRecord(temp.Dataset, "failed", 0, error);
    }

    [Theory]
    [InlineData(2, -7L)]
    [InlineData(1, 0L)]
    public async Task WriteAsync_RejectsDuplicateIdsWithinAndAcrossBatches(int bound, long duplicate)
    {
        using var temp = new TempDirectory();
        var contract = Contract();
        var error = await Assert.ThrowsAsync<DecisionProtocolException>(() =>
            DecisionArrowWriter.WriteAsync(temp.Dataset, contract, Rows([Row(duplicate, contract), Row(duplicate, contract)]), Provenance(), bound, Token));
        Assert.Equal($"Duplicate source row ID {duplicate}.", error.Message);
        DecisionArrowWriterTests.AssertFailureRecord(temp.Dataset, "failed", 1, error);
    }

    [Fact]
    public async Task WriteAsync_RejectsNullRowsWithoutPlaceholderBatch()
    {
        using var temp = new TempDirectory();
        var error = await Assert.ThrowsAsync<ArgumentNullException>(() =>
            DecisionArrowWriter.WriteAsync(temp.Dataset, Contract(), Rows([null!]), Provenance(), cancellationToken: Token));
        Assert.Equal("row", error.ParamName);
        DecisionArrowWriterTests.AssertFailureRecord(temp.Dataset, "failed", 0, error);
    }

    [Fact]
    public async Task WriteAsync_ChecksVectorCapacityBeforeGiantAllocationOrEnumeration()
    {
        using var temp = new TempDirectory();
        var candidates = Enumerable.Range(0, 4096).Select(i => new DecisionCandidate($"c{i}", $"Option {i}")).ToArray();
        var contract = new DecisionArrowSchema([new ChoiceDecisionQuestion("wide", "Choose", candidates)],
            DecisionPrecision.HighPrecision, Identity(["wide[0]"]));
        bool enumerated = false;
        async IAsyncEnumerable<DecisionArrowRow> Produce()
        {
            await Task.CompletedTask;
            enumerated = true;
            yield break;
        }
        var error = await Assert.ThrowsAsync<OverflowException>(() =>
            DecisionArrowWriter.WriteAsync(temp.Dataset, contract, Produce(), Provenance(), 65536, Token));
        Assert.False(enumerated);
        DecisionArrowWriterTests.AssertFailureRecord(temp.Dataset, "failed", 0, error);
    }
}
