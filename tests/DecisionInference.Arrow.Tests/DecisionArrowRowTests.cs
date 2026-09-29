using System.Text.Json;
using DecisionInference.Arrow;

namespace DecisionInference.Tests;

public sealed class DecisionArrowRowTests
{
    [Theory]
    [InlineData(long.MinValue)]
    [InlineData(-1L)]
    [InlineData(0L)]
    [InlineData(1L)]
    [InlineData(long.MaxValue)]
    public void Constructor_RetainsRowIdAndExactPayloadReferences(long id)
    {
        var input = new DecisionInput(JsonSerializer.SerializeToElement(new { source = id }),
            [new BinaryDecisionQuestion("bin", "Is the condition true?")]);
        var result = new DecisionResult(input, "Model", [new BinaryDecisionAnswer("bin", .25)]);
        var row = new DecisionArrowRow(id, input, result);

        Assert.Equal(id, row.RowId);
        Assert.Same(input, row.Input);
        Assert.Same(result, row.Result);
        Assert.Equal(.25, Assert.IsType<BinaryDecisionAnswer>(row.Result.Answers[0]).TrueProbability);
    }

    [Theory]
    [InlineData(true, "input")]
    [InlineData(false, "result")]
    public void Constructor_RejectsNullInputOrResult(bool nullInput, string parameter)
    {
        var input = new DecisionInput(JsonSerializer.SerializeToElement(0),
            [new BinaryDecisionQuestion("bin", "Is the condition true?")]);
        var result = new DecisionResult(input, "Model", [new BinaryDecisionAnswer("bin", .25)]);

        var error = Assert.Throws<ArgumentNullException>(() =>
            new DecisionArrowRow(0, nullInput ? null! : input, nullInput ? result : null!));
        Assert.Equal(parameter, error.ParamName);
    }
}
