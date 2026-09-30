using Microsoft.Extensions.AI;
using static DecisionInference.Tests.ContractTests;

namespace DecisionInference.Tests;

public sealed class TwoDecimalPrecisionTests
{
    private const DecisionPrecision Two = DecisionPrecision.TwoDecimalPlaces;

    [Fact]
    public void EnumValues_RetainExistingWireNumbers()
    {
        Assert.Equal(0, (int)DecisionPrecision.HighPrecision);
        Assert.Equal(1, (int)DecisionPrecision.FourDecimalPlaces);
        Assert.Equal(2, (int)Two);
        Assert.Equal(3, Enum.GetValues<DecisionPrecision>().Length);
    }

    [Fact]
    public void CanonicalGrid_AllKindsAcceptEveryHundredth()
    {
        for (int i = 0; i <= 100; i++)
        {
            // Convert decimal rationals independently: 1 - a double need not be canonical.
            double p = (double)(i / 100m), complement = (double)((100 - i) / 100m);
            var binary = new BinaryDecisionAnswer("b", p, Two);
            var choice = new ChoiceDecisionAnswer("c", "right", [new("left", complement), new("right", p)], Two);
            var score = new ScoreDecisionAnswer("s", [complement, p], p, Two);
            Assert.Equal(p, binary.TrueProbability);
            Assert.Equal("right", choice.SelectedCandidateId);
            Assert.Equal(complement, choice.Probabilities["left"]);
            Assert.Equal(p, choice.Probabilities["right"]);
            Assert.Equal(new[] { complement, p }, score.Probabilities);
            Assert.Equal(p, score.ExpectedIndex);
            Assert.Equal(p, score.NativeScore);
            Assert.Equal(Math.Max(0, (double)(i / 100m - .005m)), score.ExpectedIndexLowerBound);
            Assert.Equal(Math.Min(1, (double)(i / 100m + .005m)), score.ExpectedIndexUpperBound);
            Assert.All(new DecisionAnswer[] { binary, choice, score }, a => Assert.Equal(Two, a.Precision));
        }
    }

    [Fact]
    public void AdjacentDoubles_AllKindsRejectOffGrid()
    {
        for (int i = 0; i <= 100; i++)
        {
            double p = (double)(i / 100m), complement = (double)((100 - i) / 100m);
            foreach (double offGrid in new[] { Math.BitIncrement(p), Math.BitDecrement(p) })
            {
                Assert.Throws<DecisionProtocolException>(() => new BinaryDecisionAnswer("b", offGrid, Two));
                Assert.Throws<DecisionProtocolException>(() => new ChoiceDecisionAnswer("c", "a",
                    [new("a", offGrid), new("b", complement)], Two));
                Assert.Throws<DecisionProtocolException>(() => new ScoreDecisionAnswer("s", [offGrid, complement], precision: Two));
                Assert.Throws<DecisionProtocolException>(() => new ScoreDecisionAnswer("s", [complement, p], offGrid, Two));
            }
        }
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(-.01)]
    [InlineData(1.01)]
    [InlineData(.001)]
    [InlineData(.005)]
    [InlineData(.3333)]
    public void InvalidProbabilities_AllKindsReject(double value)
    {
        Assert.Throws<DecisionProtocolException>(() => new BinaryDecisionAnswer("b", value, Two));
        Assert.Throws<DecisionProtocolException>(() => new ChoiceDecisionAnswer("c", "a",
            [new("a", value), new("b", .5)], Two));
        Assert.Throws<DecisionProtocolException>(() => new ScoreDecisionAnswer("s", [value, .5], precision: Two));
    }

    public static IEnumerable<object[]> FeasibleDistributions()
    {
        yield return [new[] { .33, .33, .33 }, .99, .995, 1.005];
        yield return [new[] { .34, .33, .34 }, 1.01, .995, 1.005];
        yield return [new[] { .5, .49 }, .49, .495, .495]; // upper ties sum to one
        yield return [new[] { .5, .51 }, .51, .505, .505]; // lower ties sum to one
        yield return [new[] { 1d, 0d }, 0d, 0d, .005];
        yield return [new[] { 0d, 1d }, 1d, .995, 1d];
        yield return [new[] { .2, .3, .5 }, 1.3, 1.29, 1.31];
        yield return [new[] { 0d, 0d, 1d }, 2d, 1.99, 2d];
    }

    [Theory]
    [MemberData(nameof(FeasibleDistributions))]
    public void RoundedDistributions_PreserveObservationsAndExactLatentBounds(
        double[] probabilities, double expected, double lower, double upper)
    {
        var original = probabilities.ToArray();
        var choice = new ChoiceDecisionAnswer("c", "0",
            probabilities.Select((p, i) => KeyValuePair.Create(i.ToString(), p)), Two);
        var score = new ScoreDecisionAnswer("s", probabilities, precision: Two);
        probabilities[0] = .77;
        Assert.Equal(original, choice.Probabilities.Values);
        Assert.Equal(original, score.Probabilities);
        Assert.Equal(expected, score.ExpectedIndex, 14);
        Assert.Equal(lower, score.ExpectedIndexLowerBound);
        Assert.Equal(upper, score.ExpectedIndexUpperBound);
        Assert.Null(score.NativeScore);
        Assert.Equal("0", choice.SelectedCandidateId);
        Assert.Throws<NotSupportedException>(() => ((IDictionary<string, double>)choice.Probabilities)["0"] = .77);
        Assert.Throws<NotSupportedException>(() => ((IList<double>)score.Probabilities)[0] = .77);
    }

    public static IEnumerable<object[]> InfeasibleDistributions()
    {
        yield return [new[] { .32, .33, .33 }];
        yield return [new[] { .34, .34, .34 }];
        yield return [new[] { 0d, .98 }]; // clipped lower endpoint cannot supply enough mass
        yield return [new[] { .02, 1d }]; // clipped upper endpoint cannot shed enough mass
        yield return [new[] { 0d, 0d }];
        yield return [new[] { 1d, 1d }];
        yield return [new[] { .99 }]; // sum .99 isn't universally feasible
        yield return [Array.Empty<double>()];
    }

    [Theory]
    [MemberData(nameof(InfeasibleDistributions))]
    public void InfeasibleMass_IsRejectedForChoiceAndScore(double[] probabilities)
    {
        Assert.Throws<DecisionProtocolException>(() => new ChoiceDecisionAnswer("c", "0",
            probabilities.Select((p, i) => KeyValuePair.Create(i.ToString(), p)), Two));
        Assert.Throws<DecisionProtocolException>(() => new ScoreDecisionAnswer("s", probabilities, precision: Two));
    }

    [Fact]
    public void ClippedIntervals_CanPutRawExpectationAboveRubric()
    {
        var answer = new ScoreDecisionAnswer("s", [0, .01, 1], 2, Two);
        Assert.Equal(new[] { 0d, .01, 1d }, answer.Probabilities);
        Assert.Equal(2.01, answer.ExpectedIndex);
        Assert.Equal(1.995, answer.ExpectedIndexLowerBound);
        Assert.Equal(1.995, answer.ExpectedIndexUpperBound);
        Assert.Equal(2, answer.NativeScore);
    }

    [Theory]
    [InlineData(.98, false)]
    [InlineData(.99, true)]
    [InlineData(1, true)]
    [InlineData(1.01, true)]
    [InlineData(1.02, false)]
    public void NativeScore_RequiresClosedIntervalOverlap(double native, bool accepted)
    {
        // Latent expectation [.995,1.005]; .99 and 1.01 touch only at a tie.
        if (!accepted)
        {
            Assert.Throws<DecisionProtocolException>(() => new ScoreDecisionAnswer("s", [.33, .33, .33], native, Two));
            return;
        }
        var answer = new ScoreDecisionAnswer("s", [.33, .33, .33], native, Two);
        Assert.Equal(native, answer.NativeScore);
        Assert.Equal(.99, answer.ExpectedIndex);
        Assert.Equal(.995, answer.ExpectedIndexLowerBound);
        Assert.Equal(1.005, answer.ExpectedIndexUpperBound);
        Assert.Equal(new[] { .33, .33, .33 }, answer.Probabilities);
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(double.NegativeInfinity)]
    [InlineData(-.01)]
    [InlineData(2.01)]
    [InlineData(1.001)]
    [InlineData(1.005)]
    public void NativeScore_RejectsNoncanonicalAndOutOfDomainValues(double native) =>
        Assert.Throws<DecisionProtocolException>(() => new ScoreDecisionAnswer("s", [.33, .33, .33], native, Two));

    [Theory]
    [InlineData(DecisionPrecision.TwoDecimalPlaces, .995, 1.005, .99, 1.01)]
    [InlineData(DecisionPrecision.FourDecimalPlaces, .99995, 1.00005, .9999, 1.0001)]
    public void PrecisionDigits_UseDistinctHalfWidths(DecisionPrecision precision, double lower, double upper,
        double lowerNative, double upperNative)
    {
        double p = precision == Two ? .33 : .3333;
        foreach (double native in new[] { lowerNative, upperNative })
        {
            var answer = new ScoreDecisionAnswer("s", [p, p, p], native, precision);
            Assert.Equal(lower, answer.ExpectedIndexLowerBound);
            Assert.Equal(upper, answer.ExpectedIndexUpperBound);
            Assert.Equal(native, answer.NativeScore);
            Assert.Equal(precision, answer.Precision);
        }
        double half = precision == Two ? .005 : .00005;
        Assert.Equal(half, new ScoreDecisionAnswer("s", [1, 0], 0, precision).ExpectedIndexUpperBound);
        Assert.Equal(1 - half, new ScoreDecisionAnswer("s", [0, 1], 1, precision).ExpectedIndexLowerBound);
    }

    [Fact]
    public void Result_CopiesReasoningAndAnswerJsonUsage()
    {
        var usage = new UsageDetails
        {
            InputTokenCount = 11, OutputTokenCount = 12, ReasoningTokenCount = 5, TotalTokenCount = 23,
            AdditionalCounts = new() { [Jeeves.JeevesDecisionGenerator.AnswerJsonTokenCountKey] = 7 }
        };
        var result = new DecisionResult(Input(Binary()), "label", [new BinaryDecisionAnswer("b", .8, Two)], usage);
        usage.InputTokenCount = 99;
        usage.OutputTokenCount = 98;
        usage.TotalTokenCount = 97;
        usage.ReasoningTokenCount = 96;
        usage.AdditionalCounts.Clear();
        Assert.NotSame(usage, result.Usage);
        Assert.Equal(11, result.Usage!.InputTokenCount);
        Assert.Equal(12, result.Usage.OutputTokenCount);
        Assert.Equal(23, result.Usage.TotalTokenCount);
        Assert.Equal(5, result.Usage.ReasoningTokenCount);
        Assert.Equal(7, result.Usage.AdditionalCounts![Jeeves.JeevesDecisionGenerator.AnswerJsonTokenCountKey]);
        Assert.Equal(.8, result.GetAnswer<BinaryDecisionAnswer>("b").TrueProbability);
    }
}
