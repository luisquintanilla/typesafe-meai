using System.Numerics.Tensors;
using DecisionInference.Julia;
using Microsoft.ML.OnnxRuntime;

namespace DecisionInference.Tests;

public sealed class JuliaNumericSinkTests
{
    private static CancellationToken Cancel => TestContext.Current.CancellationToken;

    [Fact]
    public void DecodeToSink_MatchesPublicDoubleBitsSelectionBoundsAndOwnedSource()
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        var batch = prepare.Prepare(JuliaTestFixture.Mixed(), Cancel);
        var logits = Tensor.Create<float>([0, 0, float.NaN, -2, 0, 3, 0, 1, float.PositiveInfinity], [3, 3]);
        var decoder = new DecodeDecisions();
        var typed = decoder.Decode(batch, logits);
        var sink = new SnapshotSink();
        decoder.DecodeToSink(batch, logits, sink);
        Assert.Equal(1, sink.Calls);
        Assert.Same(batch.Input, sink.Source);
        Assert.Equal([0, -1, -1], sink.Selected);
        Assert.Equal([0, 2, 5, 7], sink.Offsets);
        var choice = Assert.IsType<ChoiceDecisionAnswer>(typed.Answers[0]);
        Assert.Equal(((ChoiceDecisionQuestion)batch.Input.Questions[0]).Candidates[0].Id, choice.SelectedCandidateId);
        var orderedChoice = ((ChoiceDecisionQuestion)batch.Input.Questions[0]).Candidates.Select(c => choice.Probabilities[c.Id]);
        var score = Assert.IsType<ScoreDecisionAnswer>(typed.Answers[1]);
        Assert.Null(score.NativeScore);
        var binary = Assert.IsType<BinaryDecisionAnswer>(typed.Answers[2]);
        Assert.Equal(orderedChoice.Select(BitConverter.DoubleToInt64Bits),
            sink.Values.Take(2).Select(BitConverter.DoubleToInt64Bits));
        Assert.Equal(score.Probabilities.Select(BitConverter.DoubleToInt64Bits),
            sink.Values.Skip(2).Take(3).Select(BitConverter.DoubleToInt64Bits));
        Assert.Equal(BitConverter.DoubleToInt64Bits(binary.TrueProbability), BitConverter.DoubleToInt64Bits(sink.Values[6]));
        Assert.Equal(new[] { score.ExpectedIndex, score.ExpectedIndexLowerBound, score.ExpectedIndexUpperBound }.Select(BitConverter.DoubleToInt64Bits),
            sink.Scores.Skip(3).Take(3).Select(BitConverter.DoubleToInt64Bits));
        double[] saved = sink.Values.ToArray();
        logits[0, 0] = 99;
        decoder.DecodeToSink(batch, logits, new SnapshotSink());
        Assert.Equal(saved, sink.Values);
        Assert.All(typed.Answers, answer => Assert.Equal(DecisionPrecision.HighPrecision, answer.Precision));
        Assert.Equal([0d, 0d], Assert.IsType<JuliaRawAnswer>(typed.Answers[0].RawRepresentation).Logits);
    }

    [Fact]
    public void DecodeToSink_LateInvalidLaneNeverCommitsPartialFrameAndRecovers()
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        var batch = prepare.Prepare(JuliaTestFixture.Mixed(), Cancel);
        var sink = new SnapshotSink();
        var decoder = new DecodeDecisions();
        var invalid = Tensor.Create<float>([0, 1, 0, 0, 1, 2, 0, float.NaN, 0], [3, 3]);
        Assert.Equal("Nonfinite Julia valid-option logit.", Assert.Throws<DecisionProtocolException>(() =>
            decoder.DecodeToSink(batch, invalid, sink)).Message);
        Assert.Equal(0, sink.Calls);
        var valid = Tensor.Create<float>([0, 1, float.NaN, 0, 1, 2, 0, 1, float.NaN], [3, 3]);
        decoder.DecodeToSink(batch, valid, sink);
        Assert.Equal(1, sink.Calls);
        Assert.All(sink.Values, value => Assert.True(double.IsFinite(value) && value is >= 0 and <= 1));
    }

    [Fact]
    public void GenerateToSink_SinkThrowCancellationAndDisposalPreserveGateRecovery()
    {
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        using var options = new SessionOptions { IntraOpNumThreads = 1, InterOpNumThreads = 1 };
        using var session = new InferenceSession(SyntheticOnnx.CreateJulia(), options);
        using var score = new ScoreOnnxDecisionModel(session);
        using var generator = new JuliaDecisionGenerator(prepare, score, new DecodeDecisions());
        var input = JuliaTestFixture.Mixed();
        var sentinel = new InvalidOperationException("numeric sink failed");
        var throwing = new SnapshotSink { Error = sentinel };
        Assert.Same(sentinel, Assert.Throws<InvalidOperationException>(() => generator.GenerateToSink(input, throwing, Cancel)));
        using var cancelled = CancellationTokenSource.CreateLinkedTokenSource(Cancel);
        cancelled.Cancel();
        Assert.Equal(cancelled.Token, Assert.ThrowsAny<OperationCanceledException>(() =>
            generator.GenerateToSink(input, new SnapshotSink(), cancelled.Token)).CancellationToken);
        var recovered = new SnapshotSink();
        generator.GenerateToSink(input, recovered, Cancel);
        Assert.Equal(1, recovered.Calls);
        generator.Dispose();
        Assert.Throws<ObjectDisposedException>(() => generator.GenerateToSink(input, recovered, Cancel));
        Assert.Equal(1, recovered.Calls);
        Assert.Equal(new nint[] { 3, 3 }, score.Score(prepare.Prepare(input, Cancel), Cancel).Lengths.ToArray());
    }

    private sealed class SnapshotSink : IJuliaNumericSink
    {
        internal int Calls;
        internal DecisionInput? Source;
        internal int[] Selected = [];
        internal int[] Offsets = [];
        internal double[] Values = [];
        internal double[] Scores = [];
        internal Exception? Error;

        public void WriteFrame(DecisionInput source, ReadOnlySpan<int> selected, ReadOnlySpan<int> offsets,
            ReadOnlySpan<double> probabilities, ReadOnlySpan<double> scoreFields)
        {
            if (Error is { } error) throw error;
            Calls++;
            Source = source;
            Selected = selected.ToArray();
            Offsets = offsets.ToArray();
            Values = probabilities.ToArray();
            Scores = scoreFields.ToArray();
        }
    }
}
