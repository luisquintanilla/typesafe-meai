using DecisionInference.Laya;
using Microsoft.ML.OnnxRuntime;
using System.Numerics.Tensors;
using System.Reflection;
using static DecisionInference.Tests.ContractTests;
using static DecisionInference.Tests.LayaStagesTests;

namespace DecisionInference.Tests;

public sealed class OnnxExtractionTests
{
    private static CancellationToken Cancel => TestContext.Current.CancellationToken;
    private static SessionOptions Options() => new() { IntraOpNumThreads = 1, InterOpNumThreads = 1 };

    [Fact]
    public void OutputPolicy_LayaAcceptsExtraGraphOutputsWhileJuliaRejectsThem()
    {
        using var options = Options();
        using var session = new InferenceSession(SyntheticOnnx.CreateLayaOutputs([0, 1], [.25f, .75f], 1, 2, extraOutput: true), options);
        Assert.Equal(["logits", "act_probs", "unused_output"], session.OutputMetadata.Keys);
        using var laya = new ScoreOnnxDecisionModel(session);
        var batch = Prepare().Prepare(Request(Choice()), Cancel);
        var result = laya.Score(batch, Cancel);
        Assert.Equal(new nint[] { 1, 2 }, result.Logits.Lengths.ToArray());
        Assert.Equal(1, result.Logits[0, 1]);
        Assert.Equal(.75f, result.ActionProbabilities[0, 1]);
        Assert.Contains("only the logits output", Assert.Throws<ArgumentException>(() =>
            new Julia.ScoreOnnxDecisionModel(session)).Message, StringComparison.Ordinal);
        Assert.Equal(.75f, laya.Score(batch, Cancel).ActionProbabilities[0, 1]);
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void OutputPolicy_LayaDefersNonfiniteValidLogitsUntilDecodeAndIgnoresPadding(float sentinel)
    {
        using var options = Options();
        using var session = new InferenceSession(SyntheticOnnx.CreateLayaOutputs(
            [0, 1, sentinel, 0, 1, 2, 0, 1, sentinel], [.5f, .5f, .5f, .5f, .5f, .5f], 3, 3), options);
        using var score = new ScoreOnnxDecisionModel(session);
        var mixed = Mixed();
        var invalid = new DecisionRequest(mixed.State,
            [new ChoiceDecisionQuestion("three", "choose", [new("a", "a"), new("b", "b"), new("c", "c")]),
             mixed.Questions[1], mixed.Questions[2]]);
        var invalidBatch = Prepare().Prepare(invalid, Cancel);
        var raw = score.Score(invalidBatch, Cancel);
        Assert.Equal(sentinel, raw.Logits[0, 2]);
        var decode = new DecodeDecisions(Profile());
        Assert.Throws<DecisionProtocolException>(() => decode.Decode(invalidBatch, raw));
        var validBatch = Prepare().Prepare(mixed, Cancel);
        var recovered = score.Score(validBatch, Cancel);
        Assert.Equal(sentinel, recovered.Logits[0, 2]);
        Assert.Equal(sentinel, recovered.Logits[2, 2]);
        var response = decode.Decode(validBatch, recovered);
        Assert.Equal("opaque-B", Assert.IsType<ChoiceDecisionAnswer>(response.Answers[0]).SelectedCandidateId);
        Assert.Equal(.7310585786300049, Assert.IsType<BinaryDecisionAnswer>(response.Answers[2]).TrueProbability, 14);
        Assert.Equal(1.5752103826044415, Assert.IsType<ScoreDecisionAnswer>(response.Answers[1]).ExpectedIndex, 14);
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void OutputPolicy_LayaDefersNonfiniteActionsUntilDecode(float sentinel)
    {
        using var options = Options();
        using var session = new InferenceSession(SyntheticOnnx.CreateLayaOutputs([0, 1], [.5f, sentinel], 1, 2), options);
        using var score = new ScoreOnnxDecisionModel(session);
        var batch = Prepare().Prepare(Request(Choice()), Cancel);
        var raw = score.Score(batch, Cancel);
        Assert.Equal(sentinel, raw.ActionProbabilities[0, 1]);
        Assert.Equal(1, raw.Logits[0, 1]);
        Assert.Throws<DecisionProtocolException>(() => new DecodeDecisions(Profile()).Decode(batch, raw));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    public void OutputPolicy_LayaRejectsStaticActionWidthAtConstruction(int width)
    {
        using var options = Options();
        using var session = new InferenceSession(SyntheticOnnx.CreateLayaOutputs([0, 1], new float[width], 1, 2, width), options);
        Assert.Equal("act_probs must have width 2.",
            Assert.Throws<ArgumentException>(() => new ScoreOnnxDecisionModel(session)).Message);
        Assert.False(options.IsClosed);
    }

    [Fact]
    public void SharedRunner_HasNoExportedTypesAndBothScorersUseTheSameAssembly()
    {
        var laya = typeof(ScoreOnnxDecisionModel).GetField("_runner", BindingFlags.Instance | BindingFlags.NonPublic)!;
        var julia = typeof(Julia.ScoreOnnxDecisionModel).GetField("_runner", BindingFlags.Instance | BindingFlags.NonPublic)!;
        Assert.Same(laya.FieldType, julia.FieldType);
        Assert.Equal("DecisionInference.Onnx.MarkerOnnxRunner", laya.FieldType.FullName);
        Assert.Empty(laya.FieldType.Assembly.GetExportedTypes());
        var friends = laya.FieldType.Assembly.GetCustomAttributes<System.Runtime.CompilerServices.InternalsVisibleToAttribute>();
        Assert.Equal(["DecisionInference.Julia", "DecisionInference.Laya"], friends.Select(a => a.AssemblyName).Order(StringComparer.Ordinal));
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public async Task SharedRunner_CoordinatesScoreCancellationAndDisposalWithPublicOwnerContext(bool julia, bool ownsSession)
    {
        using var fixture = new JuliaTestFixture();
        using var juliaPrepare = fixture.Prepare();
        using var options = Options();
        using var session = new InferenceSession(julia ? SyntheticOnnx.CreateJulia() : SyntheticOnnx.Create(), options);
        using IDisposable scorer = julia
            ? new Julia.ScoreOnnxDecisionModel(session, ownsSession)
            : new ScoreOnnxDecisionModel(session, ownsSession);
        var juliaBatch = juliaPrepare.Prepare(Request(Choice()), Cancel);
        var layaBatch = Prepare().Prepare(Request(Choice()), Cancel);
        Func<CancellationToken, Tensor<float>> run = token => julia
            ? ((Julia.ScoreOnnxDecisionModel)scorer).Score(juliaBatch, token)
            : ((ScoreOnnxDecisionModel)scorer).Score(layaBatch, token).Logits;
        Action<CancellationToken> nullBatch = token =>
        {
            if (julia) ((Julia.ScoreOnnxDecisionModel)scorer).Score(null!, token);
            else ((ScoreOnnxDecisionModel)scorer).Score(null!, token);
        };
        var first = run(Cancel);
        float expected = first[0, 1];
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        Assert.Equal("batch", Assert.Throws<ArgumentNullException>(() => nullBatch(cancelled.Token)).ParamName);

        // White-box synchronization proves both public stages actually enter the shared gate,
        // without exporting test hooks or relying on a slow model/timed native inference.
        var runner = scorer.GetType().GetField("_runner", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(scorer)!;
        var gate = (SemaphoreSlim)runner.GetType().GetField("_gate", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(runner)!;
        gate.Wait(Cancel);
        Thread? disposing = null;
        Exception? disposalError = null;
        using var entered = new ManualResetEventSlim();
        using var disposeEntered = new ManualResetEventSlim();
        try
        {
            using var pendingCancellation = new CancellationTokenSource();
            var pending = Task.Run(() => { entered.Set(); return run(pendingCancellation.Token); }, Cancel);
            Assert.True(entered.Wait(TimeSpan.FromSeconds(10), Cancel));
            pendingCancellation.Cancel();
            var error = await Assert.ThrowsAnyAsync<OperationCanceledException>(async () => await pending);
            Assert.Equal(pendingCancellation.Token, error.CancellationToken);
            disposing = new Thread(() =>
            {
                disposeEntered.Set();
                try { scorer.Dispose(); }
                catch (Exception error) { disposalError = error; }
            });
            disposing.Start();
            Assert.True(disposeEntered.Wait(TimeSpan.FromSeconds(10), Cancel));
            Assert.True(SpinWait.SpinUntil(() => !disposing.IsAlive ||
                (disposing.ThreadState & ThreadState.WaitSleepJoin) != 0, TimeSpan.FromSeconds(10)));
            Assert.True(disposing.IsAlive);
            Assert.False(SessionIsDisposed(session));
        }
        finally
        {
            gate.Release();
            if (disposing is not null) Assert.True(disposing.Join(TimeSpan.FromSeconds(10)));
        }
        Assert.Null(disposalError);
        Assert.Equal(ownsSession, SessionIsDisposed(session));
        Assert.Equal(expected, first[0, 1]);
        Assert.Equal(scorer.GetType().FullName, Assert.Throws<ObjectDisposedException>(() => run(Cancel)).ObjectName);
        Assert.Equal("batch", Assert.Throws<ArgumentNullException>(() => nullBatch(cancelled.Token)).ParamName);
        Assert.ThrowsAny<OperationCanceledException>(() => run(cancelled.Token));
        scorer.Dispose();
        Assert.False(options.IsClosed);
        if (!ownsSession)
        {
            using IDisposable reused = julia ? new Julia.ScoreOnnxDecisionModel(session) : new ScoreOnnxDecisionModel(session);
            var next = julia ? ((Julia.ScoreOnnxDecisionModel)reused).Score(juliaBatch, Cancel)
                : ((ScoreOnnxDecisionModel)reused).Score(layaBatch, Cancel).Logits;
            Assert.Equal(expected, next[0, 1]);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SharedRunner_ConstructorFailureDoesNotTakeSessionOwnershipAndZeroIsStatic(bool julia)
    {
        using var options = Options();
        byte[] invalid = julia ? SyntheticOnnx.Create() : SyntheticOnnx.CreateJulia();
        using var rejected = new InferenceSession(invalid, options);
        Assert.Throws<ArgumentException>(() =>
        {
            if (julia) new Julia.ScoreOnnxDecisionModel(rejected, ownsSession: true);
            else new ScoreOnnxDecisionModel(rejected, ownsSession: true);
        });
        Assert.False(SessionIsDisposed(rejected));

        var inputs = SyntheticOnnx.Inputs();
        inputs[0] = inputs[0] with { Shape = [0, -1] };
        using var session = new InferenceSession(julia ? SyntheticOnnx.CreateJulia(inputs) : SyntheticOnnx.Create(inputs), options);
        using var fixture = new JuliaTestFixture();
        using var prepare = fixture.Prepare();
        using IDisposable scorer = julia ? new Julia.ScoreOnnxDecisionModel(session) : new ScoreOnnxDecisionModel(session);
        var error = Assert.Throws<ArgumentException>(() =>
        {
            if (julia) ((Julia.ScoreOnnxDecisionModel)scorer).Score(prepare.Prepare(Request(Choice()), Cancel), Cancel);
            else ((ScoreOnnxDecisionModel)scorer).Score(Prepare().Prepare(Request(Choice()), Cancel), Cancel);
        });
        Assert.Contains("graph input input_ids", error.Message, StringComparison.Ordinal);
    }

    private static bool SessionIsDisposed(InferenceSession session) =>
        (bool)typeof(InferenceSession).GetField("_disposed", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(session)!;
}
