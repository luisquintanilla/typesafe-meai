using System.Runtime.CompilerServices;
using Apache.Arrow;
using DecisionInference;
using DecisionInference.Arrow;
using DecisionInference.Julia;

internal sealed class JuliaArrowExporter(DecisionArrowSchema contract, JuliaDecisionGenerator generator) : IJuliaNumericSink
{
    private DecisionArrowBatchBuilder _builder = new(contract, DecisionArrowWriter.DefaultBatchSize);
    private long _id;
    private DecisionInput? _source;

    internal async Task<RecordBatch> WarmupAsync((long Id, DecisionInput Input) state, CancellationToken token)
    {
        await Task.CompletedTask;
        _id = state.Id;
        _source = state.Input;
        try { generator.GenerateToSink(state.Input, this, token); }
        finally { _source = null; }
        return _builder.Build();
    }

    public void WriteFrame(DecisionInput source, ReadOnlySpan<int> selectedOrdinals, ReadOnlySpan<int> probabilityOffsets,
        ReadOnlySpan<double> probabilities, ReadOnlySpan<double> scoreFields)
    {
        if (!ReferenceEquals(source, _source)) throw new DecisionProtocolException("Julia numeric source association changed.");
        _builder.AppendHighPrecisionFrame(_id, selectedOrdinals, probabilityOffsets, probabilities, scoreFields);
    }

    internal async IAsyncEnumerable<RecordBatch> Batches(IAsyncEnumerable<(long Id, DecisionInput Input)> states,
        [EnumeratorCancellation] CancellationToken token)
    {
        await foreach (var state in states.WithCancellation(token))
        {
            _id = state.Id;
            _source = state.Input;
            try { generator.GenerateToSink(state.Input, this, token); }
            finally { _source = null; }
            if (_builder.Count == DecisionArrowWriter.DefaultBatchSize)
            {
                yield return _builder.Build();
                _builder = new(contract, DecisionArrowWriter.DefaultBatchSize);
            }
        }
        if (_builder.Count > 0) yield return _builder.Build();
    }
}
