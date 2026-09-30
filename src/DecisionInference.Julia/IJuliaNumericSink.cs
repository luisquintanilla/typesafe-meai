namespace DecisionInference.Julia;

internal interface IJuliaNumericSink
{
    // Borrows end when this synchronous call returns; implementations must copy any retained values.
    void WriteFrame(DecisionInput source, ReadOnlySpan<int> selectedOrdinals, ReadOnlySpan<int> probabilityOffsets,
        ReadOnlySpan<double> probabilities, ReadOnlySpan<double> scoreFields);
}
