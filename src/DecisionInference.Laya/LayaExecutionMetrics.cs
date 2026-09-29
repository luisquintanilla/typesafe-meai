namespace DecisionInference.Laya;

internal sealed record LayaExecutionMetrics(double PreparationMilliseconds, double ScoringMilliseconds,
    double DecodingMilliseconds, int GraphRows, int SequenceLength, int MarkerWidth, long ManagedTensorBytes);
