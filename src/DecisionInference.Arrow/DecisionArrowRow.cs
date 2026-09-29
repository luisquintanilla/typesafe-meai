namespace DecisionInference.Arrow;

public sealed class DecisionArrowRow
{
    public long RowId { get; }
    public DecisionInput Input { get; }
    public DecisionResult Result { get; }

    public DecisionArrowRow(long rowId, DecisionInput input, DecisionResult result)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(result);
        RowId = rowId;
        Input = input;
        Result = result;
    }
}
