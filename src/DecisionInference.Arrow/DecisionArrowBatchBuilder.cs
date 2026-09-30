using Apache.Arrow;
using Apache.Arrow.Types;

namespace DecisionInference.Arrow;

internal sealed class DecisionArrowBatchBuilder
{
    private readonly DecisionArrowSchema _contract;
    private readonly Int64Array.Builder _ids = new();
    private readonly DoubleArray.Builder[][] _numeric;
    private readonly Int32Array.Builder?[] _selected;
    private readonly FixedSizeListArray.Builder?[] _vectors;
    private readonly int _capacity;
    private readonly DecisionAnswer[] _answers;
    private readonly double[][] _values;
    public int Count => _ids.Length;

    public DecisionArrowBatchBuilder(DecisionArrowSchema contract, int capacity)
    {
        _contract = contract;
        if (capacity is < 1 or > 65536) throw new ArgumentOutOfRangeException(nameof(capacity));
        _capacity = capacity;
        _answers = new DecisionAnswer[contract.Questions.Count];
        _values = new double[contract.Questions.Count][];
        // Check every bound before allocating any probability buffer.
        foreach (int width in contract.Compiled.Widths) _ = checked(capacity * width * sizeof(double));
        _ids.Reserve(capacity);
        _numeric = new DoubleArray.Builder[contract.Questions.Count][];
        _selected = new Int32Array.Builder?[contract.Questions.Count];
        _vectors = new FixedSizeListArray.Builder?[contract.Questions.Count];
        for (int i = 0; i < contract.Questions.Count; i++)
        {
            var q = contract.Questions[i];
            int width = contract.Compiled.Widths[i];
            _values[i] = new double[width];
            _numeric[i] = Enumerable.Range(0, q.Kind == DecisionKind.Score ? 4 : q.Kind == DecisionKind.Binary ? 1 : 0)
                .Select(_ => new DoubleArray.Builder().Reserve(capacity)).ToArray();
            if (q.Kind == DecisionKind.Choice) _selected[i] = new Int32Array.Builder().Reserve(capacity);
            if (width > 0)
            {
                _ = checked(capacity * width * sizeof(double));
                _vectors[i] = new FixedSizeListArray.Builder(new Field("item", DoubleType.Default, false), width).Reserve(capacity);
                ((DoubleArray.Builder)_vectors[i]!.ValueBuilder).Reserve(checked(capacity * width));
            }
        }
    }

    public void Append(DecisionArrowRow row)
    {
        ArgumentNullException.ThrowIfNull(row);
        if (Count == _capacity) throw new InvalidOperationException("Batch capacity is exhausted.");
        if (!string.Equals(row.Result.ModelId, _contract.Identity.ModelId, StringComparison.Ordinal))
            throw new DecisionProtocolException("Model identity changed within the dataset.");
        if (row.Input.Questions.Count != _contract.Questions.Count)
            throw new DecisionProtocolException("Question definitions changed within the dataset.");
        for (int i = 0; i < row.Input.Questions.Count; i++)
            if (!_contract.Compiled.Matches(row.Input.Questions[i], i))
                throw new DecisionProtocolException("Question definitions changed within the dataset.");
        ValidateAnswers(row.Result);
        // Every definition, association and observation is validated before the first append.
        _ids.Append(row.RowId);
        for (int i = 0; i < _answers.Length; i++)
        {
            var answer = _answers[i];
            switch (answer)
            {
                case BinaryDecisionAnswer b: _numeric[i][0].Append(b.TrueProbability); break;
                case ChoiceDecisionAnswer c:
                    _selected[i]!.Append(_contract.Compiled.Candidates[i]![c.SelectedCandidateId]);
                    AppendVector(i, _values[i]);
                    break;
                case ScoreDecisionAnswer s:
                    _numeric[i][0].Append(s.ExpectedIndex);
                    _numeric[i][1].Append(s.ExpectedIndexLowerBound);
                    _numeric[i][2].Append(s.ExpectedIndexUpperBound);
                    if (s.NativeScore is { } native) _numeric[i][3].Append(native); else _numeric[i][3].AppendNull();
                    AppendVector(i, _values[i]);
                    break;
                default: throw new NotSupportedException("Unsupported answer.");
            }
        }
    }

    private void ValidateAnswers(DecisionResult result)
    {
        if (result.Answers.Count != _answers.Length)
            throw new DecisionProtocolException("Missing, extra, duplicate or null answers.");
        for (int i = 0; i < _answers.Length; i++)
        {
            var question = _contract.Questions[i];
            DecisionAnswer? answer = result.Answers[i].QuestionId == question.Id ? result.Answers[i] : null;
            if (answer is null)
                for (int j = 0; j < result.Answers.Count; j++)
                    if (result.Answers[j].QuestionId == question.Id) { answer = result.Answers[j]; break; }
            if (answer is null || answer.Kind != question.Kind)
                throw new DecisionProtocolException("Answer ID or kind mismatch.");
            _answers[i] = answer;
            switch (answer)
            {
                case BinaryDecisionAnswer b:
                    DecisionObservationValidation.Binary(b.TrueProbability, answer.Precision);
                    break;
                case ChoiceDecisionAnswer c:
                    var candidates = ((ChoiceDecisionQuestion)question).Candidates;
                    if (c.Probabilities.Count != candidates.Count)
                        throw new DecisionProtocolException("Incomplete or unexpected candidate distribution.");
                    for (int j = 0; j < candidates.Count; j++)
                        if (!c.Probabilities.TryGetValue(candidates[j].Id, out _values[i][j]))
                            throw new DecisionProtocolException("Incomplete or unexpected candidate distribution.");
                    if (!_contract.Compiled.Candidates[i]!.ContainsKey(c.SelectedCandidateId))
                        throw new DecisionProtocolException("Unknown selected candidate.");
                    DecisionObservationValidation.Distribution(_values[i], answer.Precision);
                    break;
                case ScoreDecisionAnswer s:
                    if (s.Probabilities.Count != _values[i].Length)
                        throw new DecisionProtocolException("Incomplete or unexpected rubric distribution.");
                    for (int j = 0; j < _values[i].Length; j++) _values[i][j] = s.Probabilities[j];
                    var observation = DecisionObservationValidation.Score(_values[i], s.NativeScore, answer.Precision);
                    if (s.ExpectedIndex != observation.Expected || s.ExpectedIndexLowerBound != observation.Lower ||
                        s.ExpectedIndexUpperBound != observation.Upper)
                        throw new DecisionProtocolException("Score expectation or portable rounding bounds mismatch.");
                    break;
                default: throw new NotSupportedException("Unsupported answer.");
            }
        }
        if (result.Usage is { } u && (u.InputTokenCount < 0 || u.OutputTokenCount < 0 || u.TotalTokenCount < 0 ||
            (u.InputTokenCount is { } input && u.OutputTokenCount is { } output &&
                (input > long.MaxValue - output || (u.TotalTokenCount is { } total && total != input + output)))))
            throw new DecisionProtocolException("Invalid observed usage.");
        foreach (var answer in _answers)
            if (answer.Precision != _contract.Precision) throw new DecisionProtocolException("Precision changed within the dataset.");
    }

    private void AppendVector(int index, ReadOnlySpan<double> values)
    {
        _vectors[index]!.Append();
        var builder = (DoubleArray.Builder)_vectors[index]!.ValueBuilder;
        foreach (double value in values) builder.Append(value);
    }

    internal void AppendHighPrecisionFrame(long id, ReadOnlySpan<int> selected, ReadOnlySpan<int> offsets,
        ReadOnlySpan<double> probabilities, ReadOnlySpan<double> scores)
    {
        if (_contract.Precision != DecisionPrecision.HighPrecision)
            throw new DecisionProtocolException("Numeric frames require HighPrecision.");
        if (Count == _capacity) throw new InvalidOperationException("Batch capacity is exhausted.");
        int questions = _contract.Questions.Count;
        if (selected.Length != questions || offsets.Length != questions + 1 || scores.Length != checked(questions * 3) ||
            offsets[0] != 0 || offsets[^1] != probabilities.Length)
            throw new DecisionProtocolException("Numeric frame shape mismatch.");
        for (int q = 0; q < questions; q++)
        {
            int width = _contract.Questions[q].Kind == DecisionKind.Binary ? 2 : _contract.Compiled.Widths[q];
            if (offsets[q] < 0 || offsets[q + 1] - offsets[q] != width || offsets[q + 1] > probabilities.Length)
                throw new DecisionProtocolException("Numeric probability frame shape mismatch.");
            var values = probabilities.Slice(offsets[q], width);
            DecisionObservationValidation.Distribution(values, DecisionPrecision.HighPrecision);
            switch (_contract.Questions[q].Kind)
            {
                case DecisionKind.Binary:
                    DecisionObservationValidation.Binary(values[1], DecisionPrecision.HighPrecision);
                    if (selected[q] != -1) throw new DecisionProtocolException("Unexpected binary selected ordinal.");
                    break;
                case DecisionKind.Choice:
                    if (selected[q] < 0 || selected[q] >= width) throw new DecisionProtocolException("Selected index outside candidate domain.");
                    break;
                case DecisionKind.Score:
                    if (selected[q] != -1) throw new DecisionProtocolException("Unexpected score selected ordinal.");
                    var observation = DecisionObservationValidation.Score(values, null, DecisionPrecision.HighPrecision);
                    if (scores[q * 3] != observation.Expected || scores[q * 3 + 1] != observation.Lower || scores[q * 3 + 2] != observation.Upper)
                        throw new DecisionProtocolException("Score expectation or portable rounding bounds mismatch.");
                    break;
            }
        }
        _ids.Append(id);
        for (int q = 0; q < questions; q++)
        {
            var values = probabilities.Slice(offsets[q], offsets[q + 1] - offsets[q]);
            switch (_contract.Questions[q].Kind)
            {
                case DecisionKind.Binary: _numeric[q][0].Append(values[1]); break;
                case DecisionKind.Choice:
                    _selected[q]!.Append(selected[q]);
                    AppendVector(q, values);
                    break;
                case DecisionKind.Score:
                    for (int i = 0; i < 3; i++) _numeric[q][i].Append(scores[q * 3 + i]);
                    _numeric[q][3].AppendNull();
                    AppendVector(q, values);
                    break;
            }
        }
    }

    public RecordBatch Build()
    {
        var schema = _contract.Compiled.Schema;
        var arrays = new List<IArrowArray> { _ids.Build() };
        for (int i = 0; i < _contract.Questions.Count; i++)
        {
            if (_contract.Questions[i].Kind == DecisionKind.Binary) arrays.Add(_numeric[i][0].Build());
            else
            {
                var children = new List<IArrowArray>();
                if (_selected[i] is { } selected) children.Add(selected.Build());
                children.AddRange(_numeric[i].Select(b => b.Build()));
                children.Add(_vectors[i]!.Build());
                arrays.Add(new StructArray(schema.FieldsList[i + 1].DataType, Count, children, ArrowBuffer.Empty));
            }
        }
        return new(schema, arrays, Count);
    }
}
