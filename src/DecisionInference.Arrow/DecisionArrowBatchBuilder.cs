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
    public int Count => _ids.Length;

    public DecisionArrowBatchBuilder(DecisionArrowSchema contract, int capacity)
    {
        _contract = contract;
        _ids.Reserve(capacity);
        _numeric = new DoubleArray.Builder[contract.Questions.Count][];
        _selected = new Int32Array.Builder?[contract.Questions.Count];
        _vectors = new FixedSizeListArray.Builder?[contract.Questions.Count];
        for (int i = 0; i < contract.Questions.Count; i++)
        {
            var q = contract.Questions[i];
            int width = q switch { ChoiceDecisionQuestion c => c.Candidates.Count, ScoreDecisionQuestion s => s.Rubric.Count, _ => 0 };
            _numeric[i] = Enumerable.Range(0, q.Kind == DecisionKind.Score ? 4 : q.Kind == DecisionKind.Binary ? 1 : 0)
                .Select(_ => new DoubleArray.Builder().Reserve(capacity)).ToArray();
            if (q.Kind == DecisionKind.Choice) _selected[i] = new Int32Array.Builder().Reserve(capacity);
            if (width > 0)
            {
                _ = checked(capacity * width * sizeof(double));
                _vectors[i] = new FixedSizeListArray.Builder(new Field("item", DoubleType.Default, false), width).Reserve(capacity);
            }
        }
    }

    public void Append(DecisionArrowRow row)
    {
        if (!string.Equals(row.Result.ModelId, _contract.Identity.ModelId, StringComparison.Ordinal))
            throw new DecisionProtocolException("Model identity changed within the dataset.");
        if (row.Input.Questions.Count != _contract.Questions.Count ||
            row.Input.Questions.Where((q, i) => DecisionArrowSchema.QuestionJson(q) != DecisionArrowSchema.QuestionJson(_contract.Questions[i])).Any())
            throw new DecisionProtocolException("Question definitions changed within the dataset.");
        // Revalidate association; a result may have been constructed against another input.
        var result = new DecisionResult(row.Input, row.Result.ModelId, row.Result.Answers, row.Result.Usage);
        if (result.Answers.Any(a => a.Precision != _contract.Precision))
            throw new DecisionProtocolException("Precision changed within the dataset.");
        _ids.Append(row.RowId);
        for (int i = 0; i < result.Answers.Count; i++)
        {
            var answer = result.Answers[i];
            switch (answer)
            {
                case BinaryDecisionAnswer b: _numeric[i][0].Append(b.TrueProbability); break;
                case ChoiceDecisionAnswer c:
                    var candidates = ((ChoiceDecisionQuestion)_contract.Questions[i]).Candidates;
                    _selected[i]!.Append(candidates.Select((v, index) => (v, index)).Single(v => v.v.Id == c.SelectedCandidateId).index);
                    AppendVector(i, candidates.Select(v => c.Probabilities[v.Id]));
                    break;
                case ScoreDecisionAnswer s:
                    _numeric[i][0].Append(s.ExpectedIndex);
                    _numeric[i][1].Append(s.ExpectedIndexLowerBound);
                    _numeric[i][2].Append(s.ExpectedIndexUpperBound);
                    if (s.NativeScore is { } native) _numeric[i][3].Append(native); else _numeric[i][3].AppendNull();
                    AppendVector(i, s.Probabilities);
                    break;
                default: throw new NotSupportedException("Unsupported answer.");
            }
        }
    }

    private void AppendVector(int index, IEnumerable<double> values)
    {
        _vectors[index]!.Append();
        var builder = (DoubleArray.Builder)_vectors[index]!.ValueBuilder;
        foreach (double value in values) builder.Append(value);
    }

    public RecordBatch Build()
    {
        var schema = _contract.Schema;
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
