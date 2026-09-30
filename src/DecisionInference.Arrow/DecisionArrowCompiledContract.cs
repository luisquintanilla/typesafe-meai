using System.Collections.Frozen;
using Apache.Arrow;
using Apache.Arrow.Types;

namespace DecisionInference.Arrow;

internal sealed class DecisionArrowCompiledContract
{
    internal Schema Schema { get; }
    internal IReadOnlyList<DecisionQuestion> Questions { get; }
    internal IReadOnlyList<int> Widths { get; }
    internal IReadOnlyList<FrozenDictionary<string, int>?> Candidates { get; }

    internal DecisionArrowCompiledContract(DecisionArrowSchema contract, Schema schema)
    {
        Schema = schema;
        Questions = contract.Questions;
        var widths = new int[Questions.Count];
        var candidates = new FrozenDictionary<string, int>?[Questions.Count];
        for (int i = 0; i < Questions.Count; i++)
        {
            widths[i] = Questions[i] switch
            {
                ChoiceDecisionQuestion c => c.Candidates.Count,
                ScoreDecisionQuestion s => s.Rubric.Count,
                _ => 0
            };
            if (Questions[i] is ChoiceDecisionQuestion choice)
                candidates[i] = choice.Candidates.Select((c, ordinal) => KeyValuePair.Create(c.Id, ordinal)).ToFrozenDictionary(StringComparer.Ordinal);
        }
        Widths = System.Array.AsReadOnly(widths);
        Candidates = System.Array.AsReadOnly(candidates);
    }

    internal bool Matches(DecisionQuestion question, int index)
    {
        var expected = Questions[index];
        if (ReferenceEquals(question, expected)) return true;
        if (question.Id != expected.Id || question.Kind != expected.Kind || question.Instructions != expected.Instructions) return false;
        switch (question)
        {
            case BinaryDecisionQuestion b when expected is BinaryDecisionQuestion e:
                return b.TrueDescription == e.TrueDescription && b.FalseDescription == e.FalseDescription;
            case ChoiceDecisionQuestion c when expected is ChoiceDecisionQuestion e:
                if (c.Candidates.Count != e.Candidates.Count) return false;
                for (int i = 0; i < c.Candidates.Count; i++)
                    if (c.Candidates[i].Id != e.Candidates[i].Id || c.Candidates[i].Description != e.Candidates[i].Description) return false;
                return true;
            case ScoreDecisionQuestion s when expected is ScoreDecisionQuestion e:
                if (s.Rubric.Count != e.Rubric.Count) return false;
                for (int i = 0; i < s.Rubric.Count; i++) if (s.Rubric[i] != e.Rubric[i]) return false;
                return true;
            default: return false;
        }
    }

    internal Schema CreateSchemaView() => new(Schema.FieldsList.Select(CloneField),
        Schema.Metadata?.ToDictionary(p => p.Key, p => p.Value, StringComparer.Ordinal));

    private static Field CloneField(Field field)
    {
        IArrowType type = field.DataType switch
        {
            StructType structure => new StructType(structure.Fields.Select(CloneField).ToArray()),
            FixedSizeListType vector => new FixedSizeListType(CloneField(vector.ValueField), vector.ListSize),
            _ => field.DataType
        };
        return new(field.Name, type, field.IsNullable,
            field.Metadata?.ToDictionary(p => p.Key, p => p.Value, StringComparer.Ordinal));
    }
}
