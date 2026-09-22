using System.Text;
using System.Text.Json;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.AI.Evaluation;

namespace TypeSafe.MEAI.Evaluation;

/// <summary>Structured context whose evidence is entirely stored in standard Contents.</summary>
public sealed class TypeSafeEvaluationContext : EvaluationContext
{
    /// <summary>Owns and serializes a structured context payload for reporting roundtrips.</summary>
    public TypeSafeEvaluationContext(string name, JsonElement value)
        : base(name, new DataContent(Encoding.UTF8.GetBytes(value.GetRawText()), "application/json"))
    {
    }
}
