using System.Text.Json;
using Microsoft.Extensions.AI;

namespace TypeSafe.MEAI;

/// <summary>An explicit accepted-or-abstained result, with no partial/default value on abstention.</summary>
public sealed class TypeSafeDecision<T>
{
    private readonly T? _value;
    /// <summary>Whether every field and application policy accepted the object.</summary>
    public bool Accepted { get; }
    /// <summary>The complete accepted value; accessing it on abstention throws.</summary>
    public T Value => Accepted ? _value! : throw new InvalidOperationException("An abstained decision has no value.");
    /// <summary>Full native evidence, not a calibrated joint probability.</summary>
    public TypeSafeResponse Evidence { get; }
    /// <summary>Rejected field pointers/reasons; empty on acceptance.</summary>
    public IReadOnlyList<TypeSafeRejection> Rejections { get; }
    internal TypeSafeDecision(T value, TypeSafeResponse evidence)
    {
        Accepted = true; _value = value; Evidence = evidence; Rejections = Array.Empty<TypeSafeRejection>();
    }
    internal TypeSafeDecision(TypeSafeAbstentionException exception)
    {
        Evidence = exception.Evidence; Rejections = exception.Rejections;
    }
}

/// <summary>Standard typed-output cannot return an incomplete T; this exception preserves abstention evidence.</summary>
public sealed class TypeSafeAbstentionException : Exception
{
    /// <summary>Complete native response evidence.</summary>
    public TypeSafeResponse Evidence { get; }
    /// <summary>All rejected field/application policies.</summary>
    public IReadOnlyList<TypeSafeRejection> Rejections { get; }
    internal TypeSafeAbstentionException(TypeSafeResponse evidence, IReadOnlyList<TypeSafeRejection> rejections)
        : base("TypeSafe projection abstained; no complete value is available.")
    {
        Evidence = evidence;
        Rejections = Array.AsReadOnly(rejections.ToArray());
    }
}

/// <summary>Explicit decisions using the actual MEAI typed-output entry point.</summary>
public static class TypeSafeDecisionExtensions
{
    /// <summary>Calls the supplied decorated schema client; returns abstention instead of incomplete T.</summary>
    public static async Task<TypeSafeDecision<T>> GetTypeSafeDecisionAsync<T>(this IChatClient schemaClient, TypeSafeState state,
        ChatOptions? options = null, JsonSerializerOptions? serializerOptions = null, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        try
        {
            var response = await schemaClient.GetResponseAsync<T>([state.ToMessage()],
                serializerOptions ?? AIJsonUtilities.DefaultOptions, options, useJsonSchemaResponseFormat: true, cancellationToken).ConfigureAwait(false);
            return new(response.Result, response.GetTypeSafeEvidence());
        }
        catch (TypeSafeAbstentionException e) { return new(e); }
    }

    /// <summary>Reads serialized native evidence after projection, including JsonElement cache/report roundtrips.</summary>
    public static TypeSafeResponse GetTypeSafeEvidence(this ChatResponse response)
    {
        if (response.AdditionalProperties?.TryGetValue("typesafe.v1.evidence", out var payload) != true ||
            !response.AdditionalProperties.TryGetValue("typesafe.v1.evidence.questions", out var questionPayload))
            throw new TypeSafeProtocolException("Projected response is missing native evidence metadata.");
        var json = payload is JsonElement j ? j : JsonSerializer.SerializeToElement(payload);
        var questions = questionPayload is JsonElement q ? q : JsonSerializer.SerializeToElement(questionPayload);
        return TypeSafeResponse.Parse(json.GetRawText(), questions.EnumerateObject().Select(p => TypeSafeQuestion.Parse(p.Name, p.Value)));
    }
}
