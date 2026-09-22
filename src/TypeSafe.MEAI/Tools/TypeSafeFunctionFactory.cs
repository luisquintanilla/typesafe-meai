using System.Text.Json;
using Microsoft.Extensions.AI;

namespace TypeSafe.MEAI;

/// <summary>Exposes an application-approved, bounded native question batch as an ordinary MEAI function.</summary>
public static class TypeSafeFunctionFactory
{
    /// <summary>Creates a state-only function; it cannot choose endpoint, credentials, model or rubrics.</summary>
    public static AIFunction Create(IChatClient decoratedClient, IEnumerable<TypeSafeQuestion> questions,
        string name = "evaluate_support_ticket", string description = "Evaluate a synthetic support ticket using the host-approved questions.")
    {
        ArgumentNullException.ThrowIfNull(decoratedClient);
        var snapshot = questions.ToArray();
        _ = ((ChatOptions?)null).WithTypeSafeQuestions(snapshot);
        return AIFunctionFactory.Create(
            async (string state, CancellationToken cancellationToken) =>
            {
                cancellationToken.ThrowIfCancellationRequested();
                return (await decoratedClient.GetTypeSafeResponseAsync(TypeSafeState.FromText(state), snapshot,
                    cancellationToken: cancellationToken).ConfigureAwait(false)).Json;
            },
            name, description);
    }
}
