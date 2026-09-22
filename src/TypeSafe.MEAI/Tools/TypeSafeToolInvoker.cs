using Microsoft.Extensions.AI;

namespace TypeSafe.MEAI;

/// <summary>One explicit authorized host action. No inference loop, retries, or global exactly-once claim.</summary>
public static class TypeSafeToolInvoker
{
    /// <summary>Invokes the original outer function at most once after fresh authorization and, when required, distinct approval.
    /// Context/services are host-owned, not inferred arguments. Host replay prevention remains the caller's responsibility.</summary>
    public static async ValueTask<FunctionResultContent> InvokeAsync(TypeSafeToolSelection selection,
        Func<TypeSafeToolSelection, CancellationToken, ValueTask<bool>> authorize,
        Func<TypeSafeToolSelection, CancellationToken, ValueTask<bool>>? approve = null,
        IDictionary<object, object?>? context = null, IServiceProvider? services = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(selection);
        ArgumentNullException.ThrowIfNull(authorize);
        cancellationToken.ThrowIfCancellationRequested();
        selection.EnsureExecutable();
        if (!await authorize(selection, cancellationToken).ConfigureAwait(false))
            throw new UnauthorizedAccessException("Host authorization denied the selected function.");
        cancellationToken.ThrowIfCancellationRequested();
        if (selection.Function!.GetService(typeof(ApprovalRequiredAIFunction)) is not null &&
            (approve is null || !await approve(selection, cancellationToken).ConfigureAwait(false)))
            throw new UnauthorizedAccessException("The registered function requires separate affirmative host approval.");
        cancellationToken.ThrowIfCancellationRequested();
        selection.EnsureExecutable();
        var arguments = new AIFunctionArguments(selection.CopyArguments()) { Context = context, Services = services };
        var result = await selection.Function.InvokeAsync(arguments, cancellationToken).ConfigureAwait(false);
        return new FunctionResultContent(selection.CorrelationId!, result);
    }
}
