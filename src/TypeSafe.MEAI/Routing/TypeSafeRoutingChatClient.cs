using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.AI;

namespace TypeSafe.MEAI;

/// <summary>A host-owned route. Its client is borrowed, not disposed by the router.</summary>
/// <param name="Id">Unique nonempty Choice label.</param>
/// <param name="Description">Meaningful routing semantics.</param>
/// <param name="Client">The existing downstream client/pipeline.</param>
public sealed record TypeSafeRoute(string Id, string Description, IChatClient Client);

/// <summary>Per-call routing evidence; downstream model/usage remain untouched.</summary>
/// <param name="RouteId">Accepted route, or null for an uncertain decision.</param>
/// <param name="UsedFallback">Whether the explicitly configured fallback was selected.</param>
/// <param name="Evidence">Full native classification response.</param>
public sealed record TypeSafeRouteDecision(string? RouteId, bool UsedFallback, TypeSafeResponse Evidence);

/// <summary>An uncertain route without an explicitly configured fallback.</summary>
public sealed class TypeSafeRoutingAbstentionException : InvalidOperationException
{
    /// <summary>Full native evidence for this call.</summary>
    public TypeSafeRouteDecision Decision { get; }
    /// <summary>Creates an evidence-carrying routing abstention.</summary>
    public TypeSafeRoutingAbstentionException(TypeSafeRouteDecision decision) : base("Routing abstained; no fallback was configured.") => Decision = decision;
}

/// <summary>A stateless classification policy over the existing experimental MEAI RoutingChatClient.</summary>
public sealed class TypeSafeRoutingChatClient : IChatClient
{
    private readonly IChatClient _router;
    private bool _disposed;

    /// <summary>Uses explicit confidence and optional fallback; HTTP/protocol failures always propagate.
    /// Classifier model is independent of downstream options. All clients are borrowed.
    /// A custom projector is required for non-text conversations; it must return an owned explicit state.</summary>
    public TypeSafeRoutingChatClient(IChatClient classifier, IEnumerable<TypeSafeRoute> routes,
        double minimumConfidence, string? classifierModelId = null, IChatClient? fallback = null,
        Func<IReadOnlyList<ChatMessage>, TypeSafeState>? stateProjector = null,
        Func<TypeSafeRouteDecision, CancellationToken, ValueTask>? observe = null)
    {
        ArgumentNullException.ThrowIfNull(classifier);
        if (!double.IsFinite(minimumConfidence) || minimumConfidence < 0 || minimumConfidence > 1)
            throw new ArgumentOutOfRangeException(nameof(minimumConfidence));
        var configured = routes.ToArray();
        if (configured.Length is < 1 or > 255 || configured.Any(r => r is null ||
            string.IsNullOrWhiteSpace(r.Id) || string.IsNullOrWhiteSpace(r.Description) || r.Client is null) ||
            configured.Select(r => r.Id).Distinct(StringComparer.Ordinal).Count() != configured.Length)
            throw new ArgumentException("Routes require 1-255 unique nonempty IDs, descriptions, and clients.", nameof(routes));
        if (configured.Any(r => ReferenceEquals(r.Client, classifier)) || ReferenceEquals(fallback, classifier))
            throw new ArgumentException("The classifier cannot also be a downstream route/fallback.", nameof(classifier));
        var targets = configured.ToDictionary(r => r.Id, r => r.Client, StringComparer.Ordinal);
        var question = new ChoiceQuestion("route", TypeSafeJson.Entry("Choose the best destination for this stateless conversation."),
            configured.Select(r => KeyValuePair.Create(r.Id, TypeSafeJson.Entry(r.Description))));
        stateProjector ??= ProjectText;
#pragma warning disable MEAI001 // The approved proof deliberately uses the pinned experimental router.
        _router = RoutingChatClient.Create(async (context, token) =>
        {
            token.ThrowIfCancellationRequested();
            var state = stateProjector((IReadOnlyList<ChatMessage>)context.Messages);
            var evidence = await classifier.GetTypeSafeResponseAsync(state, [question],
                new ChatOptions { ModelId = classifierModelId }, token).ConfigureAwait(false);
            var answer = (ChoiceAnswer)evidence.Answers["route"];
            bool accepted = answer.Confidence >= minimumConfidence;
            var decision = new TypeSafeRouteDecision(accepted ? answer.Choice : null, !accepted && fallback is not null, evidence);
            if (observe is not null) await observe(decision, token).ConfigureAwait(false);
            token.ThrowIfCancellationRequested();
            return accepted ? targets[answer.Choice] : fallback ?? throw new TypeSafeRoutingAbstentionException(decision);
        });
#pragma warning restore MEAI001
    }

    private static TypeSafeState ProjectText(IReadOnlyList<ChatMessage> messages)
    {
        var snapshot = messages.Select(m => new
        {
            role = m.Role.Value,
            author = m.AuthorName,
            contents = m.Contents.Select(c => c is TextContent text ? text.Text :
                throw new NotSupportedException("Default routing projection supports text only; supply an explicit state projector.")).ToArray()
        }).ToArray();
        return TypeSafeState.FromJson(JsonSerializer.SerializeToElement(snapshot));
    }

    private ChatMessage[] Prepare(IEnumerable<ChatMessage> messages, ChatOptions? options, CancellationToken token)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        token.ThrowIfCancellationRequested();
        if (options?.ConversationId is not null || options?.ContinuationToken is not null)
            throw new NotSupportedException("Routing server-side conversation/continuation handles is not supported.");
        return messages.ToArray();
    }

    /// <inheritdoc />
    public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null,
        CancellationToken cancellationToken = default) =>
        _router.GetResponseAsync(Prepare(messages, options, cancellationToken), options, cancellationToken);

    /// <inheritdoc />
    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages,
        ChatOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var snapshot = Prepare(messages, options, cancellationToken);
        await foreach (var update in _router.GetStreamingResponseAsync(snapshot, options, cancellationToken).ConfigureAwait(false))
            yield return update;
    }

    /// <inheritdoc />
    public object? GetService(Type serviceType, object? serviceKey = null) =>
        serviceKey is null && serviceType.IsInstanceOfType(this) ? this : null;

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _router.Dispose();
    }
}
