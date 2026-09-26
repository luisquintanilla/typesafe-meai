using System.Text.Json;
using System.Text.Json.Serialization.Metadata;

namespace DecisionInference;

/// <summary>Service lookup and explicit, metadata-based state serialization conveniences.</summary>
public static class DecisionClientExtensions
{
    /// <summary>Gets a service for the exact key, or the default value when unavailable or of another type.</summary>
    public static TService? GetService<TService>(this IDecisionClient client, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(client);
        return client.GetService(typeof(TService), serviceKey) is TService service ? service : default;
    }

    /// <summary>Gets a required service for the exact key, or throws <see cref="InvalidOperationException"/>.</summary>
    public static object GetRequiredService(this IDecisionClient client, Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(client);
        ArgumentNullException.ThrowIfNull(serviceType);
        var service = client.GetService(serviceType, serviceKey);
        return service is not null && serviceType.IsInstanceOfType(service) ? service :
            throw new InvalidOperationException($"A service of type '{serviceType}' is not available for the specified key.");
    }

    /// <summary>Gets a required typed service for the exact key, or throws <see cref="InvalidOperationException"/>.</summary>
    public static TService GetRequiredService<TService>(this IDecisionClient client, object? serviceKey = null) =>
        (TService)client.GetRequiredService(typeof(TService), serviceKey);

    /// <summary>Snapshots JSON state and questions, forwarding options and cancellation unchanged.</summary>
    public static Task<DecisionResponse> GetResponseAsync(this IDecisionClient client, JsonElement state,
        IEnumerable<DecisionQuestion> questions, DecisionOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(client);
        return client.GetResponseAsync(new DecisionRequest(state, questions), options, cancellationToken);
    }

    /// <summary>Serializes state using only the supplied metadata, then evaluates the questions.</summary>
    /// <remarks>Null state is serialized according to the metadata. No reflection fallback or hidden serializer configuration is used.</remarks>
    public static Task<DecisionResponse> GetResponseAsync<TState>(this IDecisionClient client, TState state,
        IEnumerable<DecisionQuestion> questions, JsonTypeInfo<TState> stateTypeInfo,
        DecisionOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(client);
        ArgumentNullException.ThrowIfNull(questions);
        ArgumentNullException.ThrowIfNull(stateTypeInfo);
        cancellationToken.ThrowIfCancellationRequested();
        return client.GetResponseAsync(JsonSerializer.SerializeToElement(state, stateTypeInfo), questions, options, cancellationToken);
    }
}
