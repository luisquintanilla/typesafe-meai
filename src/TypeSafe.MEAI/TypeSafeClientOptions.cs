namespace TypeSafe.MEAI;

/// <summary>Immutable provider settings, shared with normalization and transport.</summary>
public sealed record TypeSafeClientOptions
{
    /// <summary>HTTPS service origin, without a path, query, or fragment.</summary>
    public Uri Endpoint { get; init; } = new("https://api.typesafe.ai");
    /// <summary>Explicit default model; examples pin a version rather than a mutable alias.</summary>
    public string ModelId { get; init; } = "jev-1.13.0";
    /// <summary>Nonsecret account/tenant namespace required for opt-in caching.</summary>
    public string CacheNamespace { get; init; } = "";
    /// <summary>Maximum buffered success/error body in bytes.</summary>
    public int MaximumResponseBytes { get; init; } = 4 * 1024 * 1024;

    internal TypeSafeClientOptions Validate()
    {
        if (!Endpoint.IsAbsoluteUri || Endpoint.Scheme != Uri.UriSchemeHttps || Endpoint.AbsolutePath != "/" ||
            Endpoint.Query.Length != 0 || Endpoint.Fragment.Length != 0 || Endpoint.UserInfo.Length != 0)
            throw new ArgumentException("Endpoint must be an HTTPS origin without path, credentials, query, or fragment.");
        ArgumentException.ThrowIfNullOrWhiteSpace(ModelId);
        if (MaximumResponseBytes <= 0) throw new ArgumentOutOfRangeException(nameof(MaximumResponseBytes));
        return this;
    }
}
