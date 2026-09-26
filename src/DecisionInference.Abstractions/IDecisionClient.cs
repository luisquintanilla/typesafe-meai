namespace DecisionInference;

public interface IDecisionClient : IDisposable
{
    Task<DecisionResponse> GetResponseAsync(DecisionRequest request, DecisionOptions? options = null,
        CancellationToken cancellationToken = default);
    object? GetService(Type serviceType, object? serviceKey = null);
}

public enum DecisionKind { Binary, Choice, Score }

public sealed record DecisionOptions
{
    public string? ModelId { get; init; }
}

public sealed class DecisionClientMetadata
{
    public string ProviderName { get; }
    public string ModelId { get; }
    public IReadOnlyList<DecisionKind> SupportedKinds { get; }

    public DecisionClientMetadata(string providerName, string modelId, params DecisionKind[] supportedKinds)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(providerName);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        ArgumentNullException.ThrowIfNull(supportedKinds);
        if (supportedKinds.Length == 0 || supportedKinds.Any(k => !Enum.IsDefined(k)))
            throw new ArgumentException("At least one valid capability is required.", nameof(supportedKinds));
        ProviderName = providerName;
        ModelId = modelId;
        SupportedKinds = Array.AsReadOnly(supportedKinds.Distinct().ToArray());
    }

    public void Validate(DecisionRequest request, DecisionOptions? options, bool fixedModel = false)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (options?.ModelId is { } model)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(model);
            if (fixedModel && !string.Equals(model, ModelId, StringComparison.Ordinal))
                throw new NotSupportedException("This local client cannot switch its loaded model.");
        }
        if (request.Questions.Any(q => !SupportedKinds.Contains(q.Kind)))
            throw new NotSupportedException("The batch contains an unsupported decision kind.");
    }
}
