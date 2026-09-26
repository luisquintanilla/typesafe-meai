namespace TypeSafe.MEAI;

/// <summary>Acceptance failed before any partial/default typed value or executable call was emitted.</summary>
public sealed class TypeSafeAbstentionException : Exception
{
    /// <summary>Self-contained evidence for each completed native batch.</summary>
    public IReadOnlyList<JevResultContent> Evidence { get; }
    /// <summary>Explicit field/application rejection reasons.</summary>
    public IReadOnlyList<TypeSafeRejection> Rejections { get; }
    internal TypeSafeAbstentionException(IEnumerable<JevResultContent> evidence, IEnumerable<TypeSafeRejection> rejections)
        : base("TypeSafe decision abstained; no complete value or executable call is available.")
    {
        Evidence = Array.AsReadOnly(evidence.ToArray());
        Rejections = Array.AsReadOnly(rejections.ToArray());
    }
}
