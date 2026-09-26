using System.Collections.ObjectModel;
using System.Text.Json;

namespace TypeSafe.MEAI;

/// <summary>Optional host configuration for one ordinary ChatOptions.Tools declaration; not a function registry or executor.</summary>
public sealed class TypeSafeToolOptions
{
    /// <summary>Optional argument-pointer overrides of the client policy and schema descriptions.</summary>
    public IReadOnlyDictionary<string, TypeSafeFieldOptions> Fields { get; }
    /// <summary>Owned host-fixed scalar arguments keyed by escaped pointer.</summary>
    public IReadOnlyDictionary<string, JsonElement> FixedValues { get; }
    /// <summary>Optional scalar pointers explicitly omitted in favor of schema-declared defaults.</summary>
    public IReadOnlySet<string> OmitDefaultParameters { get; }
    /// <summary>Optional deterministic pre-emission validation; this does not authorize execution.</summary>
    public Func<JsonElement, string?>? Validate { get; }
    /// <summary>Snapshots the host configuration independently of caller-owned dictionaries/JSON documents.</summary>
    public TypeSafeToolOptions(IReadOnlyDictionary<string, TypeSafeFieldOptions>? fields = null,
        IReadOnlyDictionary<string, JsonElement>? fixedValues = null, IEnumerable<string>? omitDefaultParameters = null,
        Func<JsonElement, string?>? validate = null)
    {
        Fields = new ReadOnlyDictionary<string, TypeSafeFieldOptions>(new Dictionary<string, TypeSafeFieldOptions>(fields ?? new Dictionary<string, TypeSafeFieldOptions>(), StringComparer.Ordinal));
        FixedValues = new ReadOnlyDictionary<string, JsonElement>((fixedValues ?? new Dictionary<string, JsonElement>()).ToDictionary(p => p.Key, p => p.Value.Clone(), StringComparer.Ordinal));
        OmitDefaultParameters = System.Collections.Frozen.FrozenSet.ToFrozenSet(omitDefaultParameters ?? [], StringComparer.Ordinal);
        Validate = validate;
    }
}
