using System.Collections.ObjectModel;
using System.Text.Json;
using Microsoft.Extensions.AI;

namespace TypeSafe.MEAI;

/// <summary>One snapshotted host-registered function and its bounded argument plan.</summary>
public sealed class TypeSafeToolDefinition
{
    internal AIFunction Function { get; }
    internal string Name { get; }
    internal string Description { get; }
    internal JsonElement Schema { get; }
    internal TypeSafeCompiledSchema Plan { get; }
    internal Func<JsonElement, string?>? Validate { get; }

    /// <summary>Compiles MEAI's existing parameter schema; no independent reflection.
    /// Only bool/enum parameters may be inferred. Optional scalar parameters require explicit fixed/default-omission strategy.</summary>
    public TypeSafeToolDefinition(AIFunction function, IReadOnlyDictionary<string, TypeSafeFieldBinding>? bindings = null,
        IReadOnlyDictionary<string, JsonElement>? fixedValues = null, IEnumerable<string>? omitDefaultParameters = null,
        Func<JsonElement, string?>? validate = null)
    {
        Function = function ?? throw new ArgumentNullException(nameof(function));
        Name = function.Name;
        Description = function.Description;
        if (string.IsNullOrWhiteSpace(Name) || Name == TypeSafeToolSelector.NoMatch || string.IsNullOrWhiteSpace(Description))
            throw new ArgumentException("Tool needs a nonempty name/description; the no-match name is reserved.", nameof(function));
        Schema = function.JsonSchema.Clone();
        var ownedFixed = new ReadOnlyDictionary<string, JsonElement>((fixedValues ?? new Dictionary<string, JsonElement>())
            .ToDictionary(p => p.Key, p => p.Value.Clone(), StringComparer.Ordinal));
        Plan = TypeSafeSchemaCompiler.CompileCore(Schema, bindings ?? new Dictionary<string, TypeSafeFieldBinding>(),
            $"Infer arguments only for the selected host function '{Name}': {Description}", ownedFixed,
            (omitDefaultParameters ?? []).ToHashSet(StringComparer.Ordinal));
        if (!Plan.IsObjectRoot) throw new TypeSafeSchemaException("", "Function parameters must resolve to a closed object.");
        if (Plan.Fields.Any(f => f.Question is ScoreQuestion))
            throw new TypeSafeSchemaException("", "Tool argument inference supports Boolean/string-enum only; supply other scalars as fixed values.");
        Validate = validate;
    }

    internal void EnsureUnchanged()
    {
        if (Function.Name != Name || Function.Description != Description || !JsonElement.DeepEquals(Function.JsonSchema, Schema))
            throw new InvalidOperationException("Registered function declaration changed after plan creation.");
    }
}

/// <summary>Explicit non-executable outcomes are distinct from accepted selection.</summary>
public enum TypeSafeToolSelectionStatus
{
    /// <summary>Exactly one registered call is ready for separate host authorization.</summary>
    Accepted,
    /// <summary>The explicit no-match label was chosen.</summary>
    NoMatch,
    /// <summary>Selection/arguments failed an explicit acceptance policy.</summary>
    Abstained
}

/// <summary>One read-only selection result. Model output never authorizes execution.</summary>
public sealed class TypeSafeToolSelection
{
    internal TypeSafeToolDefinition? Definition { get; }
    internal JsonElement Arguments { get; }
    internal string? CorrelationId { get; }
    /// <summary>Accepted, no-match, or abstained.</summary>
    public TypeSafeToolSelectionStatus Status { get; }
    /// <summary>Original allowlisted function, including all wrappers; absent for no-match.</summary>
    public AIFunction? Function => Definition?.Function;
    /// <summary>At most one proposed call. Invocation rejects any subsequent mutation.</summary>
    public FunctionCallContent? FunctionCall { get; }
    /// <summary>Complete selection and optional argument-batch evidence, in request order.</summary>
    public IReadOnlyList<TypeSafeResponse> Evidence { get; }
    /// <summary>Deterministic policy rejection reasons.</summary>
    public IReadOnlyList<TypeSafeRejection> Rejections { get; }

    internal TypeSafeToolSelection(TypeSafeToolSelectionStatus status, TypeSafeToolDefinition? definition,
        JsonElement? arguments, List<TypeSafeResponse> evidence, IReadOnlyList<TypeSafeRejection>? rejections = null)
    {
        Status = status;
        Definition = definition;
        Arguments = arguments?.Clone() ?? default;
        Evidence = evidence.AsReadOnly();
        Rejections = Array.AsReadOnly((rejections ?? []).ToArray());
        if (status == TypeSafeToolSelectionStatus.Accepted)
        {
            CorrelationId = Guid.NewGuid().ToString("N");
            FunctionCall = new(CorrelationId, definition!.Name, CopyArguments());
        }
    }

    internal Dictionary<string, object?> CopyArguments() =>
        Arguments.EnumerateObject().ToDictionary(p => p.Name, p => (object?)p.Value.Clone(), StringComparer.Ordinal);

    internal void EnsureExecutable()
    {
        if (Status != TypeSafeToolSelectionStatus.Accepted || Definition is null || FunctionCall is null)
            throw new InvalidOperationException("Only an accepted registered selection can be invoked.");
        Definition.EnsureUnchanged();
        if (FunctionCall.Name != Definition.Name || FunctionCall.CallId != CorrelationId || FunctionCall.InformationalOnly ||
            !JsonElement.DeepEquals(JsonSerializer.SerializeToElement(FunctionCall.Arguments), Arguments))
            throw new InvalidOperationException("Selected function/argument binding was changed.");
        if (Definition.Validate?.Invoke(Arguments.Clone()) is { } reason)
            throw new InvalidOperationException($"Arguments failed host validation: {reason}");
    }
}

/// <summary>Selects first, then infers only the selected function's arguments in zero/one further batch. Never invokes functions.</summary>
public sealed class TypeSafeToolSelector
{
    /// <summary>Reserved Choice label, consuming one of the 255 options.</summary>
    public const string NoMatch = "__typesafe_no_match__";
    private readonly IChatClient _client;
    private readonly Dictionary<string, TypeSafeToolDefinition> _tools;
    private readonly ChoiceQuestion _question;
    private readonly double _minimumConfidence;

    /// <summary>Snapshots a reusable allowlist and explicit selection-confidence policy. The decorated native client is borrowed.</summary>
    public TypeSafeToolSelector(IChatClient decoratedClient, IEnumerable<TypeSafeToolDefinition> tools, double minimumConfidence)
    {
        _client = decoratedClient ?? throw new ArgumentNullException(nameof(decoratedClient));
        var configured = tools.ToArray();
        if (configured.Length is < 1 or > 254 || configured.Any(t => t is null) ||
            configured.Select(t => t.Name).Distinct(StringComparer.Ordinal).Count() != configured.Length)
            throw new ArgumentException("Specify 1-254 uniquely named registered functions.", nameof(tools));
        if (!double.IsFinite(minimumConfidence) || minimumConfidence < 0 || minimumConfidence > 1)
            throw new ArgumentOutOfRangeException(nameof(minimumConfidence));
        _minimumConfidence = minimumConfidence;
        _tools = configured.ToDictionary(t => t.Name, StringComparer.Ordinal);
        _question = new ChoiceQuestion("tool", TypeSafeJson.Entry("Select one eligible host function, or no-match. Selection is not authorization."),
            configured.Select(t => KeyValuePair.Create(t.Name, TypeSafeJson.Entry(t.Description)))
                .Append(KeyValuePair.Create(NoMatch, TypeSafeJson.Entry("None of these functions is appropriate."))));
    }

    /// <summary>At most two logical native calls, through the supplied decorated IChatClient; all function invocation counts remain zero.</summary>
    public async Task<TypeSafeToolSelection> SelectAsync(TypeSafeState state, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        foreach (var tool in _tools.Values) tool.EnsureUnchanged();
        // Snapshot options once so the second logical batch cannot observe intervening caller mutations.
        var snapshot = NativeRequest.SnapshotOptions(options);
        var selection = await _client.GetTypeSafeResponseAsync(state, [_question], snapshot, cancellationToken).ConfigureAwait(false);
        List<TypeSafeResponse> evidence = [selection];
        var answer = (ChoiceAnswer)selection.Answers["tool"];
        if (answer.Confidence < _minimumConfidence)
            return new(TypeSafeToolSelectionStatus.Abstained, null, null, evidence, [new("", "Tool-selection confidence is below the explicit threshold.")]);
        if (answer.Choice == NoMatch) return new(TypeSafeToolSelectionStatus.NoMatch, null, null, evidence);
        var selected = _tools[answer.Choice];
        selected.EnsureUnchanged();
        TypeSafeResponse? arguments = null;
        if (selected.Plan.Questions.Count != 0)
        {
            arguments = await _client.GetTypeSafeResponseAsync(state, selected.Plan.Questions, snapshot, cancellationToken).ConfigureAwait(false);
            evidence.Add(arguments);
        }
        var projected = selected.Plan.Project(arguments);
        if (projected.Rejections.Count != 0)
            return new(TypeSafeToolSelectionStatus.Abstained, selected, null, evidence, projected.Rejections);
        if (selected.Validate?.Invoke(projected.Value!.Value.Clone()) is { } reason)
            return new(TypeSafeToolSelectionStatus.Abstained, selected, null, evidence, [new("", reason)]);
        cancellationToken.ThrowIfCancellationRequested();
        return new(TypeSafeToolSelectionStatus.Accepted, selected, projected.Value, evidence);
    }
}
