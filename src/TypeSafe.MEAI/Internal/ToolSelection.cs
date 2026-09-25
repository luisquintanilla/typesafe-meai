using System.Text.Json;
using Microsoft.Extensions.AI;

namespace TypeSafe.MEAI;

internal static class ToolSelection
{
    internal const string NoMatch = "__typesafe_no_match__";

    internal static async Task<ChatResponse> RunAsync(TypeSafeChatClient client, TypeSafeState state, ChatOptions options,
        string? instructions, TypeSafeClientOptions settings, CancellationToken cancellationToken)
    {
        if (options.AllowMultipleToolCalls == true) throw new NotSupportedException("Only one selected call is supported.");
        if (options.ToolMode is not (null or AutoChatToolMode or RequiredChatToolMode or NoneChatToolMode))
            throw new NotSupportedException("Unsupported tool mode.");
        if (options.Tools!.Count is < 1 or > 254 || options.Tools.Any(t => t is not AIFunctionDeclaration))
            throw new NotSupportedException("Specify 1-254 uniquely named function declarations.");
        var candidates = options.Tools.Cast<AIFunctionDeclaration>().Select(function =>
        {
            string name = function.Name, description = function.Description;
            if (string.IsNullOrWhiteSpace(name) || name == NoMatch || string.IsNullOrWhiteSpace(description))
                throw new ArgumentException("Function names/descriptions must be nonempty; the no-match label is reserved.");
            var config = settings.Tools.GetValueOrDefault(name) ?? new();
            var schema = function.JsonSchema.Clone();
            var plan = TypeSafeSchemaCompiler.CompileCore(schema, config.Fields,
                $"Task: {instructions}\nInfer arguments only for '{name}': {description}", config.FixedValues, config.OmitDefaultParameters, settings.Policy);
            if (!plan.IsObjectRoot || plan.Fields.Any(f => f.Question is ScoreQuestion))
                throw new TypeSafeSchemaException("", "Tools require finite object arguments; infer only Boolean/enum values, fixing other scalars explicitly.");
            return new Candidate(function, name, description, schema, plan, config);
        }).ToArray();
        if (candidates.Select(c => c.Name).Distinct(StringComparer.Ordinal).Count() != candidates.Length)
            throw new ArgumentException("Function names must be unique.");
        if (options.ToolMode is NoneChatToolMode) return TypeSafeChatClient.Response([], []);
        string? specific = (options.ToolMode as RequiredChatToolMode)?.RequiredFunctionName;
        Candidate? selected = null;
        List<JevResultContent> evidence = [];
        if (specific is not null)
            selected = candidates.SingleOrDefault(c => c.Name == specific) ?? throw new ArgumentException("Required function is not among the supplied tools.");
        else
        {
            double threshold = settings.Policy?.MinimumConfidence ?? throw new ArgumentException("Tool selection requires an explicit client confidence policy.");
            bool required = options.ToolMode is RequiredChatToolMode;
            var labels = candidates.Select(c => KeyValuePair.Create(c.Name, TypeSafeJson.Entry(c.Description)));
            if (!required) labels = labels.Append(KeyValuePair.Create(NoMatch, TypeSafeJson.Entry("None of these functions is appropriate.")));
            var question = new ChoiceQuestion("tool", TypeSafeJson.Entry($"Task: {instructions}\nSelect one eligible host function{(required ? "." : ", or no-match.")} Selection is not authorization."), labels);
            var batch = await client.BatchAsync(state, [question], options, "tool-selection", cancellationToken).ConfigureAwait(false);
            evidence.Add(batch);
            var answer = (ChoiceAnswer)batch.Answers["tool"];
            if (answer.Confidence < threshold)
                throw new TypeSafeAbstentionException(evidence, [new("", "Tool-selection confidence is below the explicit threshold.")]);
            if (answer.Choice == NoMatch) return TypeSafeChatClient.Response(evidence.Cast<AIContent>().ToList(), evidence);
            selected = candidates.Single(c => c.Name == answer.Choice);
        }
        selected.EnsureUnchanged();
        JevResultContent? arguments = null;
        if (selected.Plan.Questions.Count != 0)
        {
            arguments = await client.BatchAsync(state, selected.Plan.Questions, options, "tool-arguments", cancellationToken).ConfigureAwait(false);
            arguments.AdditionalProperties!["typesafe.v1.fields"] = JsonSerializer.SerializeToElement(selected.Plan.Fields.ToDictionary(f => f.Question.Id, f => f.Pointer));
            evidence.Add(arguments);
        }
        var projection = selected.Plan.Project(arguments?.Native);
        if (projection.Rejections.Count != 0) throw new TypeSafeAbstentionException(evidence, projection.Rejections);
        if (selected.Config.Validate?.Invoke(projection.Value!.Value) is { } reason)
            throw new TypeSafeAbstentionException(evidence, [new("", reason)]);
        cancellationToken.ThrowIfCancellationRequested();
        selected.EnsureUnchanged();
        var contents = evidence.Cast<AIContent>().ToList();
        contents.Add(new FunctionCallContent(Guid.NewGuid().ToString("N"), selected.Name,
            projection.Value!.Value.EnumerateObject().ToDictionary(p => p.Name, p => (object?)p.Value.Clone(), StringComparer.Ordinal)));
        return TypeSafeChatClient.Response(contents, evidence, toolCall: true);
    }

    private sealed record Candidate(AIFunctionDeclaration Function, string Name, string Description, JsonElement Schema,
        TypeSafeCompiledSchema Plan, TypeSafeToolOptions Config)
    {
        internal void EnsureUnchanged()
        {
            if (Function.Name != Name || Function.Description != Description || !JsonElement.DeepEquals(Function.JsonSchema, Schema))
                throw new InvalidOperationException("The host function declaration changed during selection.");
        }
    }
}
