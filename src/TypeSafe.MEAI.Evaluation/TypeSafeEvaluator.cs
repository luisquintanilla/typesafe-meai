using System.Diagnostics;
using System.Text.Json;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.AI.Evaluation;

namespace TypeSafe.MEAI.Evaluation;

/// <summary>Batches all configured native metric questions through the configured decorated judge.</summary>
public sealed class TypeSafeEvaluator : IEvaluator
{
    private readonly TypeSafeMetricDefinition[] _definitions;
    /// <inheritdoc/>
    public IReadOnlyCollection<string> EvaluationMetricNames { get; }

    /// <summary>Creates an evaluator with distinct metric names and native question IDs.</summary>
    public TypeSafeEvaluator(IEnumerable<TypeSafeMetricDefinition> definitions)
    {
        _definitions = definitions.ToArray();
        if (_definitions.Length == 0 ||
            _definitions.Select(d => d.Name).Distinct(StringComparer.Ordinal).Count() != _definitions.Length ||
            _definitions.Select(d => d.Question.Id).Distinct(StringComparer.Ordinal).Count() != _definitions.Length)
            throw new ArgumentException("Metrics and question IDs must be nonempty and distinct.");
        EvaluationMetricNames = Array.AsReadOnly(_definitions.Select(d => d.Name).ToArray());
    }

    /// <inheritdoc/>
    public async ValueTask<EvaluationResult> EvaluateAsync(IEnumerable<ChatMessage> messages, ChatResponse modelResponse,
        ChatConfiguration? chatConfiguration = null, IEnumerable<EvaluationContext>? additionalContext = null, CancellationToken cancellationToken = default)
    {
        if (chatConfiguration?.ChatClient is not { } judge)
            throw new ArgumentException("A decorated judge ChatConfiguration.ChatClient is required.", nameof(chatConfiguration));
        cancellationToken.ThrowIfCancellationRequested();
        var conversation = messages.Select(SnapshotMessage).ToArray();
        var response = modelResponse.Messages.Select(SnapshotMessage).ToArray();
        EvaluationContext[] contexts = (additionalContext ?? []).Select(c => new SnapshotContext(c.Name, c.Contents.Select(SnapshotContent).ToArray())).ToArray();
        if (contexts.Select(c => c.Name).Distinct(StringComparer.Ordinal).Count() != contexts.Length)
            throw new ArgumentException("Evaluation context names must be distinct so reporting cannot overwrite evidence.");
        var state = TypeSafeState.FromJson(JsonSerializer.SerializeToElement(new
        {
            conversation = conversation.Select(ProjectMessage),
            response = response.Select(ProjectMessage),
            responseModel = modelResponse.ModelId,
            context = contexts.Select(c => new { name = c.Name, contents = c.Contents.Select(ProjectContent).ToArray() })
        }));
        var questions = _definitions.Select(d => d.Question).ToArray();
        long start = Stopwatch.GetTimestamp();
        var chat = await judge.GetResponseAsync([state.ToMessage()], ((ChatOptions?)null).WithTypeSafeQuestions(questions), cancellationToken).ConfigureAwait(false);
        TimeSpan duration = Stopwatch.GetElapsedTime(start);
        var native = chat.GetTypeSafeResponse(questions);
        string batch = Guid.NewGuid().ToString("N");
        List<EvaluationMetric> metrics = [];
        foreach (var definition in _definitions)
        {
            TypeSafeAnswer answer = native.Answers[definition.Question.Id];
            EvaluationMetric metric = answer switch
            {
                NoulAnswer n => new NumericMetric(definition.Name, n.Noul),
                ScoreAnswer s => new NumericMetric(definition.Name, s.Score),
                ChoiceAnswer c => new StringMetric(definition.Name, c.Choice),
                _ => throw new TypeSafeProtocolException("Unknown evaluation answer variant.")
            };
            metric.Metadata ??= new Dictionary<string, string>();
            metric.Metadata["typesafe.v1.answer"] = answer.Json.GetRawText();
            metric.Metadata["typesafe.v1.rubric"] = definition.Question.ToJson().GetRawText();
            metric.Metadata["typesafe.v1.batch"] = batch;
            metric.Metadata["typesafe.v1.envelope"] = native.Json.GetRawText();
            metric.AddOrUpdateChatMetadata(chat, duration);
            metric.AddOrUpdateContext(contexts);
            if (definition.Interpret is not null) metric.Interpretation = definition.Interpret(answer);
            metrics.Add(metric);
        }
        return new EvaluationResult(metrics);
    }

    private static ChatMessage SnapshotMessage(ChatMessage message) =>
        new(message.Role, message.Contents.Select(SnapshotContent).ToArray()) { AuthorName = message.AuthorName };

    private static AIContent SnapshotContent(AIContent content) => content switch
    {
        TextContent text => new TextContent(text.Text),
        DataContent data when string.Equals(data.MediaType, "application/json", StringComparison.OrdinalIgnoreCase) =>
            TypeSafeState.Parse(new System.Text.UTF8Encoding(false, true).GetString(data.Data.Span)).ToMessage().Contents[0],
        _ => throw new NotSupportedException("Evaluation state supports text and application/json content only. Supply explicit supported context instead.")
    };

    private sealed class SnapshotContext(string name, AIContent[] contents) : EvaluationContext(name, contents);

    private static JsonElement ProjectMessage(ChatMessage message) => JsonSerializer.SerializeToElement(new
    {
        role = message.Role.Value,
        author = message.AuthorName,
        contents = message.Contents.Select(ProjectContent).ToArray()
    });

    private static JsonElement ProjectContent(AIContent content)
    {
        if (content is TextContent text)
            return JsonSerializer.SerializeToElement(new { type = "text", value = text.Text });
        if (content is DataContent data)
        {
            using var doc = JsonDocument.Parse(data.Data);
            return JsonSerializer.SerializeToElement(new { type = "json", value = doc.RootElement });
        }
        throw new NotSupportedException("Evaluation state supports text and application/json content only.");
    }
}
