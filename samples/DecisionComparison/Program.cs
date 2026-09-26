using System.Diagnostics;
using System.Numerics.Tensors;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using DecisionInference;
using DecisionInference.Laya;
using DecisionInference.TensorSharp;
using DecisionInference.TypeSafe;
using Microsoft.Extensions.AI;
using TypeSafe.MEAI;

Console.WriteLine($"{RuntimeInformation.OSDescription}; {RuntimeInformation.ProcessArchitecture}; {RuntimeInformation.FrameworkDescription}");
using var deadline = new CancellationTokenSource(TimeSpan.FromMinutes(10));
string mode = args.FirstOrDefault() ?? "compare";
switch (mode)
{
    case "compare":
        await CompareApis(deadline.Token);
        break;
    case "provider" when args.Length == 2 && args[1] == "jev":
        using (var http = new HttpClient(new RecordedJevHandler()))
        using (var client = new TypeSafeDecisionClient(http, "offline-fixture"))
            Print(await client.GetResponseAsync(RecordedJevHandler.Request(), cancellationToken: deadline.Token));
        break;
    case "provider" when args.Length == 5 && args[1] == "laya":
        var layaClock = Stopwatch.StartNew();
        using (var client = new LayaDecisionClient(args[2], args[3], args[4]))
        {
            Console.WriteLine($"Laya load: {layaClock.Elapsed.TotalMilliseconds:F1} ms; revision={LayaProfile.Revision}");
            await Measure("Laya first call", () => client.GetResponseAsync(LocalRequest(), cancellationToken: deadline.Token));
            Print(await Measure("Laya warm", () => client.GetResponseAsync(LocalRequest(), cancellationToken: deadline.Token)));
        }
        break;
    case "stages" when args.Length is 4 or 5:
        var profile = LayaProfile.Load(args[3]);
        var (tokenizer, tokenMetadata) = LayaTokenizer.Load(args[2]);
        var prepare = new PrepareDecisionInputs(profile, tokenizer, tokenMetadata);
        var decode = new DecodeDecisions(profile);
        using (var score = ScoreOnnxDecisionModel.Load(args[1]))
        using (var client = new LayaDecisionClient(prepare, score, decode))
        {
            var request = LocalRequest();
            var batch = prepare.Prepare(request, deadline.Token);
            var raw = score.Score(batch, deadline.Token);
            var direct = decode.Decode(batch, raw);
            var viaClient = await client.GetResponseAsync(request, cancellationToken: deadline.Token);
            EqualAnswers(direct, viaClient, 1e-7);
            Print(direct);
            Console.WriteLine($"Public-stage/client parity; two native runs; B={batch.BatchSize}, L={batch.SequenceLength}, K={batch.MarkerWidth}");
            if (args.Length == 5)
                File.WriteAllText(args[4], JsonSerializer.Serialize(new
                {
                    revision = LayaProfile.Revision, batch = batch.BatchSize, length = batch.SequenceLength, width = batch.MarkerWidth,
                    input_ids = Copy(batch.InputIds), attention_mask = Copy(batch.AttentionMask),
                    marker_pos = Copy(batch.MarkerPositions), marker_mask = Copy(batch.MarkerMask), qtype = Copy(batch.QuestionTypes),
                    logits = raw.Logits.ToArray(), act_probs = raw.ActionProbabilities.ToArray(),
                    evidence = direct.Answers.Select(a => ExportEvidence(a.RawRepresentation)).ToArray(),
                    probabilities = direct.Answers.Select(a => a switch
                    {
                        BinaryDecisionAnswer b => new[] { 1 - b.TrueProbability, b.TrueProbability },
                        ChoiceDecisionAnswer c => c.Probabilities.Values.ToArray(),
                        ScoreDecisionAnswer s => s.Probabilities.ToArray(),
                        _ => throw new NotSupportedException()
                    }).ToArray(),
                    expectedIndex = direct.GetAnswer<ScoreDecisionAnswer>("severity").ExpectedIndex
                }));
        }
        break;
    case "provider" when args.Length == 3 && args[1] == "qwen":
        // Process-local initialization belongs to this executable, never the provider library.
        Environment.SetEnvironmentVariable("MAX_CONTEXT", "512");
        var qwenClock = Stopwatch.StartNew();
        using (var client = QwenDecisionClient.Load(args[2]))
        {
            Console.WriteLine($"Qwen verified hash={QwenDecisionClient.ModelSha256}; load={qwenClock.Elapsed.TotalMilliseconds:F1} ms");
            var metadata = client.GetRequiredService<DecisionClientMetadata>();
            if (metadata.ProviderName != "tensorsharp-qwen" || metadata.ProviderUri is not null ||
                metadata.DefaultModelId != QwenDecisionClient.PinnedModelId ||
                metadata.SupportedKinds is not { Count: 1 } || metadata.SupportedKinds[0] != DecisionKind.Choice)
                throw new InvalidOperationException("Qwen metadata does not describe the loaded local Choice model.");
            var fixedOptions = new DecisionOptions { ModelId = metadata.DefaultModelId };
            var first = await Measure("Qwen first forward", () => client.GetResponseAsync(ChoiceRequest(), fixedOptions, deadline.Token));
            if (first.GetAnswer("category").RawRepresentation is not QwenRawAnswer
                { Provider: "tensorsharp-qwen", ConditionalOnAnswerTokens: true, Aliases: "ABC", Logits.Count: 3, TokenIds.Count: 3 })
                throw new InvalidOperationException("Qwen answer-token evidence is missing or inconsistent.");
            string ownedEvidence = ExportEvidence(first.GetAnswer("category").RawRepresentation).GetRawText();
            if (first.RawRepresentation is not QwenRawResponse { ForwardCalls: 1, ContextLength: 512 })
                throw new InvalidOperationException("Qwen forward/context evidence is missing or inconsistent.");
            Print(first);
            await client.GetResponseAsync(ChoiceRequest("The team lunch is at noon in the usual meeting room."), cancellationToken: deadline.Token);
            var repeated = await Measure("Qwen repeated warm forward", () => client.GetResponseAsync(ChoiceRequest(), cancellationToken: deadline.Token));
            EqualAnswers(first, repeated, 1e-7);
            foreach (var invalidModel in new[] { "", " ", QwenDecisionClient.PinnedModelId.ToLowerInvariant() })
            {
                try
                {
                    await client.GetResponseAsync(ChoiceRequest(), new() { ModelId = invalidModel }, deadline.Token);
                    throw new InvalidOperationException("Qwen accepted an invalid or different loaded model.");
                }
                catch (ArgumentException) when (string.IsNullOrWhiteSpace(invalidModel)) { }
                catch (NotSupportedException) when (!string.IsNullOrWhiteSpace(invalidModel)) { }
            }
            foreach (var unsupported in new DecisionQuestion[]
                { new BinaryDecisionQuestion("unsupported"), new ScoreDecisionQuestion("unsupported", null, ["low", "high"]) })
            {
                var mixed = new DecisionRequest(ChoiceRequest().State, [ChoiceRequest().Questions[0], unsupported]);
                try
                {
                    await client.GetResponseAsync(mixed, cancellationToken: deadline.Token);
                    throw new InvalidOperationException("Qwen accepted an unsupported mixed batch.");
                }
                catch (NotSupportedException) { }
            }
            if (metadata.DefaultModelId != QwenDecisionClient.PinnedModelId || fixedOptions.ModelId != metadata.DefaultModelId ||
                first.ModelId != metadata.DefaultModelId)
                throw new InvalidOperationException("Qwen model metadata or caller options changed.");
            Console.WriteLine("Local metadata, fixed-model override, invalid models and Binary/Score rejection verified.");
            var afterRejection = await client.GetResponseAsync(ChoiceRequest(), cancellationToken: deadline.Token);
            EqualAnswers(first, afterRejection, 1e-7);
            if (ExportEvidence(first.GetAnswer("category").RawRepresentation).GetRawText() != ownedEvidence ||
                ExportEvidence(repeated.GetAnswer("category").RawRepresentation).GetRawText() != ownedEvidence)
                throw new InvalidOperationException("Owned answer-token evidence changed across model reset/reuse.");
            Console.WriteLine("One model load, four forwards, A/B/A repeated-request isolation; original owned logits unchanged.");
            var result = first.GetAnswer<ChoiceDecisionAnswer>("category");
            if (result.SelectedCandidateId != "phishing")
                throw new InvalidOperationException("Pinned example no longer selects Phishing.");
            double[] reference = [0.0659, 0.0270, 0.9071];
            if (result.Probabilities.Values.Where((p, i) => Math.Abs(p - reference[i]) > 0.01).Any())
                throw new InvalidOperationException("Pinned sample probabilities differ by more than 0.01 absolute.");
            Console.WriteLine("Pinned published sample: category and probabilities within 0.01 absolute tolerance (not a calibration test).");
        }
        break;
    case "inventory":
        foreach (var type in new[] { typeof(IDecisionClient), typeof(TypeSafeDecisionClient), typeof(LayaDecisionClient), typeof(QwenDecisionClient), typeof(TypeSafeChatClient) })
        {
            var assembly = type.Assembly;
            Console.WriteLine($"{assembly.GetName().Name}: exportedTypes={assembly.GetExportedTypes().Length}, " +
                $"declaredPublicMembers={assembly.GetExportedTypes().Sum(t => t.GetMembers(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly).Length)}");
        }
        break;
    default:
        throw new ArgumentException("Modes: compare | inventory | provider jev | provider laya <onnx> <tokenizer.json> <laya_config.json> | stages <onnx> <tokenizer.json> <laya_config.json> [receipt.json] | provider qwen <gguf>");
}

static async Task CompareApis(CancellationToken ct)
{
    var request = RecordedJevHandler.Request();
    using var decisionHandler = new RecordedJevHandler();
    using var chatHandler = new RecordedJevHandler();
    using var decisionHttp = new HttpClient(decisionHandler);
    using var chatHttp = new HttpClient(chatHandler);
    using var decision = new TypeSafeDecisionClient(decisionHttp, "offline-fixture");
    var metadata = decision.GetRequiredService<DecisionClientMetadata>();
    var decisionOptions = new DecisionOptions { ModelId = metadata.DefaultModelId }.Clone();
    if (metadata.ProviderUri != new Uri("https://api.typesafe.ai"))
        throw new InvalidOperationException("Configured Jev provider metadata differs.");
    using IChatClient chat = new TypeSafeChatClient(chatHttp, new() { ModelId = "jev-1.13.0" }, "offline-fixture");
    var options = new ChatOptions
    {
        AdditionalProperties = new()
        {
            ["typesafe.v1.questions"] = JsonSerializer.SerializeToElement(new
            {
                safe = new { type = "noul", instructions = "Is it safe?" },
                category = new { type = "choice", instructions = "Choose one category.", criteria = new { legitimate = "Legitimate", phishing = "Phishing" } },
                severity = new { type = "score", instructions = "Assess severity.", criteria = new[] { "low", "medium", "high" } }
            })
        }
    };
    var messages = new[] { new ChatMessage(ChatRole.User, request.State.GetString()) };
    // One uncached warm-up each; the fixture records the actual outgoing bodies, not reconstructed equivalents.
    await decision.GetResponseAsync(request.State, request.Questions, decisionOptions, ct);
    await chat.GetResponseAsync(messages, options, ct);
    var newResult = await Measure("Decision API (synthetic HTTP)", () => decision.GetResponseAsync(
        request.State.GetString()!, request.Questions, ComparisonJsonContext.Default.String, decisionOptions, ct));
    var oldResult = await Measure("IChatClient (synthetic HTTP)", () => chat.GetResponseAsync(messages, options, ct));
    if (!JsonElement.DeepEquals(decisionHandler.LastBody, chatHandler.LastBody))
        throw new InvalidOperationException("The native wire payloads differ.");
    if (oldResult.ModelId != newResult.ModelId ||
        oldResult.Usage?.InputTokenCount != newResult.Usage?.InputTokenCount ||
        oldResult.Usage?.OutputTokenCount != newResult.Usage?.OutputTokenCount)
        throw new InvalidOperationException("Model/usage differ.");
    var oldJson = JsonDocument.Parse(oldResult.Messages.Single().Contents.OfType<JevResultContent>().Single().Data);
    using (oldJson)
    {
        if (newResult.RawRepresentation is not JsonElement native || !JsonElement.DeepEquals(oldJson.RootElement, native))
            throw new InvalidOperationException("Owned native evidence differs.");
    }
    Print(newResult);
    Console.WriteLine($"Exact outgoing payload and complete native-evidence parity; sends decision={decisionHandler.Calls}, chat={chatHandler.Calls}; no caches.");
    Console.WriteLine("This compares API overhead, NOT live Jev latency or accuracy. Chat still provides schema projection, tools and MEAI middleware; the decision API does not.");
}

static DecisionRequest ChoiceRequest(string email = "Payroll asks for your password on a non-company sign-in page.") =>
    new(JsonSerializer.SerializeToElement("Email: " + email), [new ChoiceDecisionQuestion("category", "Choose one option.",
        [new("legitimate", "Legitimate"), new("spam", "Spam"), new("phishing", "Phishing")])]);

static DecisionRequest LocalRequest()
{
    var choice = ChoiceRequest();
    return new(choice.State, [choice.Questions[0], new BinaryDecisionQuestion("suspicious", "Does the email request a password?"),
        new ScoreDecisionQuestion("severity", "Assess the risk in the email.", ["low", "medium", "high"])]);
}

static async Task<T> Measure<T>(string name, Func<Task<T>> action)
{
    long before = GC.GetTotalAllocatedBytes(precise: true);
    var clock = Stopwatch.StartNew();
    var result = await action();
    Console.WriteLine($"{name}: {clock.Elapsed.TotalMilliseconds:F3} ms, process-wide allocated delta={GC.GetTotalAllocatedBytes(precise: true) - before} bytes; one observed call, not a benchmark");
    return result;
}

static void EqualAnswers(DecisionResponse first, DecisionResponse second, double tolerance)
{
    if (first.ModelId != second.ModelId || first.Answers.Count != second.Answers.Count) throw new InvalidOperationException("Response identity changed.");
    for (int i = 0; i < first.Answers.Count; i++)
    {
        var a = first.Answers[i]; var b = second.Answers[i];
        if (a.QuestionId != b.QuestionId || a.Kind != b.Kind) throw new InvalidOperationException("Answer correlation changed.");
        double[] left = Values(a), right = Values(b);
        if (left.Length != right.Length || left.Where((p, j) => Math.Abs(p - right[j]) > tolerance).Any())
            throw new InvalidOperationException("Repeated or composed probabilities differ.");
        if (a is ChoiceDecisionAnswer ac && b is ChoiceDecisionAnswer bc &&
            (ac.SelectedCandidateId != bc.SelectedCandidateId || !ac.Probabilities.Keys.SequenceEqual(bc.Probabilities.Keys)))
            throw new InvalidOperationException("Choice identity changed.");
        if (!JsonElement.DeepEquals(ExportEvidence(a.RawRepresentation), ExportEvidence(b.RawRepresentation)))
            throw new InvalidOperationException("Owned provider raw values changed.");
    }
    if ((first.RawRepresentation is not null || second.RawRepresentation is not null) &&
        !JsonElement.DeepEquals(ExportEvidence(first.RawRepresentation), ExportEvidence(second.RawRepresentation)))
        throw new InvalidOperationException("Response raw values changed.");
    static double[] Values(DecisionAnswer a) => a switch
    {
        BinaryDecisionAnswer b => [b.TrueProbability], ChoiceDecisionAnswer c => c.Probabilities.Values.ToArray(),
        ScoreDecisionAnswer s => [.. s.Probabilities, s.ExpectedIndex], _ => throw new NotSupportedException()
    };
}

static void Print(DecisionResponse result)
{
    Console.WriteLine($"model={result.ModelId}; inputTokens={result.Usage?.InputTokenCount}; outputTokens={result.Usage?.OutputTokenCount?.ToString() ?? "not reported"}");
    foreach (var answer in result.Answers)
        Console.WriteLine(answer switch
        {
            BinaryDecisionAnswer b => $"{b.QuestionId}: trueProbability={b.TrueProbability:G6}",
            ChoiceDecisionAnswer c => $"{c.QuestionId}: selected={c.SelectedCandidateId}; " + string.Join(", ", c.Probabilities.Select(p => $"{p.Key}={p.Value:G6}")),
            ScoreDecisionAnswer s => $"{s.QuestionId}: expectedIndex={s.ExpectedIndex:G6}; [{string.Join(", ", s.Probabilities)}]",
            _ => throw new NotSupportedException()
        });
    Console.WriteLine("Model/decoder evidence, not calibrated correctness; no acceptance thresholds applied.");
}

static T[] Copy<T>(ReadOnlyTensorSpan<T> tensor)
{
    var copy = new T[checked((int)tensor.FlattenedLength)];
    tensor.FlattenTo(copy);
    return copy;
}

static JsonElement ExportEvidence(object? raw) => raw switch
{
    JsonElement json => json.Clone(),
    LayaRawAnswer laya => JsonSerializer.SerializeToElement(laya, ComparisonJsonContext.Default.LayaRawAnswer),
    QwenRawAnswer qwen => JsonSerializer.SerializeToElement(qwen, ComparisonJsonContext.Default.QwenRawAnswer),
    QwenRawResponse qwen => JsonSerializer.SerializeToElement(qwen, ComparisonJsonContext.Default.QwenRawResponse),
    _ => throw new InvalidOperationException("Expected owned provider raw data.")
};

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
[JsonSerializable(typeof(string))]
[JsonSerializable(typeof(LayaRawAnswer))]
[JsonSerializable(typeof(QwenRawAnswer))]
[JsonSerializable(typeof(QwenRawResponse))]
internal partial class ComparisonJsonContext : JsonSerializerContext;
