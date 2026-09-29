using System.Diagnostics;
using System.Numerics.Tensors;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using DecisionInference;
using DecisionInference.Laya;
using DecisionInference.Ollaya;
using DecisionInference.TensorSharp;
using DecisionInference.TypeSafe;
using Julia = DecisionInference.Julia;
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
        using (var generator = new TypeSafeDecisionGenerator(http, "offline-fixture"))
            Print(await ConsumeChoiceAsync(generator, RecordedJevHandler.Input(), "category", null, deadline.Token));
        break;
    case "provider" when args.Length == 5 && args[1] == "laya":
        var layaClock = Stopwatch.StartNew();
        using (var generator = new LayaDecisionGenerator(args[2], args[3], args[4]))
        {
            Console.WriteLine($"Laya load: {layaClock.Elapsed.TotalMilliseconds:F1} ms; revision={LayaProfile.Revision}");
            await Measure("Laya first call", () => ConsumeChoiceAsync(generator, LocalInput(), "category", null, deadline.Token));
            Print(await Measure("Laya warm", () => generator.GenerateAsync(LocalInput(), cancellationToken: deadline.Token)));
        }
        break;
    case "stages" when args.Length is 4 or 5:
        var profile = LayaProfile.Load(args[3]);
        var (tokenizer, tokenMetadata) = LayaTokenizer.Load(args[2]);
        var prepare = new PrepareDecisionInputs(profile, tokenizer, tokenMetadata);
        var decode = new DecodeDecisions(profile);
        using (var score = ScoreOnnxDecisionModel.Load(args[1]))
        using (var generator = new LayaDecisionGenerator(prepare, score, decode))
        {
            var input = LocalInput();
            var batch = prepare.Prepare(input, deadline.Token);
            var raw = score.Score(batch, deadline.Token);
            var direct = decode.Decode(batch, raw);
            var viaGenerator = await ConsumeChoiceAsync(generator, input, "category", null, deadline.Token);
            EqualAnswers(direct, viaGenerator, 1e-7);
            Print(direct);
            Console.WriteLine($"Public-stage/generator parity; two native runs; B={batch.BatchSize}, L={batch.SequenceLength}, K={batch.MarkerWidth}");
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
        using (var generator = QwenDecisionGenerator.Load(args[2]))
        {
            Console.WriteLine($"Qwen verified hash={QwenDecisionGenerator.ModelSha256}; load={qwenClock.Elapsed.TotalMilliseconds:F1} ms");
            var metadata = generator.GetRequiredService<DecisionGeneratorMetadata>();
            if (metadata.ProviderName != "tensorsharp-qwen" || metadata.ProviderUri is not null ||
                metadata.DefaultModelId != QwenDecisionGenerator.PinnedModelId ||
                metadata.SupportedKinds is not { Count: 1 } || metadata.SupportedKinds[0] != DecisionKind.Choice)
                throw new InvalidOperationException("Qwen metadata does not describe the loaded local Choice model.");
            var fixedOptions = new DecisionGenerationOptions { ModelId = metadata.DefaultModelId };
            var first = await Measure("Qwen first forward", () => ConsumeChoiceAsync(generator, ChoiceInput(), "category", fixedOptions, deadline.Token));
            if (first.GetAnswer("category").RawRepresentation is not QwenRawAnswer
                { Provider: "tensorsharp-qwen", ConditionalOnAnswerTokens: true, Aliases: "ABC", Logits.Count: 3, TokenIds.Count: 3 })
                throw new InvalidOperationException("Qwen answer-token evidence is missing or inconsistent.");
            string ownedEvidence = ExportEvidence(first.GetAnswer("category").RawRepresentation).GetRawText();
            if (first.RawRepresentation is not QwenRawResult { ForwardCalls: 1, ContextLength: 512 })
                throw new InvalidOperationException("Qwen forward/context evidence is missing or inconsistent.");
            Print(first);
            await generator.GenerateAsync(ChoiceInput("The team lunch is at noon in the usual meeting room."), cancellationToken: deadline.Token);
            var repeated = await Measure("Qwen repeated warm forward", () => generator.GenerateAsync(ChoiceInput(), cancellationToken: deadline.Token));
            EqualAnswers(first, repeated, 1e-7);
            foreach (var invalidModel in new[] { "", " ", QwenDecisionGenerator.PinnedModelId.ToLowerInvariant() })
            {
                try
                {
                    await generator.GenerateAsync(ChoiceInput(), new() { ModelId = invalidModel }, deadline.Token);
                    throw new InvalidOperationException("Qwen accepted an invalid or different loaded model.");
                }
                catch (ArgumentException) when (string.IsNullOrWhiteSpace(invalidModel)) { }
                catch (NotSupportedException) when (!string.IsNullOrWhiteSpace(invalidModel)) { }
            }
            foreach (var unsupported in new DecisionQuestion[]
                { new BinaryDecisionQuestion("unsupported", "The message is safe."), new ScoreDecisionQuestion("unsupported", null, ["low", "high"]) })
            {
                var mixed = new DecisionInput(ChoiceInput().State, [ChoiceInput().Questions[0], unsupported]);
                try
                {
                    await generator.GenerateAsync(mixed, cancellationToken: deadline.Token);
                    throw new InvalidOperationException("Qwen accepted an unsupported mixed batch.");
                }
                catch (NotSupportedException) { }
            }
            if (metadata.DefaultModelId != QwenDecisionGenerator.PinnedModelId || fixedOptions.ModelId != metadata.DefaultModelId ||
                first.ModelId != metadata.DefaultModelId)
                throw new InvalidOperationException("Qwen model metadata or caller options changed.");
            Console.WriteLine("Local metadata, fixed-model override, invalid models and Binary/Score rejection verified.");
            var afterRejection = await generator.GenerateAsync(ChoiceInput(), cancellationToken: deadline.Token);
            EqualAnswers(first, afterRejection, 1e-7);
            if (ExportEvidence(first.GetAnswer("category").RawRepresentation).GetRawText() != ownedEvidence ||
                ExportEvidence(repeated.GetAnswer("category").RawRepresentation).GetRawText() != ownedEvidence)
                throw new InvalidOperationException("Owned answer-token evidence changed across model reset/reuse.");
            Console.WriteLine("One model load, four forwards, A/B/A repeated-input isolation; original owned logits unchanged.");
            var result = first.GetAnswer<ChoiceDecisionAnswer>("category");
            if (result.SelectedCandidateId != "phishing")
                throw new InvalidOperationException("Pinned example no longer selects Phishing.");
            double[] reference = [0.0659, 0.0270, 0.9071];
            if (result.Probabilities.Values.Where((p, i) => Math.Abs(p - reference[i]) > 0.01).Any())
                throw new InvalidOperationException("Pinned sample probabilities differ by more than 0.01 absolute.");
            Console.WriteLine("Pinned published sample: category and probabilities within 0.01 absolute tolerance (not a calibration test).");
        }
        break;
    case "provider" when args.Length is 4 or 5 && args[1] == "ollaya":
        await ProveOllaya(new Uri(args[2]), args[3], args.Length == 5 ? args[4] : null, deadline.Token);
        break;
    case "provider" when args.Length is 3 or 4 && args[1] == "julia":
        await ProveJulia(args[2], args.Length == 4 ? args[3] : null, deadline.Token);
        break;
    case "inventory":
        foreach (var assembly in new[] { typeof(IDecisionGenerator), typeof(TypeSafeDecisionGenerator), typeof(OllayaDecisionGenerator), typeof(LayaDecisionGenerator), typeof(Julia.JuliaDecisionGenerator), typeof(QwenDecisionGenerator), typeof(TypeSafeChatClient) }
            .Select(t => t.Assembly).Append(Assembly.Load("DecisionInference.SystemOne")).Append(Assembly.Load("DecisionInference.Onnx")))
        {
            Console.WriteLine($"{assembly.GetName().Name}: exportedTypes={assembly.GetExportedTypes().Length}, " +
                $"declaredPublicMembers={assembly.GetExportedTypes().Sum(t => t.GetMembers(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly).Length)}");
        }
        break;
    default:
        throw new ArgumentException("Engineering proof modes: compare | inventory | provider jev | provider laya <onnx> <tokenizer.json> <laya_config.json> | stages <onnx> <tokenizer.json> <laya_config.json> [receipt.json] | provider qwen <gguf> | provider ollaya <endpoint> <concrete-model> [receipt.json] | provider julia <asset-directory> [receipt.json]. Application examples: samples/README.md.");
}

static async Task ProveJulia(string directory, string? receiptPath, CancellationToken ct)
{
    using var sessionOptions = new Microsoft.ML.OnnxRuntime.SessionOptions { IntraOpNumThreads = 2, InterOpNumThreads = 1 };
    using var prepare = new Julia.PrepareDecisionInputs(Path.Combine(directory, "tokenizer.json"));
    using var score = Julia.ScoreOnnxDecisionModel.Load(Path.Combine(directory, "model.onnx"), sessionOptions);
    var decode = new Julia.DecodeDecisions();
    using var generator = new Julia.JuliaDecisionGenerator(prepare, score, decode);
    var questions = new DecisionQuestion[]
    {
        new ChoiceDecisionQuestion(" route \u00e9", "Which team should handle this?",
            [new(" Account ", "Account support"), new("billing", "Billing"), new("e\u0301", "Technical support")]),
        new BinaryDecisionQuestion(" described ", "Does the customer need account access?",
            trueDescription: "The customer needs help accessing their account.", falseDescription: "The customer does not need account access."),
        new ScoreDecisionQuestion(" risk ", "Assess the urgency.", ["low", "moderate", "high", "critical"])
    };
    using var structured = JsonDocument.Parse("""
        {"z": "café e\u0301 🫠 \u0378", "a": [true, null, 1, 1.0, 1e-5, 1e16, -0.0, 123456789012345678901234567890], "line": "a\tb\nc\r\n"}
        """);
    using var array = JsonDocument.Parse("""["  reset my password  ", {"white": "\u00a0hello\u2009world", "added": "<unused0> [@BOS@]"}, [2, 3.14]]""");
    JsonElement[] states = [JsonSerializer.SerializeToElement("The customer asks to reset their password.\n  café e\u0301 🫠\t"),
        structured.RootElement, array.RootElement];
    var receipts = new List<object>();
    DecisionResult? first = null;
    var firstInput = new DecisionInput(states[0], questions);
    foreach (var state in states)
    {
        var input = new DecisionInput(state, questions);
        var batch = prepare.Prepare(input, ct);
        var logits = score.Score(batch, ct);
        var direct = decode.Decode(batch, logits);
        var result = await ConsumeChoiceAsync(generator, input, questions[0].Id, new() { ModelId = Julia.JuliaDecisionGenerator.ModelId }, ct);
        EqualAnswers(direct, result, 1e-12);
        if (result.Usage is not null || result.Answers.Any(a => a.Precision != DecisionPrecision.HighPrecision))
            throw new InvalidOperationException("Julia must report high precision and unknown usage.");
        receipts.Add(JuliaEvidence(input, batch, logits, result));
        first ??= result;
        string snapshot = ExportEvidence(direct.Answers[0].RawRepresentation).GetRawText();
        logits[0, 0] = float.NaN;
        if (snapshot != ExportEvidence(direct.Answers[0].RawRepresentation).GetRawText())
            throw new InvalidOperationException("Decoded raw snapshot borrowed the caller's output tensor.");
    }
    Print(first!);
    var concurrent = await Task.WhenAll(Enumerable.Range(0, 2).Select(_ => Task.Run(() => generator.GenerateAsync(firstInput, cancellationToken: ct), ct)));
    foreach (var result in concurrent) EqualAnswers(first!, result, 1e-12);
    using (var cancelled = new CancellationTokenSource())
    {
        cancelled.Cancel();
        try { await generator.GenerateAsync(firstInput, cancellationToken: cancelled.Token); throw new InvalidOperationException("Pre-cancellation ignored."); }
        catch (OperationCanceledException) { }
    }
    foreach (var rejected in new[]
    {
        new DecisionInput(JsonSerializer.SerializeToElement(string.Concat(Enumerable.Repeat("weather ", 1200))), questions),
        new DecisionInput(JsonSerializer.SerializeToElement("text"), [questions[0], new BinaryDecisionQuestion("last", "Statement", trueDescription: "Only one criterion")])
    })
    {
        try { await generator.GenerateAsync(rejected, cancellationToken: ct); throw new InvalidOperationException("Lossy input accepted."); }
        catch (ArgumentException) { }
        catch (NotSupportedException) { }
    }
    var longInput = new DecisionInput(JsonSerializer.SerializeToElement(string.Concat(Enumerable.Repeat("weather ", 800))), questions);
    var longBatch = prepare.Prepare(longInput, ct);
    string cancellation;
    using (var inFlight = CancellationTokenSource.CreateLinkedTokenSource(ct))
    {
        inFlight.CancelAfter(TimeSpan.FromMilliseconds(75));
        try { score.Score(longBatch, inFlight.Token); throw new InvalidOperationException("Expected actual in-flight ONNX cancellation."); }
        catch (OperationCanceledException error) when (error.InnerException is Microsoft.ML.OnnxRuntime.OnnxRuntimeException)
        {
            cancellation = error.InnerException.Message;
        }
    }
    EqualAnswers(first!, await generator.GenerateAsync(firstInput, cancellationToken: ct), 1e-12);

    using var referenceCases = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "parity-cases.json")));
    var selected = referenceCases.RootElement.EnumerateArray().Where(x =>
        x.GetProperty("request").GetProperty("state").GetString() == "" &&
        x.GetProperty("request").GetProperty("type").GetString() == "choice").Take(6).ToArray();
    var referenceQuestions = selected.Select((item, i) => new ChoiceDecisionQuestion($" benchmark {i} ",
        item.GetProperty("request").GetProperty("question").GetString(),
        item.GetProperty("request").GetProperty("options").EnumerateArray().Select((o, j) => new DecisionCandidate($" id {j} ", o.GetString()!)))).ToArray();
    if (selected.Length != 6)
        throw new InvalidOperationException("Unexpected upstream bounded reference case shape.");
    var benchmarkInput = new DecisionInput(JsonSerializer.SerializeToElement(""), referenceQuestions);
    var benchmarkBatch = prepare.Prepare(benchmarkInput, ct);
    var benchmarkLogits = score.Score(benchmarkBatch, ct);
    var benchmarkResult = decode.Decode(benchmarkBatch, benchmarkLogits);
    double maxReferenceError = 0;
    for (int row = 0; row < selected.Length; row++)
    {
        var expected = selected[row].GetProperty("pytorch_logits").EnumerateArray().Select(x => x.GetDouble()).ToArray();
        var actual = ((Julia.JuliaRawAnswer)benchmarkResult.Answers[row].RawRepresentation!).Logits;
        maxReferenceError = Math.Max(maxReferenceError, expected.Select((x, i) => Math.Abs(x - actual[i])).Max());
        if (maxReferenceError > 1e-4 || Array.IndexOf(expected, expected.Max()) != Array.IndexOf(actual.ToArray(), actual.Max()))
            throw new InvalidOperationException("Pinned upstream PyTorch reference differs beyond 1e-4 or changes argmax.");
    }
    receipts.Add(JuliaEvidence(benchmarkInput, benchmarkBatch, benchmarkLogits, benchmarkResult));
    if (receiptPath is not null)
        File.WriteAllText(receiptPath, JsonSerializer.Serialize(new
        {
            model = Julia.JuliaDecisionGenerator.ModelId, revision = "82a2fadf8fccfccdc5fd4e1009ba8f1a265eb7a8",
            referenceRecipe = "Python julia/data.py strict1024/head256; json.dumps ensure_ascii=False",
            requests = receipts, concurrentCalls = 2, preCancelled = true, inFlightOnnxCancellation = cancellation,
            rejectionRecovery = true, ownedSnapshots = true, upstreamPytorchCases = selected.Length, maxReferenceError,
            upstreamTolerance = 1e-4, usage = "unknown, not zero"
        }));
    Console.WriteLine($"Julia CPU: all kinds, same consumer/stages, Unicode/structured states, full probabilities, owned snapshots, concurrency, native termination/recovery; upstream PyTorch {selected.Length} cases maxAbsLogitError={maxReferenceError:G9}.");
}

static object JuliaEvidence(DecisionInput input, Julia.JuliaInputBatch batch, Tensor<float> logits, DecisionResult result)
{
    var questions = input.Questions.ToDictionary(q => q.Id, q => new
    {
        type = q.Kind switch { DecisionKind.Choice => "choice", DecisionKind.Score => "score", _ => "noul" },
        instructions = q.Instructions,
        criteria = q switch
        {
            ChoiceDecisionQuestion c => (object)c.Candidates.ToDictionary(c => c.Id, c => c.Description),
            ScoreDecisionQuestion s => s.Rubric,
            BinaryDecisionQuestion b when b.TrueDescription is not null =>
                new Dictionary<string, string> { ["false"] = b.FalseDescription!, ["true"] = b.TrueDescription },
            _ => null
        }
    });
    return new
    {
        state = input.State, questions, batch = batch.BatchSize, length = batch.SequenceLength, width = batch.MarkerWidth,
        input_ids = Copy(batch.InputIds), attention_mask = Copy(batch.AttentionMask), marker_pos = Copy(batch.MarkerPositions),
        marker_mask = Copy(batch.MarkerMask), qtype = Copy(batch.QuestionTypes), logits = logits.ToArray(),
        answers = result.Answers.Select(a => new
        {
            id = a.QuestionId,
            probabilities = a switch
            {
                BinaryDecisionAnswer b => new[] { 1 - b.TrueProbability, b.TrueProbability },
                ChoiceDecisionAnswer c => c.Probabilities.Values.ToArray(),
                ScoreDecisionAnswer s => s.Probabilities.ToArray(),
                _ => throw new NotSupportedException()
            },
            selected = (a as ChoiceDecisionAnswer)?.SelectedCandidateId,
            expectedIndex = (a as ScoreDecisionAnswer)?.ExpectedIndex
        }).ToArray()
    };
}

static async Task CompareApis(CancellationToken ct)
{
    var input = RecordedJevHandler.Input();
    using var decisionHandler = new RecordedJevHandler();
    using var chatHandler = new RecordedJevHandler();
    using var decisionHttp = new HttpClient(decisionHandler);
    using var chatHttp = new HttpClient(chatHandler);
    using var decision = new TypeSafeDecisionGenerator(decisionHttp, "offline-fixture");
    var metadata = decision.GetRequiredService<DecisionGeneratorMetadata>();
    var decisionOptions = new DecisionGenerationOptions { ModelId = metadata.DefaultModelId }.Clone();
    if (metadata.ProviderUri != new Uri("https://api.typesafe.ai"))
        throw new InvalidOperationException("Configured Jev provider metadata differs.");
    using IChatClient chat = new TypeSafeChatClient(chatHttp, new() { ModelId = "jev-1.13.0" }, "offline-fixture");
    var options = new ChatOptions
    {
        AdditionalProperties = new()
        {
            ["typesafe.v1.questions"] = JsonSerializer.SerializeToElement(new
            {
                q0 = new { type = "noul", instructions = "Is it safe?" },
                q1 = new { type = "choice", instructions = "Choose one category.", criteria = new Dictionary<string, string> { ["0"] = "Legitimate", ["1"] = "Phishing" } },
                q2 = new { type = "score", instructions = "Assess severity.", criteria = new[] { "low", "medium", "high" } }
            })
        }
    };
    var messages = new[] { new ChatMessage(ChatRole.User, input.State.GetString()) };
    // One uncached warm-up each; the fixture records the actual outgoing bodies, not reconstructed equivalents.
    await decision.GenerateAsync(input.State, input.Questions, decisionOptions, ct);
    await chat.GetResponseAsync(messages, options, ct);
    var newResult = await Measure("Decision API (synthetic HTTP)", () => ConsumeChoiceAsync(decision, input, "category", decisionOptions, ct));
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
    if (newResult.GetAnswer<ChoiceDecisionAnswer>("category").SelectedCandidateId != "phishing" ||
        newResult.GetAnswer<BinaryDecisionAnswer>("safe").TrueProbability != 0.1 ||
        Math.Abs(newResult.GetAnswer<ScoreDecisionAnswer>("severity").ExpectedIndex - 1.6) > 1e-12)
        throw new InvalidOperationException("Mapped portable decision semantics differ.");
    Print(newResult);
    Console.WriteLine($"Explicit same-wire aliased fixture: outgoing payload/native-evidence parity; external decision IDs restored; sends decision={decisionHandler.Calls}, chat={chatHandler.Calls}; no caches.");
    Console.WriteLine("Intentional migration: the original chat API does not alias IDs. This fixture supplies matching aliases explicitly; automatic arbitrary-ID wire parity is NOT claimed.");
    Console.WriteLine("Instrumented consumer observations, NOT a performance comparison, live Jev latency or accuracy. Chat still provides schema projection, tools and MEAI middleware; the decision API does not.");
}

static async Task ProveOllaya(Uri endpoint, string model, string? receiptPath, CancellationToken ct)
{
    using var handler = new HttpClientHandler { AllowAutoRedirect = false, UseCookies = false };
    using var http = new HttpClient(handler);
    using IDecisionGenerator generator = new OllayaDecisionGenerator(http, new() { Endpoint = endpoint, ModelId = model });
    var metadata = generator.GetRequiredService<DecisionGeneratorMetadata>();
    if (metadata.ProviderName != "ollaya" || metadata.ProviderUri != endpoint || metadata.SupportedKinds is not null)
        throw new InvalidOperationException("Ollaya must describe an unknown mutable catalog, not static all-kind support.");
    var input = LocalInput();
    var options = new DecisionGenerationOptions { ModelId = model };
    var first = await ConsumeChoiceAsync(generator, input, "category", options, ct);
    var choice = first.GetAnswer<ChoiceDecisionAnswer>("category");
    _ = first.GetAnswer<BinaryDecisionAnswer>("suspicious");
    var score = first.GetAnswer<ScoreDecisionAnswer>("severity");
    if (first.Answers.Any(a => a.Precision != DecisionPrecision.FourDecimalPlaces) ||
        !choice.Probabilities.Keys.SequenceEqual(((ChoiceDecisionQuestion)input.Questions[0]).Candidates.Select(c => c.Id)) ||
        score.ExpectedIndexLowerBound < 0 || score.ExpectedIndexUpperBound > 2)
        throw new InvalidOperationException("Live precision, exact IDs or latent bounds are inconsistent.");
    string rawBefore = ExportEvidence(first.RawRepresentation).GetRawText();
    var concurrent = await Task.WhenAll(Enumerable.Range(0, 2).Select(_ =>
        generator.GenerateAsync(input.State.GetString()!, input.Questions, ComparisonJsonContext.Default.String, options, ct)));
    foreach (var result in concurrent) EqualAnswers(first, result, 0);
    if (rawBefore != ExportEvidence(first.RawRepresentation).GetRawText())
        throw new InvalidOperationException("Live raw snapshot changed across concurrent calls.");
    using var cancelled = new CancellationTokenSource();
    cancelled.Cancel();
    OllayaApiException truncation;
    try
    {
        await generator.GenerateAsync(input, cancellationToken: cancelled.Token);
        throw new InvalidOperationException("Pre-cancelled call unexpectedly completed.");
    }
    catch (OperationCanceledException) when (cancelled.IsCancellationRequested) { }
    try
    {
        await generator.GenerateAsync(ChoiceInput(string.Concat(Enumerable.Repeat("This is a long unrelated state. ", 3000))), cancellationToken: ct);
        throw new InvalidOperationException("Ollaya silently accepted a truncated state.");
    }
    catch (OllayaApiException error) when (error.StatusCode == System.Net.HttpStatusCode.UnprocessableEntity && error.Code == "STATE_TRUNCATED")
    {
        truncation = error;
    }
    var afterRejection = await ConsumeChoiceAsync(generator, input, "category", options, ct);
    EqualAnswers(first, afterRejection, 0);
    if (rawBefore != ExportEvidence(first.RawRepresentation).GetRawText())
        throw new InvalidOperationException("Raw snapshot changed after rejected-input recovery.");
    if (metadata.DefaultModelId != model || options.ModelId != model)
        throw new InvalidOperationException("Model overrides mutated caller options or defaults.");
    Print(first);
    if (receiptPath is not null)
        File.WriteAllText(receiptPath, JsonSerializer.Serialize(new
        {
            endpoint, requestedModel = model, actualModel = first.ModelId,
            proof = "live-independent-family-not-a-calibration-or-latency-comparison",
            concurrentCalls = 2, overBudgetStatus = (int)truncation.StatusCode!,
            overBudgetCode = truncation.Code, postRejectionRecovery = true, preCancellation = true,
            observedExpectedIndex = score.ExpectedIndex, lowerBound = score.ExpectedIndexLowerBound,
            upperBound = score.ExpectedIndexUpperBound, nativeScore = score.NativeScore,
            modelInfo = (JsonElement)first.AdditionalProperties!["ollaya.modelInfo"]!,
            raw = ExportEvidence(first.RawRepresentation)
        }, new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine($"Live Ollaya Binary/Choice/Score; shared typed consumer; two concurrent calls; raw ownership; pre-cancel; no-truncation {(int)truncation.StatusCode!}/{truncation.Code}; recovery.");
}

static async Task<DecisionResult> ConsumeChoiceAsync(IDecisionGenerator generator, DecisionInput input,
    string choiceQuestionId, DecisionGenerationOptions? options, CancellationToken ct)
{
    var result = await generator.GenerateAsync(input.State, input.Questions,
        ComparisonJsonContext.Default.JsonElement, options, ct);
    var answer = result.GetAnswer<ChoiceDecisionAnswer>(choiceQuestionId);
    Console.WriteLine($"Shared consumer: {answer.QuestionId} selected {answer.SelectedCandidateId}; {answer.Probabilities.Count} outcomes.");
    return result;
}

static DecisionInput ChoiceInput(string email = "Payroll asks for your password on a non-company sign-in page.") =>
    new(JsonSerializer.SerializeToElement("Email: " + email), [new ChoiceDecisionQuestion("category", "Choose one option.",
        [new("legitimate", "Legitimate"), new("spam", "Spam"), new("phishing", "Phishing")])]);

static DecisionInput LocalInput()
{
    var choice = ChoiceInput();
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

static void EqualAnswers(DecisionResult first, DecisionResult second, double tolerance)
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

static void Print(DecisionResult result)
{
    Console.WriteLine($"model={result.ModelId}; inputTokens={result.Usage?.InputTokenCount}; outputTokens={result.Usage?.OutputTokenCount?.ToString() ?? "not reported"}");
    foreach (var answer in result.Answers)
        Console.WriteLine(answer switch
        {
            BinaryDecisionAnswer b => $"{b.QuestionId}: trueProbability={b.TrueProbability:G6}",
            ChoiceDecisionAnswer c => $"{c.QuestionId}: selected={c.SelectedCandidateId}; " + string.Join(", ", c.Probabilities.Select(p => $"{p.Key}={p.Value:G6}")),
            ScoreDecisionAnswer s => $"{s.QuestionId}: observedIndexEstimate={s.ExpectedIndex:G6}; feasibleBounds=[{s.ExpectedIndexLowerBound:G6},{s.ExpectedIndexUpperBound:G6}]; precision={s.Precision}; [{string.Join(", ", s.Probabilities)}]",
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
    Julia.JuliaRawAnswer julia => JsonSerializer.SerializeToElement(julia, ComparisonJsonContext.Default.JuliaRawAnswer),
    QwenRawAnswer qwen => JsonSerializer.SerializeToElement(qwen, ComparisonJsonContext.Default.QwenRawAnswer),
    QwenRawResult qwen => JsonSerializer.SerializeToElement(qwen, ComparisonJsonContext.Default.QwenRawResult),
    _ => throw new InvalidOperationException("Expected owned provider raw data.")
};

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase)]
[JsonSerializable(typeof(string))]
[JsonSerializable(typeof(JsonElement))]
[JsonSerializable(typeof(LayaRawAnswer))]
[JsonSerializable(typeof(Julia.JuliaRawAnswer))]
[JsonSerializable(typeof(QwenRawAnswer))]
[JsonSerializable(typeof(QwenRawResult))]
internal partial class ComparisonJsonContext : JsonSerializerContext;
