using System.Collections.Concurrent;
using System.Globalization;
using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using OllamaSharp;
using OllamaSharp.Models;
using OllamaSharp.Models.Exceptions;
using NativeRequest = OllamaSharp.Models.SystemOneRequest;

namespace DecisionInference.OllamaSharp.Tests;

public sealed class SdkCharacterizationTests
{
    private static CancellationToken Token => TestContext.Current.CancellationToken;
    private static NativeRequest Request(object? state = null) => new()
    {
        Model = Fixtures.Model,
        State = state ?? JsonSerializer.SerializeToElement(Fixtures.Ticket),
        Questions = new()
        {
            ["q0"] = new SystemOneNoulQuestion
            {
                Instructions = "Account access?", Criteria = new() { No = "No access problem", Yes = "Access problem" }
            },
            ["q1"] = new SystemOneChoiceQuestion
            {
                Instructions = "Choose team", Criteria = new() { ["0"] = "Account", ["1"] = "Billing" }
            },
            ["q2"] = new SystemOneScoreQuestion { Instructions = "Rate urgency", Criteria = ["Routine", "Blocked", "Critical"] }
        }
    };

    [Fact]
    public async Task StockSdk_NativeEndpointAllKindsAndTrailingDiscriminators()
    {
        var body = Fixtures.Native(Fixtures.Input());
        foreach (var answer in body["answers"]!.AsObject().Select(p => p.Value!.AsObject()))
        {
            var type = answer["type"]!.DeepClone();
            answer.Remove("type");
            answer.Add("type", type);
        }
        using var handler = new ScriptedHandler(_ => Fixtures.Response(body.ToJsonString()));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http, Fixtures.Model);
        var response = await sdk.SystemOneAsync(Request(), Token);
        var sent = Assert.Single(handler.Requests);
        Assert.Equal(HttpMethod.Post, sent.Method);
        Assert.Equal("/v1/systemone", sent.Uri.AbsolutePath);
        Assert.Equal("application/json", sent.MediaType);
        Assert.Equal(["q0", "q1", "q2"], sent.Body.GetProperty("questions").EnumerateObject().Select(p => p.Name));
        Assert.Equal(Fixtures.Ticket, sent.Body.GetProperty("state").GetString());
        Assert.Equal(["noul", "choice", "score"], sent.Body.GetProperty("questions").EnumerateObject()
            .Select(p => p.Value.GetProperty("type").GetString()));
        Assert.IsType<SystemOneNoulAnswer>(response.Answers["q0"]);
        Assert.IsType<SystemOneChoiceAnswer>(response.Answers["q1"]);
        var score = Assert.IsType<SystemOneScoreAnswer>(response.Answers["q2"]);
        Assert.Equal(1.5752103826044412, score.Score, 14);
        Assert.Equal(1836, response.Usage.InputTokens);
        Assert.Equal(7, response.Usage.OutputTokens);
        Assert.True(sdk.IncomingJsonSerializerOptions.AllowOutOfOrderMetadataProperties);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("0s")]
    [InlineData("60s")]
    [InlineData("-1s")]
    [InlineData("2147483647s")]
    [InlineData("-2147483648s")]
    public async Task StockSdk_KeepAliveIsStringOrOmitted(string? duration)
    {
        using var handler = new ScriptedHandler(_ => Fixtures.Response(Fixtures.Native().ToJsonString()));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http);
        var request = Request();
        request.KeepAlive = duration;
        await sdk.SystemOneAsync(request, Token);
        var body = Assert.Single(handler.Requests).Body;
        if (duration is null) Assert.False(body.TryGetProperty("keep_alive", out _));
        else Assert.Equal(duration, body.GetProperty("keep_alive").GetString());
        Assert.Equal("", sdk.SelectedModel);
    }

    [Theory]
    [InlineData("\"caf\\u00e9 <tag>\"")]
    [InlineData("{\"text\":\"caf\\u00e9 <tag>\"}")]
    [InlineData("[\"caf\\u00e9 <tag>\"]")]
    [InlineData("{}")]
    [InlineData("[]")]
    public async Task StockSdk_BoxedOwnedJsonElementPreservesKindAndCapturesEscaping(string state)
    {
        JsonElement owned;
        using (var document = JsonDocument.Parse(state)) owned = document.RootElement.Clone();
        using var handler = new ScriptedHandler(_ => Fixtures.Response(Fixtures.Native().ToJsonString()));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http);
        await sdk.SystemOneAsync(Request(owned), Token);
        var sent = Assert.Single(handler.Requests);
        var emitted = sent.Body.GetProperty("state");
        Assert.Equal(owned.ValueKind, emitted.ValueKind);
        Assert.True(JsonElement.DeepEquals(owned, emitted));
        Assert.Equal(Encoding.UTF8.GetByteCount(sent.Text), sent.Bytes.Length);
        if (state.Contains("text", StringComparison.Ordinal) || owned.ValueKind == JsonValueKind.Array && owned.GetArrayLength() > 0)
            Assert.Contains("café <tag>", emitted.GetRawText());
        TestContext.Current.TestOutputHelper!.WriteLine($"state={emitted.GetRawText()}; finalBytes={sent.Bytes.Length}");
    }

    [Fact]
    public async Task StockSdk_CustomHeadersAreSerializedAndSent()
    {
        using var handler = new ScriptedHandler(_ => Fixtures.Response(Fixtures.Native().ToJsonString()));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http);
        sdk.DefaultRequestHeaders["X-Default"] = "default-value";
        var request = Request();
        request.CustomHeaders["X-Custom"] = "custom-value";
        await sdk.SystemOneAsync(request, Token);
        var sent = Assert.Single(handler.Requests);
        Assert.Equal("custom-value", sent.Headers["X-Custom"]);
        Assert.Equal("default-value", sent.Headers["X-Default"]);
        Assert.Equal("custom-value", sent.Body.GetProperty("CustomHeaders").GetProperty("X-Custom").GetString());
    }

    [Fact]
    public async Task StockSdk_MissingScalarsDuplicateKeysUnknownFieldsAndTotalAreLost()
    {
        const string source = """
            {"model":"tev1:0.8b","usage":{"input_tokens":1,"input_tokens":2,"total_tokens":999,"future":true},
             "answers":{"q0":{"noul":0.7,"noul":0.0,"type":"noul","future":"lost"},
                        "q1":{"choice":"0","probabilities":{"0":1.0,"1":0.0},"type":"choice"},
                        "q2":{"legend":{"0":"Routine","1":"Blocked","2":"Critical"},"probabilities":{"0":1,"1":0,"2":0},"type":"score"}}}
            """;
        using var handler = new ScriptedHandler(_ => Fixtures.Response(source));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http);
        var parsed = await sdk.SystemOneAsync(Request(), Token);
        Assert.Equal(0, Assert.IsType<SystemOneNoulAnswer>(parsed.Answers["q0"]).Noul);
        Assert.Equal(0, Assert.IsType<SystemOneChoiceAnswer>(parsed.Answers["q1"]).Confidence);
        var score = Assert.IsType<SystemOneScoreAnswer>(parsed.Answers["q2"]);
        Assert.Equal(0, score.Score);
        Assert.Equal(0, score.Confidence);
        Assert.Equal(2, parsed.Usage.InputTokens);
        Assert.Equal(0, parsed.Usage.OutputTokens);
        string snapshot = JsonSerializer.Serialize(parsed);
        Assert.DoesNotContain("future", snapshot);
        Assert.DoesNotContain("total_tokens", snapshot);
        Assert.NotEqual(source, snapshot);
    }

    [Theory]
    [InlineData("2147483648")]
    [InlineData("-2147483649")]
    public async Task StockSdk_OutOfInt32TokensFailDeserialization(string tokens)
    {
        var body = Fixtures.Native();
        body["usage"]!["input_tokens"] = JsonNode.Parse(tokens);
        using var handler = new ScriptedHandler(_ => Fixtures.Response(body.ToJsonString()));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http);
        await Assert.ThrowsAsync<JsonException>(() => sdk.SystemOneAsync(Request(), Token));
        Assert.Single(handler.Requests);
    }

    [Fact]
    public async Task StockSdk_Int32MaximumCountersAreIndividuallyRepresentable()
    {
        var body = Fixtures.Native();
        body["usage"]!["input_tokens"] = int.MaxValue;
        body["usage"]!["output_tokens"] = int.MaxValue;
        using var handler = new ScriptedHandler(_ => Fixtures.Response(body.ToJsonString()));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http);
        var result = await sdk.SystemOneAsync(Request(), Token);
        Assert.Equal(int.MaxValue, result.Usage.InputTokens);
        Assert.Equal(int.MaxValue, result.Usage.OutputTokens);
    }

    [Theory]
    [InlineData(400)]
    [InlineData(404)]
    [InlineData(500)]
    public async Task StockSdk_ErrorContractIsNotSanitizedTransport(int status)
    {
        using var handler = new ScriptedHandler(_ => Fixtures.Response("""{"error":"sensitive-fixture-diagnostic"}""", (HttpStatusCode)status));
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http);
        if (status == 400)
        {
            var error = await Assert.ThrowsAsync<OllamaException>(() => sdk.SystemOneAsync(Request(), Token));
            Assert.Contains("sensitive-fixture-diagnostic", error.Message);
        }
        else
        {
            var error = await Assert.ThrowsAsync<HttpRequestException>(() => sdk.SystemOneAsync(Request(), Token));
            Assert.Equal((HttpStatusCode)status, error.StatusCode);
        }
        Assert.Single(handler.Requests);
    }

    [Fact]
    public async Task StockSdk_CancellationReachesHttp()
    {
        using var source = CancellationTokenSource.CreateLinkedTokenSource(Token);
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var handler = new ScriptedHandler(async (_, token) =>
        {
            Assert.True(token.CanBeCanceled);
            entered.SetResult();
            await Task.Delay(Timeout.Infinite, token);
            throw new InvalidOperationException("Cancellation did not interrupt waiting.");
        });
        using var http = Fixtures.Http(handler);
        using var sdk = new OllamaApiClient(http);
        var pending = sdk.SystemOneAsync(Request(), source.Token);
        await entered.Task.WaitAsync(Token);
        source.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => pending);
        Assert.Single(handler.Requests);
    }
}

internal static class Fixtures
{
    internal const string Model = "tev1:0.8b";
    internal const string Ticket = "I cannot sign in after resetting my password.";
    internal static DecisionInput Input() => new(JsonSerializer.SerializeToElement(Ticket),
    [
        new BinaryDecisionQuestion("account-access", "Account access?", "Access problem", "No access problem"),
        new ChoiceDecisionQuestion("support-team", "Choose team", [new("account", "Account"), new("billing", "Billing")]),
        new ScoreDecisionQuestion("urgency", "Rate urgency", ["Routine", "Blocked", "Critical"])
    ]);
    // Independent softmax/expectation fixtures grounded in pinned server scoring, not SDK rounded examples.
    internal static JsonObject Native(DecisionInput? input = null, string model = Model, bool uniform = false)
    {
        input ??= Input();
        var answers = new JsonObject();
        for (int row = 0; row < input.Questions.Count; row++)
        {
            var question = input.Questions[row];
            JsonObject answer;
            if (question is BinaryDecisionQuestion)
                answer = new() { ["type"] = "noul", ["noul"] = 1 / (1 + Math.Exp(-2)) };
            else
            {
                int count = question is ChoiceDecisionQuestion c ? c.Candidates.Count : ((ScoreDecisionQuestion)question).Rubric.Count;
                double[] weights = Enumerable.Range(0, count).Select(i => uniform ? 1d :
                    count == 2 ? Math.Exp(i == 0 ? -2 : 0) : Math.Exp(i - (count - 1d))).ToArray();
                double sum = weights.Sum();
                double[] probabilities = weights.Select(w => w / sum).ToArray();
                var distribution = new JsonObject(probabilities.Select((p, i) =>
                    KeyValuePair.Create<string, JsonNode?>(i.ToString(CultureInfo.InvariantCulture), JsonValue.Create(p))));
                answer = new()
                {
                    ["type"] = question is ChoiceDecisionQuestion ? "choice" : "score",
                    ["probabilities"] = distribution,
                    ["confidence"] = Math.Clamp(1 + probabilities.Sum(p => p == 0 ? 0 : p * Math.Log(p)) / Math.Log(count), 0, 1)
                };
                if (question is ChoiceDecisionQuestion)
                    answer["choice"] = Array.IndexOf(probabilities, probabilities.Max()).ToString(CultureInfo.InvariantCulture);
                else
                {
                    answer["score"] = probabilities.Select((p, i) => p * i).Sum();
                    answer["legend"] = new JsonObject(((ScoreDecisionQuestion)question).Rubric.Select((description, i) =>
                        KeyValuePair.Create<string, JsonNode?>(i.ToString(CultureInfo.InvariantCulture), JsonValue.Create(description))));
                }
            }
            answers.Add("q" + row.ToString(CultureInfo.InvariantCulture), answer);
        }
        return new() { ["model"] = model, ["answers"] = answers, ["usage"] = new JsonObject { ["input_tokens"] = 1836, ["output_tokens"] = 7 } };
    }
    internal static HttpClient Http(HttpMessageHandler handler) => new(handler) { BaseAddress = new("http://127.0.0.1:11434") };
    internal static HttpResponseMessage Response(string json, HttpStatusCode status = HttpStatusCode.OK) =>
        new(status) { Content = new StringContent(json, Encoding.UTF8, "application/json") };
}

internal sealed record RecordedRequest(HttpMethod Method, Uri Uri, string? MediaType, byte[] Bytes,
    IReadOnlyDictionary<string, string> Headers)
{
    internal string Text => Encoding.UTF8.GetString(Bytes);
    internal JsonElement Body => JsonSerializer.Deserialize<JsonElement>(Bytes);
}

internal sealed class ScriptedHandler : HttpMessageHandler
{
    private readonly Func<RecordedRequest, CancellationToken, Task<HttpResponseMessage>> _respond;
    internal ConcurrentQueue<RecordedRequest> Requests { get; } = new();
    internal ScriptedHandler(Func<RecordedRequest, HttpResponseMessage> respond)
        : this((request, _) => Task.FromResult(respond(request))) { }
    internal ScriptedHandler(Func<RecordedRequest, CancellationToken, Task<HttpResponseMessage>> respond) => _respond = respond;
    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        var recorded = new RecordedRequest(request.Method, request.RequestUri!, request.Content?.Headers.ContentType?.MediaType,
            request.Content is null ? [] : await request.Content.ReadAsByteArrayAsync(cancellationToken),
            request.Headers.ToDictionary(p => p.Key, p => string.Join(",", p.Value), StringComparer.OrdinalIgnoreCase));
        Requests.Enqueue(recorded);
        return await _respond(recorded, cancellationToken);
    }
}
