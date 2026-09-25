using System.Collections.Concurrent;
using System.Net;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace TypeSafe.MEAI.Tests.Fixtures;

internal static class NativeFixture
{
    internal const string Key = "SYNTHETIC-NOT-A-CREDENTIAL";
    internal const string QuestionsKey = "typesafe.v1.questions";
    internal const string State = """{"ticket":17,"category":"billing"}""";
    internal const string Envelope = """
        {
          "model":"jev-1.13.0",
          "usage":{"input_tokens":11,"output_tokens":7,"vendor_usage":"retained"},
          "answers":{
            "follow-up":{"type":"noul","noul":0.82,"vendor_answer":{"tag":"noul-extra"}},
            "destination":{"type":"choice","choice":"billing","confidence":0.91,
              "probabilities":{"billing":0.60,"technical":0.30,"general":0.10},"vendor_answer":["choice-extra"]},
            "quality":{"type":"score","score":1.25,"confidence":0.76,
              "probabilities":{"0":0.20,"1":0.50,"2":0.30},
              "legend":{"0":"poor","1":{"level":"adequate","examples":["a","b"]},"2":null},
              "vendor_answer":"score-extra"}
          },
          "vendor_envelope":{"revision":17,"items":[null,{"kept":true}]}
        }
        """;

    internal static TypeSafeClientOptions Settings => new()
    {
        CacheNamespace = "synthetic-tenant-a",
        Policy = new(new TypeSafeBooleanPolicy(0.2, 0.8), 0.8)
    };

    internal static TypeSafeQuestion[] Questions() =>
    [
        new NoulQuestion("follow-up", TypeSafeJson.Entry("Follow up?"),
            Json("""{"true":{"when":"needs assistance"},"false":null}""")),
        new ChoiceQuestion("destination", TypeSafeJson.Entry("Select destination"),
        [
            KeyValuePair.Create("billing", TypeSafeJson.Entry("Billing")),
            KeyValuePair.Create("technical", Json("""{"description":"Technical"}""")),
            KeyValuePair.Create("general", Json("""["General","Other"]"""))
        ]),
        new ScoreQuestion("quality", TypeSafeJson.Entry("Assess quality"),
        [
            TypeSafeJson.Entry("poor"),
            Json("""{"level":"adequate","examples":["a","b"]}"""),
            Json("null")
        ])
    ];

    internal static JsonElement Json(string json)
    {
        using var doc = JsonDocument.Parse(json);
        return doc.RootElement.Clone();
    }

    internal static ChatOptions Options(IEnumerable<TypeSafeQuestion>? questions = null) => new()
    {
        AdditionalProperties = new()
        {
            [TypeSafeChatClient.QuestionsKey] = JsonSerializer.SerializeToElement(
                (questions ?? Questions()).ToDictionary(q => q.Id, q => q.ToJson()))
        }
    };
    internal static ChatMessage[] Messages(string state = State) =>
        [new(ChatRole.User, [new DataContent(Encoding.UTF8.GetBytes(state), "application/json")])];

    internal static JsonSerializerOptions ContentJson()
    {
        var json = new JsonSerializerOptions(AIJsonUtilities.DefaultOptions);
        json.AddAIContentType<JevResultContent>(JevResultContent.TypeDiscriminator);
        return json;
    }

    internal static HttpResponseMessage Response(string body = Envelope, int status = 200) =>
        new((HttpStatusCode)status) { Content = new StringContent(body, Encoding.UTF8, "application/json") };

    internal static ServiceProvider Services(RecordingHandler handler, bool cache = false,
        TypeSafeClientOptions? settings = null, TimeProvider? clock = null,
        TimeSpan? attempt = null, TimeSpan? total = null)
    {
        var services = new ServiceCollection();
        services.AddDistributedMemoryCache();
        if (clock is not null) services.AddSingleton(clock);
        services.AddTypeSafeChatClient("native", Key, settings ?? Settings, cache, attempt, total)
            .ConfigurePrimaryHttpMessageHandler(() => handler);
        return services.BuildServiceProvider();
    }

    internal static IChatClient Client(this ServiceProvider services) =>
        services.GetRequiredKeyedService<IChatClient>("native");

    internal static void AssertNative(JevResultContent response)
    {
        Assert.Equal(JevResultContent.ContentMediaType, response.MediaType);
        Assert.True(JsonElement.DeepEquals(response.Json, Json(new UTF8Encoding(false, true).GetString(response.Data.Span))));
        Assert.Equal("jev-1.13.0", response.Model);
        Assert.Equal(11, response.InputTokens);
        Assert.Equal(7, response.OutputTokens);
        Assert.Equal(18, response.TotalTokens);
        Assert.Equal(["destination", "follow-up", "quality"], response.Answers.Keys.Order());
        Assert.Equal(0.82, Assert.IsType<NoulAnswer>(response.Answers["follow-up"]).Noul);
        var choice = Assert.IsType<ChoiceAnswer>(response.Answers["destination"]);
        Assert.Equal("billing", choice.Choice);
        Assert.Equal(0.91, choice.Confidence);
        Assert.Equal(0.60, choice.Probabilities.GetProperty("billing").GetDouble());
        Assert.Equal(0.30, choice.Probabilities.GetProperty("technical").GetDouble());
        Assert.Equal(0.10, choice.Probabilities.GetProperty("general").GetDouble());
        var score = Assert.IsType<ScoreAnswer>(response.Answers["quality"]);
        Assert.Equal(1.25, score.Score);
        Assert.Equal(0.76, score.Confidence);
        Assert.Equal(0.20, score.Probabilities.GetProperty("0").GetDouble());
        Assert.Equal(0.50, score.Probabilities.GetProperty("1").GetDouble());
        Assert.Equal(0.30, score.Probabilities.GetProperty("2").GetDouble());
        Assert.Equal("poor", score.Legend.GetProperty("0").GetString());
        Assert.Equal("adequate", score.Legend.GetProperty("1").GetProperty("level").GetString());
        Assert.Equal(["a", "b"], score.Legend.GetProperty("1").GetProperty("examples").EnumerateArray().Select(x => x.GetString()));
        Assert.Equal(JsonValueKind.Null, score.Legend.GetProperty("2").ValueKind);
        Assert.Equal("noul-extra", response.Answers["follow-up"].Json.GetProperty("vendor_answer").GetProperty("tag").GetString());
        Assert.Equal("choice-extra", choice.Json.GetProperty("vendor_answer")[0].GetString());
        Assert.Equal("score-extra", score.Json.GetProperty("vendor_answer").GetString());
        Assert.Equal("retained", response.Json.GetProperty("usage").GetProperty("vendor_usage").GetString());
        Assert.Equal(17, response.Json.GetProperty("vendor_envelope").GetProperty("revision").GetInt32());
        Assert.True(response.Json.GetProperty("vendor_envelope").GetProperty("items")[1].GetProperty("kept").GetBoolean());
    }

    internal static void AssertChat(ChatResponse response)
    {
        var message = Assert.Single(response.Messages);
        Assert.Equal(ChatRole.Assistant, message.Role);
        var evidence = Assert.IsType<JevResultContent>(Assert.Single(message.Contents));
        Assert.Equal("", response.Text);
        Assert.Equal("jev-1.13.0", response.ModelId);
        Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
        Assert.Null(response.ResponseId);
        Assert.Null(response.CreatedAt);
        Assert.NotNull(response.Usage);
        Assert.Equal(11, response.Usage.InputTokenCount);
        Assert.Equal(7, response.Usage.OutputTokenCount);
        Assert.Equal(18, response.Usage.TotalTokenCount);
        AssertNative(evidence);
    }
}

internal sealed record CapturedRequest(HttpMethod Method, Uri Uri, string? Scheme, string? Key, string? MediaType, string? Body);

internal sealed class RecordingHandler(
    Func<HttpRequestMessage, int, CancellationToken, Task<HttpResponseMessage>>? respond = null) : HttpMessageHandler
{
    private int _sends;
    private int _disposals;
    internal int Sends => Volatile.Read(ref _sends);
    internal int Disposals => Volatile.Read(ref _disposals);
    internal ConcurrentQueue<CapturedRequest> Requests { get; } = new();

    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        var attempt = Interlocked.Increment(ref _sends);
        var body = request.Content is null ? null : await request.Content.ReadAsStringAsync(cancellationToken);
        Requests.Enqueue(new(request.Method, request.RequestUri!, request.Headers.Authorization?.Scheme,
            request.Headers.Authorization?.Parameter, request.Content?.Headers.ContentType?.MediaType, body));
        return respond is null ? NativeFixture.Response() : await respond(request, attempt, cancellationToken);
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing) Interlocked.Increment(ref _disposals);
        base.Dispose(disposing);
    }
}
