using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.Extensions.AI;

namespace OllamaDecisions.Tests;

public sealed class OllamaDecisionClientTests
{
    private static readonly DecisionQuestion[] Questions =
    [
        new BinaryDecisionQuestion("refund", "Is a refund requested?", trueDescription: "A refund is requested", falseDescription: "No refund"),
        new ChoiceDecisionQuestion("team", "Which team should handle this?", [new("billing", "Payments"), new("technical", "Software problems")]),
        new ScoreDecisionQuestion("urgency", "How urgent is this?", [new("low", "Routine"), new("medium", "Soon"), new("high", "Immediate")]),
    ];

    // Answers are deliberately out of request order, and probabilities out of candidate order.
    private const string AllAnswers = """
        {
          "model": "nimble",
          "answers": {
            "urgency": {"type": "score", "score": 1.25, "legend": {"0": "Routine", "1": "Soon", "2": "Immediate"},
                        "probabilities": {"2": 0.4, "0": 0.15, "1": 0.45}, "confidence": 0.12},
            "team": {"type": "choice", "choice": "billing", "probabilities": {"technical": 0.25, "billing": 0.75}, "confidence": 0.19},
            "refund": {"type": "noul", "noul": 0.9}
          },
          "usage": {"input_tokens": 174, "output_tokens": 3},
          "prompt_eval_cached_count": 100
        }
        """;

    [Fact]
    public async Task MapsJsonStateAndQuestionsToOllamaRequest()
    {
        var handler = new FakeHandler(AllAnswers);
        using var client = new OllamaDecisionClient(new Uri("http://ollama.test:11434"), "nimble", new HttpClient(handler));

        await Ask(client, new DecisionRequest(JsonDocument.Parse("""{"ticket":"Charged twice"}""").RootElement, Questions));

        Assert.Equal(new Uri("http://ollama.test:11434/v1/systemone"), handler.RequestUri);
        JsonNode expected = JsonNode.Parse("""
            {
              "model": "nimble",
              "state": {"ticket": "Charged twice"},
              "questions": {
                "refund": {"instructions": "Is a refund requested?", "type": "noul", "criteria": {"false": "No refund", "true": "A refund is requested"}},
                "team": {"instructions": "Which team should handle this?", "type": "choice", "criteria": {"billing": "Payments", "technical": "Software problems"}},
                "urgency": {"instructions": "How urgent is this?", "type": "score", "criteria": ["Routine", "Soon", "Immediate"]}
              }
            }
            """)!;
        Assert.True(JsonNode.DeepEquals(expected, handler.Body), handler.Body?.ToJsonString());
    }

    [Fact]
    public async Task MapsTextAndImageContentsToStateAndImages()
    {
        var handler = new FakeHandler("""{"model":"clef","answers":{"refund":{"type":"noul","noul":0.5}},"usage":{"input_tokens":1,"output_tokens":0}}""");
        using var client = new OllamaDecisionClient(httpClient: new HttpClient(handler));
        byte[] png = [0x89, 0x50, 0x4E, 0x47];
        var raw = new JsonObject { ["keep_alive"] = "10m", ["model"] = "ignored" };

        await Ask(client, 
            new DecisionRequest(null, [Questions[0]], [new TextContent("The customer wrote: "), new DataContent(png, "image/png"), new TextContent("refund please")]),
            new DecisionOptions { ModelId = "clef", RawRepresentationFactory = _ => raw });

        Assert.Equal(new Uri("http://localhost:11434/v1/systemone"), handler.RequestUri);
        Assert.Equal("clef", (string?)handler.Body!["model"]);
        Assert.Equal("10m", (string?)handler.Body["keep_alive"]);
        Assert.Equal("The customer wrote: refund please", (string?)handler.Body["state"]);
        Assert.Equal(Convert.ToBase64String(png), (string?)handler.Body["images"]![0]);
        Assert.Equal("ignored", (string?)raw["model"]);
    }

    [Fact]
    public async Task MapsOutOfOrderAnswersInRequestOrder()
    {
        using var client = new OllamaDecisionClient(defaultModelId: "nimble", httpClient: new HttpClient(new FakeHandler(AllAnswers)));

        DecisionResponse response = await Ask(client, new DecisionRequest(JsonDocument.Parse("\"Charged twice\"").RootElement, Questions));

        Assert.Equal(["refund", "team", "urgency"], response.Answers.Select(a => a.QuestionId));
        Assert.Equal(0.9, Assert.IsType<BinaryDecisionAnswer>(response.Answers[0]).TrueProbability);

        var team = Assert.IsType<ChoiceDecisionAnswer>(response.Answers[1]);
        Assert.Equal("billing", team.SelectedCandidateId);
        Assert.Equal([("billing", 0.75), ("technical", 0.25)], team.Probabilities.Select(p => (p.Id, p.Value)));
        Assert.Equal(0.19, team.AdditionalProperties![OllamaDecisionClient.ConfidenceKey]);

        var urgency = Assert.IsType<ScoreDecisionAnswer>(response.Answers[2]);
        Assert.Equal(1.25, urgency.Score);
        Assert.Equal([("low", 0.15), ("medium", 0.45), ("high", 0.4)], urgency.Probabilities.Select(p => (p.Id, p.Value)));

        Assert.Equal("nimble", response.ModelId);
        Assert.Equal(174, response.Usage!.InputTokenCount);
        Assert.Equal(3, response.Usage.OutputTokenCount);
        Assert.Equal(177, response.Usage.TotalTokenCount);
        Assert.Equal(100, response.Usage.CachedInputTokenCount);
        Assert.IsType<JsonElement>(response.RawRepresentation);
    }

    [Theory]
    [InlineData("""{"answers":{"refund":{"type":"choice","choice":"billing","probabilities":{"billing":1}}}}""")]
    [InlineData("""{"answers":{}}""")]
    [InlineData("""{"answers":{"refund":{"type":"noul","noul":0.5},"extra":{"type":"noul","noul":0.5}}}""")]
    [InlineData("""{"answers":{"refund":{"type":"noul","noul":1.5}}}""")]
    [InlineData("""not json""")]
    public async Task ThrowsProtocolExceptionForMismatchedAnswers(string json)
    {
        using var client = new OllamaDecisionClient(defaultModelId: "nimble", httpClient: new HttpClient(new FakeHandler(json)));

        await Assert.ThrowsAsync<DecisionProtocolException>(() =>
            Ask(client, new DecisionRequest(JsonDocument.Parse("\"text\"").RootElement, [Questions[0]])));
    }

    [Fact]
    public async Task ThrowsProtocolExceptionForUnrequestedCandidate()
    {
        const string json = """{"answers":{"team":{"type":"choice","choice":"billing","probabilities":{"billing":0.5,"sales":0.5}}}}""";
        using var client = new OllamaDecisionClient(defaultModelId: "nimble", httpClient: new HttpClient(new FakeHandler(json)));

        await Assert.ThrowsAsync<DecisionProtocolException>(() =>
            Ask(client, new DecisionRequest(JsonDocument.Parse("\"text\"").RootElement, [Questions[1]])));
    }

    public static TheoryData<AIContent[]> UnsupportedContents => new()
    {
        new AIContent[] { new DataContent(new byte[] { 1 }, "image/gif") },
        new AIContent[] { new UriContent(new Uri("https://example.test/a.png"), "image/png") },
        new AIContent[] { new DataContent(new byte[] { 1 }, "audio/wav") },
    };

    [Theory]
    [MemberData(nameof(UnsupportedContents))]
    public async Task ThrowsNotSupportedForUnsupportedContent(AIContent[] contents)
    {
        var handler = new FakeHandler(AllAnswers);
        using var client = new OllamaDecisionClient(defaultModelId: "clef", httpClient: new HttpClient(handler));

        await Assert.ThrowsAsync<NotSupportedException>(() =>
            Ask(client, new DecisionRequest(JsonDocument.Parse("\"text\"").RootElement, Questions, contents)));
        Assert.Null(handler.Body);
    }

    [Fact]
    public async Task ThrowsNotSupportedForStateWithTextOrImagesWithoutState()
    {
        using var client = new OllamaDecisionClient(defaultModelId: "clef", httpClient: new HttpClient(new FakeHandler(AllAnswers)));
        JsonElement state = JsonDocument.Parse("\"text\"").RootElement;

        await Assert.ThrowsAsync<NotSupportedException>(() =>
            Ask(client, new DecisionRequest(state, Questions, [new TextContent("more text")])));
        await Assert.ThrowsAsync<NotSupportedException>(() =>
            Ask(client, new DecisionRequest(null, Questions, [new DataContent(new byte[] { 1 }, "image/png")])));
        await Assert.ThrowsAsync<NotSupportedException>(() =>
            Ask(client, new DecisionRequest(state, [new ChoiceDecisionQuestion("team", (string?)null, [new("a", "A"), new("b", "B")])])));
    }

    [Theory]
    [InlineData(HttpStatusCode.NotFound, false)]
    [InlineData(HttpStatusCode.BadRequest, false)]
    [InlineData(HttpStatusCode.TooManyRequests, true)]
    [InlineData(HttpStatusCode.InternalServerError, true)]
    public async Task MapsHttpErrorsToClientException(HttpStatusCode status, bool transient)
    {
        var handler = new FakeHandler("""{"error":"model 'nimble' not found"}""", status);
        using var client = new OllamaDecisionClient(defaultModelId: "nimble", httpClient: new HttpClient(handler));

        var ex = await Assert.ThrowsAsync<DecisionClientException>(() =>
            Ask(client, new DecisionRequest(JsonDocument.Parse("\"text\"").RootElement, Questions)));
        Assert.Equal(transient, ex.IsTransient);
        Assert.Contains("model 'nimble' not found", ex.Message);
    }

    [Fact]
    public void ReportsMetadata()
    {
        using var client = new OllamaDecisionClient(new Uri("http://ollama.test:11434"), "nimble");

        var metadata = Assert.IsType<DecisionClientMetadata>(client.GetService(typeof(DecisionClientMetadata)));
        Assert.Equal("ollama", metadata.ProviderName);
        Assert.Equal("nimble", metadata.DefaultModelId);
        Assert.Equal([DecisionKind.Binary, DecisionKind.Choice, DecisionKind.Score], metadata.SupportedKinds);
    }

    private static Task<DecisionResponse> Ask(IDecisionClient client, DecisionRequest request, DecisionOptions? options = null) =>
        client.GetResponseAsync(request, options, TestContext.Current.CancellationToken);

    private sealed class FakeHandler(string response, HttpStatusCode status = HttpStatusCode.OK) : HttpMessageHandler
    {
        public Uri? RequestUri { get; private set; }

        public JsonNode? Body { get; private set; }

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            RequestUri = request.RequestUri;
            Body = JsonNode.Parse(await request.Content!.ReadAsStringAsync(cancellationToken));
            return new HttpResponseMessage(status) { Content = new StringContent(response, Encoding.UTF8, "application/json") };
        }
    }
}
