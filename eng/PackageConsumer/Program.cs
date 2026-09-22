using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.AI.Evaluation;
using TypeSafe.MEAI;
using TypeSafe.MEAI.Evaluation;

using var handler = new SyntheticHandler();
using var http = new HttpClient(handler);
using var native = new TypeSafeChatClient(http, new() { ModelId = "jev-1.13.0" },
    "SYNTHETIC-NOT-A-CREDENTIAL");
var decorated = new CountingClient(native);
TypeSafeQuestion[] questions = [new NoulQuestion("followup", TypeSafeJson.Entry("Does the synthetic ticket need follow-up?"))];
var state = TypeSafeState.FromText("Synthetic: a payment failed.");
var evidence = await decorated.GetTypeSafeResponseAsync(state, questions);
Require(evidence.GetAnswer<NoulAnswer>("followup").Noul == 0.9 && evidence.Json.GetProperty("synthetic_extra").GetString() == "retained",
    "Native package evidence");

var schema = new TypeSafeSchemaChatClient(decorated, new Dictionary<string, TypeSafeFieldBinding>
{
    ["/followup"] = new("Does the ticket need follow-up?", booleanPolicy: new(0.2, 0.8))
});
var typed = await schema.GetResponseAsync<Decision>(state.ToMessage());
Require(typed.Result.Followup && typed.GetTypeSafeEvidence().OutputTokens == 2, "Real MEAI typed-output package path");

IEvaluator evaluator = new TypeSafeEvaluator([new TypeSafeMetricDefinition("Followup", questions[0])]);
var evaluation = await evaluator.EvaluateAsync([new(ChatRole.User, "Synthetic prompt")],
    new(new ChatMessage(ChatRole.Assistant, "Scripted primary response")) { ModelId = "scripted-primary" },
    new ChatConfiguration(decorated),
    [new TypeSafeEvaluationContext("reference", TypeSafeJson.Entry(new { expected = "followup" }))]);
Require(evaluation.Metrics["Followup"] is NumericMetric { Value: 0.9 } metric &&
    metric.Metadata?["typesafe.v1.answer"].Contains("\"noul\":0.9", StringComparison.Ordinal) == true, "Separate evaluator package");
Require(handler.Sends == 3 && decorated.Calls == 3, "Every package path used the decorated native client");
Console.WriteLine("PACKAGE PROOF PASS: native + actual MEAI typed output + separate evaluator; 3 decorated calls, 3 synthetic HTTP sends, no network.");

static void Require(bool condition, string claim)
{
    if (!condition) throw new InvalidOperationException($"Package proof failed: {claim}.");
}

[JsonUnmappedMemberHandling(JsonUnmappedMemberHandling.Disallow)]
internal sealed class Decision
{
    [JsonPropertyName("followup")]
    public required bool Followup { get; init; }
}

internal sealed class CountingClient(IChatClient inner) : DelegatingChatClient(inner)
{
    internal int Calls;
    public override Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        Calls++;
        return base.GetResponseAsync(messages, options, cancellationToken);
    }
}

internal sealed class SyntheticHandler : HttpMessageHandler
{
    internal int Sends;
    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        Sends++;
        using var body = JsonDocument.Parse(await request.Content!.ReadAsStringAsync(cancellationToken));
        var answers = body.RootElement.GetProperty("questions").EnumerateObject().ToDictionary(p => p.Name,
            p => p.Value.GetProperty("type").GetString() == "noul" ? new { type = "noul", noul = 0.9 } :
            throw new InvalidOperationException("Unexpected package-proof question."));
        return new(HttpStatusCode.OK)
        {
            Content = new StringContent(JsonSerializer.Serialize(new
            {
                model = "jev-1.13.0", answers, usage = new { input_tokens = 5, output_tokens = 2 }, synthetic_extra = "retained"
            }), Encoding.UTF8, "application/json")
        };
    }
}
