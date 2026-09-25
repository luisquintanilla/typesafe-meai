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
using var native = new TypeSafeChatClient(http, new() { ModelId = "jev-1.13.0", Policy = new(new(0.2, 0.8), 0.8) },
    "SYNTHETIC-NOT-A-CREDENTIAL");
var decorated = new CountingClient(native);
TypeSafeQuestion[] questions = [new NoulQuestion("followup", TypeSafeJson.Entry("Does the synthetic ticket need follow-up?"))];
const string state = "Synthetic: a payment failed.";
var nativeResponse = await decorated.GetResponseAsync(state, new ChatOptions
{
    AdditionalProperties = new() { [TypeSafeChatClient.QuestionsKey] = JsonSerializer.SerializeToElement(questions.ToDictionary(q => q.Id, q => q.ToJson())) }
});
var evidence = nativeResponse.Messages.Last().Contents.OfType<JevResultContent>().Single();
Require(((NoulAnswer)evidence.Answers["followup"]).Noul == 0.9 && evidence.Json.GetProperty("synthetic_extra").GetString() == "retained",
    "Native package evidence");
var serializer = new JsonSerializerOptions(AIJsonUtilities.DefaultOptions);
serializer.AddAIContentType<JevResultContent>(JevResultContent.TypeDiscriminator);
var restored = JsonSerializer.Deserialize<ChatResponse>(JsonSerializer.Serialize(nativeResponse, serializer), serializer)!;
Require(restored.Messages.Last().Contents.OfType<JevResultContent>().Single().Json.GetProperty("synthetic_extra").GetString() == "retained", "Custom subtype package serialization");
var typed = await decorated.GetResponseAsync<Decision>(state);
Require(typed.Result.Followup && typed.Messages.Last().Contents.OfType<JevResultContent>().Single().OutputTokens == 2, "Real MEAI typed-output package path");
var boolean = await decorated.GetResponseAsync<bool>("Does this payment ticket need follow-up?");
Require(boolean.Result, "Primitive root data wrapper");

IEvaluator evaluator = new TypeSafeEvaluator([new TypeSafeMetricDefinition("Followup", questions[0])]);
var evaluation = await evaluator.EvaluateAsync([new(ChatRole.User, "Synthetic prompt")],
    new(new ChatMessage(ChatRole.Assistant, "Scripted primary response")) { ModelId = "scripted-primary" },
    new ChatConfiguration(decorated),
    [new TypeSafeEvaluationContext("reference", TypeSafeJson.Entry(new { expected = "followup" }))]);
Require(evaluation.Metrics["Followup"] is NumericMetric { Value: 0.9 } metric &&
    metric.Metadata?["typesafe.v1.answer"].Contains("\"noul\":0.9", StringComparison.Ordinal) == true, "Separate evaluator package");
int effects = 0;
var function = AIFunctionFactory.Create((bool followup) => { effects++; return followup; }, "preview", "Preview a follow-up decision, without persistence.");
using var executing = decorated.AsBuilder().UseFunctionInvocation(configure: invocation =>
{
    invocation.MaximumConsecutiveErrorsPerRequest = 0;
    invocation.FunctionInvoker = async (context, token) =>
    {
        if (!ReferenceEquals(context.Function, function) || context.CallContent.Name != function.Name || effects != 0 ||
            JsonSerializer.SerializeToElement(context.Arguments).GetProperty("followup").ValueKind != JsonValueKind.True)
            throw new UnauthorizedAccessException("Only the one harmless preview with followup=true is authorized.");
        token.ThrowIfCancellationRequested();
        context.Terminate = true;
        return await context.Function.InvokeAsync(context.Arguments, token);
    };
}).Build();
var invoked = await executing.GetResponseAsync(state, new ChatOptions { Tools = [function], ToolMode = ChatToolMode.RequireSpecific("preview") });
Require(effects == 1 && invoked.Messages.SelectMany(m => m.Contents).OfType<FunctionResultContent>().Single().Exception is null, "Real one-shot middleware package execution");
Require(handler.Sends == 5 && decorated.Calls == 5, "Every package path used the decorated provider with no extra post-execution inference");
Console.WriteLine("PACKAGE PROOF PASS: native evidence/serialization + actual MEAI DTO and primitive output + evaluator + standard one-shot tools; 5 decorated calls, 5 synthetic HTTP sends, no network.");

static void Require(bool condition, string claim)
{
    if (!condition) throw new InvalidOperationException($"Package proof failed: {claim}.");
}

internal sealed class Decision
{
    [JsonPropertyName("followup"), System.ComponentModel.Description("Does the ticket need follow-up?")]
    public bool Followup { get; init; }
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
