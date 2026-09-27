using System.Net;
using System.Text;
using System.Text.Json;
using DecisionInference;

internal sealed class RecordedJevHandler : HttpMessageHandler
{
    public int Calls { get; private set; }
    public JsonElement LastBody { get; private set; }
    public static DecisionRequest Request() => new(JsonSerializer.SerializeToElement("An email requests a password."),
        [new BinaryDecisionQuestion("safe", "Is it safe?"),
         new ChoiceDecisionQuestion("category", "Choose one category.", [new("legitimate", "Legitimate"), new("phishing", "Phishing")]),
         new ScoreDecisionQuestion("severity", "Assess severity.", ["low", "medium", "high"])]);

    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        Calls++;
        LastBody = JsonSerializer.Deserialize<JsonElement>(await request.Content!.ReadAsStringAsync(cancellationToken));
        return new(HttpStatusCode.OK)
        {
            Content = new StringContent("""
                {"model":"jev-1.13.0","usage":{"input_tokens":30,"output_tokens":6,"total_tokens":36},
                 "answers":{"q0":{"type":"noul","noul":0.1},
                 "q1":{"type":"choice","choice":"1","confidence":0.7,"probabilities":{"0":0.1,"1":0.9}},
                 "q2":{"type":"score","score":1.6,"confidence":0.7,"probabilities":{"0":0.1,"1":0.2,"2":0.7},
                 "legend":{"0":"low","1":"medium","2":"high"}}},"fixture":"synthetic-not-live"}
                """, Encoding.UTF8, "application/json")
        };
    }
}
