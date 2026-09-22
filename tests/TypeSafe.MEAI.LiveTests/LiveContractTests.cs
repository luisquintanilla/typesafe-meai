using System.Text.Json;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace TypeSafe.MEAI.LiveTests;

public sealed class LiveContractTests
{
    public static bool Approved => Environment.GetEnvironmentVariable("TYPESAFE_LIVE_APPROVAL") == "SYNTHETIC-3-REQUESTS";

    [Fact(Skip = "NOT RUN: separate approval required.", SkipUnless = nameof(Approved))]
    public async Task Models_UsesDocumentedWrapper()
    {
        using var services = CreateServices();
        var models = new TypeSafeModelClient(services.GetRequiredService<IHttpClientFactory>().CreateClient("typesafe:live"),
            new(), Environment.GetEnvironmentVariable("TYPESAFE_API_KEY")!);
        var result = await models.GetModelsAsync(TestContext.Current.CancellationToken);
        Assert.Equal(JsonValueKind.Array, result.GetProperty("models").ValueKind);
    }

    [Fact(Skip = "NOT RUN: separate approval required.", SkipUnless = nameof(Approved))]
    public async Task MixedBatch_ValidatesContractNotPredictedProbabilities()
    {
        using var services = CreateServices();
        TypeSafeQuestion[] questions =
        [
            new NoulQuestion("followup", TypeSafeJson.Entry("Does this synthetic support ticket require follow-up?")),
            new ChoiceQuestion("destination", TypeSafeJson.Entry("Which team handles the request?"),
                new Dictionary<string, JsonElement> { ["billing"] = TypeSafeJson.Entry("Payments"), ["technical"] = TypeSafeJson.Entry("Software problems") }),
            new ScoreQuestion("quality", TypeSafeJson.Entry("Rate how useful the draft response is."),
                new[] { TypeSafeJson.Entry("Unhelpful"), TypeSafeJson.Entry("Partly useful"), TypeSafeJson.Entry("Actionable") })
        ];
        var result = await services.GetRequiredKeyedService<IChatClient>("live").GetTypeSafeResponseAsync(
            TypeSafeState.Parse("""{"ticket":"Synthetic: payment failed.","draft":"Please check the payment details."}"""),
            questions, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(3, result.Answers.Count);
        Assert.NotEmpty(result.Model);
    }

    [Fact(Skip = "NOT RUN: unresolved SDK-compatible shapes require separate approval.", SkipUnless = nameof(Approved))]
    public async Task NullAndStructuredVariants_AreExplicitCompatibilityProbe()
    {
        using var services = CreateServices();
        TypeSafeQuestion[] questions =
        [
            new NoulQuestion("omitted"),
            new NoulQuestion("explicit_null", TypeSafeJson.Entry<object?>(null), TypeSafeJson.Entry<object?>(null)),
            new ScoreQuestion("structured", TypeSafeJson.Entry("Rate the absent state."),
                new[] { TypeSafeJson.Entry<object?>(null), TypeSafeJson.Entry(new { meaning = "present" }) })
        ];
        var result = await services.GetRequiredKeyedService<IChatClient>("live").GetTypeSafeResponseAsync(
            TypeSafeState.Parse("null"), questions, cancellationToken: TestContext.Current.CancellationToken);
        Assert.Equal(3, result.Answers.Count);
    }

    private static ServiceProvider CreateServices()
    {
        Assert.True(Approved, "Live approval must name the exact three synthetic requests.");
        string? key = Environment.GetEnvironmentVariable("TYPESAFE_API_KEY");
        Assert.False(string.IsNullOrWhiteSpace(key), "Explicitly approved live run requires TYPESAFE_API_KEY.");
        ServiceCollection services = new();
        services.AddTypeSafeChatClient("live", key!, new() { ModelId = "jev-1.13.0" });
        return services.BuildServiceProvider();
    }
}
