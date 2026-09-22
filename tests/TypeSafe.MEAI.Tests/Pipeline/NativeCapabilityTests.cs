using System.Text.Json;
using Microsoft.Extensions.AI;
using TypeSafe.MEAI.Tests.Fixtures;
using Xunit;

namespace TypeSafe.MEAI.Tests.Pipeline;

[Trait("Stage", "A")]
public sealed class NativeCapabilityTests
{
    public static TheoryData<string, bool> UnsupportedOptions
    {
        get
        {
            TheoryData<string, bool> data = new();
            foreach (var option in Names)
            {
                data.Add(option, false);
                data.Add(option, true);
            }
            return data;
        }
    }

    private static readonly string[] Names =
    [
        "ConversationId", "Instructions", "Temperature", "MaxOutputTokens", "TopP", "TopK",
        "FrequencyPenalty", "PresencePenalty", "Seed", "Reasoning", "StopSequences", "StopSequences-empty",
        "AllowMultipleToolCalls", "ToolMode", "ToolMode-none", "ToolMode-required", "ToolMode-specific", "Tools", "Tools-empty", "AllowBackgroundResponses",
        "ContinuationToken", "RawRepresentationFactory", "ResponseFormat-text", "ResponseFormat-schema", "reserved"
    ];

    private static void SetUnsupported(ChatOptions options, string name)
    {
        switch (name)
        {
            case "ConversationId": options.ConversationId = ""; break;
            case "Instructions": options.Instructions = ""; break;
            case "Temperature": options.Temperature = 0; break;
            case "MaxOutputTokens": options.MaxOutputTokens = 0; break;
            case "TopP": options.TopP = 0; break;
            case "TopK": options.TopK = 0; break;
            case "FrequencyPenalty": options.FrequencyPenalty = 0; break;
            case "PresencePenalty": options.PresencePenalty = 0; break;
            case "Seed": options.Seed = 0; break;
            case "Reasoning": options.Reasoning = new(); break;
            case "StopSequences": options.StopSequences = ["stop"]; break;
            case "StopSequences-empty": options.StopSequences = []; break;
            case "AllowMultipleToolCalls": options.AllowMultipleToolCalls = false; break;
            case "ToolMode": options.ToolMode = ChatToolMode.Auto; break;
            case "ToolMode-none": options.ToolMode = ChatToolMode.None; break;
            case "ToolMode-required": options.ToolMode = ChatToolMode.RequireAny; break;
            case "ToolMode-specific": options.ToolMode = ChatToolMode.RequireSpecific("synthetic"); break;
            case "Tools": options.Tools = [AIFunctionFactory.Create(() => "never-invoked", "synthetic")]; break;
            case "Tools-empty": options.Tools = []; break;
            case "AllowBackgroundResponses": options.AllowBackgroundResponses = false; break;
            case "ContinuationToken": options.ContinuationToken = ResponseContinuationToken.FromBytes(new byte[] { 17 }); break;
            case "RawRepresentationFactory": options.RawRepresentationFactory = _ => throw new InvalidOperationException("must not invoke"); break;
            case "ResponseFormat-text": options.ResponseFormat = ChatResponseFormat.Text; break;
            case "ResponseFormat-schema": options.ResponseFormat = ChatResponseFormat.ForJsonSchema(NativeFixture.Json("""{"type":"object"}""")); break;
            case "reserved": options.AdditionalProperties!["typesafe.future.unsupported"] = "value"; break;
            default: throw new ArgumentOutOfRangeException(nameof(name));
        }
    }

    [Theory]
    [MemberData(nameof(UnsupportedOptions))]
    public async Task UnsupportedOptions_RejectBeforeHttpAndBeforePrimedRealCache(string name, bool primedCache)
    {
        using var handler = new RecordingHandler();
        using var http = new HttpClient(handler, disposeHandler: false);
        using var services = NativeFixture.Services(handler, cache: true);
        using IChatClient client = primedCache ? services.Client() : new TypeSafeChatClient(http, NativeFixture.Settings, NativeFixture.Key);
        if (primedCache)
        {
            var primed = await client.GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), TestContext.Current.CancellationToken);
            NativeFixture.AssertChat(primed);
            var cached = await client.GetResponseAsync(NativeFixture.Messages(), NativeFixture.Options(), TestContext.Current.CancellationToken);
            NativeFixture.AssertChat(cached);
            Assert.Equal(1, handler.Sends);
        }
        var options = NativeFixture.Options();
        SetUnsupported(options, name);
        if (name == "RawRepresentationFactory")
        {
            var serialized = JsonSerializer.Serialize(options);
            var restored = JsonSerializer.Deserialize<ChatOptions>(serialized)!;
            Assert.Null(restored.RawRepresentationFactory);
        }
        await Assert.ThrowsAsync<NotSupportedException>(() =>
            client.GetResponseAsync(NativeFixture.Messages(), options, TestContext.Current.CancellationToken));
        Assert.Equal(primedCache ? 1 : 0, handler.Sends);
    }

    [Fact]
    public void PinnedChatOptions_PublicPropertyAuditHasNoUntestedCapability()
    {
        string[] expected =
        [
            "ConversationId", "Instructions", "Temperature", "MaxOutputTokens", "TopP", "TopK", "FrequencyPenalty",
            "PresencePenalty", "Seed", "Reasoning", "ResponseFormat", "ModelId", "StopSequences",
            "AllowMultipleToolCalls", "ToolMode", "Tools", "AllowBackgroundResponses", "ContinuationToken",
            "RawRepresentationFactory", "AdditionalProperties"
        ];
        Assert.Equal(expected.Order(), typeof(ChatOptions).GetProperties().Select(p => p.Name).Order());
    }
}
