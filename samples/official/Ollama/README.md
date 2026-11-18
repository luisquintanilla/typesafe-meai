# Ollama decisions with the proposed Microsoft.Extensions.AI APIs

This sample implements `IDecisionClient`, proposed in
[dotnet/extensions#7795](https://github.com/dotnet/extensions/pull/7795), over Ollama's decision endpoint
(`POST /v1/systemone`, Ollama 0.35.0+). It is a second implementation of the same abstraction used by the
[Azure OpenAI sample](../AzureOpenAI/README.md), written with plain `HttpClient` and no Ollama SDK.

The APIs are experimental, unreleased and may change. The packages are built locally from that pull request.

- [OllamaDecisionClient.cs](../OllamaDecisions/OllamaDecisionClient.cs): the adapter
- [OllamaDecisions.Tests](../OllamaDecisions.Tests/OllamaDecisionClientTests.cs): request/response mapping tests with a fake handler
- [Ollama.cs](Ollama.cs): the sample

It asks Binary, Choice and Score questions about a support ticket twice: once from JSON state with `nimble`,
and once from text plus [an image](../AzureOpenAI/declined-payment.png) with `clef-flash`, which accepts images
(Ollama 0.35.1+).

## Build the packages

Clone dotnet/extensions with the #7795 branch checked out, run `restore.cmd` once, then from this repository's root:

```powershell
.\eng\build-official-decision-packages.ps1 -TypedLayerCheckout <path-to-#7795-worktree>
```

`-OpenAIAdapterCheckout` is only needed for the Azure OpenAI sample. The packages are written to the ignored
`artifacts\official-feed` folder as `11.0.0-decisions.1`; [NuGet.config](../NuGet.config) reads that feed.

## Run

```powershell
ollama pull nimble
ollama pull clef-flash
dotnet run --file samples\official\Ollama\Ollama.cs
dotnet test samples\official\OllamaDecisions.Tests\OllamaDecisions.Tests.csproj
```

`OLLAMA_ENDPOINT`, `OLLAMA_DECISION_MODEL` and `OLLAMA_IMAGE_DECISION_MODEL` override the defaults
(`http://localhost:11434`, `nimble`, `clef-flash`).

Example output (Ollama 0.40.16, Windows x64):

```text
== JSON state (model nimble)
refund: P(true) = 0.3%
category: account (confidence 0.99)
  billing    0.3%
  technical  0.6%
  account    99.1%
urgency: expected score 1.59 (0=low, 2=high) (confidence 0.39)
  low        7.4%
  medium     26.0%
  high       66.6%
Tokens: 1171 in, 0 out
== Text + image (model clef-flash)
refund: P(true) = 1.8%
category: billing (confidence 0.91)
  billing    94.3%
  technical  4.3%
  account    1.4%
urgency: expected score 1.56 (0=low, 2=high) (confidence 0.34)
  low        4.2%
  medium     36.0%
  high       59.8%
Tokens: 423 in, 0 out
```

## How the request maps

| `IDecisionClient` | Ollama |
|---|---|
| `DecisionRequest.State` | `state` (JSON object, array or string) |
| `TextContent` in `Contents` | concatenated into `state` when `State` is null |
| `DataContent` (PNG, JPEG, WebP) in `Contents` | `images` (base64) |
| `BinaryDecisionQuestion` | `noul`; true/false descriptions become `criteria` |
| `ChoiceDecisionQuestion` | `choice`; candidates become `criteria` |
| `ScoreDecisionQuestion` | `score`; level descriptions become ordered `criteria` |
| `DecisionOptions.ModelId` | `model` (falls back to the client's default model) |
| `DecisionOptions.RawRepresentationFactory` returning a `JsonObject` | starting request body, for example `keep_alive` |
| `usage`, `prompt_eval_cached_count` | `DecisionResponse.Usage` |
| per-answer `confidence` | answer `AdditionalProperties["confidence"]` |

Answers are correlated by question ID and returned in request order. Missing, extra or duplicate answers,
a type that does not match the question kind, and probabilities for options that were not requested throw
`DecisionProtocolException`. Score probabilities, which Ollama keys by level index, are mapped back to level IDs.
HTTP 408, 429, 5xx, network failures and timeouts throw a transient `DecisionClientException`; other HTTP
errors throw a non-transient one with Ollama's error message.

## Limitations

- Ollama takes one `state` and a separate image list, so the order of text and images in `Contents` is not
  preserved, and `State` cannot be combined with text contents. Those requests, images without any state,
  URI content and other media types throw `NotSupportedException` rather than dropping content.
- Ollama requires instructions for every question; empty instructions throw `NotSupportedException`.
- Score level IDs are not sent to the model, only their descriptions in order.
- Question and candidate limits are enforced by the server, not by the adapter.
- Ollama has no response ID and no refusal, so `ResponseId` is null and answers are never `UnansweredDecisionAnswer`.
