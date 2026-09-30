# Ollama: three typed questions, one HTTP request

Ask about the same invented account-access ticket used by the other quickstarts,
but read **Binary, Choice and ordinal Score** together. [Ollama.cs](Ollama.cs) is
the complete standalone .NET 10 file app, not a fragment or a chat-client wrapper.
The project-owned `DecisionInference` packages are experimental, unofficial and
unpublished. No production certification or model-quality claim is implied.

## What runs where

Your .NET process serializes state/questions and validates a typed result.
A separately managed **Ollama v0.35.0 or later** server owns the tokenizer,
GGUF model and scoring runner. The adapter sends `POST /v1/systemone`; it does
not port an inference runtime, install/start a daemon, pull models or call
`/api/show`. No OllamaSharp, ONNX, TensorSharp, Python or Qevi dependency is needed.

Ollama is **not Ollaya**. The [Ollaya example](../Ollaya/README.md) uses an audited
ONNX service on port 11435; Ollama defaults to 11434 and selects an actual local
model. The [Jeeves example](../Jeeves/README.md) targets a different service on
8009, where the model field is only an echo label and does not select weights.
Even Ollama's echoed label is not proof of an immutable model digest.

## Safe first run

Use a full checkout, a .NET 10 SDK compatible with `global.json`, and PowerShell
at the **repository root**. First build may contact NuGet for code dependencies;
neither command below contacts a daemon, loads weights or downloads a model:

```powershell
dotnet run --file samples\quickstarts\Ollama\Ollama.cs -- --help
dotnet run --file samples\quickstarts\Ollama\Ollama.cs -- --check-config http://127.0.0.1:11434 tev1:0.8b
```

The second command checks argument shape only, **not** server availability,
version, model presence or compatibility. Invalid/missing configuration exits
with code 2. File apps reference local source projects and disable Native AOT;
see [shared SDK/build guidance](../../README.md#common-prerequisites-and-commands).

**Before authorizing a real call**, have the runtime owner independently provide
a supported server and trained local model, review its provenance/license/digest
and resource cost, and approve sending data to that endpoint. The pinned handler
requires compatible GGUF weights: a `qwen3.5` renderer, or an empty renderer with
the `qwen35` family, plus a scoring-capable runner. Compatible architecture alone
does not establish decision-task quality. Trained choices advertised upstream
include [nimble](https://ollama.com/library/nimble), [tev1:4b](https://ollama.com/library/tev1)
and [tev1:0.8b](https://ollama.com/library/tev1:0.8b); **not every installed model
works**. The pinned endpoint rejects cloud and does not auto-pull missing models.
This walkthrough neither acquires weights nor claims their licenses from a
training-code license.

Only after that separate authorization, this form performs a real request:

```powershell
dotnet run --file samples\quickstarts\Ollama\Ollama.cs -- http://127.0.0.1:11434 tev1:0.8b
```

Both endpoint and model are explicit. Use HTTPS or literal-loopback HTTP, never
insecure remote HTTP or `http://localhost`. The default library origin is
`http://127.0.0.1:11434`; the library has no guessed default model. There is no
fake key. The app disables redirects/cookies and does not log raw diagnostics.

## Follow the C# call

The ticket is a string; source-generated `TicketJson.Default.String` supplies
ordinary JSON serialization metadata. It is not a training or schema compiler.
Question IDs identify answers; descriptions supply meaning. The Choice IDs and
descriptions match the other support-team examples exactly. Binary asks whether
the ticket concerns account access. Score describes three **ordered** urgency
levels; it is not a percentage scale.

This is an exact excerpt from the complete app:

```csharp
var options = new OllamaGenerationOptions { KeepAliveSeconds = 60 };
DecisionResult result = await generator.GenerateAsync(ticket, questions,
    TicketJson.Default.String, options, deadline.Token);
var account = result.GetAnswer<BinaryDecisionAnswer>("account-access");
var team = result.GetAnswer<ChoiceDecisionAnswer>("support-team");
var urgency = result.GetAnswer<ScoreDecisionAnswer>("urgency");
```

The HTTP client is borrowed by the generator and disposed by the app, after the
generator. Reuse an appropriate HttpClient lifetime in a service. The 30-second
cancellation deadline covers HTTP waiting; it does not promise remote compute
was stopped. There is no success-shaped synthetic fallback on a failed call.

`KeepAliveSeconds` is **residency after the call**, not a timeout: null omits the
field and inherits the server setting; zero unloads after the call; negative
retains indefinitely; positive retains for that many seconds. This sample
explicitly chooses a finite 60 seconds. The nullable `int` is intentionally a
valid subset of upstream duration syntax, serialized as integer seconds, not
a Go duration string. Plain `DecisionGenerationOptions` still selects `ModelId`;
typed `Clone()` preserves both base fields and the residency field, with an
independent shallow extension dictionary. Nonempty input extension bags and
foreign derived options are rejected, not ignored. No thinking, temperature,
streaming or tools controls are offered.

## Read the result without overclaiming

The app prints the model label, P(true), every Choice probability, every Score
probability, the expected zero-based urgency index and observed usage. It uses
`G17` so display formatting does not pretend the protocol has a decimal grid.
**No live output was captured for this phase.** A synthetic test fixture is
protocol evidence, not a result from Nimble/Tev or a performance measurement.

Binary `noul` is a number: probability of true among false/true alternatives,
not a Boolean permission. Choice selects the highest probability, with the
first request-order option winning exact ties. Score is the probability-weighted
ordinal index; the returned legend must match the requested rubric. Complete
distributions, IDs, model echo and usage are validated; raw JSON is owned after
the response is disposed.

The server computes float64 softmax and ordinary JSON numbers. The adapter uses
existing `HighPrecision`, no rounding, renormalization or new precision enum.
The rounded probability examples in online docs are illustrative: a displayed
sum of .9999 may fail the actual high-precision distribution check. Native
`confidence = 1 - H(p)/ln(N)` measures concentration among the alternatives,
not calibrated correctness. It stays in raw evidence, not a universal API.

Usage is `Input=input_tokens`, `Output=output_tokens`, `Total=Input+Output`
with checked nonnegative integer arithmetic. Input includes each question's
complete rendered prompt even when prefixes are cached. Output counts internal
scoring, primer and retry tokens: it may exceed question count or be zero for a
direct scorer, and is **not JSON length**. Reasoning and server latency are not
reported; missing metrics are unknown, not invented zeros. This is not Jeeves'
reasoning-inclusive wire conversion.

## Limits and experiments

The provider accepts 1..64 questions, Choice/Score 2..26, nonblank string state
or JSON objects/arrays (including empty ones), and nonblank string instructions.
Even a portable Binary question with both descriptions still needs instructions
for this backend. Candidate descriptions are explicit, never null numeric
aliases. Tev's documented training range of **2..24 alternatives** differs from
the endpoint's **2..26**; accepting 26 is not evidence of trained quality at 26.

The exact final serialized UTF-8 request, including model, options and escaping,
must fit **65,536 bytes**. That is not a token budget. Each rendered prompt must
fit the actual server context with **two token positions of headroom**. The
server rejects over-budget input, never truncates it; the client does not invent
a character-to-token estimator.

Every question sees the full state and schema, but not the other predicted
answers. Adding a question can change another prediction through conditioning.
Our `q0`/numeric aliases preserve caller identity, but change the prompt relative
to a direct SDK call using semantic field keys. One HTTP request is not one
guaranteed native batched forward. Read [the beginner lesson](../../../docs/learning/05-ollama-decisions.md).

Before a separately authorized experiment, predict which description matters.
Then change one description, or remove one question, while holding the ticket
fixed. Explain what changed in the input before interpreting a changed answer.
Do not infer a benchmark, calibration, speedup or accuracy improvement from one
ticket. The app only prints predictions; application policy decides what to do.

## Troubleshooting and sources

Missing model: visible HTTP 404. Unsupported model/runner/context/schema: 400.
Oversized server request: 413. Loading/rendering/scoring failure: 500. The
adapter preserves status and bounded opt-in diagnostic text in
`OllamaApiException`; malformed success JSON/distributions raise
`DecisionProtocolException`. These are failures, not low-confidence predictions.
Check runtime ownership/configuration instead of silently switching providers.

The contract is pinned to **Ollama v0.35.0,
`cc4069396f3ad2c370c53eed2e4a42ac13adab84`**:
[compiler/answers](https://github.com/ollama/ollama/blob/cc4069396f3ad2c370c53eed2e4a42ac13adab84/decision/systemone.go),
[wire types](https://github.com/ollama/ollama/blob/cc4069396f3ad2c370c53eed2e4a42ac13adab84/decision/types.go),
[handler](https://github.com/ollama/ollama/blob/cc4069396f3ad2c370c53eed2e4a42ac13adab84/server/routes.go#L843-L926),
[scoring/usage](https://github.com/ollama/ollama/blob/cc4069396f3ad2c370c53eed2e4a42ac13adab84/llm/llama_server_score.go),
[duration parsing](https://github.com/ollama/ollama/blob/cc4069396f3ad2c370c53eed2e4a42ac13adab84/api/types.go#L1170-L1206).
The [current API docs](https://docs.ollama.com/api/systemone) may evolve beyond
that source pin. Verification here is offline fixtures, compiled safe CLI paths
and a native-free private package consumer, **not live compatibility, GPU,
accuracy or performance proof**.
