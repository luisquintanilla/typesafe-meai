# Decision-first side-by-side experiment

This is an experimental, project-owned `DecisionInference` API, **not** a
Microsoft/Agent Framework API, a replacement for `IChatClient`, or a quality
benchmark. The chat implementation at `bc2f307ed2b7e8c5ea0eac5e95dabeb4868b7c19`
remains unchanged. Build the additive experiment with
`dotnet build eng\DecisionComparison.slnx`; the original solution remains the
only solution at the repository root.

## Start with the application learning path

New to decision models? Begin with the [sample catalog](../samples/README.md)
and [beginner learning guide](learning/README.md). Six independent .NET 10
file-based quickstarts cover Julia, Laya, Jev, Ollaya, Jeeves and Qwen; scenarios
compare Julia/Laya with sequential lifetimes and Jeeves reasoning budgets.
Each has a detailed adjacent model/setup
walkthrough. This document remains the advanced architecture/evidence reference.

The project-based proof harness now lives in
[`eng\DecisionProof`](../eng/DecisionProof/README.md), separate from teaching
code. The original chat sample remains unchanged. The root README now leads
with this branch's decision experiment; the earlier chat guide is preserved at
[`docs\chat-adapter.md`](chat-adapter.md) and remains the README embedded in both
chat packages. Baseline preservation permits this deliberate documentation
exception and only the exact README source/destination item substitution in
the two chat project files. Runtime, dependencies and other original files
remain protected; historical all-91-file receipts below retain their original scope.

## Jeeves: current additive HTTP integration

`DecisionInference.Jeeves` implements the existing `IDecisionGenerator` contract
against [PostHog/jeeves at f04ec5567301450dcaae0210dd54deeb4f647f87](https://github.com/PostHog/jeeves/tree/f04ec5567301450dcaae0210dd54deeb4f647f87).
It reuses SystemOne aliasing, typed answer validation and bounded transport,
not an ONNX/CUDA port, model manager or generic reasoning framework.
The [quickstart](../samples/quickstarts/Jeeves/README.md) disables thinking;
the [budget scenario](../samples/scenarios/ReasoningBudget/README.md) uses a
finite 128-token cap. [The lesson](learning/04-reasoning-budgets.md) explains
their accounting and interpretation.

This phase makes real runtime additions: `TwoDecimalPlaces`, generalized
two/four-decimal interval arithmetic, an internal prevalidated-usage seam in
the shared decoder, and the new HTTP provider. It is not an unchanged-runtime
documentation phase. Legacy decoder calls still validate their original
input/output/optional-total usage fields and preserve their old mappings.
All 382 prior full public API entries are retained. The current reflection
inventory has 444 entries: 61 across seven new Jeeves public types and one
enum-field addition. Existing enum values stay 0 and 1; the new precision is 2.
Older loader, teaching and full-gate counts below remain historical evidence,
not results of this new phase.

The pinned API sends independently rounded two-decimal probabilities, native
Score and confidence. Its usage requires a different MEAI mapping:
`Input=input_tokens`, `Reasoning=reasoning_tokens`,
`Output=output_tokens+reasoning_tokens`, `Total=Input+Output`, with checked
nonnegative integer arithmetic. Original answer-JSON `output_tokens` remains
under `UsageDetails.AdditionalCounts["jeeves.answerJsonTokens"]` and in the
owned raw envelope. MEAI's [reasoning-count remarks](https://learn.microsoft.com/dotnet/api/microsoft.extensions.ai.usagedetails.reasoningtokencount#remarks)
include reasoning in output; these counters are not compute or billing claims.

Null/plain generation options omit the wire `options` object. Explicit
`JeevesGenerationOptions` sends all four upstream fields, including null
threshold clearing, and snapshots them before the HTTP wait; derived `Clone`
also preserves base properties. Defaults are think=true, max_think=2560,
nothink_threshold=null, return_reasoning=false. Unsupported derived options,
nonempty input extension bags, negative caps and nonfinite/out-of-range
thresholds fail before sending. Choice remains 1..255 and Score 2..10.

The response's optional reasoning map is correlated through the same question
aliases. Explicit opt-in is required; requested-but-missing or unsolicited
reasoning is rejected. Each owned `JeevesReasoning` snapshot validates the
pinned engine's thought/closed/token/text relationships, per-question caps and
aggregate equality to reasoning usage. A closing-only token legitimately has
empty text; a positive unclosed chain is valid. The immutable dictionary and
server `LatencyMilliseconds` are in `JeevesResultMetadata` at
`AdditionalProperties["jeeves.metadata"]`. Raw text is not logged and is not
a faithful-explanation guarantee. `latency_ms` excludes server lock wait and
response serialization/tokenization, unlike whole-client elapsed time.

The engine's threshold compares `max(nothink_probs)`, not returned normalized
Choice confidence. ModelId is exactly a request/echo label, never dynamic
weight selection or checkpoint attestation. There is no automatic `/v1/models`
discovery, `/api/show` profile or invented key. The default unauthenticated
origin is `http://127.0.0.1:8009`; HTTPS or literal-loopback HTTP is required,
not `http://localhost`. Isolated authorization, strict UTF-8/duplicate checks,
bounded response buffering, request IDs, cancellation and ownership reuse the
existing transport. Client cancellation does not promise GPU cancellation.
Ordinal aliases keep caller IDs opaque, but upstream renders those aliases
into prompts, so direct semantic-key SDK requests are not byte-identical.

Primary protocol evidence is
[api.py](https://github.com/PostHog/jeeves/blob/f04ec5567301450dcaae0210dd54deeb4f647f87/inference/api.py),
[types.py](https://github.com/PostHog/jeeves/blob/f04ec5567301450dcaae0210dd54deeb4f647f87/inference/types.py),
[serve.py](https://github.com/PostHog/jeeves/blob/f04ec5567301450dcaae0210dd54deeb4f647f87/inference/serve.py),
[engine.py](https://github.com/PostHog/jeeves/blob/f04ec5567301450dcaae0210dd54deeb4f647f87/inference/engine.py)
and [format.py](https://github.com/PostHog/jeeves/blob/f04ec5567301450dcaae0210dd54deeb4f647f87/prep/format.py).
The integration proof is offline fixtures, retained shared-surface regressions,
compiled file apps and a fresh Jeeves-only package consumer without native
stacks. No authorized service/GPU endpoint was supplied: no weights, daemon,
live inference, hardware measurement or calibration claim is part of this phase.

## Start with a complete local generator

For ordinary application code, load a generator rather than assembling stages:

```csharp
using DecisionInference;
using DecisionInference.Julia;

using IDecisionGenerator generator =
    JuliaDecisionGenerator.LoadFromDirectory(@"artifacts\decision-models\julia");
```

With `using DecisionInference.Laya;`, use
`LayaDecisionGenerator.LoadFromDirectory(@"artifacts\decision-models\laya")`.
Both own their resources and use the existing owning path constructors.
For thread/session configuration, each also offers a single, unambiguous
`Load(new() { ... })` overload with provider-specific required/init paths and
an optional `Action<SessionOptions> ConfigureSession`. These construction types
are distinct from per-call `DecisionGenerationOptions`.

The directory layouts are deterministic: Julia binds `model.onnx` and
`tokenizer.json`; Laya binds `laya.onnx`, `tokenizer\tokenizer.json` and
`laya_config.json`. ORT resolves graph-declared external weights (the pinned
graphs require `model.onnx.data` / `laya.onnx.data` respectively). No scanning,
substitution, download or unused config/parity-file requirement is introduced.
Paths/files are validated and made absolute before configuration. Temporary
session options are configured once and disposed even on failure; exceptions
propagate, and caller configuration is not retained.

See the [Julia](../samples/quickstarts/Julia/README.md) and
[Laya](../samples/quickstarts/Laya/README.md) quickstarts for both typed factories,
directory layouts, ownership and runnable file-app commands. The former
`generate` proof-host dispatch has been retired. Existing explicit path and advanced stage constructors are unchanged,
including borrowed-options/stage ownership defaults. The tensor-export and
cancellation commands below remain **advanced proof harnesses**, not the
application happy path.

The additive loading phase retains all 353 prior full API entries unchanged and
adds 29 entries (four factory methods and two required/init options types,
including accessors). Its scoped validation passed 241 cases: 52 new loader
cases plus 189 retained Julia/Laya stage and lifecycle cases. Both fresh
provider-only package consumers exercised directory and configured loading with
retained real CPU models, and all four then-current `generate` examples ran. Nine unchanged
external archives were checked against previously signature-verified bytes;
this is not a rerun of the historical 1,210-case/66-archive full gate below.
Native failure/recovery and temporary `SessionOptions.IsClosed` are directly
tested. Failure-time native tokenizer/session handles are not returned by the
constructors, so those unreachable allocations' cleanup relies on the unchanged
exception-safe construction paths, not a claimed native leak measurement.

Once loaded, consumption is provider-neutral:

```csharp
using DecisionInference;
using System.Text.Json.Serialization;

static async Task<ChoiceDecisionAnswer> ChooseAsync(
    IDecisionGenerator generator, CancellationToken token)
{
    DecisionQuestion[] questions =
        [new ChoiceDecisionQuestion("category", "Choose one category.",
            [new("legitimate", "Legitimate email"), new("phishing", "Credential theft")])];
    var metadata = generator.GetRequiredService<DecisionGeneratorMetadata>();
    var options = new DecisionGenerationOptions { ModelId = metadata.DefaultModelId };
    DecisionResult result = await generator.GenerateAsync(
        "An email asks for a password.", questions, DecisionJsonContext.Default.String,
        options.Clone(), token);
    return result.GetAnswer<ChoiceDecisionAnswer>("category");
}
// Application policy, not the generator, decides whether to act.

[JsonSerializable(typeof(string))]
internal partial class DecisionJsonContext : JsonSerializerContext;
```

## Generator naming migration

The capability-oriented `IDecisionGenerator.GenerateAsync` names align with
MEAI's [embedding generator](https://learn.microsoft.com/dotnet/api/microsoft.extensions.ai.iembeddinggenerator-2.generateasync)
and [image generator](https://learn.microsoft.com/dotnet/api/microsoft.extensions.ai.iimagegenerator.generateasync).
"Generator" describes producing typed decision results; it does not imply
autoregressive text generation. The deliberate `DecisionInput`/`DecisionResult`
vocabulary is ours: MEAI image generation still uses
`ImageGenerationRequest`/`ImageGenerationResponse`, so Microsoft does **not**
require Input/Result. No generic generator type is needed for these fixed
decision kinds.

This is an explicit source/binary-breaking rename of the **unreleased
experimental API**, with coordinated consumers and no compatibility aliases.
Package IDs, namespaces, assembly identities, question/answer types and
`Answers`, `GetAnswer`, `GetService`, `Dispose`, usage and extension ownership
remain unchanged. Construction options are distinct from per-call options.

| Before | Current |
|---|---|
| `IDecisionClient` | `IDecisionGenerator` |
| `DecisionRequest` / `DecisionResponse` | `DecisionInput` / `DecisionResult` |
| Decision `GetResponseAsync` overloads | `GenerateAsync` |
| `DecisionOptions` | `DecisionGenerationOptions` |
| `DecisionClientMetadata` | `DecisionGeneratorMetadata` |
| `DecisionClientExtensions` / `DecisionResponseExtensions` | `DecisionGeneratorExtensions` / `DecisionResultExtensions` |
| `TypeSafeDecisionClient` | `TypeSafeDecisionGenerator` |
| `OllayaDecisionClient` | `OllayaDecisionGenerator` |
| `LayaDecisionClient` | `LayaDecisionGenerator` |
| `JuliaDecisionClient` | `JuliaDecisionGenerator` |
| `QwenDecisionClient` | `QwenDecisionGenerator` |
| `TypeSafeDecisionOptions` / `OllayaDecisionOptions` | `TypeSafeDecisionGeneratorOptions` / `OllayaDecisionGeneratorOptions` |
| `LayaInputBatch.Request` / `JuliaInputBatch.Request` | `LayaInputBatch.Input` / `JuliaInputBatch.Input` |
| `QwenRawResponse` | `QwenRawResult` |
| Decision parameters `request` / `client` / `response` | `input` / `generator` / `result` |

The original `IChatClient.GetResponseAsync`, actual HTTP request/response
concepts, provider wire fields, model IDs and diagnostic metadata keys are not
renamed. Historical evidence remains immutable. This document was previously
`docs\decision-client-design.md`; the extraction commit
`087e2fcab58bcb62224179ada0abcc6f97954b2f` retained the old public names.
This later naming-only commit changes those names separately, not the extraction.

The settled naming gate retained **1,210 passing cases** (650 original + 560
decision), with no added cases, failures or skips. Fresh, separate-process
reflection snapshots retain all **353 full API entries across eight assemblies**:
143 entries change under the explicit type/member/parameter map and none change
otherwise. Both internal assemblies still export zero types. Every token in the
40 runtime/test/host C# files matches the recorded semantic rename and narrow
contextual corrections, including literals except enumerated diagnostic wording;
no operators, calls or assertion logic are erased by the comparison.

Both Laya and Julia managed proof JSON files are byte-identical to extraction,
and both independent oracle results are exactly equal. The combined package host
and isolated Laya/Julia/Ollaya consumers retain identical package/version closures;
66 external archives were hash/signature-verified. Of the 151 final non-document
inputs, ten C# paths are renamed and 29 contents change; all 91 original baseline
fingerprints remain unchanged. Runtime accounting remains 25 C# files / 2,169
physical lines. One preliminary gate was interrupted during baseline tests to
incorporate the final two-token sample local rename, then the final input
snapshot completed the settled gate. No model download or daemon restart was
needed, and there is no new performance or calibration claim.

## Decision semantics

Instructions, descriptions and rubric levels are ordinary strings. State is
owned JSON, not a chat message or a schema. Question/candidate IDs are nonblank,
ordinal, unique and never trimmed or normalized. IDs do not supply instructions
or candidate meaning. Null Choice instructions resolve to `"Choose one option."`;
null Score instructions resolve to `"Assess the state against the ordered rubric."`.
Binary needs a nonblank statement or both meaningful outcome descriptions;
explicitly supplied empty/whitespace Binary descriptions are rejected.
Candidate/rubric descriptions must be nonblank; explicit empty instructions on
described choices/scores remain explicit. These are intentional decision-only
constructor migrations, not changes to the original chat contract.
Portable collections and
distributions are snapshotted and read-only. `UsageDetails` is the
existing MEAI primitive, copied with its standard `Add` method. Unknown counts
remain null; local prepared token counts are not billable Jev token counts.

Binary returns the probability of true, Choice an exact selected ID and a
complete exclusive distribution, and Score a complete ordered distribution
with `ExpectedIndex = sum(index * observedProbability)`. All supplied probabilities
must be finite and in `[0,1]`. Precision is intrinsic to each answer:

* `HighPrecision` preserves the original distribution tolerance `1e-6` and native
  Score agreement tolerance `1e-6 * (levelCount-1)`. Jev and embedded models use it.
* `FourDecimalPlaces` requires the exact nearest-double representation of a
  .0001 grid value, without additional epsilon. Each observation denotes a
  clipped `[p-.00005,p+.00005]` interval.
  Decimal arithmetic must establish that a normalized latent distribution exists.
  Inclusive endpoints conservatively admit exact rounding ties; this is not
  extra model uncertainty, calibration, or a caller-supplied arbitrary tolerance.
* `TwoDecimalPlaces` uses the same rules on the .01 grid with clipped
  `[p-.005,p+.005]` intervals. It does not relax the other precision modes.

Ollaya selects four-decimal precision and Jeeves two-decimal precision automatically. Values/raw JSON are never
normalized or overwritten. For Score, `ExpectedIndexLowerBound` and
`ExpectedIndexUpperBound` are the minimum/maximum latent ordinal expectation
compatible with normalized mass in those intervals. An independently rounded
`NativeScore` must have an overlapping rounding interval. For high precision,
the bounds reflect the retained native-score validation tolerance, clipped to
the rubric domain; they are not statistical confidence intervals.
The observed `ExpectedIndex` is only an estimate for rounded data and can lie
**outside** the rubric range: `[0,.0001,1]` gives `2.0001` although the feasible
latent expectation is at most 2. Consumers must not treat the observed estimate
as a normalized, exact, calibrated expectation.

Thus `[.3333,.3333,.3333]` is valid but `[.3332,.3332,.3332]` cannot carry unit
mass. Observations `[.1234,.2344,.6421]` admit separately rounded native score
`1.5187`; requiring equality to observed `1.5186` would invent false precision.
Bounds depend on cardinality and ordinal weights, not one inflated epsilon.
For Jeeves, `[.33,.33,.33]` and `[.04,.43,.54]` are feasible despite sums .99
and 1.01. The latter's observed expectation is 1.51, while its latent bounds
are 1.495..1.505, compatible with independently rounded native score 1.50.
Malformed responses fail completely. Ordinal expectation does not assert equal
real-world distances between rubric categories.

There is no portable confidence, policy threshold, abstention, tool calling,
DTO/schema compilation, chat projection, fallback, registry, scheduler or
middleware framework. Every request succeeds completely or fails. One logical
batch does not promise a single HTTP call/forward, joint answer consistency,
or common token accounting.

## MEAI conventions and extension ownership

`IDecisionGenerator.GenerateAsync(DecisionInput, ...)` produces a `DecisionResult`;
typical callers
can instead pass state and questions through extensions. Typed state requires
the caller's explicit `JsonTypeInfo<TState>` (including source-generated metadata);
there is no reflection fallback, object serialization overload, or hidden
serializer configuration. A `JsonElement` overload snapshots existing JSON.
Null typed state follows the supplied serializer metadata. Options and cancellation
are forwarded, not replaced. Exact-ID `GetAnswer` throws for a missing ID;
`GetAnswer<TAnswer>` also throws for a wrong answer type. Ordered `Answers` remain
available for batch correlation. Service helpers forward the exact service key.

`DecisionGenerationOptions` is extensible and mutable. `Clone` preserves `ModelId` and
shallow-clones `AdditionalPropertiesDictionary`; dictionary values remain shared
references. Derived types override `Clone` to preserve their own properties.
Providers never mutate caller options and capture the needed model ID once.
All current providers recognize **no** additional-option keys: this bag
is an extension point, not a promise to interpret settings. Jeeves explicitly
rejects nonempty input bags and uses its typed options instead. No
`RawRepresentationFactory` is exposed because no current provider needs a
per-call native SDK request factory.

Metadata is descriptive: nullable `ProviderName`, `ProviderUri` and
`DefaultModelId` follow MEAI. An override never rewrites the default, while
`DecisionResult.ModelId` carries the provider-reported model identity or label;
for Jeeves it is only the request echo. `SupportedKinds`
is an owned snapshot: null means unknown, empty means none supported. Embedded
generators report their exact fixed-model kinds. Jev and Ollaya report unknown
because catalogs/model overrides are not static guarantees of arbitrary-question
support; unknown does not mean supported. Ollaya performs selected-model discovery;
Jev preserves native server rejection for selected-model restrictions it cannot
know locally. Each provider still enforces its known preflight rules.
Jeeves describes its pinned server's Binary/Choice/Score protocol subset, not
verified loaded weights or arbitrary future server capabilities.
HTTP generators report configured endpoints; local generators do not invent service URIs.

Answers and results expose mutable `RawRepresentation` (`[JsonIgnore]`) and
`AdditionalProperties`. Arbitrary assigned objects/dictionaries are **not**
deep-cloned or promised immutable. Portable validated payloads remain read-only.
Jev, Ollaya and Jeeves explicitly clone native answer/envelope `JsonElement`s before disposing
the source document. Julia's `JuliaRawAnswer`, Laya's `LayaRawAnswer` and Qwen's `QwenRawAnswer`/
`QwenRawResult` retain managed snapshots of every prior raw value, never live
handles, borrowed spans or model-owned buffers. Read-only lists wrap per-call
owned arrays. JSON evidence export belongs to the host/proof boundary, not the
local inference libraries. No response ID is fabricated from request IDs or
counters; there is no current native response ID to expose.

## Dependencies and capabilities

| Library | Direct dependencies beyond the shared contract | Capability |
|---|---|---|
| Abstractions | MEAI.Abstractions 10.10.0 (`UsageDetails`, `AdditionalPropertiesDictionary`) | Contract only |
| SystemOne | Shared contract only | Internal wire/HTTP code, no exported framework |
| TypeSafe | SystemOne | Binary/Choice/Score native wire; selected model may reject |
| Ollaya | SystemOne | Audited NLI/GLiClass profiles; per-model checks, unknown static capabilities |
| Jeeves | SystemOne | Pinned Binary/Choice/Score HTTP subset; optional reasoning, two-decimal observations |
| Onnx | ORT.Managed 1.23.2, Tensors 10.0.9 | Internal shared marker-model execution, zero exported types |
| Laya | Onnx, Tokenizers 2.0.0, ORT.Managed 1.23.2, Tensors 10.0.9 | Local Binary/Choice/Score |
| TensorSharp | TensorSharp.Models 2026.9.1, Tensors 10.0.9 | Qwen Choice only, 2–26 aliases |
| Julia | Onnx, Tokenizers.HuggingFace 3.23.1, ORT.Managed 1.23.2, Tensors 10.0.9 | Native Binary, Choice/Score 2–20; strict 1024/256 |

The executable selects CPU ORT 1.23.2. The comparison host alone intentionally
joins old and new APIs. No new library references the chat implementation or
`Microsoft.ML`/`MLNet.TextInference.Onnx`. Tokenizers and ORT are standalone
libraries; there are no DataViews, estimators, `Fit`, or ML.NET persistence.
New central entries are scoped to `Decision*` projects: otherwise adding
Tensors centrally changes the classification in existing transitive lockfiles.
The combined host resolves Tensors 10.0.12 through the unchanged chat graph;
the independent Laya/Qwen/Julia libraries resolve their 10.0.9 pin.
TensorSharp.Models has a substantial transitive media/native/backend graph;
the experiment does not hide that cost by counting only direct dependencies.
`artifacts\decision-proof\dependency-graphs.json` records resolved closures.

**Jev:** the caller supplies `HttpClient` and an explicit key. The host owns
nonredirecting handlers, standard resilience and a deadline covering body
reads; `HttpClient.Timeout` alone with `ResponseHeadersRead` is insufficient.
Both declared and observed body lengths are bounded. The retained
per-`ReadAsync` cancellation algorithm is intentional: .NET 10
`LoadIntoBufferAsync` on derived `StreamContent` dropped cancellation in the
original regression. HTTP failures retain status and opt-in, bounded diagnostic
text (replacement decoding on errors only); success JSON requires strict UTF-8.
Do not log diagnostic bodies or provider evidence automatically.

**Shared wire identity:** both HTTP generators assign question aliases `q0`, `q1`, ...
and numeric candidate aliases `0`, `1`, ... in caller order, always sending
resolved instructions. Responses must correlate fully before exact external
IDs are restored. Native raw JSON retains native aliases. Owned read-only maps
are provider metadata under `systemone.questionIds` (wire to external) and
`systemone.candidateIds` (wire question to wire/external candidate map);
`systemone.requestId` is HTTP correlation, not a fabricated response identity.
Consumers never need these maps for ordinary typed answer access.

**Ollaya:** inject an isolated `HttpClient` with automatic redirects and cookies
disabled. Default endpoint is `http://127.0.0.1:11435`; insecure HTTP is allowed
only for literal loopback IPs, not DNS hostnames or remote hosts. HTTPS remains
required remotely. Optional explicit `apiKey` never comes from Jev settings;
ambient Authorization is rejected. The caller owns injected handlers and must
not change them during use. Post-response redirect detection is not a substitute
for disabling redirects before sending sensitive state/credentials.
HTTP status, optional error `Code`, request ID and bounded opt-in diagnostic
text are preserved, including malformed/non-UTF8 error bodies.

Every call captures its selected model, validates known whole-batch limits,
then performs read-only `POST /api/show`. It never pulls, creates, loads or
unloads models through management APIs. Audited current profiles are
`nli/nli-pairs-v1` and `gliclass/gliclass-uni-v1`, ONNX only. This narrow list
is an adapter limitation, not a generic abstraction constraint.
The pinned server's capabilities advertise all kinds even for fixed-schema
guards, so family/layout/engine/architecture and requested kinds are checked.
Routers, unknown profiles and Qwen3Guard fixed schemas are explicitly unsupported;
choose a concrete compatible model. Embedded question presets alone do not
prove a restriction. A different actual model ID is retained and rediscovered;
incompatible profiles fail. Catalog mutation cannot be eliminated atomically
by discovery, so complete response correlation and numerical validation remain
mandatory. No discovery cache promises permanent capabilities.

Only `/v1/systemone` is used: it rejects state truncation with 422, unlike the
native route. Known limits are state string/object/array, 1–256 questions,
Choice 2–255, Score 2–10, and an 8 MiB serialized body; body size is checked again
after semantic preparation. Token budgets remain server-owned; a rejected request
can still consume server compute. Ollaya's Laya layout silently clips instructions/
options without reporting that as state truncation, so this adapter rejects it:
use embedded `LayaDecisionGenerator` for strict pre-inference budget enforcement.
GLiClass empty string state is rejected because its truncation flag would miss
head-only truncation. Reserved markers are checked in decoded JSON strings/keys,
instructions/descriptions and GLiClass's direct instruction/state concatenation.
This refuses rewriting rather than silently sanitizing input.

NLI/GLiClass reject one-sided Binary criteria, which upstream would discard.
NLI's per-model templates can ignore instructions for described outcomes:
the adapter explicitly prepends nonempty instructions plus newline to each
Choice/Score description and each paired Binary description, preserving both
inputs. This is a documented semantic adaptation, not native prompt parity.
The original portable request is unchanged; owned selected-model discovery and
actual prepared wire JSON are available as `ollaya.modelInfo`,
`ollaya.actualModelInfo` (when needed), and `ollaya.wireRequest`.
NLI and GLiClass mappings are adapter-defined and uncalibrated.
Concurrent HTTP calls are independent; disposal only disposes the HTTP client
when ownership was transferred. Cancellation cancels HTTP waiting/body reads,
not a guarantee to interrupt model execution on the daemon.

**Laya:** `PrepareDecisionInputs`, `ScoreOnnxDecisionModel`, and `DecodeDecisions`
are the exact public stages invoked by `LayaDecisionGenerator`. Preparation uses
`Microsoft.ML.Tokenizers.Tokenizer`; execution uses `InferenceSession`,
`OrtValue`, `SessionOptions`, and per-call `RunOptions`; outputs are owned
`Tensor<float>` values, with standard read-only tensor spans for input views.
There is no alternate tokenizer/tensor/runner interface.

The five inputs are Int64 IDs/masks `[B,L]`, Int64 marker positions `[B,K]`,
Boolean marker validity `[B,K]`, and Int64 qtype `[B]` (`0=choice,1=score,2=noul`).
Outputs are Float32 logits `[B,K]` and already-normalized action probabilities
`[B,2]`. Padding is excluded from option softmax; the action head is not
softmaxed again. Logits, effective temperature and action probabilities remain
provider evidence, not calibrated accuracy.

Framing is `[CLS] head [SEP] ([MASK] option)* [SEP] state [SEP]`.
Choice uses numeric aliases (`0: description`) rather than opaque IDs for
meaning. Score renders `level i: description`; Binary renders false then true.
The pinned config uses `max_len=512`, `head_max_len=192`. Each option has at
most 49 text tokens; option markers/text must leave at least 16 head tokens;
the actual head and remaining state must fit their budgets. All questions are
prepared before inference. No truncation or mask rewriting is permitted.
Reserved framing strings/IDs fail explicitly. The selected source decoder's
temperature clamp `[0.5,5]` is retained: the raw `choice:11+` value
`0.1005828083` becomes `0.5`; raw config values and effective answer evidence
make this visible.

Path-based Laya generators/scorers own the created session. Injected sessions and
scorers are borrowed unless ownership is explicitly transferred; the caller
must ensure injected graph/profile identity and not dispose borrowed resources
during use. Caller `SessionOptions` remain borrowed. Calls and disposal are
serialized; cancellation sets `RunOptions.Terminate`. Native buffers are
disposed before returning managed copies: this is not end-to-end zero-copy.
Neither a `Task` return type nor an `Async` suffix makes the local computation
nonblocking; this pass introduces no offloading or scheduling layer.

**Julia:** the new `DecisionInference.Julia` package has its own model-specific
`PrepareDecisionInputs`, `ScoreOnnxDecisionModel` and `DecodeDecisions`; these
are exactly the stages composed by `JuliaDecisionGenerator`. The graph has the
same five named input shapes/types, opset 18 and dynamic batch/token/option axes,
but **only** Float32 `logits[B,K]`, not Laya's action head. Metadata, static input
dimensions, actual output shape/type and finite valid-option logits are checked.
Unused padded logits are excluded. There is no generic profile/engine abstraction
and no change to the existing Laya stages.

Julia follows the pinned Python **named typed API**, not its presentation API.
Choice labels are descriptions alone, Score labels are rubric strings in order,
and Binary labels are `[falseDescription, trueDescription]`, or exactly
`["false", "true"]` when both descriptions are absent. One-sided descriptions
are explicitly rejected. Resolved instructions are always passed verbatim;
question/candidate IDs never enter labels or tensors. Choice and Score accept
2–20 alternatives locally; the shared contract and other providers retain their
own limits. No router shortlisting or conditional survivor distribution is used.

The decoder uses stable double-precision full softmax over the Float32 logits.
It retains small nonzero probabilities and never snaps a high maximum to one.
Upstream `probabilities.py`/JavaScript `predict` instead suppress values below
.01, renormalize, and can snap a maximum above .95 to one; those display values
are **not** portable probabilities or four-decimal observations. Julia answers
use `HighPrecision`, first-maximum Choice selection, zero-based expected Score,
and Binary `p[1]`. Raw logits are owned, uncalibrated managed snapshots. Native
usage is unknown (`Usage == null`), not zero or a fabricated tokenizer count.

The reference recipe is Python `julia/data.py` `sequence(strict=True,1024,256)`
plus `Collator`, not the bundled Rust encoder's compact JSON default.
Object/array state is rendered like `json.loads` followed by
`json.dumps(ensure_ascii=False)`: insertion order, `", "` and `": "` separators,
Unicode preserved, integer versus binary64-float formatting preserved
(`1`, `1.0`, `-0.0`, `1e-05`). Large JSON integers remain integers.
Duplicate object keys, nonfinite parsed doubles, invalid Unicode and scalar
number/Boolean/null state are rejected instead of silently losing information.
Plain string state is not JSON-quoted. This rendering is model input, never an
identity normalization or a replacement for the owned caller JSON.

The complete batch is prepared before ORT. Framing is
`[2] head [1] ([4] option)* [1] state [1]`, with a leading space when encoding
each option, at most 48 option tokens excluding its marker, at least 16 reserved
head tokens, and no option/head/state truncation. Input length is padded upward
to a multiple of eight, capped at 1024; marker padding preserves row order.
All special-token text/IDs are rejected in caller content, including decoded
JSON keys/values. No mask rewriting occurs. The 1024/256 policy is deliberately
fixed: Python's advertised 8192 context is not a claim of tested 8K ONNX support.

Julia's tokenizer is Replace-space/U+2581, Metaspace (always prepend, split),
BPE with byte fallback/fused unknowns, and 249 added tokens. The pinned file
has 256,000 vocabulary entries and 580,604 merges. Framing metadata validates
`<pad>=0`, `<eos>=1`, `<bos>=2`, `<unk>=3`, `<mask>=4`; the file's remaining
normalization, merges and added-token behavior are executed by the established
HuggingFace Rust tokenizer, not a new BPE implementation. Microsoft Tokenizers
2.0.0's `BpeOptions` lacks byte fallback, so this provider alone uses the
community `Tokenizers.HuggingFace` binding. It adds Google.Protobuf 3.34.1 and
native assets for win/linux/osx x64/arm64. **Only win-x64 was executed here.**
The combined sample/test closure consequently resolves Protobuf 3.34.1 rather
than 3.30.2; existing provider-library and original chat lockfiles stay unchanged.

Preparation owns and disposes its native tokenizer and serializes preparation/
disposal. Scoring owns per-call `OrtValue`s and `RunOptions`, copies managed
`Tensor<float>` output, and uses `RunOptions.Terminate` for cancellation.
Generator calls/disposal are serialized too. Path constructors own their stages;
injected stages/sessions are borrowed unless explicitly transferred, and must
not be concurrently disposed by the borrower. `SessionOptions` stay borrowed.
Cancellation is checked between tokenizations; an individual native tokenizer
call is synchronous and not interruptible. No worker scheduler, global context
mutation, model reload per request, Python, Node, browser or server is shipped.

**Qwen:** the library never mutates `MAX_CONTEXT`. The host sets 512 before
loading the hash-pinned CPU GGUF once. A..Z must each encode to one distinct
valid token; arbitrary candidate labels are never reduced to their first
token. The complete prompt for every question is checked against the model's
effective context before the first reset/forward. Reserved prompt delimiters
are rejected. The adapter serializes model access, resets KV state before and
after each forward, and copies selected logits before runtime storage can be
reused. A borrower must provide exclusive access to the injected model.
`ForwardRefill(int[])` has no cancellation token: checks around/between
forwards do **not** promise in-flight interruption. The distribution is
conditional on answer tokens, not the entire vocabulary or correctness.

## Reproducible proof

**Original implementation receipt:** the first tested implementation was commit
`54b61ddcdc1e791addc333c8766162c98b1cf074`; its final combined proof log is
`artifacts\decision-full-gate.log` in this worktree (also retained in the session
files). The original verification and initial receipt-only follow-up were
completed locally, without pushing a branch, opening a PR or publishing packages.
The subsequent MEAI-conventions pass changes extension APIs and managed raw
representations, not decision semantics or the frozen chat baseline. Its full
combined proof is retained at `artifacts\idiomatic-full-gate.log`; the durable
execution receipt records exact source fingerprints and tested/final commits.

```powershell
.\eng\verify-decisions.ps1 -Offline -IncludeBaseline -Packages
# With explicitly provided local assets; neither command downloads them:
.\eng\verify-decisions.ps1 -Offline -Packages `
  -LayaDirectory <local-laya-directory> -QwenPath <local-gguf> `
  -JuliaDirectory <local-julia-directory> -IndependentReference
```

Default model-free runs explicitly print which real-model gates were not run.
The optional independent Laya check requires already-installed Python
`tokenizers`, `numpy`, and `onnxruntime`; these are not library dependencies.
The offline gate performs no live/paid HTTP inference. Local package proof verifies cached
external archive hashes/signatures before using an isolated offline feed with
explicitly accepted local unsigned packages; no global trust setting changes.

The one host supports `compare` (explicitly aliased synthetic Jev envelopes through both
APIs), `provider jev`, `provider laya <onnx> <tokenizer.json> <config>`,
`provider qwen <gguf>`, `stages <onnx> <tokenizer.json> <config> [receipt.json]`,
`provider ollaya <endpoint> <concrete-model> [receipt.json]`, and `inventory`.
Julia adds `provider julia <asset-directory> [receipt.json]`.
`compare` captures real outgoing bodies and checks exact semantic JSON equality,
model, usage, complete owned native evidence and restored portable IDs.
The old chat fixture is deliberately given the same aliases: original automatic
arbitrary-ID wire parity no longer holds after the opaque-ID hardening.
Single-call allocation/latency observations include process-wide allocations;
they are not statistical benchmarks or real Jev latency. Local load, first
inference and warm calls are labeled separately.

Actual local evidence on Windows x64/.NET 10.0.12:
Qwen's published phishing example reproduced probabilities
`0.0658742, 0.0270162, 0.907110`; repeated A/B/A calls preserve evidence with
one model load. Laya public-stage/generator parity passed for all three kinds.
Independent HF tokenizers 0.22.2/NumPy 1.26.4/ORT 1.20.1 CPU reproduced all
five prepared tensors exactly, raw logits within `1e-4`, and probabilities/
expected index/action output within `1e-5`. This uses the **new numeric alias**
recipe, not old label-based golden outputs. Laya selected Spam whereas Qwen
selected Phishing for this example: this is not an adapter failure and one
example establishes neither backend quality nor calibration.
The unchanged baseline gate passes its original 622 core + 28 evaluation
cases, eight sample scenarios and package-only proof.
The prior conventions decision suite had 239 passing VSTest cases (no skips): the
original 202 behavior cases plus 37 focused conventions cases. Those
are separate from, not additions to, the baseline's 650 cases.
The Ollaya hardening suite passes **465 cases, zero failures/skips** in focused
Release/VSTest execution: 239 retained definitions plus 226 additions, across
138 methods (84 original, 54 new). Intent/alias/capability expectation migrations
affect 26 original methods/55 case definitions; no original case is deleted.
Canonical-grid neighbors now fail rather than accepting fixed epsilon slack;
explicit blank Binary descriptions fail even with a statement. Unknown string
capabilities remain extensible, while null/non-string discovery entries fail
before inference. These are documented contract migrations, not removed checks.
The settled combined gate also passed the unchanged **650 baseline cases**,
all 465 decision cases, both embedded CPU models, the independent Laya oracle,
and package-only host/Laya/Ollaya consumers with **64 external archives**
hash/signature-verified. Logs are `artifacts\ollaya-full-gate.log` and
`artifacts\ollaya-live-gate.log`. The latter was refreshed after the final
discovery-validation fix. All 136 snapshotted non-document inputs stayed
byte-identical across those gates, and all 91 protected baseline hashes match.
The verified local implementation commit is
`ce61cb8c774c2ab75ba5572b055ad3383800c5d6`; the receipt-only follow-up changes
documentation, not the tested runtime, tests, host, locks or proof scripts.

The Ollaya live gate is separate and opt-in; it never installs or downloads:

```powershell
.\eng\verify-ollaya.ps1 -Endpoint http://127.0.0.1:11439 -Model nli:modernbert-large
```

One actual sample function, `ConsumeChoiceAsync`, performs the identical
`IDecisionGenerator` typed-state/questions/options call and typed Choice lookup
without provider branches for synthetic Jev, real embedded Laya/Qwen and real Ollaya operations
where their capabilities intersect. Positive live Binary/Choice/Score, concurrent
calls, raw retention, pre-cancellation and an observed
`422/STATE_TRUNCATED` rejection plus normal-call recovery are checked.
Agreement across independent model probabilities/categories is **not** expected.
Package proof includes an Ollaya-only consumer whose graph excludes TypeSafe,
ORT, TensorSharp and ML.NET; its protocol fixture is explicitly not live evidence.

Julia also executes that same consumer function, including typed Choice lookup;
it now uses source-generated `JsonElement` state metadata for both text and
structured JSON without provider branches. The Julia-only package consumer
uses only Julia/Abstractions plus CPU ORT and their verified closure. It compiles
and executes all three kinds through packaged public stages and generator APIs,
excluding TypeSafe, Ollaya, SystemOne, Laya, TensorSharp and ML.NET framework.

The actual Ollaya run used official **v0.7.2**, release commit
`17b85bf86d4427b790858074434911aa6557b183`, on CPU in existing Ubuntu 24.04 WSL.
HOME, storage and PID files were isolated under ignored `artifacts\ollaya-proof`;
the Windows caller used loopback port 11439. The Linux runtime archive was
21,713,246 bytes, SHA256
`0219b94b0b30841fda043bc89bb9b4067ebb46a6328a1108ab66453603b428b4`.
The deployed release is not assumed identical to the research pin:
`release-source-audit.json` records 18/19 focused files byte-identical; the HTTP
file's remaining change adds CPU launch configuration, not truncation semantics.
This is release/source verification, not an independently reproduced binary build.

Independent model `nli:modernbert-large` uses
`MoritzLaurer/ModernBERT-large-zeroshot-v2.0@a51e07b524299e309dd2b88d48b0cfa2bd9ec598`.
Its complete manifest layers total 798,926,208 bytes; manifest SHA256
`0c984ae95685d511953341fd75e5cf49e31bae5110574dba78999c5f2267cd7f`,
weights SHA256 `9e88e3ba0721cee770645d262f10dbe8125ed592ff349b7ba2eedaaded97230a`.
Exact official URLs, sizes and every layer hash are in
`artifacts\ollaya-proof\download-receipt.json`. No guard model or model survey
was downloaded; guard restrictions are pinned protocol fixtures, not live proof.

`artifacts\ollaya-live-gate.log` and `ollaya-proof\managed-live.json` record the
real shared-consumer result: Choice `.2777/.3592/.3631`, Binary `.9676`,
Score observations `.2957/.3880/.3162`, observed index `1.0204`, feasible bounds
`[1.02045,1.02055]`, native Score `1.0205`, and input/output usage `205/0`.
The measured native/observed difference is exactly the quantized-contract case
the hardening was designed to retain. Separate native probes also established
an independently rounded Score `2.2051` versus reconstructed `2.2052`;
they are evidence of wire behavior, not quality or cross-provider calibration.

| Required behavior | Representative exact test |
|---|---|
| Exact IDs and owned request/result snapshots | `OpaqueIds_PreserveSpacesCaseAndDistinctUnicodeNormalization`; `Result_OwnsAnswerDistributionsEvidenceAndUsageSnapshots` |
| Complete distributions and ordinal expectation | `Result_RequiresExactCompleteCandidateAndRubricDistributions`; `Score_ExpectedIndexAndNativeToleranceScaleWithRubric` |
| Native transport and real per-read cancellation | `TypedInput_UsesNativeStringEnvelopeAndDecodesAllThreeKinds`; `DerivedStreamContent_StalledReadReceivesCancellationAndCompletesBeforeDisposal` |
| Actual synthetic ORT execution with independent numeric assertions | `SyntheticOrt_ScoresActualDynamicGraphAndMatchesPublicStageGeneratorForAllKinds` |
| Bounded single-token scoring recipe | `AliasIds_RejectsDuplicateMultitokenNegativeOrMissingTokens`; `Prepare_ExactLiteralPromptUsesDescriptionsNotOpaqueIdsAndOwnsTokens` |
| Shallow options clone and derived dispatch | `Options_CloneSeparatesDictionaryEntriesButRetainsNestedReferences`; `Options_DerivedCloneDispatchesVirtuallyAndPreservesDerivedState` |
| Descriptive metadata and model overrides | `Metadata_DistinguishesUnknownFromExplicitEmptyCapabilities`; `Metadata_DefaultOverrideAndObservedModelRemainDistinct` |
| Exact keyed services and supplied state metadata | `ServiceExtensions_ForwardExactTypeAndKeyAndReturnSameService`; `TypedStateExtension_UsesSuppliedMetadataIncludingNullAndForwardsExactCallArguments` |
| Exact-ID answer access and clear failures | `AnswerExtensions_UseExactIdsIncludingCaseSpacesAndUnicode`; `AnswerExtensions_ReportWrongTypesAndReturnExactMatchingSubtypes` |
| Ignored raw serialization and shallow extension references | `RawRepresentation_IsIgnoredByJsonSerializationForResultAndEveryAnswerKind`; `Extensions_RetainMutableCallerReferencesWithoutChangingValidatedPayload` |
| Four-decimal mass feasibility without normalization | `RoundedChoice_255CandidatesAllowFeasibleDriftWithoutRenormalizing`; `RoundedScore_AcceptsIndependentNativeRoundingAndKeepsObservedEstimate` |
| Canonical grid and explicit intent | `FourDecimalGrid_AcceptsJsonLiteralButRejectsAdjacentDoublesAndSubGridOffsets`; `BinaryIntent_ExplicitInstructionPreservesValidDescriptionsAndRejectsBlanks` |
| Selected-model discovery and restricted schemas | `MutableCatalog_SameModelIsRevalidatedAndUnsafeSecondCallNeverInfers`; `ModelCatalog_RejectsUnauditedOrRestrictedProfilesBeforeAnyInference` |
| Ollaya cancellation and independent concurrent calls | `Cancellation_ReachesEveryShowAndInferenceReadBeforeDisposal`; `ConcurrentInputs_KeepModelsAliasesAndCancellationIndependent` |

The synthetic suite does not impersonate trained-model proof. Real Qwen KV
reset/repetition and the independent Laya oracle are executed by the explicit
local-model gates. Package proof runs both supplied local backends through a
package-only copy of the host, plus a separate Laya-only public-stage consumer
whose resolved closure excludes chat, TensorSharp and ML.NET.
The loaded Qwen gate also asserts Choice-only metadata, null service URI, actual
model identity, fixed-model override acceptance, rejection of invalid models and
both unsupported kinds, and populated managed answer/response raw values.

## Provenance and honest scope accounting

| Source | Pin | Reuse |
|---|---|---|
| MEAI conventions | Abstractions `10.10.0`, source `02107c65bab30aad9e35b5133ed643eaa77bccd8` (nuspec) | Options copy/clone, descriptive metadata and service-helper patterns; actual `UsageDetails` and `AdditionalPropertiesDictionary` primitives |
| Existing chat proof | `bc2f307ed2b7e8c5ea0eac5e95dabeb4868b7c19` | Native transport/protocol algorithms source-adapted; now shared internally by decision HTTP providers, never a dependency on the chat assembly |
| [Ollaya](https://github.com/ollaya-dev/ollaya/tree/152ad20c88f8ea9b6d1acf3ed0e06b002d38b2b4) | `152ad20c88f8ea9b6d1acf3ed0e06b002d38b2b4` | Normative `docs/api.md`, `decide.rs`, `question.rs`, `answer.rs`, `views.rs`, layouts and family docs; protocol/profile semantics, no engine port |
| [Embeddings prototype](https://github.com/luisquintanilla/onnx-embeddings-prototype) | `79b71da1e0ae44139983fca03734974a2f2f8c80` | Direct primitives, owned output copies, borrowed options/session, `Terminate` cancellation |
| [Domain kernel](https://github.com/luisquintanilla/mlnet-text-inference-custom-transforms) | `e9596f64640e0ca39a5ad931bffd785696f4962f` | Laya rendering/framing, graph contract, temperature policy, BPE configuration; no ML.NET layer |
| [Laya export](https://huggingface.co/receptron/laya-onnx/tree/68f27dfe5a27a54fb2b1fefc432f43f972e90868) | `68f27dfe5a27a54fb2b1fefc432f43f972e90868` | Five hash-pinned local files; hashes in verification entry |
| [Qwen sample](https://gist.github.com/luisquintanilla/3d1a22ca8dbab1ae9a80caff4f2bf612) | Models `2026.9.1`; GGUF `7f403014…fe2e1fa` | Prompt/selected-logit scoring adapted; bounded aliases, ownership and reset added |
| TensorSharp package source | `70909d16321c51a78667ed9a4e243afe925071a1` (nuspec) | Package's actual APIs compiled and executed; not assumed from latest source |
| [Julia typed/framing source](https://huggingface.co/SupersonicLabs/Julia-1/tree/a85b127321d580d65176c89ced8273f305745d85) | `a85b127321d580d65176c89ced8273f305745d85` | Strict sequence, collator and named typed API semantics; no display postprocessor or router |
| [Julia ONNX export](https://huggingface.co/SupersonicLabs/Julia-1-ONNX/tree/82a2fadf8fccfccdc5fd4e1009ba8f1a265eb7a8) | `82a2fadf8fccfccdc5fd4e1009ba8f1a265eb7a8` | Pinned opset-18 graph, external F32 weights, tokenizer and recorded PyTorch cases; model card declares Apache-2.0 |
| [HuggingFace .NET binding](https://github.com/IgnaciodelaTorreArias/Tokenizers.HuggingFace.DotNet/tree/9743e169ba445a90f473475311a0a7b4afa8147c) | `3.23.1`, nuspec commit `9743e169ba445a90f473475311a0a7b4afa8147c` | Apache-2.0 community binding; upstream Rust tokenizer via length-delimited protobuf FFI, not bespoke tokenization |

The Ollaya pass measured 19 C# files/1,699 physical lines: contract 467,
shared SystemOne 288, Jev 75, Ollaya 218, Laya 476, Qwen 175.
The prior conventions pass was 15 files/1,304 lines. This pass adds one
precision enum, four Ollaya exported types (generator, settings and two errors),
and a shared assembly with no exported types. The conventions pass's two
extension classes and three raw-result types remain included, not hidden.
All adapted/duplicated source remains counted. The original
1,796 lines/25 files include schema projection, policies, tool support,
middleware integration and evaluation that the new API intentionally lacks.
At the Julia implementation commit, Julia added 5 C# files/480 physical lines,
for **24 files/2,179 lines** across seven decision libraries. This included the explicit Python-compatible
JSON renderer and adapted local ownership/ORT mechanics; duplicated patterns
are counted, not hidden behind a new framework.
Do not compare all five new providers with one old provider and call the
difference a like-for-like reduction. Tests, host, verification scripts,
documentation, lockfiles and package dependencies are separate costs.
`inventory` reports exported types and declared public reflection members,
not a subjective API simplicity score.

| Responsibility (before the internal ONNX extraction below) | Nontrivial exact reference-line matches | Changed/new nontrivial lines | Exported types / declared public members |
|---|---:|---:|---:|
| Contract, precision and MEAI conventions | 11 | 240 | 19 / 106 |
| Shared native transport/protocol | 36 | 133 | 0 / 0 |
| Jev connection and errors | 13 | 33 | 4 / 26 |
| Ollaya discovery and semantic preparation | 0 | 138 | 4 / 28 |
| Adapted Laya kernels and new generator/ownership | 22 | 274 | 10 / 77 |
| Adapted Qwen recipe and new bounded adapter | 1 | 104 | 3 / 32 |
| Julia typed framing, explicit JSON recipe and local ORT mechanics | 107 | 185 | 6 / 37 |

Literal reuse is a conservative textual measurement: trim lines, exclude
comments/usings/namespaces and lines shorter than 20 characters, then count
each matching occurrence against the pinned source set. Renamed/restructured
algorithm reuse appears in the changed/new column, not as independently
invented code. All other lines still contribute to the physical totals.
Julia's reference set is the unchanged Laya stage/mechanism source plus its
pinned Python source; its counts do not imply 107 lines of new independently
invented inference code.
The original chat assembly exposes 24 types/228 declared public members;
these numbers include substantially different responsibilities.

[dotnet/extensions#7764](https://github.com/dotnet/extensions/issues/7764) and
[microsoft/agent-framework#8563](https://github.com/microsoft/agent-framework/pull/8563)
(`fc582740539b88ba8aa043e922f45d2b40521296`) are prior proposal/experimental
implementation evidence, not a shipped/accepted contract. The comparison is
published for review in [PR #2](https://github.com/luisquintanilla/typesafe-meai/pull/2),
stacked on the unchanged original PR. The separately verified Ollaya follow-up
received explicit publication authorization after its local gates passed. No package was
published and no license or repository visibility was changed.

## Julia native proof (separately verified follow-up)

Pinned files remain under ignored `artifacts\decision-models\julia`.
`model.onnx` is 2,988,923 bytes, SHA256
`97141d0cfb1da6204e9f8f24d581af72eaeb82cda21149d83eaa6df7160fbcd9`;
`model.onnx.data` is 576,782,336 bytes, SHA256
`fd915be810d7ebfb80fb05a48dd33c9484d17ae1b6bcb9e1f544cbaaa913ded1`.
Tokenizer SHA256 is
`609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f`.
The asset receipt records exact pinned HF URLs, sizes, source Git blob IDs
and SHA256 hashes. No floating-main model download or training is involved.
Resolve model/tokenizer/recorded-case files from
`https://huggingface.co/SupersonicLabs/Julia-1-ONNX/resolve/82a2fadf8fccfccdc5fd4e1009ba8f1a265eb7a8/<file>`;
the oracle's `julia/data.py` and `julia/typed.py` come from
`https://huggingface.co/SupersonicLabs/Julia-1/resolve/a85b127321d580d65176c89ced8273f305745d85/<file>`.
Preserve those relative paths under the asset directory. The gate rejects
wrong bytes, and never obtains assets implicitly.

```powershell
# Existing local assets only. The oracle also needs the pinned julia/data.py,
# julia/typed.py and parity-cases.json from the reference revisions above.
dotnet run --project eng\DecisionProof -c Release -- `
  provider julia artifacts\decision-models\julia artifacts\decision-proof\julia-managed.json
python eng\verify-julia-reference.py artifacts\decision-models\julia `
  artifacts\decision-proof\julia-managed.json
```

The Python oracle uses already-installed `transformers` 5.3.0, `tokenizers`
0.22.2, `torch` 2.10.0+cpu (collation only), NumPy 1.26.4 and ORT 1.20.1 CPU.
None is a shipped Julia package dependency. It executes the hash-verified,
unmodified upstream strict sequence, collator and `predict_typed`, independently
recreates all five tensors, then executes the same pinned ONNX weights through
Python ORT. Its fixed proof requires four requests, 15 answers, exact answer
order/cardinalities and 20 correctly shaped input tensors; empty requests,
shortened probabilities and wrong shapes were explicitly verified to fail.

Actual .NET ORT 1.23.2 CPU evidence: **20/20 tensors exactly equal, 15/15 typed
answers agree, maximum logit error 0**, maximum full-softmax error
`1.0673970388119791e-16`, maximum Score error `2.220446049250313e-16`.
This includes text, object, array, Unicode/whitespace, described Boolean and
mixed option counts. The tolerances are explicit: `1e-4` logits, `5e-5`
independent probabilities, `1e-4*(K-1)` Score, and `1e-12` softmax from identical
logits. These bound implementation comparison, not calibration or quality.
The proof also checks native in-flight termination (an ORT inner exception),
pre-cancellation, concurrent/repeated calls, owned outputs and post-rejection/
post-cancellation recovery. It runs the very same public stages as the generator.

Six **upstream-recorded** PyTorch cases (first six Choice cases with empty state,
to form one shared-state batch) agree in argmax and have maximum logit difference
`8.22544098e-5`, below `1e-4`. This is not a local PyTorch model reproduction,
not a rerun of the upstream 100-case parity study, and not a new accuracy
benchmark. Upstream accuracy figures and 8K claims are not local evidence.
In one actual Choice, full probabilities were approximately
`.000356666, .00133131, .998312`; retaining those tails demonstrates that the
presentation API's one-hot suppression is not being substituted.

The focused Julia suite adds **79 cases/33 methods**, all passing, while
retaining the prior 465 cases unchanged. Representative tests are
`Prepare_IdRenamingLeavesFiveTensorsIdenticalAndRestoresKeys`,
`Prepare_PythonJsonRecipeSurvivesDocumentDisposal`,
`Prepare_OptionBudgetCountsTextNotMarker`,
`Generator_PreflightsEntireMixedBatchBeforeIncompatibleScorer`,
`Decode_DoubleProbabilitiesRetainTailsAndZeroBasedScore`,
`Score_WiresAllFiveBuffersToTheirNamedInputs` and
`Generator_RepeatedConcurrentCallsAreDeterministicAndOwnRawSnapshots`.

The settled combined gate passed **1,194 cases = 650 unchanged chat + 544
decision (465 retained + 79 Julia)**, zero failures/skips. It reran real Laya
and Qwen with their prior assets, both independent ONNX oracles, the package-only
combined host, and Laya-only/Ollaya-only/Julia-only consumers. **66 external
archives** were checked against locked hashes and repository signatures.
All **147 frozen non-document inputs** and **91 original baseline hashes**
remained unchanged throughout the gate. The full log, native/reference JSON,
package closure/native-RID inspection and final local commit are linked by
`files\julia-execution-receipt.json` in the implementation session.
Ollaya's runtime/shared HTTP source did not change, so its already-verified
NLI live evidence was retained rather than reinstalling or restarting its
scoped daemon. This pass makes no new live Ollaya claim.

On this machine an unrelated ancestor `C:\Dev\Directory.Build.targets` imports
Arcade. All Julia build/gate commands scoped
`$env:ImportDirectoryBuildTargets='false'` to their shell process; no ancestor
file or global profile was edited. This workaround is unnecessary in a normal
checkout without that unrelated ancestor. Julia was implemented and verified
locally at `cc3921dac784378221e6729035d818fa8055ea1b`. Publication to the existing
PR was separately authorized after that proof. The documentation-only
publication commit `ba33de18089ce3302faea02487283739ff290e7d` also recorded the
then-unimplemented extraction plan. That publication did not represent another
build/model execution, merge either PR or publish packages. The subsequent
mechanical extraction below was separately authorized. The naming migration
above was authorized only after that extraction was independently verified.

## Internal ONNX execution extraction

Both public `ScoreOnnxDecisionModel` stages now compose the same internal
`DecisionInference.Onnx.MarkerOnnxRunner`. Julia was adapted first and checked
before adapting Laya. This is **one maintained native execution implementation**,
not a general-purpose inference framework or a performance claim.

The compiled `DecisionInference.Onnx` package follows SystemOne's friend-assembly
precedent: **zero exported types**, friends only Laya and Julia, no test friend.
It is a real transitive NuGet dependency of both providers, not hidden with
`PrivateAssets`. Its dependencies are managed ORT **1.23.2**, Tensors **10.0.9**
and Abstractions for the existing `DecisionProtocolException` (including
Abstractions' existing MEAI dependency). No dependency version changed.
Neither tokenizer, HTTP providers, TensorSharp nor native CPU/GPU runtime
selection belongs to it; executables still select the native runtime.

The shared implementation validates the five named inputs and tensor metadata,
checks every fixed dimension including zero, binds the existing managed arrays,
and owns each call's `OrtValue`s, `RunOptions`, cancellation registration and
native results. Individual `using` declarations preserve exception-safe partial
allocation cleanup. It serializes scoring/session disposal, uses `Terminate`,
and copies requested outputs into managed tensors after actual shape/type checks.
Small internal input/output bindings add no input-array copies or new public
tensor format. No native handles escape.

Provider adapters retain output selection and policy. Julia's finite-value
validation callback executes **under the shared scoring lock**, before releasing
native call resources. Public batch null checks still precede cancellation;
disposed exceptions identify the public scorer, not the internal runner.
Constructor validation completes before acquiring session ownership, while
`Load` still cleans up a newly created session on validation failure. Disposal
is idempotent and does not dispose the coordination semaphore. Laya's formerly
unqualified static-input/cancellation messages now include `Laya`; exception
categories, output errors and validation stages are preserved.

Tokenizers and their lifetimes, JSON rendering, semantic descriptions,
framing/markers, complete-batch preflight, context limits, packing/padding,
decoding, usage, raw evidence and outer generator locks were source-unchanged
by the extraction. The subsequent naming migration changes their decision
symbols only, not these mechanics.
Observable differences remain intentional:
Laya accepts additional graph outputs and returns logits plus width-two
`act_probs`; Julia requires exactly `logits`. Laya's raw score stage can expose
nonfinite values until decoding, while Julia's score stage rejects nonfinite
valid-option logits; neither rejects unused padded lanes merely for being
nonfinite. Laya uses exact-longest-row padding and its temperature/action policy;
Julia pads to multiples of eight and uses full uncalibrated softmax.

The extraction's API snapshot compares **353 complete reflection entries**, including
constructors, parameter names/defaults, accessors, generic constraints and
attributes, rather than just type counts. All existing entries match; the new
assembly contributes none. The later rename keeps all 353 entries modulo the
explicit type/member/parameter map above. `OnnxExtractionTests` added 16 focused cases:
output-policy/validation-timing characterization, static action width,
zero-dimensional rejection, constructor-failure ownership and both scorers'
shared locking/cancellation/disposal with owned and borrowed sessions. The
synchronization tests inspect the private shared gate without adding test hooks.
Existing tests and independent oracle code/tolerances are unchanged.

Runtime accounting after extraction is **25 C# files/2,169 physical lines**
across eight libraries: Laya 420, Julia 433, shared ONNX 93; other libraries
unchanged. The net reduction is only ten lines. Model-specific loading,
output policy and public result construction deliberately remain in wrappers;
the benefit is a single location for native correctness, not a claim that every
similar line disappeared. Package and test costs remain separately visible.

The verifier includes the shared package, rejects tokenizer/provider/native
runtime leakage into it, and checks isolated Laya/Julia consumers resolve the
transitive assembly with zero exports. Laya cannot acquire Julia's Rust tokenizer,
Julia cannot acquire Laya's tokenizer, and HTTP-only dependency closures remain
native-free. The unchanged native/oracle commands and scoped ancestor-import
workaround above apply to the extraction as well.

The settled extraction gate passed **1,210 cases = 650 original + 560 decision**,
retaining all previous 1,194 cases and adding 16. Both retained real ONNX models
and Qwen executed. Laya retained exact five-tensor equality and its existing
`1e-4` logit / `1e-5` probability, action and Score tolerances. Julia retained
15 rows / 20 exact tensors, zero maximum Python-ORT logit difference,
`1.0673970388119791e-16` maximum probability error and
`2.220446049250313e-16` maximum Score error. Its managed proof JSON is
byte-identical to the pre-extraction receipt; no local PyTorch model execution
is claimed.

The package-only host and isolated Laya, Ollaya and Julia consumers passed,
including native CPU stage/generator parity, with **66 external archives**
hash/signature-verified. The 151 final non-document inputs and 91 original
baseline fingerprints were unchanged after the gate. Of the prior 147-input
manifest, 12 expected paths changed and 135 remained identical; four new
files account for the final 151. The prior manifest remains historical, not
misrepresented as the tested extraction. Only the two scoring stages changed
among existing provider runtime files. No models were downloaded and no
Ollaya daemon was restarted.

No generic inference interface, base generator, registry, manifest DSL, tokenizer
or softmax unification, scheduler, pooling, GPU migration or zero-copy promise
was introduced. **No abstraction renames were included in the extraction
commit**; the naming migration is a distinct follow-up.
