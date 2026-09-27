# Decision-first side-by-side experiment

This is an experimental, project-owned `DecisionInference` API, **not** a
Microsoft/Agent Framework API, a replacement for `IChatClient`, or a quality
benchmark. The chat implementation at `bc2f307ed2b7e8c5ea0eac5e95dabeb4868b7c19`
remains unchanged. Build the additive experiment with
`dotnet build eng\DecisionComparison.slnx`; the original solution remains the
only solution at the repository root.

```csharp
using System.Text.Json.Serialization;

DecisionQuestion[] questions =
    [new ChoiceDecisionQuestion("category", "Choose one category.",
        [new("legitimate", "Legitimate email"), new("phishing", "Credential theft")])];
var metadata = client.GetRequiredService<DecisionClientMetadata>();
var options = new DecisionOptions { ModelId = metadata.DefaultModelId };
DecisionResponse response = await client.GetResponseAsync(
    "An email asks for a password.", questions, DecisionJsonContext.Default.String,
    options.Clone(), token);
var answer = response.GetAnswer<ChoiceDecisionAnswer>("category");
// Application policy, not the provider, decides whether to act.

[JsonSerializable(typeof(string))]
internal partial class DecisionJsonContext : JsonSerializerContext;
```

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

Ollaya selects four-decimal precision automatically. Values/raw JSON are never
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
Malformed responses fail completely. Ordinal expectation does not assert equal
real-world distances between rubric categories.

There is no portable confidence, policy threshold, abstention, tool calling,
DTO/schema compilation, chat projection, fallback, registry, scheduler or
middleware framework. Every request succeeds completely or fails. One logical
batch does not promise a single HTTP call/forward, joint answer consistency,
or common token accounting.

## MEAI conventions and extension ownership

`IDecisionClient` retains its request/response method shapes; typical callers
can instead pass state and questions through extensions. Typed state requires
the caller's explicit `JsonTypeInfo<TState>` (including source-generated metadata);
there is no reflection fallback, object serialization overload, or hidden
serializer configuration. A `JsonElement` overload snapshots existing JSON.
Null typed state follows the supplied serializer metadata. Options and cancellation
are forwarded, not replaced. Exact-ID `GetAnswer` throws for a missing ID;
`GetAnswer<TAnswer>` also throws for a wrong answer type. Ordered `Answers` remain
available for batch correlation. Service helpers forward the exact service key.

`DecisionOptions` is extensible and mutable. `Clone` preserves `ModelId` and
shallow-clones `AdditionalPropertiesDictionary`; dictionary values remain shared
references. Derived types override `Clone` to preserve their own properties.
Providers never mutate caller options and capture the needed model ID once.
All four current providers recognize **no** additional-option keys: this bag
is an extension point, not a promise to interpret settings. No
`RawRepresentationFactory` is exposed because no current provider needs a
per-call native SDK request factory.

Metadata is descriptive: nullable `ProviderName`, `ProviderUri` and
`DefaultModelId` follow MEAI. An override never rewrites the default, while
`DecisionResponse.ModelId` identifies the actual result model. `SupportedKinds`
is an owned snapshot: null means unknown, empty means none supported. Embedded
clients report their exact fixed-model kinds. HTTP clients report unknown
because catalogs/model overrides are not static guarantees of arbitrary-question
support; unknown does not mean supported. Ollaya performs selected-model discovery;
Jev preserves native server rejection for selected-model restrictions it cannot
know locally. Each provider still enforces its known preflight rules.
HTTP clients report configured endpoints; local clients do not invent service URIs.

Answers and responses expose mutable `RawRepresentation` (`[JsonIgnore]`) and
`AdditionalProperties`. Arbitrary assigned objects/dictionaries are **not**
deep-cloned or promised immutable. Portable validated payloads remain read-only.
Jev and Ollaya explicitly clone native answer/envelope `JsonElement`s before disposing
the source document. Laya's `LayaRawAnswer` and Qwen's `QwenRawAnswer`/
`QwenRawResponse` retain managed snapshots of every prior raw value, never live
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
| Laya | Tokenizers 2.0.0, ORT.Managed 1.23.2, Tensors 10.0.9 | Local Binary/Choice/Score |
| TensorSharp | TensorSharp.Models 2026.9.1, Tensors 10.0.9 | Qwen Choice only, 2–26 aliases |

The executable selects CPU ORT 1.23.2. The comparison host alone intentionally
joins old and new APIs. No new library references the chat implementation or
`Microsoft.ML`/`MLNet.TextInference.Onnx`. Tokenizers and ORT are standalone
libraries; there are no DataViews, estimators, `Fit`, or ML.NET persistence.
New central entries are scoped to `Decision*` projects: otherwise adding
Tensors centrally changes the classification in existing transitive lockfiles.
The combined host resolves Tensors 10.0.12 through the unchanged chat graph;
the independent Laya/Qwen libraries resolve their 10.0.9 pin.
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

**Shared wire identity:** both HTTP clients assign question aliases `q0`, `q1`, ...
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
use embedded `LayaDecisionClient` for strict pre-inference budget enforcement.
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
are the exact public stages invoked by `LayaDecisionClient`. Preparation uses
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

Path-based Laya clients/scorers own the created session. Injected sessions and
scorers are borrowed unless ownership is explicitly transferred; the caller
must ensure injected graph/profile identity and not dispose borrowed resources
during use. Caller `SessionOptions` remain borrowed. Calls and disposal are
serialized; cancellation sets `RunOptions.Terminate`. Native buffers are
disposed before returning managed copies: this is not end-to-end zero-copy.
Neither a `Task` return type nor an `Async` suffix makes the local computation
nonblocking; this pass introduces no offloading or scheduling layer.

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
  -LayaDirectory <local-laya-directory> -QwenPath <local-gguf> -IndependentReference
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
one model load. Laya public-stage/client parity passed for all three kinds.
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
`IDecisionClient` typed-state/questions/options call and typed Choice lookup
without provider branches for synthetic Jev, real embedded Laya/Qwen and real Ollaya operations
where their capabilities intersect. Positive live Binary/Choice/Score, concurrent
calls, raw retention, pre-cancellation and an observed
`422/STATE_TRUNCATED` rejection plus normal-call recovery are checked.
Agreement across independent model probabilities/categories is **not** expected.
Package proof includes an Ollaya-only consumer whose graph excludes TypeSafe,
ORT, TensorSharp and ML.NET; its protocol fixture is explicitly not live evidence.

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
| Exact IDs and owned request/result snapshots | `OpaqueIds_PreserveSpacesCaseAndDistinctUnicodeNormalization`; `Response_OwnsAnswerDistributionsEvidenceAndUsageSnapshots` |
| Complete distributions and ordinal expectation | `Response_RequiresExactCompleteCandidateAndRubricDistributions`; `Score_ExpectedIndexAndNativeToleranceScaleWithRubric` |
| Native transport and real per-read cancellation | `TypedRequest_UsesNativeStringEnvelopeAndDecodesAllThreeKinds`; `DerivedStreamContent_StalledReadReceivesCancellationAndCompletesBeforeDisposal` |
| Actual synthetic ORT execution with independent numeric assertions | `SyntheticOrt_ScoresActualDynamicGraphAndMatchesPublicStageClientForAllKinds` |
| Bounded single-token scoring recipe | `AliasIds_RejectsDuplicateMultitokenNegativeOrMissingTokens`; `Prepare_ExactLiteralPromptUsesDescriptionsNotOpaqueIdsAndOwnsTokens` |
| Shallow options clone and derived dispatch | `Options_CloneSeparatesDictionaryEntriesButRetainsNestedReferences`; `Options_DerivedCloneDispatchesVirtuallyAndPreservesDerivedState` |
| Descriptive metadata and model overrides | `Metadata_DistinguishesUnknownFromExplicitEmptyCapabilities`; `Metadata_DefaultOverrideAndObservedModelRemainDistinct` |
| Exact keyed services and supplied state metadata | `ServiceExtensions_ForwardExactTypeAndKeyAndReturnSameService`; `TypedStateExtension_UsesSuppliedMetadataIncludingNullAndForwardsExactCallArguments` |
| Exact-ID answer access and clear failures | `AnswerExtensions_UseExactIdsIncludingCaseSpacesAndUnicode`; `AnswerExtensions_ReportWrongTypesAndReturnExactMatchingSubtypes` |
| Ignored raw serialization and shallow extension references | `RawRepresentation_IsIgnoredByJsonSerializationForResponseAndEveryAnswerKind`; `Extensions_RetainMutableCallerReferencesWithoutChangingValidatedPayload` |
| Four-decimal mass feasibility without normalization | `RoundedChoice_255CandidatesAllowFeasibleDriftWithoutRenormalizing`; `RoundedScore_AcceptsIndependentNativeRoundingAndKeepsObservedEstimate` |
| Canonical grid and explicit intent | `FourDecimalGrid_AcceptsJsonLiteralButRejectsAdjacentDoublesAndSubGridOffsets`; `BinaryIntent_ExplicitInstructionPreservesValidDescriptionsAndRejectsBlanks` |
| Selected-model discovery and restricted schemas | `MutableCatalog_SameModelIsRevalidatedAndUnsafeSecondCallNeverInfers`; `ModelCatalog_RejectsUnauditedOrRestrictedProfilesBeforeAnyInference` |
| Ollaya cancellation and independent concurrent calls | `Cancellation_ReachesEveryShowAndInferenceReadBeforeDisposal`; `ConcurrentRequests_KeepModelsAliasesAndCancellationIndependent` |

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

The decision runtime is 19 C# files/1,699 physical lines: contract 467,
shared SystemOne 288, Jev 75, Ollaya 218, Laya 476, Qwen 175.
The prior conventions pass was 15 files/1,304 lines. This pass adds one
precision enum, four Ollaya exported types (client, settings and two errors),
and a shared assembly with no exported types. The conventions pass's two
extension classes and three raw-result types remain included, not hidden.
All adapted/duplicated source remains counted. The original
1,796 lines/25 files include schema projection, policies, tool support,
middleware integration and evaluation that the new API intentionally lacks.
Do not compare all four new providers with one old provider and call the
difference a like-for-like reduction. Tests, host, verification scripts,
documentation, lockfiles and package dependencies are separate costs.
`inventory` reports exported types and declared public reflection members,
not a subjective API simplicity score.

| Responsibility | Nontrivial exact reference-line matches | Changed/new nontrivial lines | Exported types / declared public members |
|---|---:|---:|---:|
| Contract, precision and MEAI conventions | 11 | 240 | 19 / 106 |
| Shared native transport/protocol | 36 | 133 | 0 / 0 |
| Jev connection and errors | 13 | 33 | 4 / 26 |
| Ollaya discovery and semantic preparation | 0 | 138 | 4 / 28 |
| Adapted Laya kernels and new client/ownership | 22 | 274 | 10 / 77 |
| Adapted Qwen recipe and new bounded adapter | 1 | 104 | 3 / 32 |

Literal reuse is a conservative textual measurement: trim lines, exclude
comments/usings/namespaces and lines shorter than 20 characters, then count
each matching occurrence against the pinned source set. Renamed/restructured
algorithm reuse appears in the changed/new column, not as independently
invented code. All other lines still contribute to the physical totals.
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
