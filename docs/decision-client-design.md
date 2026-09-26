# Decision-first side-by-side experiment

This is an experimental, project-owned `DecisionInference` API, **not** a
Microsoft/Agent Framework API, a replacement for `IChatClient`, or a quality
benchmark. The chat implementation at `bc2f307ed2b7e8c5ea0eac5e95dabeb4868b7c19`
remains unchanged. Build the additive experiment with
`dotnet build eng\DecisionComparison.slnx`; the original solution remains the
only solution at the repository root.

```csharp
var request = new DecisionRequest(
    JsonSerializer.SerializeToElement("An email asks for a password."),
    [new ChoiceDecisionQuestion("category", "Choose one category.",
        [new("legitimate", "Legitimate email"), new("phishing", "Credential theft")])]);
DecisionResponse response = await client.GetResponseAsync(request, cancellationToken: token);
var answer = (ChoiceDecisionAnswer)response.Answers[0];
// Application policy, not the provider, decides whether to act.
```

Instructions, descriptions and rubric levels are ordinary strings. State is
owned JSON, not a chat message or a schema. Question/candidate IDs are nonblank,
ordinal, unique and never trimmed or normalized. Collections are snapshotted;
returned distributions and JSON evidence are owned. `UsageDetails` is the
existing MEAI primitive, copied with its standard `Add` method. Unknown counts
remain null; local prepared token counts are not billable Jev token counts.

Binary returns the probability of true, Choice an exact selected ID and a
complete exclusive distribution, and Score a complete ordered distribution
with `ExpectedIndex = sum(index * probability)`. Finite probabilities must be
in `[0,1]` and sum to one within `1e-6` absolute. A distinct native Score must
agree with the expectation within `1e-6 * (levelCount-1)`. Malformed responses
are rejected, never repaired or renormalized. Ordinal expectation does not
assert equal real-world distances between rubric categories.

There is no portable confidence, policy threshold, abstention, tool calling,
DTO/schema compilation, chat projection, fallback, registry, scheduler or
middleware framework. Every request succeeds completely or fails. One logical
batch does not promise a single HTTP call/forward, joint answer consistency,
or common token accounting.

## Dependencies and capabilities

| Library | Direct dependencies beyond the shared contract | Capability |
|---|---|---|
| Abstractions | MEAI.Abstractions 10.10.0 (`UsageDetails`) | Contract only |
| TypeSafe | None | Binary/Choice/Score over System One |
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

**Implementation receipt:** the tested implementation is commit
`54b61ddcdc1e791addc333c8766162c98b1cf074`; its final combined proof log is
`artifacts\decision-full-gate.log` in this worktree (also retained in the session
files). This experiment and its documentation follow-up are local and unpushed;
no PR, publication or remote branch was changed. This receipt-only follow-up
leaves all runtime, test and proof inputs at the tested implementation.

```powershell
.\eng\verify-decisions.ps1 -Offline -IncludeBaseline -Packages
# With explicitly provided local assets; neither command downloads them:
.\eng\verify-decisions.ps1 -Offline -Packages `
  -LayaDirectory <local-laya-directory> -QwenPath <local-gguf> -IndependentReference
```

Default model-free runs explicitly print which real-model gates were not run.
The optional independent Laya check requires already-installed Python
`tokenizers`, `numpy`, and `onnxruntime`; these are not library dependencies.
No live/paid HTTP inference is performed. Local package proof verifies cached
external archive hashes/signatures before using an isolated offline feed with
explicitly accepted local unsigned packages; no global trust setting changes.

The one host supports `compare` (same synthetic Jev envelopes through both
APIs), `provider jev`, `provider laya <onnx> <tokenizer.json> <config>`,
`provider qwen <gguf>`, `stages <onnx> <tokenizer.json> <config> [receipt.json]`,
and `inventory`. `compare` captures real outgoing bodies and checks exact
semantic JSON equality, model, usage and complete owned native evidence.
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
The independent decision suite adds 202 passing VSTest cases (no skips); those
are separate from, not additions to, the baseline's 650 cases.

| Required behavior | Representative exact test |
|---|---|
| Exact IDs and owned request/result snapshots | `OpaqueIds_PreserveSpacesCaseAndDistinctUnicodeNormalization`; `Response_OwnsAnswerDistributionsEvidenceAndUsageSnapshots` |
| Complete distributions and ordinal expectation | `Response_RequiresExactCompleteCandidateAndRubricDistributions`; `Score_ExpectedIndexAndNativeToleranceScaleWithRubric` |
| Native transport and real per-read cancellation | `TypedRequest_UsesNativeStringEnvelopeAndDecodesAllThreeKinds`; `DerivedStreamContent_StalledReadReceivesCancellationAndCompletesBeforeDisposal` |
| Actual synthetic ORT execution with independent numeric assertions | `SyntheticOrt_ScoresActualDynamicGraphAndMatchesPublicStageClientForAllKinds` |
| Bounded single-token scoring recipe | `AliasIds_RejectsDuplicateMultitokenNegativeOrMissingTokens`; `Prepare_ExactLiteralPromptUsesDescriptionsNotOpaqueIdsAndOwnsTokens` |

The synthetic suite does not impersonate trained-model proof. Real Qwen KV
reset/repetition and the independent Laya oracle are executed by the explicit
local-model gates. Package proof runs both supplied local backends through a
package-only copy of the host, plus a separate Laya-only public-stage consumer
whose resolved closure excludes chat, TensorSharp and ML.NET.

## Provenance and honest scope accounting

| Source | Pin | Reuse |
|---|---|---|
| Existing chat proof | `bc2f307ed2b7e8c5ea0eac5e95dabeb4868b7c19` | Native transport/protocol algorithms source-adapted into three TypeSafe files; no shared binary |
| [Embeddings prototype](https://github.com/luisquintanilla/onnx-embeddings-prototype) | `79b71da1e0ae44139983fca03734974a2f2f8c80` | Direct primitives, owned output copies, borrowed options/session, `Terminate` cancellation |
| [Domain kernel](https://github.com/luisquintanilla/mlnet-text-inference-custom-transforms) | `e9596f64640e0ca39a5ad931bffd785696f4962f` | Laya rendering/framing, graph contract, temperature policy, BPE configuration; no ML.NET layer |
| [Laya export](https://huggingface.co/receptron/laya-onnx/tree/68f27dfe5a27a54fb2b1fefc432f43f972e90868) | `68f27dfe5a27a54fb2b1fefc432f43f972e90868` | Five hash-pinned local files; hashes in verification entry |
| [Qwen sample](https://gist.github.com/luisquintanilla/3d1a22ca8dbab1ae9a80caff4f2bf612) | Models `2026.9.1`; GGUF `7f403014…fe2e1fa` | Prompt/selected-logit scoring adapted; bounded aliases, ownership and reset added |
| TensorSharp package source | `70909d16321c51a78667ed9a4e243afe925071a1` (nuspec) | Package's actual APIs compiled and executed; not assumed from latest source |

The new runtime is 13 C# files/1,147 physical lines: contract 275, Jev 268,
Laya 454, Qwen 150. All adapted/duplicated source remains counted. The original
1,796 lines/25 files include schema projection, policies, tool support,
middleware integration and evaluation that the new API intentionally lacks.
Do not compare all three new backends with one old provider and call the
difference a like-for-like reduction. Tests, host, verification scripts,
documentation, lockfiles and package dependencies are separate costs.
`inventory` reports exported types and declared public reflection members,
not a subjective API simplicity score.

| Responsibility | Nontrivial exact reference-line matches | Changed/new nontrivial lines | Exported types / declared public members |
|---|---:|---:|---:|
| New contract | 0 | 164 | 16 / 84 |
| Adapted Jev transport/protocol | 57 | 92 | 4 / 26 |
| Adapted Laya kernels and new client/ownership | 22 | 259 | 9 / 69 |
| Adapted Qwen recipe and new bounded adapter | 1 | 88 | 1 / 7 |

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
implementation evidence, not a shipped/accepted contract. No package was
published and no license, repository visibility, PR or remote branch changed.
