# Ordinary OllamaSharp native decision adapter

**Experimental and unofficial.** `DecisionInference.OllamaSharp` is a separate,
optional implementation of the existing `IDecisionGenerator`, designed for
applications that already use an ordinary `IOllamaApiClient`. It does not
implement chat/embedding or recreate the custom provider's transport.
Start with the [all-three-kind quickstart](../../samples/quickstarts/OllamaSharp/README.md).

## Version and evidence boundary

| Component | Pin |
|---|---|
| Existing custom-provider baseline | `04afdaee7892d362893502bea0a3b165d3b1b318` |
| SDK | OllamaSharp NuGet **5.5.0**, source `135abcc3e31ac6a1ec535ff4c891e11ed1328df4` |
| Server protocol/compiler | Ollama **v0.35.0**, `cc4069396f3ad2c370c53eed2e4a42ac13adab84` |
| Framework/MEAI | net10.0; full Microsoft.Extensions.AI and Abstractions **10.10.0** |

The actual released package, not just repository source, is restored and used
by the offline proof. Its net10.0 group requires full MEAI and Abstractions
>=10.8.0; both resolve to 10.10.0. The resolved runtime closure is three private
packages (`DecisionInference.OllamaSharp`, SystemOne, Abstractions) and eight
external packages: OllamaSharp 5.5.0, MEAI/Abstractions 10.10.0, Caching.Abstractions,
DependencyInjection.Abstractions, Logging.Abstractions, Primitives and
System.Numerics.Tensors 10.0.12. The managed Tensors dependency comes from MEAI,
not TensorSharp/model/native inference. The SDK also contains a source-generator
analyzer. No ONNX/tokenizer/native assets or model files belong to this graph.

Offline fixtures validate integration, not installed-model compatibility,
accuracy, calibration, GPU performance, trimming or Native AOT. No server was
installed/started, no model was pulled/loaded, and no live inference was used.
Measured test/API/package identities belong to the executor's ready receipt;
historical test counts elsewhere are not evidence for this adapter.

## Conventional adapter contract

```text
owned DecisionInput
  -> shared Ollama preflight and SystemOne.Encode(maximumScoreLevels: 26)
  -> fresh, aliased OllamaSharp.Models.SystemOneRequest
  -> caller-supplied IOllamaApiClient.SystemOneAsync(request, token)
  -> detached known-field SDK DTO snapshot
  -> shared SystemOne.Decode(HighPrecision) and Ollama postchecks
  -> validated DecisionResult
```

`OllamaSharpDecisionGenerator(IOllamaApiClient client, bool ownsClient = false)`
accepts a stock SDK client. There is no guarded subclass, HTTP capture handler,
second custom transport, chat projection, discovery, model management, automatic
retry, fallback or hidden reference to `DecisionInference.Ollama`.

The SDK default model and immutable URI value are captured at construction.
An empty/null default means per-call model required; nonempty whitespace fails.
Every invocation supplies an explicit model on a fresh request. Overrides are
nonsticky and never mutate `SelectedModel`; later client changes do not silently
alter captured metadata. Model echo is checked ordinally, not as digest attestation.

Null, plain `DecisionGenerationOptions`, and `OllamaSharpGenerationOptions`
are supported. Other derived options and nonempty extensions fail before the
SDK call. Derived `Clone` retains nullable `KeepAliveSeconds` and clones the
extension dictionary using the portable shallow-value convention. Residency
is invariant signed integer text plus `s`, including both Int32 bounds; null
omits it. Unlike the custom provider's exact integer wire value, the SDK uses
a duration **string**. A bare `"60"` is not used as a seconds substitute.

Shared preflight retains 1..64 questions, Choice/Score 2..26, nonblank string
instructions, and string/object/array state with nonblank string state. Binary
descriptions do not replace required instructions. State is an owned boxed
`JsonElement`, not `GetRawText()` passed as a string. Questions use ordered
`q0`, `q1`, ... aliases; choices use ordered numeric aliases and explicit
descriptions; score rubrics remain ordered lists. Caller IDs remain opaque,
ordinal and case/Unicode-sensitive.

The new internal `OllamaDecisionProtocol` only extracts existing input/model,
first-tie-winner and exact-legend checks. It changes no portable contracts,
custom transport, integer `keep_alive`, final 65,536-byte request cap, original
raw JSON, exception API or legacy lockfile/version. SystemOne exports no types;
its encoder still defaults to ten Score levels for older providers.

## Results and services

Known DTO fields are projected explicitly into detached native-field JSON.
Unsupported answer subtypes and null DTO envelopes/collections fail explicitly.
The shared decoder retains double probabilities and **HighPrecision** with
the unchanged global `1e-6` distribution tolerance, validates complete aliases,
answer kinds, normalized finite mass and ordinal expectation. Shared postchecks
require the exact model echo, first request-order exact Choice maximum, and
exact Score rubric legend. Nothing is rounded, repaired or partially returned.

Result/answer `RawRepresentation` values are owned `JsonElement` snapshots.
`ollamasharp.rawRepresentationKind = "sdk-dto-snapshot"` states their provenance;
existing `systemone.questionIds`/`candidateIds` metadata remains. DTO dictionaries
and usage are not retained as mutable result payloads. SDK Int32 counters become
Int64 portable counts; two individually representable maxima validly sum to
4,294,967,294. Negative counters fail. Reasoning/server latency stay unknown.
SDK confidence remains backend snapshot evidence, not calibrated correctness.

Recognized SDK JSON parsing and local mapping failures become
`DecisionProtocolException` without sensitive parser details/inner exception.
Genuine SDK/HTTP errors and cancellation propagate; no status/request ID is
invented. SDK 400 errors can contain server diagnostics; applications must
choose explicitly whether/how to inspect them securely.

Exact services are `IDecisionGenerator`, `OllamaSharpDecisionGenerator`,
`DecisionGeneratorMetadata` and `IOllamaApiClient`. A non-null key returns null.
No automatic chat/embedding/concrete-client/object service is exposed.
The SDK is borrowed by default. Ownership requires `IDisposable` at
construction and disposes it once; no separate HTTP disposal occurs. Stock
SDK construction over borrowed HTTP does not own that HTTP. Calls after
adapter disposal fail. Disposal during active calls is unsupported.

## Deliberate SDK limitations

| Area | Honest guarantee / difference |
|---|---|
| Scalar presence | Missing `noul`, `score`, `confidence`, input/output counters can become zero in SDK DTOs. Validating the snapshot cannot detect the missing source property. |
| Usage domain | Out-of-Int32 source counters fail SDK JSON parsing, unlike the custom provider's wider Int64 wire domain. Widening a successful DTO does not recover lost information. |
| Duplicate/unknown fields | Duplicate keys can overwrite; unknown fields and optional original `total_tokens` disappear. No original duplicate-free/total-consistency claim is possible. |
| Raw evidence | Snapshot only: no original JSON/unknown fields, lexical numeric spellings or byte-for-byte provenance. |
| Final request size | The SDK owns final serialization. No local adapter body cap or intermediate-schema size estimate is advertised. The server can reject oversize requests. |
| Response buffering | Stock SDK uses `ResponseContentRead`, with no adapter-level response cap. |
| HTTP policy | Caller owns endpoint safety, auth/headers, redirects, cookies, UTF-8 handling, timeouts and SDK configuration. |
| Error diagnostics | Genuine SDK errors are not the custom provider's sanitized/opt-in diagnostic contract; no automatic sensitive-message logging. |
| Extra wire data | The pinned SDK serializes `CustomHeaders`, even when empty, and also uses it for actual headers. Characterization captures real emitted bytes. |
| Serialization/prompt | Default `UnsafeRelaxedJsonEscaping` and caller configuration can change Unicode/HTML escaping for object/array state. The wrapper does not mutate borrowed serializer options. |
| AOT | Initial proof does not certify trimming/Native AOT. |

The pinned server unescapes **string** state but uses `json.Compact` on
object/array raw JSON. That compacts whitespace without unescaping string
values. SDK `{"text":"café <tag>"}` and custom
`{"text":"caf\u00E9 \u003Ctag\u003E"}` can be semantically equal JSON but different
model context. The primary comparison therefore uses the same simple string
ticket, schema aliases and descriptions. Unicode/object/array cases record
actual state fragments and final byte counts separately; they do not manufacture
parity by reconfiguring the client.

This wrapper is useful for SDK-stack reuse, **not** a drop-in substitute for
the custom provider's stricter source/transport contract. Choose the custom
provider when those guarantees are required. Raw capture, transport hardening,
upstream fidelity fixes, live comparison and publication remain separate work.

## Independent offline verification

```powershell
.\eng\verify-ollamasharp.ps1 -Offline
.\eng\verify-ollamasharp.ps1 -Offline -Packages
```

`eng\OllamaSharpProof.slnx` builds the adapter and its stock-SDK HTTP tests.
The package-only consumer is deliberately built separately **after** packing
into a fresh private feed/cache; it has no source project references or original
provider reference. Package mode verifies actual external archive content hashes,
fresh private DLL/archive identities and closure. Old/new full API signatures
are compared with the exact baseline; an accepted baseline JSON may be passed
with `-ApiBaseline`, otherwise a temporary archive of the immutable baseline is
built read-only. No shared Git refs or protected worktrees are modified.

The existing default decision solution/test/proof/verifier graph remains SDK-free.
The retained legacy package path reconfirms the original Ollama-only four-package
closure. The full offline gate runs without model paths; no live/native-model
operation is enabled by `-Offline` or `-Packages`.

### Primary evidence

- [Released SDK NuGet 5.5.0](https://www.nuget.org/packages/OllamaSharp/5.5.0)
- [Pinned native interface](https://github.com/awaescher/OllamaSharp/blob/135abcc3e31ac6a1ec535ff4c891e11ed1328df4/src/OllamaSharp/IOllamaApiClient.cs)
- [Pinned native DTOs](https://github.com/awaescher/OllamaSharp/blob/135abcc3e31ac6a1ec535ff4c891e11ed1328df4/src/OllamaSharp/Models/SystemOne.cs)
- [Pinned SDK HTTP/serializer/lifetime implementation](https://github.com/awaescher/OllamaSharp/blob/135abcc3e31ac6a1ec535ff4c891e11ed1328df4/src/OllamaSharp/OllamaApiClient.cs)
- [Pinned request/header base](https://github.com/awaescher/OllamaSharp/blob/135abcc3e31ac6a1ec535ff4c891e11ed1328df4/src/OllamaSharp/Models/OllamaRequest.cs)
- [Pinned server compiler/answer semantics](https://github.com/ollama/ollama/blob/cc4069396f3ad2c370c53eed2e4a42ac13adab84/decision/systemone.go)
- [Pinned server duration parser](https://github.com/ollama/ollama/blob/cc4069396f3ad2c370c53eed2e4a42ac13adab84/api/types.go)
