# Architecture

Consumers call the **supplied decorated IChatClient** with ordinary MEAI APIs.
There is one public TypeSafe provider, not separate schema/tool/router facades:

```text
messages + schema OR tools OR JSON native questions
  -> TypeSafeChatClient: validation / owned inputs / compiler / tool selection
  -> native batch: effective model / questions / nonsecret cache scope
  -> existing MEAI logging + OpenTelemetry
  -> optional DistributedCachingChatClient
  -> internal native IChatClient: raw protocol validation / HTTP body
  -> HttpClient total timeout / retry / attempt timeout
  -> bounded body handler -> HTTP
```

The keyed registration does not replace an application's conversational client.
Public dispatch validates and snapshots caller input collections and JSON before
cache lookup. Each native batch resolves the effective model and cache scope
before invoking the decorated pipeline. There is no second normalization
middleware reconstructing that batch.

Cache identity includes state, complete questions/rubrics, effective model,
provider, normalized origin, protocol version and explicit nonsecret account
namespace. It excludes keys and key hashes. Caching is opt-in; examples use a
version-pinned model and process-local distributed memory cache. There is no
assumed TTL or single-flight behavior. Concurrent misses may call HTTP more than
once. Cached usage is original inference usage, not new billing.

Native cache results contain `JevResultContent`, not projected DTOs or executable
call IDs. Policies run after cache; tool correlation IDs are created afresh after
inference. Provider-added batch/path metadata is applied to owned copies.

The raw native client still validates its standard messages/options and response
correlation independently: it is a different trust boundary, not a trusted-state
shortcut. It builds one owned HTTP body from the validated question JSON rather
than recreating every question's wire representation and unrelated option
metadata. Public emission checks correlation again because a cache hit bypasses
the raw client. No request marker or service lookup bypasses middleware.

Compared with the original registered path, a cache miss now has one
`NativeRequest.Create` instead of two, and one `SnapshotOptions` instead of three.
A cache hit has no `NativeRequest.Create`; caller snapshots and public response
correlation remain. State still crosses a standard MEAI message boundary, so the
raw client deliberately retains its independent state validation.

## One canonical evidence content

`JevResultContent : DataContent` owns complete native UTF-8 JSON. Computed
read-only answer/model/usage views are `[JsonIgnore]`; the inherited data URI is
the serialized truth. It validates itself without a question snapshot. Provider
response correlation is a separate check. No consumer response extension,
RawRepresentation or service-unwrapping is needed:

```csharp
var evidence = response.Messages.Last().Contents.OfType<JevResultContent>().Single();
var answer = (NoulAnswer)evidence.Answers["followup"];
```

Custom AIContent types require registration, including their DataContent base:

```csharp
var json = new JsonSerializerOptions(AIJsonUtilities.DefaultOptions);
json.AddAIContentType<JevResultContent>(JevResultContent.TypeDiscriminator);
```

The keyed registration sets these copied options on the existing
DistributedCachingChatClient. Global defaults are never mutated. A separately
configured serializer or **outer** cache also needs explicit registration;
Unregistered serialization as AIContent (including ChatResponse contents) fails
explicitly. Statically serializing as DataContent can instead preserve bytes/media
while losing the custom subtype; it is not a substitute for registration.
An outer cache would also cache projections/call IDs, so do not place one around
the tool/execution workflow or expect threshold-only reuse there.

The public provider returns native-only content or projected JSON TextContent
**alongside evidence in the same last assistant message**. It never appends a
second native JSON text or a trailing evidence-only message that could confuse
actual MEAI `ChatResponse<T>` parsing.

Per-call evidence copies own their bytes and deeply snapshot mutable annotations
and additional metadata using MEAI's serialization contract; immutable decoded
native views are shared. This avoids serializing the complete evidence to a
base64 data URI and decoding/parsing it again merely to add batch/path metadata.
`Name` is preserved and JSON-ignored `RawRepresentation` is not copied. Actual
cache serialization still uses the registered subtype and validates payloads
when deserializing.

MEAI calls the operation chat even though it is native decision inference.
Streaming duration is whole-response duration. The registration explicitly
disables OpenTelemetry sensitive capture and prevents Trace-level MEAI message
logging, even with a permissive logger. HTTP factory loggers are removed to avoid
auth/request leakage. Caches and evaluation reports still contain sensitive
application data; generated artifacts are ignored and should remain private.

## Reproducible dependency bootstrap

SDK `10.0.401`, MEAI/Abstractions/Evaluation/Reporting/Http.Resilience `10.10.0`,
xUnit v3 `3.2.2`, VSTest SDK `17.14.1`, runner `3.1.5`, and local aieval
`10.10.0` are pinned. Commit lockfiles; restore in locked mode.

This machine's NuGet v3 endpoint fails TLS handshake. Official HTTPS NuGet v2
and CDN work without bypassing certificate checks. Repository configuration
clears inherited sources, disabled sources and mappings; signature validation
requires the verified NuGet repository signers (including its timestamped
historical certificate). No machine trust configuration is changed.
The final fresh-package check also required the official 2021-2024 repository
certificate `5A2901D6ADA3D18260B9C6DFE2133C95D74B9EEF6AE0E5DC334C8454D1477DF4`
for timestamped `System.Threading.RateLimiting 8.0.0`. Its fingerprint is
published in the [Microsoft certificate-rotation announcement](https://devblogs.microsoft.com/dotnet/the-nuget-org-repository-signing-certificate-will-be-updated-as-soon-as-april-8th-2024/).
It is explicitly pinned with `allowUntrustedRoot=false`; `require`, certificate
chain, timestamp and TLS validation stay enabled. An already-cached package had
masked this missing historical entry during initial restore.

The SDK's default dependency resolver locally produced NU1102 for `>=10.0.12`
while reporting `10.0.12` as its nearest version. The supported
`RestoreUseLegacyDependencyResolver=true` property fixes this local discrepancy.
[First-party guidance](https://learn.microsoft.com/nuget/consume-packages/package-references-in-project-files#nuget-dependency-resolver).
This is not a diagnosed upstream defect. Remove the repository-local workaround
only after the same graph restores with the default resolver and compare assets.

## Evaluation and conversational tools

`TypeSafe.MEAI.Evaluation` is separate so native consumers do not acquire an
evaluation dependency. Its IEvaluator snapshots supported message/context
contents and submits all configured metric questions in one native batch through
the supplied `ChatConfiguration.ChatClient`. Noul and fractional Score become
NumericMetric; Choice becomes StringMetric. No reason, rating or BooleanMetric
is invented. Interpretation requires an explicit host policy.

Full answer/rubric/envelope JSON is stored as namespaced **strings** in metric
metadata. Existing AddOrUpdateChatMetadata supplies model, token and duration
metadata. A shared `typesafe.v1.batch` ID identifies repeated metric metadata:
do not sum each metric's batch token counts as separate inferences. Additional
context is stored in standard Contents, not only derived CLR properties.
Native evaluation state uses `conversation[i].role` and
`conversation[i].contents[j].{type,value}`, with the same shape for `response`.
`context[i]` contains `name` and the same `contents` shape. A text item has
`type:"text"` and a literal string value; application/json data has `type:"json"`
and its actual structured JSON value (including JSON strings/null). JSON data
never becomes a base64 data URI in the inference state. Normal DataContent is
retained separately for standard reporting serialization. This preserves the
difference between literal JSON-looking text, a JSON string, and a JSON object.

The sample uses existing disk ReportingConfiguration and disposal persistence.
Reporting-layer response caching is explicitly disabled; native caching remains
the sole opt-in cache owner. Both a fixed scripted primary answer (Jev as judge)
and a finite Jev decision as primary response are saved. Jev is never asked to
generate free-form evaluated text. CompositeEvaluator runs children
independently and does not batch their distinct HTTP calls. Child error
diagnostics remain visible and are not equivalent to accepted quality ratings.
Existing evaluators expecting generated explanations are not drop-in compatible
with this constrained provider.

Pinned reporting owns readonly internal serializer options with no custom type
registration hook. The host's `ReportSnapshot` explicitly replaces Jev content
with ordinary DataContent **only at reporter ingress**, preserving bytes, media,
annotations, metadata, message/response IDs, usage and timestamps. It then makes
an owned standard serialization snapshot; the live custom response is unchanged.
Reports therefore contain plain DataContent; reconstruct typed views explicitly
with `new JevResultContent(data.Data)` when reading matching media. There is no
custom reporting framework, global serializer mutation or hidden lossy fallback.

The `tool` scenario uses ordinary `AIFunctionFactory.Create` over an
application-approved question set and a decorated judge. Only a string state is
exposed, not endpoint, credentials or rubrics. Existing UseFunctionInvocation on
a scripted conversational client calls that closure and consumes its complete
native JSON result. No second paid model or custom tool loop is needed.

## Composition boundaries

```text
MEAI-generated typed schema + explicit client policy + optional field overrides
  -> schema compiler / native questions
  -> supplied decorated native pipeline (and optional native cache)
  -> JevResultContent -> policy projection / complete DTO or abstention

original stateless downstream request
  -> separate native Choice classification -> existing RoutingChatClient
  -> exactly one borrowed target, unchanged response/model/usage

ordinary ChatOptions.Tools / AIFunction.JsonSchema
  -> selection Choice -> selected arguments only (zero/one batch)
  -> evidence + at most one FunctionCallContent, no execution
  -> existing UseFunctionInvocation + host authorization + standard approval
  -> original outer AIFunction once; Terminate=true and zero error recovery
```

Every native classification/argument batch goes through the same supplied
decorated `IChatClient`; adapters never find a raw provider service and bypass
middleware. Projection remains outside native caching. See
[decision policies](decision-policies.md) for the exact schema subset, pointer
bindings, inclusive policies, ownership, and approval boundaries.

## Local package-only proof

`eng\verify.ps1 -All -Offline` invokes the package verification helper only after
the prerequisite-inclusive tests, reports and scenarios. It packs the two
libraries locally; nothing is published. Package contents are allowlisted to
the library DLL/XML, README and NuGet metadata, and dependency IDs are checked.
No tests, fixtures, sample credentials, source archives or reports are packed.
The copied consumer project uses two exact `PackageReference` versions and
no implementation `ProjectReference`. It executes native inference, the actual
MEAI generic DTO/primitive API, custom-content serialization, the separate evaluator,
and one-shot ordinary tools through a counting decorator.
Its HTTP handler is entirely synthetic.

The repository requires signed external packages. Our own private proof
artifacts are deliberately **unsigned**, not production-signed. A fresh
`artifacts\package-proof\<run-id>` directory contains the isolated filesystem
feed, consumer configuration and package cache. The consumer alone permits the
two local unsigned packages; it has no remote package/audit sources. The
repository still requires trusted signatures (including the historical
certificate above); global configuration, machine certificates and shared caches
are never altered.

Only the external dependency closure named in the two library lockfiles is
copied from the existing package cache (not a broad cache copy). Each archive's
identity and **NuGet content hash** are recomputed against its lockfile; signed
packages use NuGet's canonical content hash, not a naive hash of the signed ZIP.
The pinned SDK's public `NuGet.Packaging.PackageArchiveReader.GetContentHash`
provides that algorithm. All copied external signatures are then verified with
the unchanged repository configuration. Raw archive SHA512 and actual cached
source provenance are also recorded; some initial bootstrap archives came via
the previously configured Microsoft package proxy and retain that provenance.

The fresh consumer restores only from this verified feed, creates a lockfile,
restores again in locked mode, checks its resolved external hashes against the
original closure, and verifies its assets contain no project reference or remote
source. This narrowly scoped unsigned exception cannot introduce an unverified
external dependency. `artifacts\package-proof\latest.txt` points to the successful
receipt; `provenance.json`, `external-signatures.log`, generated consumer lockfile
and packages remain local ignored evidence. No license or signing/publication
policy is selected by this proof.
