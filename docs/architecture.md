# Architecture

Native helpers call the **supplied decorated IChatClient**:

```text
one explicit state + JSON questions
  -> shared native validation / snapshot / effective defaults
  -> existing MEAI logging + OpenTelemetry
  -> optional DistributedCachingChatClient
  -> TypeSafeChatClient
  -> HttpClient total timeout / retry / attempt timeout
  -> bounded body handler -> HTTP
```

The keyed registration does not replace an application's conversational client.
The provider and outer normalization share validation. Caller input collections
and JSON documents are snapshotted, not retained as mutable inference state.

Cache identity includes state, complete questions/rubrics, effective model,
provider, normalized origin, protocol version and explicit nonsecret account
namespace. It excludes keys and key hashes. Caching is opt-in; examples use a
version-pinned model and process-local distributed memory cache. There is no
assumed TTL or single-flight behavior. Concurrent misses may call HTTP more than
once. Cached usage is original inference usage, not new billing.

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
the sole opt-in cache owner. A fixed scripted primary answer is evaluated; Jev
is the judge, not a generative primary model. CompositeEvaluator runs children
independently and does not batch their distinct HTTP calls. Child error
diagnostics remain visible and are not equivalent to accepted quality ratings.
Existing evaluators expecting generated explanations are not drop-in compatible
with this constrained provider.

`TypeSafeFunctionFactory` exposes only a string state parameter over an
application-approved immutable question set. Model, endpoint, credentials and
rubrics cannot be changed by the conversational caller. The sample places
UseFunctionInvocation on a scripted conversational IChatClient, not on Jev.
The real middleware calls the function, feeds back its complete native result,
and obtains a scripted final response. There is no custom tool loop.

## Composition boundaries

```text
MEAI-generated typed schema + explicit field bindings
  -> schema compiler / native questions
  -> supplied decorated native pipeline (and optional native cache)
  -> full native envelope -> policy projection / complete DTO or abstention

original stateless downstream request
  -> separate native Choice classification -> existing RoutingChatClient
  -> exactly one borrowed target, unchanged response/model/usage

registered AIFunction.JsonSchema + bounded bindings
  -> selection Choice -> selected arguments only (zero/one batch)
  -> evidence + at most one FunctionCallContent, no execution
  -> separate host authorization + required approval -> outer AIFunction once
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
MEAI generic typed API and the separate evaluator through a counting decorator.
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
