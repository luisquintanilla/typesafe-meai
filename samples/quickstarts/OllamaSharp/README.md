# Reuse your OllamaSharp client for typed decisions

This standalone example wraps an ordinary **OllamaSharp 5.5.0**
`IOllamaApiClient` with `OllamaSharpDecisionGenerator`. It asks Binary, Choice
and Score questions about the same invented ticket used by the
[custom HTTP Ollama example](../Ollama/README.md). It prints predictions;
it never routes a ticket.

The adapter is experimental, unofficial and unpublished. It is not an
`IChatClient`, a new inference runtime, or a replacement for the custom
provider's stricter transport and original-evidence guarantees.

## Prerequisites and safe first commands

Use the repository's .NET 10 SDK (`global.json`) from the repository root.
Restoring code dependencies may access the official NuGet feed; models are
not included or acquired by any build.
The file app explicitly names its assembly/package
`DecisionInference.OllamaSharp.Quickstart` so its `OllamaSharp.cs` filename
cannot collide with the SDK's package or assembly identity.

```powershell
dotnet run --file samples\quickstarts\OllamaSharp\OllamaSharp.cs -- --help
dotnet run --file samples\quickstarts\OllamaSharp\OllamaSharp.cs -- --check-config http://127.0.0.1:11434 YOUR-INSTALLED-MODEL
```

Both paths return before constructing HTTP/SDK clients. Configuration checking
does **not** check server availability, version, model installation or scoring
compatibility. Missing/invalid arguments exit 2. The sample accepts only an
HTTPS origin or literal-loopback HTTP, without user information/query/fragment;
that is a teaching guard, **not a policy enforced by the SDK adapter**.

A future, separately authorized live call would require an independently
managed Ollama **0.35.0+** server and an explicitly installed compatible local
System One model. Not every hosted model supports native decisions; a model
label is not a verified digest. Follow the existing
[Ollama model/setup guide](../Ollama/README.md) and review model licenses.
This example neither installs/upgrades/starts Ollama nor pulls models.

Only when you deliberately intend to send the synthetic ticket:

```powershell
dotnet run --file samples\quickstarts\OllamaSharp\OllamaSharp.cs -- http://127.0.0.1:11434 YOUR-INSTALLED-MODEL
```

No live call was used to validate this proof. File apps explicitly disable
Native AOT; SDK marketing is not adapter trimming/AOT certification.

## The important change is construction

The complete app is [OllamaSharp.cs](OllamaSharp.cs). Its ordinary SDK client is
caller-owned:

```csharp
using var ollama = new OllamaApiClient(http, installedModel);
IOllamaApiClient client = ollama;
using IDecisionGenerator generator = new OllamaSharpDecisionGenerator(client);
```

`http` and `installedModel` are supplied by the caller. `GenerateAsync` consumes
the same portable questions and typed answers as the custom provider. The
adapter uses fresh `q0`, `q1`, `q2` question aliases and ordered numeric candidate
aliases, preserves all descriptions, and makes exactly one native
`SystemOneAsync` call. No chat, model discovery, retry or fallback occurs.

For comparison, the bare SDK shape for **one** binary question is:

```csharp
var request = new OllamaSharp.Models.SystemOneRequest
{
    Model = installedModel,
    State = input.State.Clone(), // boxed JSON, not an object encoded as a string
    Questions = new()
    {
        ["q0"] = new OllamaSharp.Models.SystemOneNoulQuestion
        {
            Instructions = "Does this ticket concern account access?"
        }
    },
    KeepAlive = "60s"
};
var native = await client.SystemOneAsync(request, cancellationToken);
```

This shorter snippet explains DTO mapping, not the full sample's schema or an
additional request in the adapter. The full question schema affects model
conditioning. Binary returns P(true), Choice returns a candidate-relative
distribution, and Score returns a distribution over ordinal indices plus its
expectation. Concentration/confidence is not calibrated correctness.

## Models, residency and ownership

The generator captures `SelectedModel` once. Per-call `ModelId` overrides it
without changing the SDK setting; later SDK mutations do not alter its captured
default. An empty SDK default requires a nonblank per-call model. Caller
options are cloned. Do not mutate SDK/serializer settings during concurrent
calls.

`KeepAliveSeconds = 60` becomes `"60s"` in the SDK body. Null omits it, zero
unloads **after** the request, and negative retains. Residency is not the
30-second cancellation deadline and is not an unload/pull API call.
Cancellation stops waiting; it does not promise to interrupt server inference.

The sample disposes generator, SDK client, then HTTP client using ordinary
reverse `using` order. The adapter borrows by default. `ownsClient: true`
requires `IDisposable`, disposes the SDK once, and never separately disposes
its HTTP client. A stock SDK built over borrowed HTTP does not own that HTTP.

## Read the reduced guarantees before choosing a provider

`RawRepresentation` is an owned **SDK DTO-derived JSON snapshot**, identified
by `ollamasharp.rawRepresentationKind = "sdk-dto-snapshot"`, not the original
response. Unknown/duplicate properties and original optional totals may be
lost; missing numeric fields may already have become zero. Input/output counts
are SDK Int32 fields, although the adapter sums representable counts into Int64.
Reasoning and server latency remain unknown.

The caller owns endpoint safety, headers, redirects, cookies, serialization,
buffering and deadlines. The ordinary wrapper does not impose final request
or response byte caps. The sample disables redirects/cookies but that does
not bound stock SDK `ResponseContentRead` buffering. SDK HTTP errors can carry
sensitive diagnostics; the sample reports the failure type and a nonzero exit,
not arbitrary exception messages.

The simple **string** teaching ticket avoids a known prompt-spelling trap.
Object/array state remains JSON (never double-encoded), but the SDK's relaxed
encoder and caller settings can emit different Unicode/HTML escaping from the
custom provider. The pinned server compacts object/array JSON without
unescaping it: semantic JSON equality is not necessarily equal model context.
See [the precise adapter contract](../../../docs/design/ollamasharp-adapter.md)
for tested divergences and immutable source pins. Continue using the custom
provider when its stricter guarantees matter.

## Offline proof and troubleshooting

```powershell
.\eng\verify-ollamasharp.ps1 -Offline -Packages
```

This independently builds/tests the SDK path and a fresh package-only consumer,
not a daemon. The legacy solution/verifier stays free of the SDK.
If an unrelated ancestor build file imports Arcade, use the documented
[process-scoped workaround](../../README.md#troubleshooting-shared-setup),
never a global setting or certificate bypass.

A live failure can mean wrong endpoint/model, unsupported server/model scoring
or a genuine SDK/HTTP error. Inspect diagnostics explicitly in a secure
application context; do not log ticket-bearing server messages automatically.
No successful empty result, guessed model, rounding repair or custom-provider
fallback hides failure.
