# Ollaya: use a concrete model on a separately managed server

[Sample catalog](../../README.md) | [Beginner guide](../../../docs/learning/README.md)

## What you will learn

[Ollaya.cs](Ollaya.cs) shows explicit endpoint/model selection, safe HTTP resource
ownership, and the same typed Choice question used by the other examples.
Knowing that a server is reachable is not enough: its model and input layout
must also be supported.

**Ollaya is a server, not a decision model.** It loads models and exposes an HTTP
API. This app uses `OllayaDecisionGenerator` to discover a selected model through
`/api/show`, validate an audited layout, then call `/v1/systemone`.
The demonstrated model is **`nli:modernbert-large`**, not the server's Laya router.
NLI means *natural language inference*: a model scores relationships between
text and described alternatives. Ollaya maps that model to decision-shaped
results; it is not the identical Julia/Laya specialized-head architecture.

The repository's historical live evidence used
[Ollaya v0.7.2 at 17b85bf](https://github.com/ollaya-dev/ollaya/tree/17b85bf86d4427b790858074434911aa6557b183)
and `nli:modernbert-large`. Its source names Moritz Laurer's ModernBERT-large NLI
classifier and declares that model Apache-2.0. Review the model's actual
manifest/source/license as well as the daemon license. Read the
[pinned API documentation](https://github.com/ollaya-dev/ollaya/blob/17b85bf86d4427b790858074434911aa6557b183/docs/api.md)
and [model registry](https://ollaya.dev/search). Current website features/models
are not automatically supported by this experimental adapter.

## Prepare the server deliberately

Checklist: .NET 10 SDK and full checkout for the client; an explicitly managed
compatible Ollaya server; the concrete model installed there; network access
to its endpoint. The C# client does not need the server's weights or ORT
libraries. The server does need its own supported platform/runtime and enough
disk/RAM for its chosen model; this tutorial does not infer resource
requirements from HTTP success.

If you administer the machine and choose to set it up, obtain the platform
release from [v0.7.2 releases](https://github.com/ollaya-dev/ollaya/releases/tag/v0.7.2),
review its install instructions, and intentionally prepare the model:

```powershell
ollaya --version
ollaya pull nli:modernbert-large
ollaya show nli:modernbert-large
```

**These are setup operations, not C# build steps**: `pull` downloads weights,
and Ollaya CLI operations may automatically start its daemon. Do not run them
on someone else's managed server without authorization. Consult the pinned
server documentation for `serve`, host binding and authentication; do not expose
an unauthenticated loopback development service publicly. Record the concrete
model manifest/hash for repeatable deployments: a tag alone is not immutable.
No daemon was started/reconfigured and no weights were pulled to verify this
new sample.

## Run the client

From the repository root, **only when you have authorized the request** and the
server/model already exist:

```powershell
dotnet run --file samples\quickstarts\Ollaya\Ollaya.cs -- --help
dotnet run --file samples\quickstarts\Ollaya\Ollaya.cs -- http://127.0.0.1:11435 nli:modernbert-large
```

The provider default origin is `http://127.0.0.1:11435`, but the tutorial requires
you to supply it explicitly so the destination is visible. It accepts HTTPS
origins or HTTP with a literal loopback IP (`127.0.0.1` or `[::1]`), not
`http://localhost`, insecure remote HTTP or credential-bearing URLs. This simple
client targets a server that does not require a key. For an
authenticated deployment, wire privately supplied credentials through the
provider's explicit `apiKey` constructor parameter; never embed them in a URL
or `DefaultRequestHeaders.Authorization` (the provider rejects preexisting auth).

`--help`, missing arguments and invalid origins fail/return before networking.
First build restores HTTP dependencies via source project references, not
model stacks or public `DecisionInference.*` packages. AOT publishing is disabled.

## Follow the code

1. Validate both origin and concrete model. A syntactically valid name still
   needs discovery; validation does not claim every daemon model is supported.
2. Own an `HttpClient` with redirects/cookies disabled and lend it to the
   generator. Reverse `using` order releases the generator before the client.
3. An explicit 30-second cancellation token spans discovery and inference body
   reads. `HttpClient.Timeout` is infinite to avoid a second competing deadline.
   Cancelling the wait does **not** prove server inference stopped.
4. State is the invented sign-in ticket. The question ID `"support-team"` and
   candidate IDs are exact C# lookup keys. Human-readable descriptions explain
   account access, billing and technical problems to the model.
5. Inline JSON source generation describes serialization of the typed string.
   Await `GenerateAsync`, then retrieve `ChoiceDecisionAnswer` by the question
   ID. No dynamic JSON or chat parser is involved in application consumption.

## Interpret results without inventing proof

**Illustrative shape only; this new file app was not run against a live server:**

```text
Provider: ollaya
Model: nli:modernbert-large
Selected ID: account
account: 0.7000
billing: 0.1000
technical: 0.2000
```

The selected ID belongs to your offered candidates. These are conditional
category scores, not evidence that 70% of comparable tickets are routed
correctly. The supported daemon protocol emits four-decimal observations;
the adapter preserves the declared precision semantics rather than claiming
extra knowledge. The app also formats to four decimals. Display rows need not
sum to exactly 1. A failed discovery/protocol call is an exception, not an
uncertain prediction. No real routing takes place.

## Exercises and troubleshooting

When live calls are authorized, predict then try "I was charged twice", "Help"
and "Tell me a joke". Explain why an out-of-scope input still produces a
candidate. Change a candidate description and observe whether probability mass
moves; no answer flip is promised. Add application-owned manual review only
after considering representative labels and error costs, not a universal cutoff.

Connection refused means the destination server is unavailable; the C# sample
does not start it. Missing model means deliberate server provisioning is
needed; the client does not pull it. Discovery rejection can mean a router,
fixed-schema guard or unaudited layout was selected. Use the explicitly supported
concrete model rather than bypassing validation.

The adapter audits NLI and nonempty GLiClass layouts. Historical **live** proof
covered NLI; GLiClass support is fixture-tested. Router/guard/unknown layouts and
the daemon's Laya layout are rejected (the latter would risk truncation). This
does not prevent using [embedded Laya](../Laya/README.md). Input budgets are
validated/rejected rather than silently truncated. Native server failures belong
to server administration, not adding ORT to this HTTP app.
See [shared SDK troubleshooting](../../README.md#troubleshooting-shared-setup).

State leaves this process even on loopback. A remote endpoint receives it over
the network; review server logging and privacy. This phase compiled the app and
checked help/config failure paths only; historical protocol/live evidence remains
separate. Next: [local versus HTTP deployment](../../../docs/learning/03-running-responsibly.md).
