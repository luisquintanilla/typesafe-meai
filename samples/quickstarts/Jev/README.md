# Jev: a typed decision from a hosted service

[Sample catalog](../../README.md) | [Beginner guide](../../../docs/learning/README.md)

## What you will learn

[Jev.cs](Jev.cs) teaches the same typed support-routing call as the local examples,
but the model runs **off-process at TypeSafe**, not on your CPU. You own an
`HttpClient`, lend it to a disposable generator, provide a key privately and
await a typed answer with an explicit deadline. No AI background is needed.

Jev is TypeSafe's hosted decision model. The **provider** here is
`TypeSafeDecisionGenerator`, an experimental adapter maintained in this project.
Its current defaults are endpoint `https://api.typesafe.ai`, model
`jev-1.13.0`, and `POST /v1/systemone`. See the primary
[service introduction](https://docs.typesafe.ai/introduction),
[HTTP API](https://docs.typesafe.ai/api) and
[primitives](https://docs.typesafe.ai/primitives).
Those are living service documents; they are not a promise that every future
model/service change is compatible with this adapter.

The hosted model's weights, internal architecture, parameter count and training
details are not established by this repository. Do not infer them from Julia,
Laya or the shared wire format. No model file, tokenizer or ONNX runtime is
installed for this example. The service performs inference remotely; your code
does not train it. Review the service's current access terms, costs and data
handling policies rather than treating hosted access as an open-weights license.

## Prerequisites and private configuration

Checklist: compatible .NET 10 SDK, full checkout, network access to the endpoint,
and **your own authorized TypeSafe API key** with access to the selected model.
Obtain credentials using the service's account/onboarding process linked from
its documentation. No key is provided by this repository, and there is no
recorded-response fallback if yours is missing.

From the repository root in PowerShell 7, set the key in this process without
putting the literal in shell history or a checked-in file:

```powershell
$env:TYPESAFE_API_KEY = Read-Host "TypeSafe API key" -MaskInput
dotnet run --file samples\quickstarts\Jev\Jev.cs --
Remove-Item Env:TYPESAFE_API_KEY
```

Environment variables are **not a secret vault**: child processes inherit them.
Clear the variable/close this shell when done and use your application's secret
management facility in a real deployment. Do not paste keys into issue reports.
The sample neither prints the key nor installs it in global HTTP headers.

Help is safe without credentials and does not send a request:

```powershell
dotnet run --file samples\quickstarts\Jev\Jev.cs -- --help
dotnet build samples\quickstarts\Jev\Jev.cs
```

First build can restore NuGet code dependencies. Only the HTTP provider and its
shared contract/protocol dependencies are referenced, not local model stacks.
The provider package is unpublished, hence `#:project`; AOT is explicitly off.

## Walk through the call

1. Reject unexpected arguments, a missing key or whitespace in the key before
   constructing the HTTP client. Failure exits 2 with actionable instructions.
2. `SocketsHttpHandler` disables redirects and cookies. A credential-bearing
   request should not follow a redirected destination silently. The caller
   owns transport policy; no retry/policy framework is hidden in this sample.
3. The `HttpClient` has an infinite built-in timeout. Instead, the explicit
   30-second `CancellationTokenSource` supplies a deadline to **the whole
   generator call**, including bounded response-body reads. That is not a
   guarantee of cancellation on the remote server.
4. The generator borrows `http` (the constructor default); reverse `using`
   disposal order disposes the generator first and the client afterward.
   A real application generally reuses an appropriately managed HTTP client.
5. The invented sign-in ticket is **state**, the question asks for a team, and
   `"support-team"` is its exact lookup ID. `account`, `billing`, `technical`
   are your application keys; their descriptions carry the meaning.
6. The inline source-generated `TicketJson` context serializes the typed string.
   `GenerateAsync` returns `DecisionResult`; `GetAnswer<ChoiceDecisionAnswer>`
   supplies typed consumption rather than parsing chat prose.

## What output would mean

**Illustrative only; no live Jev request was made during this sample phase:**

```text
Provider: typesafe
Model: jev-1.13.0
Selected ID: account
account: 0.7000
billing: 0.1000
technical: 0.2000
```

Actual provider/model metadata and values come from the generator/result, not
this illustration. Each number describes the model's distribution **among the
offered categories**. It is not measured routing accuracy, and vendor-specific
confidence evidence must not be compared as if it were another provider's
calibrated probability of correctness. Four-decimal display rounding can make
printed rows sum to slightly more/less than 1. No ticket is actually routed.

An HTTP error, cancellation, malformed/oversized response or unsupported model
is an exception, not a low-confidence answer. Preserve that distinction in
application logging and UX. Avoid logging raw diagnostic response bodies or
request state indiscriminately; they may contain sensitive information.

## Exercises, troubleshooting and boundaries

Before an **authorized** run, predict what "I was charged twice", "Help", and
"Tell me a joke" might do. Keep the ticket synthetic. Change one description
or state at a time, run and explain the entire distribution, not just whether
your preferred ID won. An out-of-domain question still has only your offered
alternatives. A manual-review rule belongs to the application and needs
evaluation data; there is no universal 0.8/0.9 threshold.

Missing-key errors require private configuration, not edits to the source.
401/403 requires account/key/model access checks. Network/TLS/deadline failures
require endpoint/connectivity checks, not disabling certificate validation or
turning on redirects. Service rejection of a model/question/size is explicit;
the sample does not silently select another model or truncate state.
For SDK selection/build issues see the [catalog](../../README.md#troubleshooting-shared-setup).

The contract has Binary, Choice and Score wire forms; the selected service/model
can reject unsupported inputs. The tutorial only asks Choice. Request state,
questions and candidates leave the process and reach the hosted service.
Synthetic data is intentional. Production privacy, budgets, access controls,
failure/retry policy and action authorization are your responsibilities.

This file was compiled and its help/missing-configuration paths exercised.
Existing recorded protocol tests are separate engineering evidence, **not live
service proof**. Next: [Ollaya](../Ollaya/README.md) for a separately managed
server, or [Julia](../Julia/README.md) for in-process inference.
