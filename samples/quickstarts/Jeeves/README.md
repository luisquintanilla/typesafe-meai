# Jeeves: a typed decision without a reasoning chain

[Sample catalog](../../README.md) | [Beginner guide](../../../docs/learning/README.md)

## What you will learn

[Jeeves.cs](Jeeves.cs) sends the same invented sign-in ticket and three support
teams as the other quickstarts. It explicitly requests **`Think = false`**.
You get a complete typed Choice distribution without first learning how to
inspect generated reasoning. This is an experimental HTTP adapter, not a new
model runtime or an official Microsoft/PostHog SDK.

Jeeves is a Qwen3.5-9B-derived decision model with merged LoRA training changes
and a **pointer head**: a learned component that scores the offered alternatives.
It can generate a reasoning chain before scoring them. The upstream server
uses a custom Python/CUDA engine and a speculative drafter; it is **not** the
small Qwen3 GGUF/TensorSharp example, an ONNX model loaded by this library, or a
general chat/agent framework. Optional reasoning does not change the portable
Binary/Choice/Score contract.

Our protocol reference is
[PostHog/jeeves at f04ec55](https://github.com/PostHog/jeeves/tree/f04ec5567301450dcaae0210dd54deeb4f647f87).
The [code license is MIT](https://github.com/PostHog/jeeves/blob/f04ec5567301450dcaae0210dd54deeb4f647f87/LICENSE);
the [released model card](https://huggingface.co/PostHog/jeeves) declares the
Qwen3.5-derived weights Apache-2.0. Review both and record your actual deployed
weight/drafter revisions. A source-protocol pin is not proof of a deployed
checkpoint, its license compliance, hardware compatibility or calibration.

## Prepare a service deliberately

You need a full checkout, a .NET 10 SDK compatible with `global.json`, and an
**already running, independently authorized** compatible service. The .NET
client needs no CUDA, ONNX Runtime, tokenizer or weights. First build may
restore NuGet code dependencies; provider packages are not published.
Native AOT is explicitly disabled.

The server owner must separately follow the
[pinned upstream setup](https://github.com/PostHog/jeeves/blob/f04ec5567301450dcaae0210dd54deeb4f647f87/README.md#quickstart):
Python 3.12, supported CUDA GPU/software, model and drafter files, and intentional
server configuration. Upstream documents Hopper for its FP8 path and a
`--no-fp8` bf16 path. This guide does not assert a tested VRAM minimum or that
your GPU is supported. The client neither installs nor starts that service,
pulls weights, nor checks its hardware.

The pinned service is **unauthenticated** and defaults to
`http://127.0.0.1:8009`. Keep it on an appropriately controlled network; do not
expose it publicly as-is. This adapter does not invent an API key or pick up
ambient `Authorization`. The origin must be HTTPS, or HTTP with a literal
loopback IP (`127.0.0.1` or `[::1]`); `http://localhost`, remote HTTP and origins
with credentials, paths, queries or fragments are rejected.

## Run the client

From the repository root:

```powershell
dotnet run --file samples\quickstarts\Jeeves\Jeeves.cs -- --help
```

Help builds if necessary but sends nothing. Missing/invalid configuration
returns exit code 2 before networking. Only after authorizing a real request:

```powershell
dotnet run --file samples\quickstarts\Jeeves\Jeeves.cs -- http://127.0.0.1:8009
```

The tutorial requires the origin explicitly, despite the provider's matching
default, so the destination is visible. Every successful non-help run sends
your state to that server; loopback is still outside the C# process.

## Follow the C# call

The app owns `HttpClient`, disables cookies/redirects, and lends it to a
disposable `JeevesDecisionGenerator`. A 30-second cancellation token covers
the request and body reads. Cancellation ends the client wait; it does not
guarantee cancellation of the server's GPU operation.

The state is a string serialized with explicit source-generated JSON metadata.
`"support-team"` identifies the question; `"account"`, `"billing"` and
`"technical"` identify answers in C#. Descriptions communicate their meaning.
The adapter maps IDs to ordinal wire aliases and maps results back exactly.
Because upstream includes option keys in its model prompt, this deliberately
does not claim byte-identical prompting to a direct Python call with semantic
keys such as `"account"`.

This **excerpt from the complete app** makes the no-thinking intent visible:

```csharp
var options = new JeevesGenerationOptions { Think = false };
DecisionResult result = await generator.GenerateAsync(ticket, questions,
    TicketJson.Default.String, options, deadline.Token);
ChoiceDecisionAnswer answer = result.GetAnswer<ChoiceDecisionAnswer>("support-team");
```

A typed `JeevesGenerationOptions` sends all four fields: `think=false`,
`max_think=2560`, `nothink_threshold=null`, `return_reasoning=false` here.
`Think=false` disables reasoning even though a cap is present. A null threshold
explicitly clears a server threshold; it is not an omitted setting.
Passing **null or plain `DecisionGenerationOptions` instead omits `options`**
and accepts server defaults, which normally enable thinking. The two choices
are not interchangeable. `Clone()` retains the derived settings without
mutating your object. No magic `AdditionalProperties` input keys are supported.

`ModelId` is a **request/echo label**. The pinned server does not switch loaded
weights when you change it; the response must echo it exactly. The adapter
posts to `/v1/systemone` without automatic discovery. Upstream's `/v1/models`
listing is available to an administrator, not used here as weight attestation.

## Read output without inventing accuracy

**Illustrative shape only, not captured live output:**

```text
Provider: jeeves
Model label (not verified weights): jeeves-latest
Selected ID: account
account: 0.72
billing: 0.10
technical: 0.18
```

All candidates are present, but those numbers are not measured correctness.
They are independently rounded two-decimal observations. Valid rows can sum
to 0.99 or 1.01 when their rounding intervals contain a normalized distribution;
the adapter preserves observations rather than repairing them.
The selected ID must belong to the offered set. No ticket is routed.

Binary and ordinal Score are also supported, with the existing System One
subset: Choice 1..255 candidates and Score 2..10 rubric levels. Upstream's
wider Score limit does not silently widen this adapter. Native confidence is
provider-specific evidence, not a universal score for choosing an action.
See [reading answers](../../../docs/learning/02-reading-answers.md).

## Exercises, evidence and troubleshooting

Before an authorized run, predict the selected team. Then try an invented
billing ticket and an out-of-scope message such as "Tell me a joke". A forced
choice is not permission to act on either. Keep deterministic permissions and
manual-review policy in application code, evaluated against representative
labels rather than an invented confidence threshold.

Connection refused means you need the server owner, not a hidden C# daemon
manager. A 422 response can indicate server input/context validation; state is
not silently truncated by this adapter. A model-label, precision, correlation
or usage error is an explicit protocol failure, not an empty successful answer.
Response bodies are bounded and successful JSON must be strict UTF-8 without
duplicate properties. HTTP exceptions preserve status/request ID; the explicit
diagnostic body may contain sensitive data and is never logged automatically.
See [shared SDK troubleshooting](../../README.md#troubleshooting-shared-setup).

This integration is checked with source-grounded **offline fixtures** and
compiled help/configuration paths, not a deployed GPU service. No Jeeves weights
were acquired and no server inference was run for it. Next, study the
[reasoning lesson](../../../docs/learning/04-reasoning-budgets.md) and
[finite-budget scenario](../../scenarios/ReasoningBudget/README.md).
