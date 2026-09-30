# Reasoning is an optional budget, not a correctness guarantee

[Learning path](README.md) | [Jeeves quickstart](../../samples/quickstarts/Jeeves/README.md)

You already know a method can do extra work before returning a value.
Jeeves offers a model-specific version: generate some intermediate tokens,
then score the allowed answers with a learned decision head. A **token** is
a model's unit of text, not necessarily a word or a C# character.
The result is still a typed Binary, Choice or ordinal Score answer.

Do not confuse this with asking your application to run an agent loop.
There are no tool calls, plans for executing actions, extra input types or
new common reasoning API here. The generator makes one HTTP decision request
to a separately managed service. Application policy still decides what to do.

## Start with the boring baseline

The [Jeeves quickstart](../../samples/quickstarts/Jeeves/README.md) explicitly
sets `Think=false` and prints a full Choice distribution for an invented
account-access ticket. This makes the baseline visible rather than accidentally
accepting server defaults that normally enable thinking.

The [reasoning-budget scenario](../../samples/scenarios/ReasoningBudget/README.md)
then tries the same input with no thinking and with a cap of 128 tokens.
Predict whether the selected team will change before running it on an
authorized service. Either outcome is an observation, not an accuracy result.

## Understand the four controls

| Typed property | Wire field | What it controls |
|---|---|---|
| `Think` | `think` | Whether the engine may use a thinking budget |
| `MaxThink` | `max_think` | Nonnegative maximum generated chain tokens per question |
| `NoThinkThreshold` | `nothink_threshold` | Optional skip-thinking gate based on maximum no-think probability |
| `ReturnReasoning` | `return_reasoning` | Whether sensitive diagnostic text is returned, not whether thinking happens |

An explicit `JeevesGenerationOptions` sends a **complete snapshot**. Its defaults
are `true`, `2560`, `null`, `false`, matching the pinned upstream option type.
A null threshold is sent as JSON null and clears a server threshold. To use
server defaults instead, pass null or ordinary `DecisionGenerationOptions`;
that omits the wire options object. Plain `ModelId` still works, but is only a
request/echo label, not a control for selecting loaded weights.

The default maximum is not a promise to generate that many tokens. Remaining
context can reduce the cap; `Think=false` or `MaxThink=0` gives no generated
chain. A cap of 128 is small relative to the default but still not a time limit.
Long state can be rejected. The client does not truncate it to force success.

## Thresholds are not portable confidence

For three candidates, suppose the unrounded no-think maximum is 0.46.
Jeeves's native Choice confidence formula subtracts the uniform baseline
and rescales: `(0.46 - 1/3) / (1 - 1/3)`, reported as 0.19 after rounding.
Its **skip-thinking threshold uses 0.46, not 0.19**. Thinking may subsequently
produce a different final distribution, so returned confidence cannot tell
you which no-think value was used by that gate.

These quantities answer different questions. Neither is a measured probability
that your application is correct, and neither can be compared indiscriminately
with another provider's confidence. Upstream fitted temperature and published
evaluation results do not establish calibration on your data. A strong number
can still accompany an unsuitable forced choice.

## Read counters and text separately

Jeeves reports answer-JSON and reasoning tokens separately. MEAI defines
reasoning as part of output tokens, so this adapter deliberately maps:

```text
InputTokenCount     = input_tokens
ReasoningTokenCount = reasoning_tokens
OutputTokenCount    = output_tokens + reasoning_tokens
TotalTokenCount     = input_tokens + output_tokens + reasoning_tokens
```

The original answer-JSON count remains in the documented
`jeeves.answerJsonTokens` additional-count entry and the owned raw envelope.
All counts are checked nonnegative integers with checked addition. They are
server-reported bookkeeping, not portable cost or workload measurements.
See the [MEAI reasoning-count remarks](https://learn.microsoft.com/dotnet/api/microsoft.extensions.ai.usagedetails.reasoningtokencount#remarks).

You can receive counts without receiving text. The default scenario does
exactly that. Explicit `--show-reasoning` opts into the per-question diagnostic
snapshot:

| Observation | What it means, and does not mean |
|---|---|
| `Thought=false`, zero tokens, empty text, `Closed=false` | No generated chain; not proof that no inference work occurred |
| `Thought=true`, positive tokens, `Closed=false` | The chain did not reach its closing token; still a valid decision at the cap |
| `Closed=true` | A closing token occurred; not proof the answer is correct |
| One token and empty text | Can be the closing token alone; tokens include it but text omits it |

Text may reproduce private input, be misleading, or fail to explain the actual
cause of a score. It is not a guaranteed faithful explanation and is not an
instruction for your application to execute. Do not automatically log it.
The adapter rejects missing, extra or inconsistent reasoning when opted in,
and unsolicited reasoning when not explicitly requested. If a custom server
enables text by default, use explicit typed options to turn it off.

## Measure the right thing

`latency_ms` measures synchronized engine work inside the pinned server's
lock. It excludes queue/lock wait and response serialization/tokenization.
The scenario's client stopwatch includes the whole `GenerateAsync` wait,
including network and decoding. Neither is a hardware-independent benchmark.

HTTP cancellation stops awaiting network work; the server may continue GPU
work. Use an explicit client deadline as well as a finite thinking budget.
Do not start extra attempts automatically because the original call timed out.

Two runs on one synthetic ticket cannot show improvement, general latency or
calibration. A useful evaluation needs held-out labels, representative inputs,
fixed deployment/settings, repeated measurements and explicit costs for
mistakes. Before adding a policy, revisit
[deployment and evaluation](03-running-responsibly.md).

The current Jeeves integration evidence is source-grounded and offline.
Examples compile and their help/configuration paths run without a service;
no GPU deployment, weights, live result or calibration claim is implied.
