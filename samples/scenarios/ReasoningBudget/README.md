# Compare no thinking with a small reasoning budget

[Sample catalog](../../README.md) | [Reasoning lesson](../../../docs/learning/04-reasoning-budgets.md)

## Purpose and prerequisites

[ReasoningBudget.cs](ReasoningBudget.cs) makes **two sequential Jeeves HTTP
calls** for one synthetic account-access ticket: first no thinking, then
thinking with a cap of **128 generated tokens per question**. The requested
model label, state, question and candidates stay the same; keeping actual
weights/configuration fixed is the server owner's responsibility. This teaches
options and accounting, not a one-example quality or speed comparison.

Read the [Jeeves quickstart](../../quickstarts/Jeeves/README.md) for the pinned
protocol, licenses, .NET 10 setup, separately managed Python/CUDA service,
endpoint restrictions and privacy boundary. No server, dependencies for that
server, or model weights are provisioned by this file app. Its sole provider
reference is the HTTP-only `DecisionInference.Jeeves` source project. NuGet code
restore may need a network connection; building/help does not run a model.

## Commands and safety

From repository root, inspect help without networking:

```powershell
dotnet run --file samples\scenarios\ReasoningBudget\ReasoningBudget.cs -- --help
```

Only with an authorized, already running endpoint:

```powershell
dotnet run --file samples\scenarios\ReasoningBudget\ReasoningBudget.cs -- http://127.0.0.1:8009
```

Raw reasoning text is **off by default**. To deliberately request and print it
for synthetic, non-sensitive input:

```powershell
dotnet run --file samples\scenarios\ReasoningBudget\ReasoningBudget.cs -- http://127.0.0.1:8009 --show-reasoning
```

That flag opts in on both calls; the no-think entry will have an empty chain.
Treat generated text as untrusted diagnostic data that can expose input or
contain plausible but incorrect statements. Do not copy it into general
production logs. There is no automatic result/metadata dump.

Both calls share one explicit two-minute deadline. The token cap is **not** a
latency guarantee. Timeout cancels client network waiting, not necessarily GPU
work. Invalid arguments exit with code 2 before networking. Real failures
remain exceptions; no local synthetic fallback replaces a failed service.

## Follow the experiment

The complete app creates one disposable HTTP-backed generator and the same
`"support-team"` Choice question as the quickstarts. Each iteration constructs
an independent options snapshot. This **excerpt** shows the deliberate controls:

```csharp
var options = new JeevesGenerationOptions
{
    Think = think, MaxThink = thinkingBudget, NoThinkThreshold = null, ReturnReasoning = showReasoning
};
```

`thinkingBudget` is 128, not an unbounded server default. `NoThinkThreshold=null`
clears any server threshold so that this is not accidentally a threshold-gated
comparison. The server may reduce the cap to fit remaining context. It rejects
overlong input rather than promising that every request gets 128 useful tokens.
Reaching the cap still produces a decision; it does not mean a chain completed.

The app prints selected ID and every candidate's two-decimal observation for
each call. It also prints counters and two deliberately different timers:

| Field | Meaning |
|---|---|
| Reported input tokens | Upstream prompt accounting, including its shared-prefix rules |
| Answer-JSON tokens | Upstream `output_tokens`, tokenization of the serialized answers, **excluding reasoning** |
| Reasoning tokens | Upstream chain-token count, including a closing token when generated |
| MEAI output / total | Output = answer JSON + reasoning; total = input + output |
| Server inference ms | Upstream `latency_ms`, measured inside the server lock around synchronized engine work |
| Whole GenerateAsync elapsed ms | Local stopwatch covering the complete awaited client call, including transport/response decoding |

The server metric excludes lock waiting and response serialization/tokenization.
Neither timer isolates GPU arithmetic or proves a speed advantage. The
answer-JSON count is retained under
`UsageDetails.AdditionalCounts[JeevesDecisionGenerator.AnswerJsonTokenCountKey]`.
These are reported accounting units, not invoices, compute-cost measurements,
or estimates reconstructed by the C# client.

When text is requested, `JeevesResultMetadata.Reasoning` is an immutable owned
dictionary keyed by **caller question IDs**, not wire aliases. `Thought` says
the engine used a positive thinking cap; `Closed` says the chain reached its
closing token. `Tokens` includes that token, while `Text` excludes it: one
closing token can legitimately mean `Thought=true`, `Closed=true`, `Tokens=1`
and empty text. An unclosed positive-length chain is valid at a budget limit.
The adapter checks per-question/aggregate counts and requested budgets.
Metadata survives disposal of HTTP/JSON resources; no borrowed JSON lifetime
leaks into application code.

## Predict, observe, decide

Before running, predict whether either answer or distribution will change.
Afterward ask which printed fields actually support your explanation. If the
answer stays the same, thinking was not proven useless; if it changes, it was
not proven better. Increasing a budget does not guarantee improvement.

For a later, separately authorized evaluation, use representative labeled
examples, repeated measurements, fixed server/weight/drafter revisions and
recorded settings. Vary one control at a time; reserve examples not used to
choose a policy. Do not pick a production threshold or claim calibration from
the two observations in this example. The upstream threshold compares maximum
**no-think probability**, not returned normalized Choice confidence; the
[lesson](../../../docs/learning/04-reasoning-budgets.md#thresholds-are-not-portable-confidence)
works through the difference.

This app has offline compilation/help/configuration evidence only; no live
Jeeves timings, answers, calibration or hardware claims are supplied here.
Read the [deployment lesson](../../../docs/learning/03-running-responsibly.md)
before making this part of a real workflow.
