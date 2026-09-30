# Ollama decisions: reuse a server, keep the questions typed

Think of the server as a process implementing a method, not as the method's
return type. Your C# app supplies data and questions; an already managed Ollama
runtime owns the model and scoring. `IDecisionGenerator` returns typed answers
without converting them to chat messages. It remains an experimental interface
owned by this project, not a built-in Microsoft.Extensions.AI API.

Start with the [Ollama quickstart](../../samples/quickstarts/Ollama/README.md).
Its help and `--check-config` commands run without a daemon. A real request
requires separate authorization, Ollama **0.35.0+**, a supported installed local
GGUF model and a scoring-capable runner. This lesson has **offline evidence
only**; it does not install software, fetch weights or demonstrate model quality.

## Three questions about one ticket

The invented ticket is "I cannot sign in after resetting my password."
Binary asks whether it concerns account access, Choice chooses a support team,
and Score assesses ordered urgency levels. Before running a model, explain why
"cannot sign in" supports the account-access description but does not establish
a widespread outage. Instructions and descriptions communicate this distinction;
opaque IDs are just exact lookup keys in your program.

The complete program prints all candidate probabilities. It never routes a
ticket or grants access. A high P(true) is not authorization, a concentrated
Choice distribution is not proof of correctness, and an urgency expected index
of 1.4 would be between rubric levels, not "140% urgent." See
[reading answers](02-reading-answers.md) for the arithmetic.

## One request is not one independent question

Ollama presents each requested question with the **full state and question
schema**. It does not feed the answer to question one into question two.
Nevertheless, changing the schema can change the conditioning of all questions.
Adding a question is not guaranteed to leave the other predictions unchanged.

Our adapter replaces external keys with ordered wire aliases (`q0`, `q1`, and
numeric candidate keys) and restores exact caller IDs afterward. It always
sends meaningful candidate descriptions. Those aliases occur in the rendered
schema, so a direct SDK request with semantic keys is not byte-identical prompt
input. This matters for fair comparisons, even when the ticket text matches.

The runtime can reuse prefixes and score rows in order. **One HTTP request is
not a guarantee of one native batched forward**, nor a measured latency gain.
Compare the same task and report runtime conditions before making performance
claims; this integration does not report any benchmark.

## Similar names, different responsibilities

| Provider | What selects execution | Important distinction |
|---|---|---|
| Ollama, normally port 11434 | Requested local model and Ollama scoring runner | No discovery/pull; compatible trained models, not every model in the catalog |
| Ollaya, normally port 11435 | Audited ONNX model/profile selected on its own service | Four-decimal observations and model-layout discovery |
| Jeeves, normally port 8009 | Weights already loaded by its separate service | Request model label alone does not select weights; optional reasoning, two-decimal observations |

Ollama's existing runtime already supplies scoring; the .NET adapter does not
need a Python/Qevi port or new media contract. Objects/arrays are JSON text state,
not images or chat messages. The server's echoed model label still is not a
verified digest. Server version, weight provenance and actual runtime
compatibility remain the runtime owner's responsibility.

## Two different budgets

The exact serialized request can be at most **65,536 UTF-8 bytes**, including
JSON escaping, model selection and `keep_alive`. Separately, each rendered
prompt must fit the server's token context with two positions left for scoring.
Bytes and characters are not token estimates. Neither adapter nor server
silently truncates over-budget input.

The endpoint supports 1..64 questions and 2..26 Choice/Score alternatives.
The documented Tev training range is 2..24 alternatives: protocol acceptance
does not establish quality outside the training range. Every question needs
nonblank string instructions; string state must also be nonblank, while empty
objects and arrays are valid state.

`OllamaGenerationOptions.KeepAliveSeconds` controls how long the model stays
resident **after** a request. Null inherits the server setting, zero unloads,
negative retains indefinitely, and positive means seconds. The sample uses
60, not an indefinite default. This is separate from the app's 30-second
HTTP cancellation deadline. Canceling the wait does not prove remote work stopped.

## Numbers tell you what was observed, not whether it was right

Ollama's pinned answer code applies stable float64 softmax and ordinary JSON
encoding. `HighPrecision` preserves those numbers without forced decimal
rounding or renormalization. Rounded documentation examples are not protocol
fixtures. Native confidence is normalized entropy concentration
`1 - H(p)/ln(N)`, not the chance your answer is correct.

Input tokens sum full rendered prompts across questions, including shared
prefixes even when cached. Output tokens are internal scoring/primer/retry
tokens, not JSON response length; the count can exceed the question count.
The existing MEAI `UsageDetails` maps these to input/output and their checked
sum. Reasoning and server latency are not reported. Unknown is not zero, and
these usage counts are not a bill or a compute-time measurement.

Keep raw diagnostics opt-in: they may contain supplied data. A 400, 404, 413 or
500 is an explicit failure, not a synthetic successful answer. Only HTTPS or
literal-loopback HTTP origins are accepted, and samples disable redirects and
cookies. Review [responsible deployment](03-running-responsibly.md) before real data.

## Predict, observe, decide

In a later authorized experiment, change only one team description and predict
which probabilities might move. Then add or remove an urgency question and
remember that the full schema changed. Record the actual server version/model
digest and do not turn one ticket into an accuracy, calibration or speed claim.
Finally, design an application policy using labeled held-out examples, not a
universal confidence threshold.

The [quickstart's source references](../../samples/quickstarts/Ollama/README.md#troubleshooting-and-sources)
pin v0.35.0 at `cc4069396f3ad2c370c53eed2e4a42ac13adab84`.
The [advanced design](../decision-generator-design.md#ollama-current-offline-http-integration)
separates the narrow new adapter from preserved historical evidence.
