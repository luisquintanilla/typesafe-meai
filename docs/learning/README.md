# Decision models, one C# concept at a time

This guide is for .NET developers who have never worked with AI. Start with
methods, data and resource lifetimes you already know; introduce model math
only when it helps explain an observable result.

The `DecisionInference` types are **experimental APIs owned by this project**,
inspired by Microsoft.Extensions.AI. They are not shipped Microsoft decision
types and are not production-certified. You will run inference, not train a
model, and your first example will print a prediction rather than take action.

## Reading and doing

1. [From C# rules to a learned decision](01-from-rules-to-decisions.md):
   choose an appropriate problem, understand state/questions/candidates, then
   follow one method call across the model/application boundary.
2. Run the [Julia quickstart](../../samples/quickstarts/Julia/README.md), or pick
   another setup from the [sample catalog](../../samples/README.md).
   Predict an answer first, run, and explain what the result does not prove.
3. [Read Binary, Choice and Score answers](02-reading-answers.md):
   worked distributions, opaque IDs, token limits, rounding, optional logits
   and softmax, and why "confidence" needs a precise definition.
4. [Deploy and evaluate responsibly](03-running-responsibly.md):
   files versus services, native lifetimes/cancellation, provider selection,
   held-out labels, manual review and changing real-world data.
5. Run [Julia and Laya side by side](../../samples/scenarios/CompareProviders/README.md).
   Reuse an interface without claiming equivalent model quality.

Each lesson defines terms before using them. The short [glossary](03-running-responsibly.md#glossary)
is for revisiting concepts, not a prerequisite memorization task.
Every sample has an adjacent walkthrough, explicit setup, exercises and
troubleshooting. Use synthetic data first; HTTP examples send state off-process.

## Advanced material is separate

The [technical design](../decision-generator-design.md) explains domain rules,
architecture, exact dependency/reference pins and historically scoped evidence.
The [engineering proof harness](../../eng/DecisionProof/README.md) contains
stage parity, recorded protocol cases and tensor export. Those are maintenance
tools, not required application plumbing. The original chat sample remains
unchanged and is distinguished in the [catalog](../../samples/README.md).

No lesson claims an accuracy score, universal threshold or model leaderboard.
Actual captured quickstart outputs are labeled with their local provenance;
hosted examples use explicitly illustrative output where no live call was made.
Static SVG diagrams have text equivalents in the lessons and need no scripts,
external fonts, model downloads or active content.
