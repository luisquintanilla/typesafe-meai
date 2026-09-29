# Decision-first inference for .NET

This branch explores **`DecisionInference`**, a project-owned API for asking
models typed questions through `IDecisionGenerator.GenerateAsync`. It keeps
decision-shaped work separate from chat-shaped work.

**Experimental and unofficial:** these are not built-in Microsoft.Extensions.AI
decision types, a Microsoft or model-vendor SDK, or production-certified
components. The provider packages are **unpublished**; examples reference their
source projects in this repository.

## Start here

Read the **[sample catalog](samples/README.md)** for runnable examples, or the
**[beginner learning path](docs/learning/README.md)** if you know C# but are new
to AI. Each quickstart has a complete file-based app and an adjacent README
covering the model, setup, code, output, exercises and troubleshooting.

From the root of a full checkout, with a **.NET 10 SDK compatible with
`global.json`**, try:

```powershell
dotnet run --file samples\quickstarts\Julia\Julia.cs -- --help
```

This builds the example if necessary and prints its arguments; it does **not**
load a model or run inference. First build may access NuGet to restore code
dependencies. Models are separate, are not bundled, and never download
automatically. Continue with the [Julia quickstart](samples/quickstarts/Julia/README.md)
for the exact pinned assets, license review, native-platform prerequisites and
the command that actually runs the model. The examples explicitly disable
Native AOT rather than claim unverified support.

## What a decision call means

Think of it as a typed method call with four responsibilities:

1. **State:** supply the information to assess, such as a synthetic support
   ticket, using ordinary typed data and explicit JSON serialization metadata.
2. **Question:** describe the task and its allowed answers. Binary asks about
   one proposition; Choice selects among described candidates; Score assesses
   ordered rubric levels.
3. **Typed answer:** await `GenerateAsync`, then read a `BinaryDecisionAnswer`,
   `ChoiceDecisionAnswer` or `ScoreDecisionAnswer` from the result. Exact IDs
   identify questions/candidates; descriptions communicate meaning to the model.
4. **Application policy:** your code decides whether to act, request review or
   decline. The samples only print predictions; a model does not authorize
   actions or replace deterministic permission checks.

Candidate-relative probabilities are not measured correctness or proof of
calibration. A model can confidently choose an unsuitable option. There is no
universal confidence threshold, and provider-specific evidence is not a shared
accuracy scale. [Learn to read the answers](docs/learning/02-reading-answers.md)
before using them in a workflow.

## Providers and examples

| Quickstart | What runs | Scope |
|---|---|---|
| [Julia](samples/quickstarts/Julia/README.md) | Julia-1 ONNX in-process with CPU ONNX Runtime | Binary, Choice and ordinal Score; strict model-specific budgets |
| [Laya](samples/quickstarts/Laya/README.md) | Pinned English Laya ONNX in-process | Binary, Choice and ordinal Score; matching tokenizer/profile |
| [Jev](samples/quickstarts/Jev/README.md) | TypeSafe hosted HTTP service | Native decision protocol; requires your authorized key and service access |
| [Ollaya](samples/quickstarts/Ollaya/README.md) | A separately managed server and concrete supported model | Audited NLI/GLiClass layouts, not every model the server can host |
| [Jeeves](samples/quickstarts/Jeeves/README.md) | A separately managed Jeeves HTTP service | Binary/Choice/Score with two-decimal observations; quickstart explicitly disables thinking |
| [Qwen](samples/quickstarts/Qwen/README.md) | Exact-hash Qwen3 GGUF through TensorSharp CPU | Choice-only answer-token adapter, not a specialized decision-head model |

The **[Julia/Laya comparison](samples/scenarios/CompareProviders/README.md)**
uses the same input and interface, loading and disposing the models sequentially.
It demonstrates portable consumption, not equivalent quality, calibration or
performance. Provider setup, capabilities and cancellation limits still differ.
The guides distinguish actual local runs from recorded or unperformed live
service evidence. HTTP sends supplied state off-process; use synthetic data
while learning.

For a provider-specific advanced exercise, the
[reasoning-budget scenario](samples/scenarios/ReasoningBudget/README.md) compares
no thinking with a small explicit cap. Read the
[reasoning lesson](docs/learning/04-reasoning-budgets.md) for usage, latency and
diagnostic-text limits. Jeeves has source-grounded offline integration evidence,
not a live GPU or calibration result.

## Relationship to Microsoft.Extensions.AI

The experiment follows capability-oriented conventions: a generator produces
results, exposes metadata/services and has an explicit resource lifetime.
It reuses MEAI primitives such as `UsageDetails` and
`AdditionalPropertiesDictionary`, alongside ordinary .NET types.
That does **not** make `IDecisionGenerator` an official MEAI interface.
The decision contract does not force questions into chat messages, tool calls
or a DTO-schema projection merely to obtain typed decisions.

## Builds and deeper references

The additive decision solution is separate from the original chat solution:

```powershell
dotnet build eng\DecisionComparison.slnx
```

The root `TypeSafe.MEAI.slnx` remains the **earlier chat-adapter solution**; it is
not the decision experiment's build entry point. For advanced architecture,
pins and historically scoped evidence, see the
[technical design](docs/decision-generator-design.md). Correctness tooling lives
under [`eng\DecisionProof`](eng/DecisionProof/README.md), not on the beginner
application path.

The earlier `IChatClient` experiment remains intact and useful for its distinct
chat/schema/tool integration purpose. Its [preserved guide](docs/chat-adapter.md)
documents that API, original sample, constraints and verification commands;
the two legacy chat packages continue embedding that guide as their README.
