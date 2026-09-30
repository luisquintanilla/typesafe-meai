# Compare Julia and Laya: portable consumption, not a leaderboard

[Sample catalog](../../README.md) | [Beginner guide](../../../docs/learning/README.md)

## What this example teaches

[CompareProviders.cs](CompareProviders.cs) loads Julia, asks one Choice question,
prints the typed result and disposes Julia. **Only then** does it load Laya and
ask the identical question. The method `ShowDecisionAsync`, visible in the same
file, depends on `IDecisionGenerator`, not a concrete model. It is a small local
method, not a hidden shared runner or provider factory.

This demonstrates one reusable **application contract**, not equivalent model
architecture, probability calibration, accuracy, memory use or speed.
No credentials, Jev access, Ollaya server or Qwen artifact are required.
The question, state and candidate descriptions are the same short synthetic
support-routing example used by every individual quickstart.

## Prerequisites and model preparation

Use .NET 10 and this full repository. Complete the acquisition/hash checks in
[Julia](../../quickstarts/Julia/README.md#prepare-the-directory-once) and
[Laya](../../quickstarts/Laya/README.md#prepare-the-directory) first. Those pages
document exact pinned revisions, graph-declared sidecars, tokenizer/profile
files, license review and native prerequisites. No assets download implicitly.

You need both prepared directories:

```text
artifacts\decision-models\
  julia\     model.onnx, model.onnx.data, tokenizer.json
  laya\      laya.onnx, laya.onnx.data, laya_config.json,
             tokenizer\tokenizer.json
```

Julia-1 and English Laya run in-process through CPU ORT, with their respective
matching tokenizers/profiles. Sequential disposal avoids intentionally holding
both model sessions at once, but is not a guarantee about OS cache reclamation
or an exact peak-memory bound. Keep sufficient disk/RAM for each model.
This scenario was exercised on Windows x64; other native platforms are unverified.

## Run from the repository root

```powershell
dotnet run --file samples\scenarios\CompareProviders\CompareProviders.cs -- --help
dotnet run --file samples\scenarios\CompareProviders\CompareProviders.cs -- artifacts\decision-models\julia artifacts\decision-models\laya
```

First-run NuGet restore is separate from model acquisition. `#:project` references
the two unpublished provider projects; ORT CPU 1.23.2 is explicit. The scenario
does not load every provider transitively and does not use Native AOT.
Missing arguments/directories return exit 2; real load/inference exceptions
remain failures. The second model is not a fallback when the first fails.

## Read it in order

1. Validate both directories before doing work.
2. Define one typed string state and one question array. `"support-team"` is
   the retrieval key; `account`, `billing`, `technical` are opaque candidate IDs.
   Descriptions, rather than the IDs themselves, describe the team's purpose.
3. Create a cooperative two-minute token for the sequence. Synchronous loading
   cannot be interrupted by that token; it is **not a hard process timeout**.
4. Each `using` block owns one generator. The first block has ended before the
   second starts, so the first session is disposed before the second is loaded.
5. The visible helper receives only the interface plus the same state/questions.
   Inline `TicketJson` supplies source-generated serialization metadata.
6. Print provider, model, selected ID and full candidate distribution from a
   typed answer. No business action is taken.

Provider-neutral **consumption** does not mean provider-neutral **construction**:
Julia and Laya need different files and have different token limits. It also
does not mean that every provider supports every question kind.

## Actual side-by-side output

Captured on Windows x64 with the pinned retained assets; full model ID lines
are shown by the app and documented in the individual quickstarts:

| Provider | Selected ID | account | billing | technical |
|---|---|---:|---:|---:|
| julia | account | 1.0000 | 0.0000 | 0.0000 |
| laya | account | 0.7236 | 0.0762 | 0.2002 |

Both selected account **on this one input**. This is not evidence that they agree
in general, or that the sharper distribution is better. `F4` is a display choice:
1.0000 is not guaranteed certainty, 0.0000 need not be exact zero, and rounded
rows need not sum exactly to 1. Underlying provider precision and calibration
are separate questions. Learn [how to evaluate answers](../../../docs/learning/02-reading-answers.md)
before comparing model quality.

## Experiments and troubleshooting

Predict the results for "I was charged twice", an ambiguous "Help!", and an
out-of-domain "What is the weather?" before running. Compare distributions,
not just labels. Change a description without changing its ID; then add an
explicitly described review candidate. Why did the comparison itself change?
Neither a review candidate nor a confidence threshold guarantees safe abstention.
Use held-out representative labels to design an application review policy.

Use the individual [Julia](../../quickstarts/Julia/README.md#troubleshooting-and-limits)
and [Laya](../../quickstarts/Laya/README.md#experiments-and-troubleshooting)
pages to diagnose missing external weights, profile/tokenizer mismatch, native
architecture or context errors. Both reject over-budget inputs instead of
truncating. A request that fits Julia's budget might not fit Laya's; interface
compatibility does not erase that limit.

This scenario ran offline against existing assets; there were no new model
downloads or hosted requests. It is neither a benchmark nor a full evaluation.
Local inference does not make initial dependency acquisition offline.
For correctness machinery, move to the [engineering proof harness](../../../eng/DecisionProof/README.md)
only after understanding the [learning path](../../../docs/learning/README.md).
