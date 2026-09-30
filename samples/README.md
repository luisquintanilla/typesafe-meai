# Start here: decision models for .NET developers

You know C#, methods, interfaces and `using`. You do **not** need to know AI,
tensors or model training. These examples ask a model which support team should
handle a short, invented ticket, then read a typed answer. They never route a real
ticket. Read [the beginner guide](../docs/learning/README.md) alongside the code.

`DecisionInference` is this repository's **experimental, project-owned API**,
inspired by Microsoft.Extensions.AI patterns. It is not a built-in Microsoft
decision API, a replacement for `IChatClient`, or production certification.
The provider packages are not published. The examples reference their source
projects in this checkout; do not try `#:package DecisionInference.Julia`.

## Pick a starting point

| Example | What runs | What you need | Main lesson |
|---|---|---|---|
| [Julia](quickstarts/Julia/README.md) | Julia-1 ONNX in your process | Pinned graph, external weights and tokenizer | Load a disposable generator; consume a typed Choice |
| [Laya](quickstarts/Laya/README.md) | English Laya ONNX in your process | Pinned graph, weights, profile and tokenizer | Same application contract, different model/setup |
| [Jev](quickstarts/Jev/README.md) | TypeSafe hosted service | Your API key and network access | Own an HTTP client; remote inference with a deadline |
| [Ollaya](quickstarts/Ollaya/README.md) | A separately managed server | Explicit endpoint and supported installed model | Server is not model; discover a concrete capability |
| [Jeeves](quickstarts/Jeeves/README.md) | Separately managed Jeeves HTTP service | Explicit authorized endpoint | Start without thinking; preserve two-decimal observations |
| [Ollama](quickstarts/Ollama/README.md) | Separately managed Ollama 0.35.0+ | Explicit endpoint and compatible installed local model | Binary/Choice/Score in one request; residency is not a timeout |
| [Qwen](quickstarts/Qwen/README.md) | Qwen3 GGUF through TensorSharp CPU | Exact hash-pinned GGUF | Adapt next-token scores to Choice, with important limits |
| [Compare providers](scenarios/CompareProviders/README.md) | Julia then Laya | The two local directories only | Share consumption, not model setup or calibration |
| [Reasoning budget](scenarios/ReasoningBudget/README.md) | Two sequential Jeeves calls | The same authorized endpoint | Finite thinking budget, reported usage and distinct latency metrics |

Recommended reading: [problem and first call](../docs/learning/01-from-rules-to-decisions.md),
Julia, [answers and probabilities](../docs/learning/02-reading-answers.md), Laya,
then comparison. Read [deployment and evaluation](../docs/learning/03-running-responsibly.md)
before connecting real data. No example requires every provider to be installed.

## Common prerequisites and commands

Use a complete clone of this branch and a **.NET 10 SDK** compatible with
`global.json` (`dotnet --version` checks SDK selection). Run commands from the
**repository root**, not from an example directory. Commands here use PowerShell.

```powershell
dotnet run --file samples\quickstarts\Julia\Julia.cs -- --help
dotnet build samples\quickstarts\Julia\Julia.cs
```

Each is a real [file-based C# app](https://learn.microsoft.com/dotnet/core/sdk/file-based-apps),
not a dispatch into a larger sample project. `--file` is important: it prevents a
nearby project from consuming the filename as its own argument. After `--`, the
arguments go to the application. `--help` builds if necessary, but neither loads
a model nor sends an inference request. Missing required config exits with code 2.
The exception from a failed real load/request remains visible; it is not a
successful empty answer.

First build restores **NuGet code dependencies**, possibly using the network.
It does not download model weights. Each README gives a separate, explicit
acquisition step for the assets/services you choose. Review third-party
licenses before acquiring or redistributing models. No credentials or weights
are checked in. Retained asset folders belong under ignored `artifacts`.

`#:project` selects only that example's provider. Local ONNX examples explicitly
pin CPU `Microsoft.ML.OnnxRuntime` 1.23.2. HTTP examples do not import the ONNX,
tokenizer or TensorSharp stacks. The two new subtrees have scoped
`Directory.Packages.props` files because the root central pins are named for
`Decision*` projects; referenced libraries still use their own original pins.
File-app lockfile generation is disabled, and SDK output uses its per-file
temporary build cache. The original chat sample is outside this configuration.
`PublishAot=false` is explicit: these examples do not claim Native AOT support.
Nullable checking and the repository's warning rules still apply.

## What is intentionally not on the beginner path

[Engineering proofs](../eng/DecisionProof/README.md) live under `eng`, including
recorded transport fixtures, tensor exports, explicit stage parity and package
consumer checks. You do not need to read them to use a generator.
The [technical design](../docs/decision-generator-design.md) records architecture,
source pins, limitations and **historically scoped** proof results.

The unchanged [TypeSafe.MEAI.Sample](TypeSafe.MEAI.Sample/Program.cs) demonstrates
the earlier `IChatClient` projection, documented in the
[preserved chat-adapter guide](../docs/chat-adapter.md). It has a distinct
chat-oriented purpose; it is not the entry point for the decision-generator tutorials.

## Troubleshooting shared setup

If `--file` is unrecognized, inspect SDK selection before changing source.
If an unrelated ancestor `Directory.Build.targets` on your development machine
imports missing Arcade/build infrastructure, the verified local workaround is
`$env:ImportDirectoryBuildTargets='false'` in the **current PowerShell process**
before the command. Use it only for that machine-specific issue; do not edit
global/ancestor settings. Close that shell afterward to discard the override.
It is not a model setting.

Native runs during the earlier six-app teaching phase used Windows x64, SDK 10.0.401 and retained
Julia/Laya/Qwen assets. All six file apps were compiled; Jev/Ollaya were exercised
only through help/configuration failures, **not live inference**. A successful
synthetic call is evidence that the example runs, not that it is accurate on
your tickets. See each README for actual output and provider-specific limitations.

The later Jeeves quickstart and reasoning-budget scenario have offline
compilation/help/configuration evidence only, not live model calls. The
[reasoning lesson](../docs/learning/04-reasoning-budgets.md) explains why extra
tokens, diagnostic text and two observed answers do not prove improvement.

The later [Ollama quickstart](quickstarts/Ollama/README.md) likewise has offline
protocol/package and compiled help/configuration evidence only. No Ollama daemon
or weights were installed or invoked. Read the
[Ollama lesson](../docs/learning/05-ollama-decisions.md) before a separately
authorized live experiment; Ollama and Ollaya are distinct services.
