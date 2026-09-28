# Load a local decision generator

This experimental .NET 10 sample consumes **existing local model files**. It never
downloads or discovers models. Choose one provider; there is no fallback.
The executable supplies the native CPU ONNX Runtime dependency.

## Start with a complete generator

```csharp
using DecisionInference;
using DecisionInference.Julia;
using System.Text.Json.Serialization;

using IDecisionGenerator generator =
    JuliaDecisionGenerator.LoadFromDirectory(@"artifacts\decision-models\julia");

DecisionQuestion[] questions =
[
    new ChoiceDecisionQuestion("team", "Which team should handle this?",
        [new("account", "Account support"), new("billing", "Billing")])
];
DecisionResult result = await generator.GenerateAsync(
    "The customer asks to reset their password.", questions,
    StateJson.Default.String, cancellationToken: CancellationToken.None);
ChoiceDecisionAnswer answer = result.GetAnswer<ChoiceDecisionAnswer>("team");
Console.WriteLine(answer.SelectedCandidateId);

[JsonSerializable(typeof(string))]
internal partial class StateJson : JsonSerializerContext;
```

For Laya, use `using DecisionInference.Laya;` and replace only construction:

```csharp
using IDecisionGenerator generator =
    LayaDecisionGenerator.LoadFromDirectory(@"artifacts\decision-models\laya");
```

Both factories are synchronous and return a generator owning its session and
provider resources. Use `using`. You do not have to construct or retain stages.
The runnable [LocalGenerationExample.cs](LocalGenerationExample.cs) contains both
providers with identical source-generated state and typed answer consumption:

```powershell
dotnet run --project samples\DecisionComparison -c Release -- generate julia artifacts\decision-models\julia
dotnet run --project samples\DecisionComparison -c Release -- generate laya artifacts\decision-models\laya
```

## Exact directory layout

| Provider | Files bound by `LoadFromDirectory` |
|---|---|
| Julia | `model.onnx`, `tokenizer.json` |
| Laya | `laya.onnx`, `tokenizer\tokenizer.json`, `laya_config.json` |

These names come from the pinned exports used by this repository. Their graphs
also require `model.onnx.data` (Julia) or `laya.onnx.data` (Laya) alongside the
graph. ORT resolves the graph's declared external data; those sidecars are **not**
universal requirements for self-contained ONNX graphs. A missing referenced
sidecar fails during native loading, without substitution or retry.
`tokenizer_config.json`, parity files and Julia's Python config are not consumed.
No scanning, alternate filenames, tokenizer autodetection or downloads occur.
Use the explicit paths below for a different layout.

## Configure native sessions with typed construction options

Only construction changes; generation and answer consumption remain the same.
The two snippets assume their provider's namespace is imported.

```csharp
using var generator = JuliaDecisionGenerator.Load(new()
{
    ModelPath = @"artifacts\decision-models\julia\model.onnx",
    TokenizerJsonPath = @"artifacts\decision-models\julia\tokenizer.json",
    ConfigureSession = options =>
    {
        options.IntraOpNumThreads = 2;
        options.InterOpNumThreads = 1;
    }
});
```

```csharp
using var generator = LayaDecisionGenerator.Load(new()
{
    ModelPath = @"artifacts\decision-models\laya\laya.onnx",
    TokenizerJsonPath = @"artifacts\decision-models\laya\tokenizer\tokenizer.json",
    ProfilePath = @"artifacts\decision-models\laya\laya_config.json",
    ConfigureSession = options =>
    {
        options.IntraOpNumThreads = 2;
        options.InterOpNumThreads = 1;
    }
});
```

`JuliaDecisionGeneratorOptions` / `LayaDecisionGeneratorOptions` have required,
init-only paths and an optional init-only callback. These are **construction**
options, distinct from per-call `DecisionGenerationOptions`.
`LoadFromDirectory(directory, options => { options.IntraOpNumThreads = 2; })`
also supports session configuration without spelling out the files. Omit the
callback to preserve ORT defaults. Thread counts above illustrate configuration,
not recommended performance settings.

The loader validates nonblank paths and required files and resolves them to
absolute paths **before** calling configuration. It creates temporary native
`SessionOptions`, invokes the callback exactly once, then uses the existing
owning path constructor. Temporary options are disposed on success or failure;
neither the config object nor the callback is retained for generation. The
callback must not retain or dispose the options it receives. Callback exceptions
propagate unchanged. Invalid content, graph metadata or referenced sidecars fail
explicitly through the existing tokenizer/profile/ORT validation and cleanup.
Existence checks do not lock files or prevent concurrent file replacement.

```powershell
dotnet run --project samples\DecisionComparison -c Release -- generate julia artifacts\decision-models\julia configured
dotnet run --project samples\DecisionComparison -c Release -- generate laya artifacts\decision-models\laya configured
```

On the development machine with unrelated ancestor Arcade imports, prefix each
command process with `$env:ImportDirectoryBuildTargets='false'`; do not edit
ancestor build files or global settings.

## Advanced composition and proof commands

Existing explicit path constructors and stage constructors remain supported
with unchanged signatures/defaults. Path constructors borrow caller-supplied
`SessionOptions`; the caller still disposes those options. Stage constructors
borrow by default (`ownsStages: false` for Julia, `ownsScorer: false` for Laya).
Public preparation/scoring/decoding stages are useful for custom composition,
not required for the ordinary path above.

The host's `stages ...` and `provider julia ... [receipt.json]` commands are
**advanced parity/proof harnesses**: they intentionally expose tensor export,
raw evidence, cancellation and lifecycle checks. They are not beginner loading
examples. These harnesses are retained separately from `generate`.
See the [design and evidence](../../docs/decision-generator-design.md) for pinned
revisions, exact limits and proof commands. HTTP and Qwen construction is
unchanged; directory loaders are only for Julia and Laya.
