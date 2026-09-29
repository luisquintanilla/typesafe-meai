# Laya: the same C# question, another local decision model

[Sample catalog](../../README.md) | [Beginner guide](../../../docs/learning/README.md)

## What you will learn

[Laya.cs](Laya.cs) is a complete application, not a proof harness. It asks where
to route a synthetic support ticket and reads a typed Choice answer. If you ran
Julia first, notice how little the **consumption** code changes, while setup and
model limits remain provider-specific. You need ordinary C# knowledge, not ML.

**Model versus runtime:** the
[pinned Laya export card](https://huggingface.co/receptron/laya-onnx/blob/68f27dfe5a27a54fb2b1fefc432f43f972e90868/README.md)
describes an English FP32 ModernBERT-large encoder with Laya's decision head
(421M parameters). Those are upstream architecture facts, not quality claims.
The exported graph and weights are the **model**. CPU ONNX Runtime 1.23.2
executes it in your process. `LayaDecisionGenerator` adapts it to this project's
typed API, using Microsoft.ML.Tokenizers 2.0.0 for the matching standard BPE
tokenizer. The standalone tokenizer package does **not** introduce ML.NET's
training/DataView framework.

This is embedded inference, **not a call to Ollaya's Laya server layout**.
Compatibility is pinned to `receptron/laya-onnx` revision
`68f27dfe5a27a54fb2b1fefc432f43f972e90868`. The card declares Apache-2.0 for
weights and MIT for export code; review the actual licenses before use.

## Prepare the directory

Checklist: .NET 10 SDK, complete checkout, compatible native CPU ORT environment,
and disk space for about 1.69 GB of external weights plus graph/tokenizer/packages
and additional runtime memory. This sample was exercised on Windows x64.
Other platforms are not certified by that run.

```text
artifacts\decision-models\laya\
  laya.onnx
  laya.onnx.data
  laya_config.json
  tokenizer\
    tokenizer.json
```

After reviewing the [pinned files/license](https://huggingface.co/receptron/laya-onnx/tree/68f27dfe5a27a54fb2b1fefc432f43f972e90868),
you can explicitly acquire them from the repository root:

```powershell
$base = "https://huggingface.co/receptron/laya-onnx/resolve/68f27dfe5a27a54fb2b1fefc432f43f972e90868"
$dir = "artifacts\decision-models\laya"
New-Item -ItemType Directory -Force "$dir\tokenizer" | Out-Null
foreach ($file in "laya.onnx", "laya.onnx.data", "laya_config.json") {
    Invoke-WebRequest "$base/$file" -OutFile (Join-Path $dir $file)
}
Invoke-WebRequest "$base/tokenizer/tokenizer.json" -OutFile "$dir\tokenizer\tokenizer.json"
Get-FileHash "$dir\laya.onnx", "$dir\laya.onnx.data", "$dir\laya_config.json", "$dir\tokenizer\tokenizer.json" -Algorithm SHA256
```

Stop on a hash mismatch; do not substitute latest files:

| File | SHA256 |
|---|---|
| `laya.onnx` | `a874eb254b58b0fcb1e7ad56fbb188c29d64e08c9a46b689433e1f52c66dba1e` |
| `laya.onnx.data` | `487746363a8da57bcadb4345352997d22a0fb90d70aa22c6856668d023242aba` |
| `laya_config.json` | `5049005dc6ae3ca5e82cc7d85c421357d5c543817300c8e8c5281ddbc69bb561` |
| `tokenizer\tokenizer.json` | `6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30` |

The profile supplies the model's packing/temperature settings, not your business
policy. ORT resolves the pinned graph's `laya.onnx.data` reference. That sidecar
is mandatory **for this external-data graph**, not for every self-contained ONNX
graph. Missing referenced data throws; there is no download or fallback.
`tokenizer_config.json` is not a loader input.

## Run and read the code

From the repository root, after explicit asset preparation:

```powershell
dotnet run --file samples\quickstarts\Laya\Laya.cs -- --help
dotnet run --file samples\quickstarts\Laya\Laya.cs -- artifacts\decision-models\laya
```

First build may restore NuGet packages, not models. The file references the
local provider project and the pinned CPU native ORT package; packages for other
providers are not required. Native AOT publishing is disabled.

1. Validate the directory, then own the generator using `LoadFromDirectory`.
   Disposal deterministically releases the scorer/ORT session. Laya's tokenizer
   is managed and non-disposable (unlike Julia's native tokenizer). A real service would
   usually retain one loaded instance rather than reload per ticket.
2. `ticket` is the **state**. Its invented sign-in problem supplies evidence.
   The question text describes a task, while `"support-team"` is an exact key
   for retrieving its answer.
3. Candidates use IDs `account`, `billing`, `technical`. Descriptions, not the
   spelling of these opaque IDs, communicate the categories to the model.
4. `TicketJson` is source-generated JSON serialization metadata for a string,
   not AI training code. The typed-state overload calls the generator with that
   state and the explicit question list.
5. Await `GenerateAsync` with a 30-second cancellation token, retrieve the
   `ChoiceDecisionAnswer`, then print the whole distribution. The deadline
   starts after synchronous loading; it is not a load timeout.

For an explicit layout/session configuration, replace the loading line:

```csharp
using IDecisionGenerator generator = LayaDecisionGenerator.Load(new()
{
    ModelPath = @"artifacts\decision-models\laya\laya.onnx",
    TokenizerJsonPath = @"artifacts\decision-models\laya\tokenizer\tokenizer.json",
    ProfilePath = @"artifacts\decision-models\laya\laya_config.json",
    ConfigureSession = options => options.IntraOpNumThreads = 1
});
```

Omit `ConfigureSession` for default ORT settings. The thread count is only an
illustration. Paths normalize before that callback; temporary options are
disposed, not stored for later calls. Existing path/stage constructors remain
available for advanced composition.

## Annotated actual output

Captured on Windows x64 CPU with the pinned retained assets:

```text
Provider: laya
Model: receptron/laya-onnx@68f27dfe5a27a54fb2b1fefc432f43f972e90868
Selected ID: account
account: 0.7236
billing: 0.0762
technical: 0.2002
```

The selected ID is one of the supplied candidates. `0.7236` is the four-decimal
display of its **candidate-relative** probability, not a measured 72.36% chance
that your application will route correctly. Laya's separate action head is
provider-specific evidence, not an application authorization or a portable
accuracy score; this beginner app does not use it. Rounded display rows need
not sum exactly to 1; display rounding is not the underlying provider precision.

## Experiments and troubleshooting

Before running, predict the result for "I was charged twice", then for "Help!",
then for "Tell me a joke". Change only the state, observe the full distribution,
and explain whether the categories cover the input. Now overlap the account and
technical descriptions. Compare again without assuming the answer must flip.
If manual review is required, define it in application policy (or describe an
explicit review candidate); the model does not automatically abstain.

Missing files/profile errors are setup failures, not predictions. Missing
referenced weights must be repaired with the exact asset, never replaced by a
silent default. A tokenizer/model mismatch can change every token; use the
matching files. An ORT native-library error requires an appropriate CPU package
and process architecture, not an arbitrary model download. See the
[catalog troubleshooting](../../README.md#troubleshooting-shared-setup) for SDK
or unrelated ancestor build configuration.

This profile has a 512-token total budget, a bounded question head and a
49-token option budget. Framing/state/instructions all count; over-budget
input is rejected before inference, not truncated or rewritten. Shorten state
deliberately at the application boundary, preserving needed evidence.
Binary, Choice and ordinal Score are supported, but none guarantees correctness.
Cancellation requests ORT termination rather than a hard wall-clock stop.
Separate startup cost from inference when investigating first-call latency.

This exact synthetic example ran locally with retained assets; it is not a
new calibration/accuracy or portability study. No model was downloaded during
sample verification. Data stays in-process during inference; initial package/
model acquisition may use the network. Next: [compare Julia and Laya](../../scenarios/CompareProviders/README.md)
and [interpret probabilities](../../../docs/learning/02-reading-answers.md).
