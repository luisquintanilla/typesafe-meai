# Julia: your first local typed decision

[Sample catalog](../../README.md) | [Beginner guide](../../../docs/learning/README.md)

## What you will learn

Read [Julia.cs](Julia.cs) from top to bottom. A normal C# string becomes state,
three described candidates become a Choice question, and an awaited method
returns `ChoiceDecisionAnswer`. `using` owns the model lifetime. You do not
assemble tensors or parse generated prose. No prior AI experience is assumed.

**What runs where:** Julia-1 is a trained decision model that scores supplied
options, not a chat bot. `JuliaDecisionGenerator` is this project's adapter.
ONNX is the graph/weight interchange format; **ONNX Runtime (ORT)** executes it
on CPU inside this process. Julia's matching tokenizer converts text into the
model's token IDs through the pinned Hugging Face native binding. Loading does
not train or update the weights.

Compatibility is tied to
[Julia-1-ONNX revision 82a2fad](https://huggingface.co/SupersonicLabs/Julia-1-ONNX/tree/82a2fadf8fccfccdc5fd4e1009ba8f1a265eb7a8)
and the [Julia-1 semantics at a85b127](https://huggingface.co/SupersonicLabs/Julia-1/tree/a85b127321d580d65176c89ced8273f305745d85).
The ONNX repository describes a browser WebGPU adapter; **this C# example uses
the same pinned graph with CPU ORT**, not its browser code. Its published
benchmark numbers are not measurements of this example. The model card declares
Apache-2.0; review its license and all dependency licenses for your intended use.
Latest upstream exports are not automatically interchangeable with this pin.

## Prepare the directory once

Checklist: compatible .NET 10 SDK; this full checkout; a Windows x64 environment
for the native configuration verified here; space for at least the 577 MB weight
file plus graph/tokenizer, NuGet packages and runtime memory. Disk size is not a
RAM requirement estimate. Other platforms need compatible ORT and tokenizer
native binaries and have not been exercised by this tutorial.

Expected directory (only these loader inputs are needed):

```text
artifacts\decision-models\julia\
  model.onnx
  model.onnx.data
  tokenizer.json
```

No model is bundled or downloaded by `LoadFromDirectory`. To acquire the pinned
files yourself, review the linked model/license first, then run this explicit
PowerShell download from the **repository root**:

```powershell
$base = "https://huggingface.co/SupersonicLabs/Julia-1-ONNX/resolve/82a2fadf8fccfccdc5fd4e1009ba8f1a265eb7a8"
$dir = "artifacts\decision-models\julia"
New-Item -ItemType Directory -Force $dir | Out-Null
foreach ($file in "model.onnx", "model.onnx.data", "tokenizer.json") {
    Invoke-WebRequest "$base/$file" -OutFile (Join-Path $dir $file)
}
Get-FileHash "$dir\model.onnx", "$dir\model.onnx.data", "$dir\tokenizer.json" -Algorithm SHA256
```

Compare the hashes **before running**; stop if a file differs:

| File | SHA256 |
|---|---|
| `model.onnx` | `97141d0cfb1da6204e9f8f24d581af72eaeb82cda21149d83eaa6df7160fbcd9` |
| `model.onnx.data` | `fd915be810d7ebfb80fb05a48dd33c9484d17ae1b6bcb9e1f544cbaaa913ded1` |
| `tokenizer.json` | `609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f` |

The pinned graph **references** `model.onnx.data`, so keep it alongside the
graph. A self-contained compatible ONNX graph need not have a sidecar: external
data is a property of the graph, not a universal loader rule. ORT resolves it;
missing referenced weights fail explicitly without fallback. `tokenizer_config`,
Python scripts, parity cases and browser files are not required for this app.

## Run

From the repository root:

```powershell
dotnet run --file samples\quickstarts\Julia\Julia.cs -- --help
dotnet run --file samples\quickstarts\Julia\Julia.cs -- artifacts\decision-models\julia
```

The first build can restore NuGet packages, including CPU ORT 1.23.2 and the
provider's Tokenizers.HuggingFace 3.23.1 dependency. This is distinct from the
explicit model acquisition above. Source project references are used because
the provider packages are not published. Native AOT is explicitly disabled.

## Walk through the application

1. Validate the argument before loading. A nonexistent directory is configuration
   failure, not an uncertain model answer.
2. `LoadFromDirectory` creates an owning generator. The `using` declaration
   disposes its tokenizer/session. For a long-running application, load once,
   reuse it for calls, and dispose when the owning service shuts down.
3. The **state** is the invented ticket: "I cannot sign in after resetting my
   password." The **question** asks which support team should handle it.
   `"support-team"` is your exact lookup key, not a model instruction.
4. Candidate IDs (`account`, `billing`, `technical`) are stable application keys.
   Their descriptions explain their meanings to the model. Rename a UI caption
   independently; do not rely on an opaque ID to convey semantic meaning.
5. `GenerateAsync(ticket, questions, TicketJson.Default.String, ...)` serializes
   the typed state and runs inference. The small inline `JsonSerializerContext`
   generates serialization metadata at build time; it is not training and
   does not guarantee a correct classification.
6. `GetAnswer<ChoiceDecisionAnswer>("support-team")` retrieves the typed answer.
   Printing the selected ID is not routing a ticket or authorizing any action.
   A 30-second token bounds the inference request, not synchronous initial load.

### When you need explicit paths or ORT settings

Replace only the loading line, adding `using Microsoft.ML.OnnxRuntime;` if
needed for ORT types in your own code:

```csharp
using IDecisionGenerator generator = JuliaDecisionGenerator.Load(new()
{
    ModelPath = @"artifacts\decision-models\julia\model.onnx",
    TokenizerJsonPath = @"artifacts\decision-models\julia\tokenizer.json",
    ConfigureSession = options => options.IntraOpNumThreads = 1
});
```

The callback is optional. Omit it to preserve default ORT settings; one thread
is an illustration, not a performance recommendation. Required paths become
absolute before the callback, temporary session options are disposed, and neither
callback nor temporary options is retained for generation. Old explicit-path
and advanced stage constructors still exist; they are not needed here.

## Read the output, carefully

Captured from this exact example on Windows x64 CPU with the pinned assets:

```text
Provider: julia
Model: SupersonicLabs/Julia-1-ONNX
Selected ID: account
account: 1.0000
billing: 0.0000
technical: 0.0000
```

`F4` rounds the full probabilities for display. Printed 1.0000 is **not**
certainty, and 0.0000 need not be mathematical zero. The distribution is relative
to these three supplied alternatives, not a measured probability of being right
on arbitrary tickets. The provider preserves full softmax values rather than
snapping them to one-hot output. Rounded display rows need not sum exactly to 1.
Different inputs/models may give other answers.

## Try, predict, then explain

Predict the most likely team before each edit to `ticket`, run again, then
compare the entire distribution: "I was charged twice"; "It is broken";
"What is the weather tomorrow?" The last is outside the offered task: Julia
still chooses among your candidates. Add a meaningful manual-review candidate
or an application review rule if your workflow needs one; neither guarantees
correct abstention. Try making two descriptions overlap. Why might that change
their relative probabilities? Do not expect every wording change to flip the ID.

## Troubleshooting and limits

Missing `.data` means the graph cannot load; restore the matching referenced
file rather than renaming an unrelated one. Tokenizer mismatch can fail packing
or silently change meaning, so verify the pin. Native load/architecture errors
require compatible **both** ORT and tokenizer libraries; do not replace packages
with unpinned versions. For SDK/ancestor build issues see the [catalog](../../README.md#troubleshooting-shared-setup).

The adapter enforces a strict 1,024-token total context, 256-token head,
48-token option budget and 2-20 Choice/Score options. Tokens are not characters;
framing consumes space. Over-budget inputs fail, not truncate. Binary has its
own paired-description rules. See [answer semantics](../../../docs/learning/02-reading-answers.md).
Cancellation requests ORT termination; it is not a hard real-time guarantee.
Model load/first inference may take longer than later calls; this example is
not a benchmark. Usage counts are unknown, not zero.

This run used retained assets without model downloads. No new accuracy,
calibration or cross-platform study was performed. Inference stays in this
process, but dependency/model acquisition can use the network. Use synthetic
data while learning; validate policy and privacy before real deployment.
Next: [Laya](../Laya/README.md), then [compare](../../scenarios/CompareProviders/README.md).
