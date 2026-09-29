# Qwen: adapting a text model to a Choice question

[Sample catalog](../../README.md) | [Beginner guide](../../../docs/learning/README.md)

## What you will learn

[Qwen.cs](Qwen.cs) keeps the same typed support-routing call while using a
different mechanism: a **generative language model's next-token scores**.
It demonstrates API portability, not an assertion that all providers use the
same architecture or produce comparable confidence.

The model is Qwen3-0.6B in one specific Q4_K_M **GGUF** artifact. GGUF stores model
weights/metadata; Q4_K_M is a quantized representation, reducing the precision
of stored weights. The pinned **TensorSharp.Models 2026.9.1 CPU runtime** loads it.
`QwenDecisionGenerator` is this project's Choice-only adapter. It frames the
candidates as verified distinct single-token aliases A, B, C, and normalizes
their selected logits (raw model scores). It does not generate/read a prose
answer and does not assume arbitrary category names are single tokens.

This is **not** Julia/Laya's specialized decision-head architecture. See the
[original TensorSharp recipe](https://gist.github.com/luisquintanilla/3d1a22ca8dbab1ae9a80caff4f2bf612)
and [upstream Qwen model card](https://huggingface.co/Qwen/Qwen3-0.6B).
The recipe is inspiration, not the complete ownership/reset/validation contract.
Upstream model capabilities and long context claims are not automatically
capabilities of this 512-context prototype. Review upstream Qwen/quantization
licenses before using or redistributing third-party weights.

## Acquire the exact bytes, not just a matching name

Checklist: .NET 10 SDK, full checkout, CPU environment compatible with the
pinned TensorSharp stack, 522,640,096 bytes of disk for this GGUF plus packages
and working memory. Windows x64 was exercised here; no GPU/server is required.

The loader requires SHA256:

```text
7f4030143c1c477224c5434f8272c662a8b042079a0a584f0a27a1684fe2e1fa
```

The [public Ollama registry manifest](https://registry.ollama.ai/v2/library/qwen3/manifests/0.6b)
identified this model layer when these docs were checked. Its `0.6b` tag can
change, so acquisition below uses the **content-addressed blob**, not a floating
tag. This is asset distribution only: the example does not need an Ollama daemon.
Another publisher's similarly named Q4_K_M file can have a different hash and
is deliberately rejected. The official upstream GGUF page's Q8_0 file is also
not a substitute.

After reviewing the model/license, explicitly download from the repository root:

```powershell
$hash = "7f4030143c1c477224c5434f8272c662a8b042079a0a584f0a27a1684fe2e1fa"
$dir = "artifacts\decision-models\qwen"
New-Item -ItemType Directory -Force $dir | Out-Null
$file = Join-Path $dir "qwen3-0.6b-q4_k_m.gguf"
Invoke-WebRequest "https://registry.ollama.ai/v2/library/qwen3/blobs/sha256:$hash" -OutFile $file
if ((Get-FileHash $file -Algorithm SHA256).Hash -ne $hash) {
    throw "Wrong GGUF bytes; do not run the model."
}
```

Alternatively, use an **existing** registry blob whose hash matches; its filename
need not end in `.gguf`. Verification in this phase used such a retained blob,
not a new download. The model embeds its matching tokenizer; do not pair it
with an unrelated tokenizer file. The app itself never downloads a model.

## Run

From the repository root:

```powershell
dotnet run --file samples\quickstarts\Qwen\Qwen.cs -- --help
dotnet run --file samples\quickstarts\Qwen\Qwen.cs -- artifacts\decision-models\qwen\qwen3-0.6b-q4_k_m.gguf
```

First build may restore TensorSharp and its dependency packages. This file
references only the Qwen provider project and its transitive requirements, not
the ONNX providers or HTTP server. Public provider packages are not available;
source references are intentional. AOT publishing is disabled.

## Walk through the file

1. Validate the file argument. `Load` separately checks the exact SHA256, so
   an existing wrong file is still rejected.
2. Set `MAX_CONTEXT` to `512` **in this app process before model creation**.
   TensorSharp reads it during initialization. The library does not secretly
   mutate process-wide configuration for its callers; the host makes the choice.
3. Own the loaded generator using `using`. It serializes access to its stateful
   model and resets the KV cache (previous token state) per forward. This is
   not a multi-user conversation memory or a reload of weights on every call.
4. Supply the synthetic sign-in ticket and the same account/billing/technical
   descriptions. IDs are exact keys; descriptions supply the meaning.
5. The source-generated `TicketJson` context serializes a typed string.
   Await `GenerateAsync` and get the `ChoiceDecisionAnswer` by `"support-team"`.
   This application call is ordinary C# despite the different scoring mechanism.

The 30-second token starts **after loading**. The pinned runtime's synchronous
`ForwardRefill` has no cancellation parameter. Cancellation is checked around/
between forward calls, **not inside a running forward**; the deadline is not a
hard 30-second process timeout. Do not use it as a hard resource-kill mechanism.

## Actual output and why it needs care

Captured on Windows x64 CPU with the pinned retained GGUF (runtime startup
diagnostics omitted):

```text
Provider: tensorsharp-qwen
Model: Qwen3-0.6B-Q4_K_M@7f4030143c1c477224c5434f8272c662a8b042079a0a584f0a27a1684fe2e1fa
Selected ID: account
account: 0.9864
billing: 0.0128
technical: 0.0009
```

These are probabilities **conditional on the allowed answer aliases**, not
probabilities over the model's whole vocabulary and not a calibrated chance of
correct routing. Rounded display rows here sum to 1.0001; that is display
rounding, not new precision or a probability repair. A high score can still be
confidently wrong when none of the candidates fits.

## Exercises and troubleshooting

Predict, edit `ticket`, run, explain: "I was charged twice", "It does not work",
then "Tell me a joke". Which evidence distinguishes the candidates? Change one
description while keeping its ID fixed; scores can change even though your
application key does not. Do not promise an answer flip. Treat out-of-scope
tickets through an explicitly evaluated application manual-review policy,
not an assumed native abstention or a universal confidence cutoff.

Wrong-hash errors require the pinned bytes, not disabling verification.
OOM/native/platform failures require checking runtime compatibility and available
memory; disk file size alone does not predict peak RAM. Long prompts fail the
active token/context budget rather than being silently truncated. Framing,
instructions and descriptions consume tokens too. Keep synthetic experiments
short. For SDK/build problems see the [catalog](../../README.md#troubleshooting-shared-setup).

Only Choice with 2-26 candidates is supported. Binary and Score are not
implemented by this adapter. First load/hash verification and model initialization
are different costs from inference; this example is not a benchmark.
One bounded short Choice call ran during this phase with retained assets.
No accuracy/calibration study, new download or daemon call was performed.
Inference is local; NuGet and deliberate artifact acquisition can still use
the network. Next: [probabilities and softmax](../../../docs/learning/02-reading-answers.md).
