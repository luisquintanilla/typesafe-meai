# Decision-to-Arrow experiment

This is additive, dependent experiment work on the existing decision branch.
It does not replace either existing solution or change HTTP providers, native
stack metadata, root/global NuGet policy, model/tokenizer/ORT versions, or public
package publication. Jevaro, cloud calls and compute kernels are outside this
adapter. The separate Arrow and ML.NET experiments own kernels and predictors.

## Portable boundary

`DecisionInference.Arrow` depends only on the portable
`DecisionInference.Abstractions` and pinned `Apache.Arrow` core (including its
portable Scalars dependency). It has no provider, compute, ONNX, tokenizer or
ML.NET training dependency. Runtime inference belongs to the sample.

Schema version 1 stores non-null `row_id: int64`, binary float64, choice
`struct<selected_index:int32, probabilities:fixed_size_list<float64>[N]>`,
and score `struct<expected_index:float64, expected_index_lower_bound:float64,
expected_index_upper_bound:float64, native_score:float64?,
probabilities:fixed_size_list<float64>[N]>`. Parent structs/lists and probability
elements are non-null. Only absent native score is a legitimate null.

Question/candidate/rubric order, case and exact strings are preserved. The
selected index is an ordinal into the captured candidate IDs, not a predicted
argmax or ML.NET key. HighPrecision, FourDecimalPlaces and TwoDecimalPlaces
observations remain doubles, without normalization or inferred confidence.
Rounded expected indices may exceed the rubric domain. Stored bounds are the
portable rounding-feasibility bounds, not confidence intervals.

`decision_arrow.*` metadata captures definitions, precision and the canonical
contract. The SHA-256 feature fingerprint covers schema version, float64
representation, definitions, model/provider, sorted named asset hashes,
preprocessing/decoder identifiers and ordered feature projection. It never
trims, normalizes or rewrites caller strings. Package versions are separate
provenance and do not redefine features.

```csharp
using DecisionInference.Arrow;

var contract = new DecisionArrowSchema(questions, DecisionPrecision.HighPrecision,
    new DecisionArrowIdentity("laya", modelId, assetHashes,
        preprocessing, decoder, featureProjection));

await DecisionArrowWriter.WriteAsync(outputDirectory, contract, rows, provenance);
using var reader = await DecisionArrowDatasetReader.OpenAsync(
    manifestPath, expectedContract, expectedSourceIds, cancellationToken);
while (await reader.ReadNextRecordBatchAsync(cancellationToken) is { } batch)
{
    using (batch)
    {
        // Standard Arrow RecordBatch; consumer owns this returned batch.
    }
}
```

Each row associates the caller's `long` ID, original `DecisionInput`, and
validated `DecisionResult`. Definitions/precision/model must match the frozen
contract. No raw-answer JSON is parsed.

The writer emits schema first, builds numeric columns directly in bounded
batches (default 256), flushes the final short batch and writes explicit partial
files. ID uniqueness tracking grows with row count; result/builder storage
does not. Checked width/capacity arithmetic precedes allocations. Existing
output directories are never replaced. On failure/cancellation the error
propagates and `run.partial.json` records failed/cancelled status. Only success
renames the data and atomically publishes `manifest.json` last.

The reader checks completion, canonical contract, physical nested schema,
metadata, hash/size, IPC end marker, count, unique IDs and numerical semantics
before exposing any batches. Supply expected source IDs to reject missing/extra
records against the consumer's source manifest. It scans twice using one
read-only, write-excluding file handle; returned batches remain caller-owned.
It is an ordinary `Apache.Arrow.Ipc.IArrowArrayStream`, not a competing stream.

## Reproducible restore and packaging

Keep root `NuGet.config` unchanged. See the adjacent
[sample instructions](../../samples/experiments/DecisionArrow/README.md).
`eng\experiments\dependencies.lock.json` freezes the baseline source and all
three actual Arrow package identities, sizes and hashes. `Restore-DecisionArrow.ps1`
requires explicit unsigned-local opt-in, validates clean immutable producer
receipts and nuspec identities/exact project dependency ranges, writes a
separate config, maps exact experiment IDs to local feeds, maps other IDs to
public nuget.org v2, uses an isolated cache, validates cached source/bytes and
verifies public package signatures with the unchanged root trust configuration.
The public v2 endpoint avoids this host's v3 TLS handshake failure; TLS is not
disabled. Initial dependency updates require `-UpdateLocks`; later restores
are locked. No floating PR heads or other checkout references are used.

`Publish-DecisionArrow.ps1` refuses a dirty tree or an existing destination.
It packs only the adapter and necessary portable abstractions from committed
source, pins project-reference nuspec dependency ranges exactly, and records
versions/hash/size/source/SDK/dependency evidence. Unsigned artifact hashes
identify bytes, not author-signature authenticity.

## Local corpus and measurement gates

The ML.NET owner freezes exact five-question JSON and separate label-free
states JSONL (`{"rowId":1,"state":"original text"}`). The exporter accepts only
rowId/state, never labels/splits. Preflight checks all rows against strict
existing framing, options, token budgets and reserved-token rules without
inference or rewriting. Any unsupported row stops the experiment with IDs and
budget diagnostics in an explicitly partial record.

The scalar exporter verifies all pinned English FP32 Laya assets locally,
uses one owned CPU session and the existing single-state API, and appends
answers directly into Arrow. No asset acquisition is implicit. Existing
source/version/asset identities and actual measurements appear in provenance.
The graph, external weights, tokenizer JSON and profile are semantic assets.
The optional, unused `tokenizer_config.json` is verified and recorded separately
when present; changing it does not redefine features or impose a new loader
prerequisite beyond the existing documented Laya layout.

Timing scopes are explicit: `loadMilliseconds` is session/model setup.
The exporter warms one scalar state or one full native state batch before the
writer measurement. `firstResultMilliseconds` measures that cold first native
call; `warmupStateCount` and `warmupOnnxCalls` identify its scope and count.
`firstExportResultMilliseconds` measures the first published observation after
warmup and overlaps the inference scope. Completed data contains each requested
ID once, including the states used for warmup.
`inferenceAndAppendMilliseconds` includes generation and consumer append time.
`exportBeforeManifestPublicationMilliseconds` includes writer setup,
generation, append, IPC/hash/finalize, but excludes preflight/model load/final
manifest serialization. The whole writer-call duration is additionally printed as
`writerWallMilliseconds`; per-native-call preparation/scoring/decoding and
source correlation are streamed to `batch-measurements.jsonl` before completion.
Each call records actual graph rows, padded sequence length and marker width,
not only the model's declared dynamic axes. Inference GC collection deltas and
writer process-wide allocation bytes are separate from native workspace.
Do not add overlapping measurements or interpret a
synthetic fixture as measured extractor cost.

`transport` compares identical generated portable content (all IDs, selected
indices, probabilities, score fields and shared contract). JSON is a synthetic
transport comparator, never a provider answer parser. It reports full encode,
decode/validation, and summary/parse times separately, bytes/envelope sizes,
process-wide allocations and cumulative process peak working set. The first
in-process observation is not a fresh-process cold benchmark. Arrow's completed
manifest/hash work and JSON's single envelope are explicitly different
envelopes, not an omitted-field optimization. Kernel-only throughput, zero-copy,
probability calibration and predictive improvements are not claimed.

Real corpus/model execution and native parity/performance remain separate
acceptance gates from offline fixtures. `LayaDecisionGenerator.GenerateBatchAsync`
is a provider-specific addition; the portable single-state interface remains
unchanged. It holds the existing generator resource gate once, prepares each
original input intact, collates with the actual pad ID, executes one combined
native tensor call through the existing runner, and decodes owned per-state
slices with the original decoder. It never recursively calls the gated scalar
method or uses `Task.WhenAll`. Mixed definitions/models are rejected before
inference and caller options are not mutated.

The checked 64 MiB cap includes prepared/combined managed tensor buffers and
both combined/per-state copied output buffers, excluding model/native workspace
and returned decision objects. Preparation also checks available input-buffer
budget before allocating tensor arrays. Existing graph name/type/rank and
static dimension checks run before native execution. Captured actual graph
dimensions, not a guessed dynamic-batch capability, are recorded by the sample.
Export provenance includes the actual isolated-cache runtime package closure
(including unchanged ORT/tokenizer versions and hashes); the unselected compute
artifact is excluded. The portable adapter/abstractions receipt separately
identifies downstream interoperability artifacts. The producer sample uses
same-repository project references, identified by its compiled source commit.
Native state batching defaults to 16 in the sample; the parity sweep is
1/4/16/32 with a short final batch. Generator CPU defaults remain unchanged;
the sample's declared thread count is identical in scalar/native comparisons.

The coordinator clarified that a deterministic synthetic-state Arrow export
through the retained real scalar model can unlock provider implementation/parity
work. It is not a reduced SMS study or replacement for full-corpus artifacts.
The `parity` command uses a separately declared bounded synthetic reference,
warms scalar/native shapes, and stops on any value/selection/precision/bounds
mismatch under the existing tolerance, without widening it.
Full-corpus inference failures may
not become completed datasets, dropped rows or success-shaped placeholders.

`compare` validates two completed datasets with the trusted contract, matching
input/question hashes, counts and ordered source correlation. It compares every
stored numeric observation, selected ordinal and native-score null exactly or
under the unchanged `1e-6` representation tolerance. It retains a partial
counterexample report on failure; neither dataset is rewritten.

## Recorded producer evidence

Clean source `ab4eebf29b9fdca49ce13e3b4afac0ec16b22983` produced the immutable
adapter/abstractions version `0.1.0-exp.decisions.1.gab4eebf29b9f`. Their
SHA-256 hashes are respectively
`6acc213b85879d1a34c8b4c80ba2450f7aeac4a8259e402bb57beffb201b85e9`
(30,356 bytes) and
`fa62a54b4399416df6a7ac5bc5fd0e9fcef8f99964d25a9bbe40c8dc1bbff88e`
(13,869 bytes). The package-only consumer closure contains only these two,
baseline Arrow core/Scalars and Microsoft.Extensions.AI.Abstractions.
All three clean 257-row precision fixtures passed independent ML.NET import.

The retained real model completed the separately declared 33-state synthetic
scalar export: 67,279.8665 ms writer wall, 66,773.6751 ms native scoring, 33
ONNX calls, 3,234.4368 ms model load, 2,354.2992 ms first result and
1,729,425,408 bytes process peak working set. This initial pre-warmup export
includes first-call cost in its writer scope; later warmed exports declare
their separate warmup. Its 11,536-byte IPC SHA-256 is
`25a5e115975517b94aae152768c15cf4712d6f8d3a833e7317679613d74ff43e`.
Real-model native sweeps 1/4/16/32 passed against all 33 scalar results without
widening tolerance. These are producer gates, not the 5,574-record SMS study
or evidence of predictor quality. Full-corpus outputs remain a separate gate.

The warm 33-state scalar comparison took 48,113.1213 ms. Native batches
1/4/16/32 took 48,344.5754 / 53,312.9615 / 63,545.5875 / 68,792.6017 ms,
with 33 / 9 / 3 / 2 native calls and maximum absolute difference zero.
Fewer calls did not improve CPU throughput on this synthetic reference:
batch 16 was 32.1% slower and batch 32 was 43.0% slower than scalar.
These timings are not extrapolated as corpus results.

### Completed full-corpus scalar export

Runtime source `915b85b8dbe123e465523120b0cc2c4c7d57f7b6` completed all
5,574 frozen original states with the five unchanged questions, CPU intra-op
4/inter-op 1, ORT 1.23.2 and tokenizers 2.0.0. No labels/splits entered the
generator and no state was truncated, excluded or substituted. The immutable
portable package source remains `ab4eebf...`; it is not the runtime commit.

| Observation | Measured value |
|---|---:|
| Writer wall, excluding load/warmup | 8,848,373.8602 ms (147.47 min; approximately 0.630 source rows/s) |
| Native scoring total | 8,845,290.2273 ms |
| Preparation / decode totals | 1,025.3441 / 388.7792 ms |
| Model load / cold first call | 2,508.2789 / 1,907.4477 ms |
| First exported result after warmup | 1,780.1850 ms |
| Data calls / separate warmup calls | 5,574 / 1 |
| Process peak working set | 2,163,466,240 bytes |
| Maximum estimated managed tensor buffers | 26,825 bytes |
| Inference generation 0 / 1 / 2 collections | 48 / 10 / 2 |
| Writer process-wide allocated bytes | 576,380,472 |
| Completed IPC size | 759,744 bytes |

The standard reader from the actual immutable package, not a project
reference, accepted expected source IDs 1..5574 and read every typed batch.
All IDs occurred once and in original order; all metadata, values, counts,
file integrity and whole-stream completion checks passed. The feature
fingerprint is
`72acebb6036bfa8fdaa1d47a1490f5ea1bfc6e08fe71e7e84e8c8922ec4de514`.
The data SHA-256 is
`fae87f65effbb7cac23f6e9de8af382dd6dc1b145bf843c40001947f159dff46`,
and manifest SHA-256 is
`76748fdf24a0a33efb1a6509e00d25fbc8caddd3b002005cb1bf77e2911c8c2c`.

Private `datasets\laya\scalar-corpus-915b85b.receipt.json` has SHA-256
`829dea0cccfeef32015798896fa51037d7586e3f2585559315653be98f8c2e58`.
It records relative artifact names, hashes/sizes, frozen input/question hashes,
source/package distinction and measured scopes. It was handed to ML.NET before
releasing a bounded consumer training/evaluation/head timing slot.
ML.NET returned the separate CPU lease after its real study; the complete
native-16 run below then proceeded without overlapping consumer timings.

### Completed full-corpus native-16 comparison

The same compiled runtime source, frozen inputs/questions/assets and CPU
settings completed all 5,574 native rows. There were 349 data ONNX calls plus
one separate 16-state warmup. The final data call contained six states
(30 graph rows); other calls contained 16 states (80 graph rows). Recorded
padded sequence lengths were 99..330 and marker width was five. No bucketing,
session pooling, GPU execution or changes to provider defaults were introduced.

| Observation | Scalar | Native-16 |
|---|---:|---:|
| Writer wall (ms) | 8,848,373.8602 | 12,845,323.7374 |
| Writer wall (minutes) | 147.47 | 214.09 |
| Source rows/second | 0.630 | 0.434 |
| Native scoring total (ms) | 8,845,290.2273 | 12,844,063.3281 |
| Preparation / decode total (ms) | 1,025.3441 / 388.7792 | 525.4535 / 71.5794 |
| Cold first result (ms; batch-specific scope) | 1,907.4477 (one state) | 43,087.8526 (16 states) |
| First exported result after warmup (ms) | 1,780.1850 | 41,329.5899 |
| Data calls / separate warmup calls | 5,574 / 1 | 349 / 1 |
| Process peak working set (bytes) | 2,163,466,240 | 9,846,378,496 |
| Process peak working set (GiB) | 2.0149 | 9.1702 |
| Maximum managed tensor estimate (bytes) | 26,825 | 576,720 |
| Inference generation 0 / 1 / 2 collections | 48 / 10 / 2 | 50 / 13 / 2 |
| Writer process-wide allocated bytes | 576,380,472 | 614,842,496 |
| Completed IPC bytes | 759,744 | 759,744 |

Native-16 was **45.17% slower** with approximately **4.55 times** the process
peak working set. Fewer native calls are not a throughput win. The 64 MiB
managed tensor bound was respected; it does not cap resident model/native
workspace. Keep scalar as the experiment CLI default. These are one completed
corpus pair plus the separately warmed 33-state sweep, not repeated full-corpus
trials or a statistically estimated speedup. Padding is a plausible cost
contributor, not a separately isolated causal result.

The actual immutable-package public reader accepted native expected IDs
1..5574. Full typed comparison checked every source ID, binary observation,
selected ordinal, probability coordinate, score expectation/bounds and
native-score null/value under the unchanged `1e-6` tolerance. Maximum absolute
difference was **zero**. Both complete IPC files have the same SHA-256
`fae87f65effbb7cac23f6e9de8af382dd6dc1b145bf843c40001947f159dff46`.
No scalar-trained predictor was retrained or retuned for this byte-identical
native dataset.

Private native manifest SHA-256:
`ce699c353065766af8c49a1ca94e407663ba03bc318afb24b589a87cbf52ceed`.
`datasets\laya\native16-corpus-915b85b.receipt.json`:
`e4187c63b2ff1b10dbc31fd8aaa5fb073dcda40b5ce3e4e4d9941287fff188a6`.
`datasets\laya\scalar-native16-corpus-915b85b.parity.json`:
`eb54e61eebe4a91cdb3d059251034e7c52491e99553c4ff614405122573d10a0`.
The hardware receipt identifies Windows 11 Enterprise, Intel i9-11950H
(8 physical/16 logical cores), 68,412,301,312 bytes physical RAM and the same
4/1 thread settings. Both corpus processes recorded .NET 10.0.12.

### Transport, summaries and downstream boundary

The separate synthetic transport study covered 0, 1, 257 and 65,536 rows with
three observations each, validating equivalent IDs, all probabilities,
selections, score fields and canonical metadata. The larger artifact is not
real extractor timing. The 257-row transport report uses FourDecimalPlaces;
the 0/1/65,536-row reports use HighPrecision, so this is not a uniform-precision
cross-size scaling comparison. All three precision declarations have separate
clean 257-row interoperability fixtures. At 65,536 rows Arrow IPC was 8,865,080 bytes versus
20,832,398 bytes JSON (Arrow completion envelope approximately 4,055 bytes,
shared UTF-8 contract 2,078 bytes). Warm Arrow/JSON encode measurements were
2,567/2,482 and 2,438/2,581 ms; decode-plus-content-validation 809/949 and
659/709 ms; summary-including-validation/parse 241/181 and 230/220 ms.
Warm process-wide encode allocation deltas were approximately 4.084 GB Arrow
versus 303.6 MB JSON. No universal encoding, allocation or summary advantage
is claimed. The first observation is explicitly first-in-process, not a
fresh-process cold protocol; file-cache effects and unequal completion
envelopes remain disclosed limitations.

Completed scalar/native summaries agree for every stored coordinate. Summary
means are observations, not confidence or calibration evidence. The optional
fork compute prototype was deliberately **not selected** in this sample; its
separate fork study/package-only compatibility belongs to the Arrow owner.
Neither adapter nor predictor gained a compute/provider/native dependency.

ML.NET's real 15-arm/curve study used this scalar receipt and frozen splits,
with its training freeze pinned before holdout and 1,000 group-bootstrap
resamples of saved predictions. Consumer source
`a3f1b7e6bb12c4163d82f125e1a470ed72edc67f` and private
`predictors\real-scalar-v1\study.complete.receipt.json` (SHA-256
`3a253ccac138e6db362b2ae1172626d4bfeff9d4c89bf515b0131782bc936b21`)
identify the independent learning results. Predictive claims, conversion
boundary, uncertainty and contamination limitations belong to that report,
not to the Arrow transport or native-call count.

## Julia-first optimization (separate dependent experiment)

The completed Laya results above are unchanged. Julia is a new extractor and
feature identity, not a relabelled Laya artifact. Full Julia extraction and the
new consumer study are **not complete**. Fictional fixtures, preparation-only
preflight, static CUDA builds and old saved-model replay are not substitutes.

Before changing shared mathematics, the independent original Julia executable,
provider/Abstractions/Onnx closure and original source were frozen at
`6dda59d19a60fdad2fc14152e3774d1d5ad9b345`. The immutable v2 closure contains
83 verified files. The original strict label-free preparation preflight accepted
all 5,574 original IDs, with zero failures and padded lengths 72..304. Corpus,
question, split, state, model, weights and tokenizer hashes are unchanged.
The failed v1 harness receipt attempt is retained separately.

### Portable implementation and measured scopes

An internal immutable compiled contract now captures question order, ordinal
lookups, widths, canonical metadata and schema once. Public schema views are
defensive copies. Public answer constructors and Arrow checks share the same
span mathematics: sequential sums, unchanged `1e-6`, ToEven grids, ULP behavior,
half-unit intervals, endpoint clamping, rounded feasibility and native bounds.
Selections still need only be valid members, not argmax. The independent
4,161-case original-binary oracle matched acceptance/error messages and every
Double bit; it did not compare two refactored implementations.

The sequential reader still validates the entire completed dataset before
exposure, including trusted identity, hash/size, nested offsets/types/nulls,
all values, exact expected IDs, full EOS, trailing and concatenated IPC guards.
There is no public index, rewind, snapshot, mmap or validation-cache API.

Typed `WriteAsync` remains available. Additive `WriteBatchesAsync` accepts
standard owned `RecordBatch` values, validates each whole batch and the global
ID set, and disposes transferred batches on success, cancellation or failure.
The producer must not mutate or dispose a batch after transfer. Buffers are
bounded and checked before reservation; complete manifest publication is last.
Expected-ID/runtime checks in the Julia sample run before that publication.

Five fresh-process AB/BA pairs used the same profiler harness in both arms and
verified frozen original portable DLLs in the reference arm. Positive row
counts 1/257/65,536 across all three precisions passed all 18 matched append/
validation gates: at least 50% allocation reduction and no greater than 10%
median time regression. Minimum allocation reduction was 90.28%; the worst
median time change was a 22.66% improvement.

| 65,536-row median scope | Original ms | Optimized ms |
|---|---:|---:|
| HighPrecision append | 1,711.37 | 75.86 |
| FourDecimalPlaces append | 1,534.91 | 202.97 |
| TwoDecimalPlaces append | 1,388.44 | 189.75 |
| HighPrecision validation | 109.11 | 66.12 |
| FourDecimalPlaces validation | 215.71 | 125.64 |
| TwoDecimalPlaces validation | 229.17 | 133.85 |

Raw schema, encode, completed-open, replay, file/hash, GC, CPU and working-set
observations are retained separately, not added to overlapping scopes.
Some original scope CVs were 12..68%; these are engineering medians, not a
population speedup estimate. Warm append measurement excludes builder/buffer
construction, typed observation creation, Build and IPC. Its zero row-scope
allocation is **not** a total zero-allocation or zero-copy claim.
Private acceptance receipt `profile\paired-v1\paired.acceptance.json`:
`bad84fdbad5d14d0b54610c798b12a8f7fa829a6b372600dfd29360050296efe`.

### Julia ownership, explicit commands and runtime isolation

The sample bridges an internal synchronous Julia numeric sink to owned
256-row Arrow buffers. Public `GenerateAsync`, `Score`, `Decode`, source
association and owned raw-logit snapshots remain unchanged. The original
100-byte native output copy is retained. There is no native lease/token cache,
shared runner change, escaping span or host access to a GPU pointer.
Julia and portable Arrow have no dependency on one another.

Julia retains Python-style JSON, strict total1024/head256/option48/min16,
reserved IDs 0..4 and pad8. The graph remains FP32, with stable full Double
softmax over valid lanes, first argmax, binary p[1], zero-based score/bounds,
HighPrecision and nativeScore NULL. Laya temperature/action/rounding is not
applied. Original/candidate prepared-tensor hashes use the same byte framing.

The new explicit commands, available through `julia-help`, are:

```text
julia-contract <frozen-questions> <explicit-profile> <new-directory>
julia-fixture <frozen-questions> <explicit-profile> <new-directory>
julia-preflight <assets> <frozen-questions> <all5574-label-free-states> <new-directory>
julia-control <assets> <frozen-questions> <controls> <trusted-contract> <profile> <new-directory> [diagnostic]
julia-compare <reference-manifest> <reference-contract> <candidate-manifest> <candidate-contract> <controls> <new-report.json>
julia-export <assets> <frozen-questions> <all5574-label-free-states> <trusted-contract> <selected-profile> <new-directory>
```

Existing Laya positional commands and the beginner route are preserved.
Commands never acquire models. Real commands require an immutable portable
package receipt, exact assets/questions, only label-free rowId/state fields,
an explicit profile and a new output directory. A separate one-state warmup
is excluded from extraction timing, but its state still appears once in output.
Two separate warmup calls are recorded: one checks actual host-readable output
memory before any span access, then one warms the synchronous numeric export.
Timing includes preparation, synchronized Run/transfers, decoding, builder,
IPC and full validation; separate transfer duration is unmeasured.
Failure/cancellation exits nonzero and cannot publish a complete manifest.

The CPU host has only native ORT 1.23.2. The separate
`samples\experiments\DecisionArrow.Cuda` host has only ORT Gpu 1.23.2, its own
bin/obj/lock, and explicitly shared Julia harness sources; it does not reference
the CPU executable. Libraries use ORT Managed only. Restore/build the CUDA
host through `eng\DecisionArrowCuda.slnx`, the existing scoped mapped restore
helper and a new isolated cache, with SDK 10.0.401 and the shell-local
`ImportDirectoryBuildTargets=false` workaround on this host.
The additive CPU lock's Protobuf 3.34.1 refresh is intentional for Julia; other
existing model/tokenizer/ORT pins and global signing/trust policy are unchanged.

Profiles use exact `scalar-cpu` or `scalar-cuda`, intra4/inter1. CUDA requires
an explicit NVIDIA index/UUID and existing process-local native search
directories, `use_tf32=0`, default synchronized stream/copies and pinned V2
CUDA provider options. No global PATH/install, TensorRT, graph rewrite,
custom stream, FP16/BF16 or quantization is selected. Static restored binaries
are not initialization/placement proof. Loaded native paths/versions/hashes,
actual package closure, options and graph metadata are separate receipts.
Diagnostic actual CUDA scoring kernels and every CPU placement must be
reported; a mixed graph is HYBRID, never GPU-only.

The RTX 3080 Laptop 16 GiB/SM8.6/driver596.47 candidate initialized successfully
with the signed ORT1.23.2 GPU package, actually loaded CUDA12.6/cuDNN9.8 and
TF32-off options. Its diagnostic trace proved CUDA scoring and enumerated CPU
placements: **HYBRID**, not GPU-only. Default synchronized output was actually
host-readable Cpu FP32 [5,5], verified before span access. CUDA_PATH13.4 was not
used as readiness evidence; no global setting/install changed. The exact
official ORT v1.23.2 Windows recipe uses CUDA12.2/native86; actual signed
packages/imports/loaded DLLs, rather than a generic version table, established
this candidate's initialization.

**CUDA was rejected at the unchanged numerical gate.** On the first of eight
externally selected readiness states, original ID1926, binary coordinate0,
CPU/CUDA absolute difference was `4.9192878088799574e-6`, above `1e-6`.
No tolerance widening, model/options rewrite, CUDA128, longest-shape VRAM
qualification, paired CUDA timing or full CUDA export followed. The original
failure, complete bounded controls, loaded modules/options, trace and telemetry
are retained. Diagnostic8 timings are not clean paired performance evidence.

The approved fallback explicitly selects **scalar-cpu**, not an automatic
fallback inside a GPU-labelled process/dataset. Independent frozen original
CPU versus optimized CPU passed both eight and 128 externally selected states
with maximum difference **zero**, exact IDs/selection/precision/native-null,
and exact 128 prepared input/mask/marker/qtype/shape hashes. Runtime source
`fd14d3b1caf89886a5f6cc29fa6c01dcef199e01` and its complete frozen CPU closure
are separate from portable package source65880cdf and later documentation HEAD.
Private selected CPU qualification SHA:
`6602c1f86a251855eb2f647dd700c6402870a0c1960c81d45d7484c2cfc50f07`;
selected-profile SHA:
`a393ea3b2c93f97e19edbc15a6ffac25e49a58d2c9759dc7207a26a7a413f11f`.
Qualification status is `QUALIFIED-CPU-FALLBACK-FULL-GATED`, **not GO-READY**.

### Immutable handoff and remaining gates

CPU canonical HighPrecision contract SHA/fingerprint is
`78ed3dc0344f71f1d8515f5923dc2c1de3604e59e6b4a1c9cb89851e2b3cf05d`.
Three separate completed fictional precision fixtures have authoritative
expected IDs **1..257** each, with output batches 256+1. They are explicitly
not real Julia observations. GPU arithmetic has its own trusted canonical
identity; comparison validates each artifact with its own contract before
comparing unchanged questions/values. Consumer code must not invent hashes.

All new private evidence is under `julia-optimization-v1`, separate from old
packages, data and negative results. Source-derived portable package pins,
nuspec/hash/size closure, measured source, runtime source and later docs HEAD
are recorded separately. Package-only three-precision reader/ABI checks and
the consumer's compact-storage/prediction acceptance remain required.
Use the publisher's explicit `-StagingDirectory` under the new namespace to
keep this round's package staging separate from historical repository artifacts.

The immutable portable source is
`65880cdfd33580d23d0e833ac977f79acb18b1b0`, version
`0.1.0-exp.decisions.1.g65880cdfd335`. Abstractions is 14,532 bytes, SHA-256
`a7a8742ea8d35a8732a91531db38c14bb82c0fb9e37f936b3524412fe149cf6e`;
Arrow is 37,059 bytes, SHA-256
`f44cd8e08de3cfc774fa1315dbfc13e7ed90d999a90f9f65057d235061f0e788`.
The isolated locked package-only closure is exactly these two, baseline
Arrow/Scalars and MEAI.Abstractions. All three fictional 257-row fixtures passed
the public package reader and owned standard-batch writer ABI, including exact
IPC byte hashes. The package-only helper's `roundtrip <manifest> <contract>
<new-output>` command deliberately tests those 257 independently expected IDs.
Private portable receipt SHA-256:
`1baa85ce0b1be99d306713d8a19f287b6a8a599bcf5726169c965f842d94b03c`.

Bounded real scoring required a coordinator quiet CPU/GPU/model lease, now
released after CPU fallback qualification. The completed stage8 and CPU128
controls above used the unchanged `1e-6`. Prepared
tensors, IDs, selections, precision and native-null must match exactly.
Longest-preflight, most-option and valid1024 synthetic VRAM probes would have
been required to accept CUDA, but were not run after its hard numeric rejection.
CUDA selection would additionally require actual scoring placement,
memory stability, five clean warmed AB/BA pairs on the fixed32 subset, at
least 10% median end-to-end benefit, CV at most 10% and 4/5 faster pairs.
Only the predeclared maximum seven-pair/six-faster extension is permitted.
Otherwise CPU is explicitly selected and failure/negative evidence retained.

Only after joined core/runtime/package gates and coordinator GO may one
optimized 5,574-row Julia export run, state1/output256, with no full original
comparison. A CUDA failure cannot mix CPU rows; any explicit whole-CPU retry
uses a new directory/profile/identity. The public pinned package reader must
verify hash/schema/count/expected IDs/full EOS before consumer handoff.
Package-only consumer three-precision interoperability and 153 focused cases
passed with the final pin. Consumer projection gates passed, but its text/
combined repeated-materialization allocation/time gates remain blocked.
No full-run or new-study success is currently claimed; producer waits for
joined acceptance and coordinator GO rather than bypassing those failures.
