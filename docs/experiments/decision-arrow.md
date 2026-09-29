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

Timing scopes are explicit: `loadMilliseconds` is session/model setup;
`firstResultMilliseconds` overlaps the inference scope.
`inferenceAndAppendMilliseconds` includes generation and consumer append time.
`exportBeforeManifestPublicationMilliseconds` includes writer setup,
generation, append, IPC/hash/finalize, but excludes preflight/model load/final
manifest serialization. The whole writer-call duration is additionally printed as
`writerWallMilliseconds`; per-native-call preparation/scoring/decoding and
source correlation are streamed to `batch-measurements.jsonl` before completion.
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
