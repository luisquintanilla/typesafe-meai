# Test Generation Research

## Project Overview
- **Path**: `C:\Dev\copilot-worktrees\typesafe-meai\luisquintanilla-didactic-pancake`
- **Language / framework**: C#, `net10.0`, Apache Arrow IPC; nullable and implicit usings enabled, warnings treated as errors.
- **Test Framework**: Offline xUnit v3. Central versions: `xunit.v3` **3.2.2**, `xunit.runner.visualstudio` **3.1.5**, `Microsoft.NET.Test.Sdk` **17.14.1**. The new test project uses these versionless references, `OutputType=Exe`, `IsTestProject=true`, `IsPackable=false`, `Using Include="Xunit"` and a project reference to Arrow only, matching the inspected `tests/DecisionInference.Tests/DecisionInference.Tests.csproj`.
- **Project system**: SDK-style.
- **Dependency format and versions**: Central `PackageReference` via `Directory.Packages.props`. Arrow references `Apache.Arrow` **`[23.0.0-exp.decisions.1.g6aafc634d65c]`** and `../DecisionInference.Abstractions/DecisionInference.Abstractions.csproj`; the latter references `Microsoft.Extensions.AI.Abstractions` **10.10.0**. No inference/compute dependency or mocking library is needed.
- **SDK / runner**: `global.json` pins **10.0.401**, `latestPatch`; no MTP runner selection there. The central test packages support VSTest discovery; parent must confirm the actual test project.
- **New-file registration**: SDK implicit compile glob; no explicit compile items or Arrow `InternalsVisibleTo` found. Test the internal builder through public writer/reader APIs, not reflection or production changes.

## Scope
- **Boundary**: ONLY `src/DecisionInference.Arrow`, all six C# files below. Abstractions reads were limited to `DecisionInput.cs`, `DecisionResult.cs`, the `DecisionKind` declaration, and their project manifest for fixture construction/dependency interpretation.
- **Approved plan evidence**: ONLY section 4 (portable contract v1) and section 9.1 (isolated transport study) content in `C:\Users\luquinta\.copilot\session-state\64dc898c-ad00-4d88-81ce-534017782670\plan.md`.
- **Targets**: Exact paths in the priority table below.
- **Representative existing tests**: `tests/DecisionInference.Tests/ContractTests.cs` establishes sealed classes, `DecisionInference.Tests` namespace, facts/theories, async tests, concrete equality/collection assertions and immutable abstraction constructors.
- **Restrictions**: Research only; no production/config/test/version-control edits, installs, restores, or builds. Only this document may be written. Bugs are reported, not fixed.
- **Snapshot**: Writer was reread after concurrent parent changes added exporter measurements; observations below include those additions. No compile/runtime success is claimed.

## Dependency Graph
- **Leaf types** (no dependencies on other Arrow types): `DecisionArrowIdentity`; `DecisionArrowPackageArtifact`; `DecisionArrowRow` (uses external abstraction payloads).
- **Mid-layer types**: `DecisionArrowProvenance` → artifact record; `DecisionArrowSchema` → identity + immutable abstraction questions; `DecisionArrowManifest` → provenance + schema version; internal `DecisionArrowBatchBuilder` → schema + row + abstraction answers.
- **Top-layer types**: static `DecisionArrowWriter` → schema, builder, rows, manifests; `DecisionArrowDatasetReader` → schema, manifest, abstraction answer validation.
- **Interfaces**: No in-scope interfaces or injectable abstractions. Reader implements external `Apache.Arrow.Ipc.IArrowArrayStream`/`IDisposable`. External collaborators are Arrow arrays/IPC, file APIs, SHA-256, System.Text.Json, Stopwatch/GC measurements, and immutable decision payloads.
- **Strategy**: Test leaves directly; test layers using real local IPC files and tiny deterministic async row generators. No provider/model calls, native binaries, tokenizers, networking, or mocks.

## Build & Test Commands
Initial research recorded commands without execution. The parent subsequently supplied
the exact isolated restore helper and documented host workaround. Final execution
results are in `status.md`.

Every restore/build/test shell sets `$env:ImportDirectoryBuildTargets='false'`.
This is the documented shell-local workaround for unrelated external
`C:\Dev\Directory.Build.targets` importing Arcade. No ancestor/global file was edited.
The approved restore uses `eng/experiments/Restore-DecisionArrow.ps1`, the verified
`arrow-baseline` feed, and `artifacts/decision-arrow-tests-restore` isolated cache.

- **Build (scoped)**: `dotnet build tests/DecisionInference.Arrow.Tests/DecisionInference.Arrow.Tests.csproj --no-restore -v:q`
- **Test (scoped — fix cycles)**: `dotnet test tests/DecisionInference.Arrow.Tests/DecisionInference.Arrow.Tests.csproj --no-restore -v:q`
- **Test (harness-equivalent — root/default discovery check)**: `dotnet test TypeSafe.MEAI.slnx --list-tests --no-build --no-restore`
- **Isolated experiment discovery**: `dotnet test eng/DecisionArrowExperiments.slnx --list-tests --no-build --no-restore`
- **Final isolated build**: `dotnet build eng/DecisionArrowExperiments.slnx --no-restore --no-incremental -v:q`
- **Lint**: No scope-specific checked-in lint command established; do not run a formatting command that edits sources.

**Discovery blocker:** `eng/DecisionArrowExperiments.slnx` already names the Arrow source and `../tests/DecisionInference.Arrow.Tests/DecisionInference.Arrow.Tests.csproj`, but that test project did not exist at inspection. Root `TypeSafe.MEAI.slnx` has no Arrow/Arrow-test entry. Thus the isolated command is the relevant scoped entry point, while a generic bare root `dotnet test` will not discover Arrow tests. Parent must resolve/explicitly report this harness mismatch; this research does not alter solutions or configs.

**Resolved within the authorized scope:** The new test project now exists and the
existing isolated solution discovers all 173 cases. Root/default discovery remains
outside the approved Arrow-only entry point; neither solution was edited here.

### Base .NET extension guidance to reuse
Read ONLY `code-testing-extensions/SKILL.md` and its base `extensions/dotnet.md`, at the supplied plugin location; no examples were read and unavailable skill invocation was not retried.
- Classify project system first; preserve central versions and the established framework.
- Inspect chosen test project's references and compile-item rules; SDK glob inclusion does not imply solution registration.
- Scoped project builds/tests during implementation; final selected-solution build with `--no-incremental`.
- Use `--no-restore` after parent's restore; `--no-build` only after a successful build.
- Harness discovery must enumerate the new tests and show the expected count delta through the externally selected entry point, not merely a passing targeted project.
- Do not configure/run coverage tools or upgrade dependencies to match examples.

## Files to Test

### High Priority
| File | Classes/functions | Testability | Estimated coverage | Notes |
|---|---|---|---|---|
| `src/DecisionInference.Arrow/DecisionArrowSchema.cs` | `DecisionArrowIdentity`, `DecisionArrowSchema` | High | Unverified | Canonical semantics, metadata, schema, fingerprints, input rejection. |
| `src/DecisionInference.Arrow/DecisionArrowWriter.cs` | `DecisionArrowWriter.WriteAsync` | High with temp directory | Unverified | Bounded batches, publication order, failures/cancellation, directory protection. |
| `src/DecisionInference.Arrow/DecisionArrowDatasetReader.cs` | `DecisionArrowDatasetReader.OpenAsync`, stream API | High with raw IPC fixture helper | Unverified | Eager whole-dataset validation and corruption matrix. |
| `src/DecisionInference.Arrow/DecisionArrowBatchBuilder.cs` | internal builder constructor, `Append`, `Build`, `Count` | High indirectly | Unverified | Ordering, fidelity, consistency checks and checked allocation via writer. |

### Medium Priority
| File | Classes/functions | Testability | Estimated coverage | Notes |
|---|---|---|---|---|
| `src/DecisionInference.Arrow/DecisionArrowRow.cs` | `DecisionArrowRow` | High | Unverified | Constructor guards and retained input/result references; association checked later. |
| `src/DecisionInference.Arrow/DecisionArrowManifest.cs` | artifact, provenance, manifest records; `JsonOptions` | High | Unverified | Required members, exact wire naming, provenance independence and JSON rejection. |

### Low Priority / Skip
None of these six files is generated. Do not expand into inference providers, predictors, samples, comparison tools, or abstraction tests.

## Existing Tests & Coverage Classification
The mandated analyzer was **executed exactly once**, rooted ONLY at `src/DecisionInference.Arrow`:

```text
python -B C:\Users\luquinta\.copilot\installed-plugins\dotnet-agent-skills\dotnet-test\skills\find-untested-sources\scripts\find_untested_sources.py C:\Dev\copilot-worktrees\typesafe-meai\luisquintanilla-didactic-pancake\src\DecisionInference.Arrow --lang csharp --include-tested
```

It exited **2**: `ERROR: tree-sitter-language-pack is not installed.` No installs, restores, analyzer retries, or JSON-file writes followed. Roslyn's source script requires `Microsoft.CodeAnalysis.CSharp@4.14.0` (that dependency is cached); no ready-to-run analyzer binary was found in the skill scripts, and offline Roslyn execution without a file-app build/restore/other writes was not established. This is an analyzer blocker, not evidence that the product cannot build.

- **Per target pairing**: All six target paths above have **unknown paired-test/coverage classification** because no pairing JSON was produced. Do not label them untested/partial/substantial or invent percentages.
- A target-only analyzer root would not see tests outside that root even if it ran. Parent's separate convention/project inspection supplies that context.
- Static pairing is a heuristic, never evidence of line/branch coverage. No test source was manually inventoried.

## Existing Test Projects
- No existing test project referencing Arrow was found by the narrow test-manifest reference search.
- **Intended project**: `tests/DecisionInference.Arrow.Tests/DecisionInference.Arrow.Tests.csproj` (absent at inspection).
- **Target source project**: `src/DecisionInference.Arrow/DecisionInference.Arrow.csproj`.
- **Solution registration**: Intended path is already present in `eng/DecisionArrowExperiments.slnx`, not the root solution.
- **Test files / conventions**: Not inspected; parent owns this step. No sibling project's tests are research targets.

## Exact Public API Map
Namespace for all APIs below: **`DecisionInference.Arrow`**.

**Concurrent parent updates, confirmed during final validation:** provenance now
also includes optional `ModelInputShapes : IReadOnlyDictionary<string,int[]>?` and
`AuxiliaryAssetHashes : IReadOnlyDictionary<string,string>?`. Writer reserves its
two measurement keys before output-directory creation, and deletes the partial
record before publishing the completed manifest. The suite tests these current
APIs without editing production.

| Type | Public API |
|---|---|
| `sealed DecisionArrowIdentity` | Constructor `(string provider, string modelId, IEnumerable<KeyValuePair<string,string>> assetHashes, string preprocessing, string decoder, IEnumerable<string> featureProjection)`. Get-only `Provider`, `ModelId`, `Preprocessing`, `Decoder : string`; `AssetHashes : IReadOnlyDictionary<string,string>`; `FeatureProjection : IReadOnlyList<string>`. |
| `sealed DecisionArrowSchema` | Constants `Version = 1`, `MetadataPrefix = "decision_arrow."`. Constructor `(IEnumerable<DecisionQuestion> questions, DecisionPrecision precision, DecisionArrowIdentity identity)`. Get-only `Questions : IReadOnlyList<DecisionQuestion>`, `Precision`, `Identity`, `CanonicalJson : string`, `FeatureFingerprint : string`, `Schema : Apache.Arrow.Schema` (fresh schema per access). Static `FromCanonicalJson(string json)` returns schema; `ParseQuestions(JsonElement questions)` returns `IReadOnlyList<DecisionQuestion>`. |
| `sealed DecisionArrowRow` | Constructor **`(long rowId, DecisionInput input, DecisionResult result)`**; get-only `RowId`, `Input`, `Result`. Do not construct it with answers alone. |
| `sealed record DecisionArrowPackageArtifact` | Positional `(string Id, string Version, string Sha256, long Size)`. |
| `sealed record DecisionArrowProvenance` | Required init `SourceRepository`, `SourceCommit`, `InputSha256`, `QuestionsSha256`, `Machine`, `Runtime`, `ExecutionMode : string`; required `Packages : IReadOnlyList<DecisionArrowPackageArtifact>`. Optional init `NativeStateBatchSize`, `CpuThreads : int?`; `Measurements : IReadOnlyDictionary<string,double>?`. |
| `sealed record DecisionArrowManifest` | Init `SchemaVersion : int = 1`; required `Status`, `ContractJson`, `FeatureFingerprint`, `DataFile : string`; optional `DataSha256 : string?`, `DataSize : long?`, `Error : string?`; required `RowCount : long`, `OutputBatchSize : int`, `Provenance`. Static get-only `JsonOptions : JsonSerializerOptions` (camelCase, indented, unknown-member rejection). |
| `static DecisionArrowWriter` | `DefaultBatchSize = 256`. **`Task<DecisionArrowManifest> WriteAsync(string directory, DecisionArrowSchema contract, IAsyncEnumerable<DecisionArrowRow> rows, DecisionArrowProvenance provenance, int batchSize = 256, CancellationToken cancellationToken = default)`**. |
| `sealed DecisionArrowDatasetReader : IArrowArrayStream` | Get-only `Manifest`, `Contract`, `Schema`. **`Task<DecisionArrowDatasetReader> OpenAsync(string manifestPath, DecisionArrowSchema expectedContract, IEnumerable<long> expectedRowIds, CancellationToken cancellationToken = default)`**; overload omits expected IDs. `ValueTask<RecordBatch> ReadNextRecordBatchAsync(CancellationToken cancellationToken = default)`; `void Dispose()`. End of stream is null despite the non-nullable public signature. |

Internal `DecisionArrowBatchBuilder`, `QuestionJson`, hash helpers and validators are not test-facing APIs. The reader's first argument is the **manifest file path**, not its directory. Acceptance should use the expected-ID overload.

## Canonical Contract, Metadata, and Schema

### Canonical JSON / semantic identity
Canonical serialization is compact System.Text.Json `Utf8JsonWriter` output, no whitespace normalization. Root property order is exactly:

```text
schemaVersion, numericRepresentation, precision, questions,
provider, modelId, assetHashes, preprocessing, decoder, featureProjection
```

- Literal `schemaVersion: 1`, `numericRepresentation: "float64"`.
- Precision strings exactly **`"HighPrecision"`**, **`"FourDecimalPlaces"`**, **`"TwoDecimalPlaces"`**.
- Questions remain in declared order. Every question object's first properties are **`id`, `kind`, `instructions`**; kind literals **`"binary"`, `"choice"`, `"score"`**.
- Binary then emits **`trueDescription`, `falseDescription`**, including explicit JSON nulls.
- Choice then emits **`candidates`** in declared order, each object **`id`, `description`** in that order.
- Score then emits **`rubric`** in declared order.
- Asset hashes are copied/sorted by **ordinal asset key**; dictionary insertion order must not change canonical output/fingerprint.
- Feature projection retains caller order and exact strings. Coordinates are binary **`questionId`**, or choice/score **`questionId[zeroBasedIndex]`**. Subsets and an empty projection are currently permitted; duplicate/unknown coordinates are not.
- All accepted IDs/descriptions/instructions/provider/model/asset keys/preprocessing/decoder/projection strings retain exact case and whitespace. Distinguish null descriptions from strings. Assert decoded strings as well as a manually specified golden JSON to cover escaping without accidental trimming.
- Fingerprint is **lowercase SHA-256 of UTF-8 canonical JSON bytes**, exactly 64 hex characters. Package versions, artifact hashes/sizes, source commits, machine/runtime, row IDs, output batch size and measurements are **not** fingerprint inputs.
- Changes to question order/content, candidate ID/order/description, rubric order/content, precision, provider/model, asset key/hash, preprocessing/decoder or projection order/content must change semantic identity (choose distinguishable fixtures).
- `FromCanonicalJson` reconstructs and requires **ordinal byte-for-string equality** with generated canonical JSON: whitespace/property-order changes, omitted optional-null properties, extra properties and alternate encodings are rejected. `ParseQuestions` alone is a parser, not that canonicality check.

### Exact Arrow metadata
Schema has exactly these owned entries:

```text
decision_arrow.version             = "1"
decision_arrow.contract            = CanonicalJson
decision_arrow.feature_fingerprint = FeatureFingerprint
```

Each question field has exactly:

```text
decision_arrow.definition = compact question JSON using the order above
decision_arrow.precision  = one of the three exact precision strings
```

`row_id` and nested child/item fields have no owned metadata. No `jevaro.*` or custom extension registration is used. Dictionary key insertion order is not a reader requirement; exact key/value sets are.

### Fixed physical schema (field order is contractual)
| Field | Type / children in exact order | Nullability |
|---|---|---|
| `row_id` (first) | `Int64` | Non-null |
| Binary question ID | `Double` (`float64`) | Non-null |
| Choice question ID | `Struct`: `selected_index : Int32`; `probabilities : FixedSizeList<Double>[candidateCount]` | Struct, both children, list elements non-null |
| Score question ID | `Struct`: `expected_index : Double`; `expected_index_lower_bound : Double`; `expected_index_upper_bound : Double`; `native_score : Double`; `probabilities : FixedSizeList<Double>[rubricCount]` | Struct and all fields/elements non-null **except nullable `native_score`** |

Every probability list's value field is named **`item`**, type `Double`, non-nullable. Question fields follow declared question order. Selected index is a **zero-based ordinal into candidate IDs**, not an argmax or ML.NET key.

## Fixture Construction: Immutable Abstractions Only
Namespace **`DecisionInference`**:
- `DecisionKind`: `Binary`, `Choice`, `Score`.
- `BinaryDecisionQuestion(string id, string? instructions = null, string? trueDescription = null, string? falseDescription = null)`.
- `ChoiceDecisionQuestion(string id, string? instructions, IEnumerable<DecisionCandidate> candidates)`; candidate constructor `(string id, string description)`.
- `ScoreDecisionQuestion(string id, string? instructions, IEnumerable<string> rubric)`; at least two meaningful rubric strings.
- `DecisionInput(JsonElement state, IEnumerable<DecisionQuestion> questions)` clones state and snapshots the question list. `JsonSerializer.SerializeToElement(new { id })` is sufficient explicit state.
- `BinaryDecisionAnswer(string questionId, double trueProbability, DecisionPrecision precision = HighPrecision)`.
- `ChoiceDecisionAnswer(string questionId, string selectedCandidateId, IEnumerable<KeyValuePair<string,double>> probabilities, DecisionPrecision precision = HighPrecision)`.
- `ScoreDecisionAnswer(string questionId, IEnumerable<double> probabilities, double? nativeScore = null, DecisionPrecision precision = HighPrecision)`.
- `DecisionResult(DecisionInput input, string modelId, IEnumerable<DecisionAnswer> answers, UsageDetails? usage = null)` validates association and orders answers by input questions.
- Construct **`new DecisionArrowRow(id, input, result)`**. `RowId` accepts every `long`, including zero, negative values and extremes.

Questions/candidates have **get-only** properties and snapshot their collections; no object initializer can set IDs, kinds, instructions or candidate/rubric lists. Schema also copies its question list; identity copies hashes/projection. Mutating original lists/dictionaries after construction must not alter these retained definitions. Rows retain the exact input/result references. Do not invent mutable question APIs or use reflection/uninitialized payloads.

Abstraction constructor constraints matter when choosing fixtures: ordinal unique nonblank IDs; nonempty question/candidate sets; binary needs meaningful instructions or both descriptions; choice defaults instructions to `"Choose one option."`, score to `"Assess the state against the ordered rubric."` when null; choice/score non-null instruction strings are retained verbatim. Probabilities must be finite in [0,1].

### Fidelity fixtures (independent expected observations)
- Mixed three-question row; supply result answers in reverse question order to prove association ordering. Supply choice probability pairs in an order different from candidate order.
- Choice candidate IDs such as `["z", "A", "a"]`, probabilities in schema order `[0.1, 0.8, 0.1]`, **selected `"a"` → stored index 2** (not argmax index 1). Preserve case-sensitive IDs in metadata. One-candidate choice is also valid.
- High-precision binary **`0.12345678901234567`**: compare retained double bits, not a loose tolerance or a cast to float32. Add exact four-/two-decimal binary fixtures **`0.2718`**, **`0.27`**.
- Rounded unnormalized-but-feasible vectors: two-decimal **`[0.33,0.33,0.33]`**, four-decimal **`[0.3333,0.3333,0.3333]`**. Preserve observations, never renormalize.
- Score high precision **`[0.25,0.5,0.25]`**: expected 1, bounds 1 ± `2e-6`; test native 1 and native null separately.
- Rounded score beyond rubric maximum: two-decimal **`[0,0.01,1]`** → observed expectation **2.01**, both bounds **1.995**, native **2** valid; four-decimal **`[0,0.0001,1]`** → observed expectation **2.0001**, both bounds **1.99995**, native **2** valid. These must round-trip; do not clamp expected index or reinterpret bounds as confidence intervals.
- High-precision distribution tolerance is absolute **`1e-6`**. Rounded observations use grid feasibility intervals with half steps **0.005** / **0.00005**, clipped to the domain, with inclusive ties; invalid grid/domain/distribution/native values should be injected through raw IPC rather than bypassing validated answer constructors.

`Input.State`, usage, raw representations and additional properties are not Arrow row fields. Provider is a dataset contract identity, not a property on `DecisionResult`; a per-row provider-change fixture cannot be constructed through this API. Test provider mismatch through expected contract/metadata instead.

## Validation Branch Map

### Identity / schema / row / writer
| Entry | Reject / branch | Observable result |
|---|---|---|
| Identity | Null/blank provider, modelId, preprocessing, decoder; null assets/projection | Argument guards; strings are not trimmed |
| Identity | Blank asset key, hash not 64 lowercase `[0-9a-f]`, duplicate asset key, zero assets | `ArgumentException` (duplicate via sorted dictionary) |
| Identity | Blank/null or ordinal duplicate projection coordinate | `ArgumentException`; empty projection currently allowed |
| Schema constructor | Null questions/identity; undefined precision | Argument null / out-of-range |
| Schema constructor | Empty/null-containing question list, ordinal duplicate question ID, exact reserved ID **`"row_id"`** | `ArgumentException`; `"ROW_ID"` is not reserved |
| Schema constructor | Projection references an unknown coordinate | `ArgumentException` |
| Canonical parser | Invalid JSON, missing/wrong-typed properties, unsupported version/numeric representation/kind/precision, noncanonical serialization | Version/numeric/kind/noncanonical branches explicitly use `InvalidDataException`; JSON/property/enum/constructor failures can have their native exception types |
| Row constructor | Null input/result | `ArgumentNullException`; mismatched association is intentionally not checked here |
| Writer guards | Blank directory; null contract/rows/provenance; batch size outside **1..65536** | Argument guards, before failure-record try block |
| Writer output directory | Any existing file **or subdirectory**, including a previous completed dataset | `IOException`, no replacement; existing empty directory allowed |
| Writer loop | Null row; duplicate source ID including across batches; cancellation | Null guard / `DecisionProtocolException` / cancellation; no placeholder rows |
| Builder append | Result model differs from contract (ordinal) | `DecisionProtocolException("Model identity changed within the dataset.")` |
| Builder append | Input question count/order/serialized definition differs | `DecisionProtocolException("Question definitions changed within the dataset.")` |
| Builder append | Result was built against another input with missing/extra answers, ID/kind mismatch, candidate ID-set mismatch or score vector width mismatch | Reconstructs `new DecisionResult(row.Input, ...)`; `DecisionProtocolException` |
| Builder append | Any answer's precision differs | `DecisionProtocolException("Precision changed within the dataset.")` |
| Builder construction | `checked(capacity * width * sizeof(double))` overflows | `OverflowException`, failed record; exercise safely via one choice with **4096 candidates**, batch size **65536**, and zero rows (check occurs before vector allocation) |

For association rejection, make a **valid result against a different valid input**, then pair it with the contract's input in the row. Missing/duplicate/null answers cannot exist inside a normally constructed result; don't duplicate abstraction-only constructor tests as Arrow coverage. Unknown derived question/answer kinds are not externally constructible (`private protected` bases plus sealed concrete classes).

### Reader: all validation occurs in `OpenAsync`, before any reader is returned
Use **`OpenAsync(manifestPath, expectedContract, expectedIds, token)`**.

| Layer | Corruption / rejection branch |
|---|---|
| Expected IDs / arguments | Null expected IDs/contract; duplicate expected IDs → `ArgumentException` with the ID; enumerable exceptions propagate. Manifest path errors come from file APIs. |
| JSON envelope | Missing file, malformed JSON, unknown property, missing required members, JSON null manifest. JSON null explicitly raises `"Missing manifest."`; serializer/file failures retain native types. |
| Completion | SchemaVersion != 1; status != exact **`"complete"`**; non-null Error; RowCount < 0; OutputBatchSize < 1 or > 65536; null DataSize; invalid/missing/uppercase DataSha256 → incomplete/invalid manifest. |
| Feature identity | ContractJson invalid/noncanonical; its recomputed fingerprint differs from manifest fingerprint; canonical contract differs from expected contract. Covers changed questions, candidate/rubric order/content, provider/model, assets, precision and projection/semantics. |
| Filename | Blank, path-bearing or anything other than exact **`"decisions.arrow"`** rejected. Resolve relative to manifest directory. |
| File integrity | Actual length != DataSize; SHA-256 != DataSha256; length < 8; absent/wrong terminal marker. Marker is exactly **`FF FF FF FF 00 00 00 00`**. Missing data file also rejects. |
| Schema/metadata | Schema metadata key/value set or field count differs; any field name/order/type/nullability/metadata differs; struct child count/order differs; fixed-list width or item name/type/nullability/metadata differs. Runs per batch **and on the stream schema for zero rows**. Extra metadata is a mismatch, except null and empty dictionaries compare equivalent. |
| Batches / row IDs | Batch exceeds declared output bound; wrong column count; top-level array length mismatch or any top-level null; duplicate source IDs across/within batches; unexpected IDs; final RowCount mismatch; expected set not exactly equal to seen set (missing IDs, including expected nonempty vs zero rows). |
| Binary | Null/nonfinite observation → `InvalidDataException`; finite value outside [0,1] or invalid declared rounding grid → abstraction `DecisionProtocolException`. |
| Choice vector | Null vector, null/nonfinite element, selected null/index < 0 or >= width; invalid probability domain, distribution sum/rounding feasibility or declared grid. Selection **need not be argmax**. |
| Score vector / fields | Null vector/element, nonfinite probabilities; invalid distribution/grid; null/nonfinite expected/bound fields; stored expectation or either bound != reconstructed value; nonfinite/out-of-domain/incompatible/off-grid native score. Legitimate native null is accepted. |
| Cancellation | Token is passed through manifest JSON, hash, marker and batch reads; failures dispose the internally opened stream/file and rethrow. |

The reader reuses abstraction answer constructors for probability/native validation, so do **not** universally assert `InvalidDataException`: some content errors are `DecisionProtocolException`, malformed IPC may be an Arrow exception, and malformed JSON may be `JsonException`.

## Batch, Stream and Publication Ownership
- Default source-row batch bound **256**, configurable **1..65536**. Required sequences: zero → **no record batches**, one → **[1]**, 256 → **[256]**, 257 → **[256,1]**; add several full batches plus a final short and a custom small bound. No empty placeholder batch.
- Builder retains only the current batch's numeric values; each written `RecordBatch` is disposed by writer and builder is replaced after a full batch. Source ID uniqueness still uses an **O(total rows)** hash set; reader similarly tracks IDs. Do not claim total constant memory.
- Writer writes schema via `WriteStartAsync` **before enumerating rows**, and writes end marker even for zero rows. An async generator can check partial artifacts and absence of completed manifest before its first row. Partial data is opened with `FileShare.None`: do not try to read it concurrently to inspect schema.
- Publication order: `run.partial.json` (status **`"partial"`**) → `decisions.arrow.partial` with schema/batches/end → flush including disk → hash → rename to **`decisions.arrow`** → publish **`manifest.json`** via **`manifest.json.tmp`** rename/replace → delete `run.partial.json`. Success returns the same completed contents; final SHA, size/count/bound/contract/provenance must agree.
- All JSON publications use `path + ".tmp"`, flush to disk, then `File.Move(..., overwrite: true)`. Completed manifest is the last **published dataset artifact**; deletion of the partial record follows.
- Failure/cancellation in the try block records status exactly **`"failed"`** / **`"cancelled"`** in `run.partial.json`, Error exactly **`"{exception.GetType().Name}: {exception.Message}"`**, observed accepted-row count, null DataSha256/DataSize. Normal catch uses `throw;`: assert **same exception instance** for an injected producer exception, not just its type.
- Any `OperationCanceledException` is classified cancelled, including an injected one. Use both pre-cancelled token and deterministic producer-triggered cancellation; honor/observe enumerator disposal. Argument/nonempty-directory guards occur outside the catch and need not create failure records.
- If recording also fails, writer throws **`AggregateException`** with original and recording errors in that order; original must remain accessible. Failure records are not completed manifests; no success-shaped rows or completed manifest on normal producer failure/cancellation.
- New exporter measurements are merged into a copied ordinal dictionary: exact keys **`exportBeforeManifestPublicationMilliseconds`**, **`exportProcessAllocatedBytes`**. Caller measurements are retained and not mutated. Assert presence/finiteness/nonnegative observations, **not exact timing/allocation numbers**.
- Reader validates hash and **every batch/ID/content** before returning, disposes validation batches, rewinds, and opens a new Arrow reader. First exposed batch must therefore be the dataset's first batch. A corrupt final batch must fail `OpenAsync`, not a later read.
- Reader owns underlying file/Arrow reader; **each returned batch belongs to the caller**. Access through `IArrowArrayStream`, retain an earlier batch while advancing and after stream disposal, then dispose retained batches yourself. Null at EOS, including zero-row datasets. Use `using`, not an invented async-dispose API.
- On Windows the data file is opened `FileShare.Read`, protecting the validated file from concurrent writers/deletion for the reader lifetime. After failed `OpenAsync`/reader disposal, a test should be able to exclusively reopen/delete data.

## Corruption Fixture Method
Create valid datasets first with public APIs; for payload/schema corruption use a separate helper that writes **real raw IPC** using `ArrowStreamWriter`, `RecordBatch`, `Int64Array`, `DoubleArray`, `Int32Array`, `StructArray`, and `FixedSizeListArray`. Keep schema metadata canonical unless metadata itself is the target. Use well-formed array buffers/lengths so the intended validator, not accidental construction failure, is reached.

After editing IPC content/schema, update **both DataSize and lowercase DataSha256 in the manifest** independently; otherwise every test only exercises integrity short-circuiting. Conversely, intentionally leave them wrong in length/hash cases. Use manifest `with` copies and `DecisionArrowManifest.JsonOptions`; change one invariant at a time. Don't use the validated writer to attempt construction of intrinsically invalid answers.

For every relevant branch include a corrupt **later batch** after a valid first batch, asserting that `OpenAsync` itself rejects without exposing the first batch. For metadata validation with zero rows, write a schema-only IPC stream with EOS. Exact expected IDs are sets (enumeration order need not equal file row order); stored rows must retain producer order.

## Testing Patterns
- Use requested xUnit v3 `[Fact]`/`[Theory]`, async Task tests, explicit expected metadata strings and schema children, exact numerical/bit assertions, and exception-instance assertions.
- Unique temp directory per test; cleanup in finally/fixture disposal. No shared dataset, global culture/serializer mutation, timing sleeps, network or model dependencies.
- Small deterministic producers allow yield-time assertions, throw/cancel after a known count, and signal enumerator disposal. Use cancellation tokens explicitly.
- Golden canonical JSON/hash should have independently specified expected structure/content, not be generated by the very API under test. Inspect physical Arrow arrays as well as returned manifests.
- Cover immutable snapshots and exact strings with caller collections mutated **after** construction, mixed case, leading/trailing spaces, and meaningful escaped/Unicode text.
- Parent reads conventions; no test examples or unrelated test files were consulted here.

## Acceptance Checklist
- [ ] Canonical metadata keys/values, property/question/candidate/rubric order and exact untrimmed strings; deterministic asset-key order and lowercase UTF-8 SHA-256.
- [ ] Provider/model, asset hashes, preprocessing/decoder, numeric representation, all three precisions and projection semantics/order affect feature identity; package/source/provenance version changes alone do not.
- [ ] Reserved `row_id`, duplicate question/candidate/asset/projection/source/expected IDs; ordinal/case-sensitive identities; valid negative/zero/extreme long row IDs.
- [ ] Exact float64 probability preservation; candidate-ID lookup with non-argmax selected index and probability input enumeration reordered.
- [ ] Exact portable score expectation/bounds and native/null preservation, including both rounded expected-outside-range examples; no normalization/clamping/confidence invention.
- [ ] All three precision declarations accepted and exact metadata retained; corrupted grid/domain/nonfinite/distribution/native values rejected.
- [ ] Default 256 output bound: zero, one, 256, 257, multiple-full/final-short and configurable bound; checked allocation exercised without giant allocations.
- [ ] Schema before row enumeration; finalized data/hash first and completed manifest published last; SHA/size/count/filename/status agree.
- [ ] Producer failure and cancellation record original exception/status/count, no completed manifest or success-shaped placeholder; exceptional recording failure preserves original as aggregate inner exception.
- [ ] Reader rejects incomplete/failed/cancelled envelope, hash/size/end-marker, schema/metadata, batch/count/ID sets, feature contract, vector/selected/null/nonfinite/score corruption.
- [ ] `OpenAsync(path, expectedContract, expectedIds, token)` completes all validation before exposure, including corrupt final batch and empty-stream schema.
- [ ] Standard `IArrowArrayStream` compatibility, first batch replayed after validation, caller-owned batches and disposed file handles.
- [ ] Existing files/subdirectories/completed datasets protected byte-for-byte; empty existing output directory accepted.
- [ ] Immutable abstraction fixtures and collection snapshots; every row uses `new DecisionArrowRow(id,input,result)`.
- [ ] New tests discoverable through the agreed external harness entry point; explicitly resolve/report the root/default solution gap.

## Recommendations / Bugs and Blockers (Report, Do Not Fix)
1. Implement schema/identity/fidelity tests first, then writer batch/publication/failure tests, then the reader corruption matrix; row/manifest checks are small supporting tests. No production seams are required.
2. **Static publication bug:** `DecisionArrowWriter.cs:81-82` publishes `manifest.json` then calls `File.Delete(run.partial.json)` inside the same try. A deletion error enters failure handling **after a completed manifest exists**, and the catch does not remove/invalidate it. Failure recording may also aggregate because the same file is locked. This contradicts the unconditional no-completed-manifest-on-error gate. Report a failing acceptance case to parent; don't make the test bless the bug.
3. **Measurement collision behavior:** the two new exporter measurement keys are added with `Dictionary.Add` after final data rename. Supplying either key already in provenance causes `ArgumentException` and a failed export; there is no documented guard/reservation. On this or another post-rename/pre-manifest error, the failure record may still name `decisions.arrow.partial` although data has moved to `decisions.arrow`. Report behavior; no fix applied.
4. **Unverified reader risk:** it checks the physical last eight bytes, but never verifies parser position at first EOS equals file length. Whether extra bytes followed by another EOS are accepted depends on the pinned Arrow reader; add an offline probe if broader corruption acceptance includes trailing content. No runtime conclusion is claimed.
5. **Observed boundaries, not invented gates:** provenance/package/hash/measurement records have no semantic validation beyond JSON required members; the reader does not verify provenance hashes against source files. Empty projections are accepted, and ID syntax can create coordinate aliases (e.g. binary `q[0]` vs choice `q` coordinate). These must not be silently converted into new rejection requirements without parent approval.
6. **Execution blockers:** intended Arrow test project absent; root/default harness omits Arrow; pinned experimental Arrow package not in the default user NuGet cache; analyzer dependency missing in the invoked Python engine. Parent provides isolated artifact/restore setup. No build, tests, benchmark, coverage, dependency/config change, or compile fix was attempted.
7. Section 9.1's study needs equivalent JSON/Arrow portable content (same IDs/probabilities/score fields, shared metadata/envelopes accounted separately), zero/one/short/larger deterministic datasets, and separate cold/warm encoding/decoding/bytes/allocations/peak-memory/summary measurements. This research supports fixtures, **not performance threshold assertions or inference comparisons**. No Python/Jevaro/cross-language reader is required.

### Final dispositions

- Findings 2 and 3 above describe the initial research snapshot, not remaining
  defects. Parent-owned updates addressed both. The cleanup sharing-lock acceptance
  test and reserved-measurement prepublication tests pass.
- Initial absent-project/restore blockers in finding 6 are resolved; the mandated
  analyzer's missing offline dependency remains explicitly recorded.
- Trailing content after a first IPC EOS (finding 4), coordinate aliases and
  provenance semantic validation (finding 5) are not enumerated acceptance gates.
  No unsupported production bug or mutation-score claim is made for these areas.
- Public Arrow serialization rejects a missing-column `RecordBatch` before reader
  input can be produced. A named fixture-feasibility test records that boundary,
  not reader column-count coverage. Other structural/length corruption reaches
  the reader with independently refreshed integrity.
