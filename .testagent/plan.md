# Test Implementation Plan

## Overview

Implement a broad, offline xUnit v3 suite for **only `src/DecisionInference.Arrow`**, in **one substantial implementation phase**. All six source files belong to that phase. Coverage classifications remain **unknown**, not “untested”: the source-pairing analyzer ran once and was blocked by its missing tree-sitter package. Do not retry it, install dependencies, or invent coverage percentages.

This plan uses only `.testagent/research.md` and the caller's confirmed conventions. It does not authorize production changes. The parent/user owns `tests/DecisionInference.Arrow.Tests`, including its project manifest and restore setup. No new, separate test project is needed.

- Test namespace: **`DecisionInference.Tests`**, matching the confirmed sibling convention.
- Framework/packages: centrally managed `xunit.v3` **3.2.2**, `xunit.runner.visualstudio` **3.1.5**, `Microsoft.NET.Test.Sdk` **17.14.1**.
- Parent-owned SDK test project: `OutputType=Exe`, `IsTestProject=true`, `IsPackable=false`, `Using Include="Xunit"`, and a project reference to **Arrow only**. Arrow supplies the transitive abstraction dependency. Inherit nullable, warnings-as-errors, and `net10.0`; do not change props or central versions.
- Use `[Fact]`/`[Theory]`, async `Task` tests, real local Arrow IPC, immutable abstraction fixtures, and exact assertions. No mocking, networking, providers, inference, native binaries, coverage setup, performance thresholds, or cross-language readers.
- This planning task creates **only `.testagent/plan.md`**. Later implementation is confined to the parent-approved Arrow test files. Do not edit production, props, `eng`, samples, other tests, solutions, or version control.
- Do the work directly, sequentially within this phase; **no agents, delegation, or nesting**. Do not reload the language base extension already summarized in research.

## Commands

The initial planning snapshot awaited execution arguments. The parent subsequently
supplied the verified restore helper and shell-local
`$env:ImportDirectoryBuildTargets='false'` workaround. The one implementation
phase and final selected-workspace validation are now complete; exact results and
test-name acceptance mappings superseding estimates below are in `status.md`.
No feeds, root/global settings, packages or solution entries were independently changed.

| Purpose | Command after parent setup | State |
|---|---|---|
| Restore | Parent supplies the exact command/arguments; no substitute command planned | Waiting |
| **Build: scoped fix cycles** | `dotnet build tests/DecisionInference.Arrow.Tests/DecisionInference.Arrow.Tests.csproj --no-restore -v:q` | Waiting for supplied arguments/restored assets |
| **Test: scoped fix cycles** | `dotnet test tests/DecisionInference.Arrow.Tests/DecisionInference.Arrow.Tests.csproj --no-restore -v:q` | Waiting for supplied arguments/restored assets |
| Selected-workspace discovery | `dotnet test eng/DecisionArrowExperiments.slnx --list-tests --no-build --no-restore` | Waiting; run only after a successful build |
| **Build: final full selected workspace** | `dotnet build eng/DecisionArrowExperiments.slnx --no-restore --no-incremental -v:q` | Waiting for supplied arguments and parent confirmation of this entry point |
| **Test: final full selected workspace** | `dotnet test eng/DecisionArrowExperiments.slnx --no-restore -v:q` | Waiting; fresh build required, no `--no-build` |
| Root/default discovery comparison, if requested by parent | `dotnet test TypeSafe.MEAI.slnx --list-tests --no-build --no-restore` | Waiting; not an Arrow validation substitute |
| **Lint** | No scope-specific command established in research | Not established; do not run a formatter that changes sources |

`eng/DecisionArrowExperiments.slnx` already registers the intended Arrow test project; the root `TypeSafe.MEAI.slnx` does not. Do not edit either solution. The parent must agree that the isolated solution is the external harness entry point, or report the root/default discovery gap explicitly. Record the new test identities and discovery-count delta, not just a successful targeted run. If the parent selects another workspace entry point, use its supplied full build/test commands instead of silently treating root discovery as sufficient.

## Phase Summary

| Phase | Focus | Files | Est. Tests |
|---|---|---|---|
| 1 | Support helper; schema/identity; writer, indirect builder, fidelity and lifecycle; adversarial eager reader validation | All 6 target source files; 1 helper and 9 test files | Approximately 65–80 named methods and 200–300 executions after theory expansion; optional/blocker probes reported separately |

The estimate counts theory executions separately from methods. Centralized fixture/corruption case data should avoid hundreds of handwritten test bodies.

---

## Phase 1: Complete Offline Arrow Contract and Transport Tests

### Overview

Start with deterministic support and leaf contracts, then exercise the internal builder through the public writer, and finally validate reader rejection using independently written IPC. This is implementation order **inside one phase**, not additional phases. Every public type/member is covered directly or by a named round-trip/serialization test; the internal builder receives public-API coverage without reflection or production seams.

### Support Helper

- **File**: `tests/DecisionInference.Arrow.Tests/DecisionArrowTestSupport.cs`
- **Type**: internal `DecisionArrowTestSupport`, with a disposable test-local temporary-directory owner as needed.
- This is test support, not another production target or project.

Implement these small, reusable operations:

1. `CreateContract` / `CreateProvenance`: immutable mixed binary/choice/score fixtures, valid lowercase asset hashes, explicit question/candidate/rubric order, and meaningful strings. Parameterize precision and semantic mutations. Do not invent validation of provenance/package values.
2. `CreateRow`: construct valid `DecisionInput`, valid `DecisionResult`, and **`new DecisionArrowRow(id, input, result)`**. Use `JsonSerializer.SerializeToElement` for state. Reverse answer enumeration and reorder choice probability pairs deliberately. For mismatched association, pair the contract's input with a result valid against a different valid input.
3. `ProduceRows`: small deterministic async generator with before-first-yield, after-known-yield, exhaustion, cancellation, and disposal signals. Exceptions are test-owned instances. No sleeps, races, wall-clock assumptions, or shared state.
4. `WriteValidDatasetAsync`: call the public writer into a unique directory; return paths and expected IDs. The reader always receives the **manifest path**, never the directory.
5. `WriteRawIpcAsync`: write real IPC using public `ArrowStreamWriter`, `RecordBatch`, `Int64Array`, `DoubleArray`, `Int32Array`, `StructArray`, and `FixedSizeListArray` APIs. Expose explicit null bitmaps, selected indices, score fields, vector values, and ordered schema/child/metadata mutations. Build structurally coherent buffers so the chosen validator is reached.
6. `RewriteManifest` / `RefreshDataIntegrity`: serialize manifest `with` copies using `DecisionArrowManifest.JsonOptions`; independently calculate actual UTF-8/file SHA-256 and byte length. **After every schema/payload rewrite, update both `DataSha256` and `DataSize`**, except when integrity itself is the tested fault.
7. `ReadPhysicalRowsAsync` / `AssertSchemaMatchesGolden`: inspect Arrow arrays, field order, child structure, nullability, full metadata key/value sets, and scalar bits. These assertions must use explicit expected data, not merely another schema generated by the subject.
8. `AssertDataFileReleased`: exclusively reopen the data file after failed open or disposal, then delete within the test directory. All returned batches and test-owned handles are explicitly disposed.

Use no reflection, uninitialized objects, byte-patching guesses, mutable question properties, invented async-disposal API, or unrelated abstraction-only suites. If a public Arrow constructor/writer refuses a proposed malformed layout, report that exact case as a fixture blocker; do not count its constructor failure as reader coverage.

### Files to Test

#### 1. DecisionArrowSchema.cs — identity, canonical contract and physical schema

- **Source**: `src/DecisionInference.Arrow/DecisionArrowSchema.cs`
- **Test Files**:
  - `tests/DecisionInference.Arrow.Tests/DecisionArrowIdentityTests.cs`
  - `tests/DecisionInference.Arrow.Tests/DecisionArrowSchemaTests.cs`
- **Test Classes**: `DecisionArrowIdentityTests`, `DecisionArrowSchemaTests`
- **Methods/API**: `DecisionArrowIdentity(...)` and all get-only properties; `DecisionArrowSchema(...)`, `Version`, `MetadataPrefix`, `Questions`, `Precision`, `Identity`, `CanonicalJson`, `FeatureFingerprint`, `Schema`, `FromCanonicalJson`, `ParseQuestions`.

**Identity test names and cases**

- `Constructor_RetainsExactStringsAndSnapshotsCollections`: preserve provider/model/preprocessing/decoder case and surrounding whitespace; mutate original hashes/projection after construction; retained values stay unchanged. Assert all properties and exact ordered projection.
- `Constructor_SortsAssetsOrdinally`: reverse asset insertion order; keys such as `A`, `a`, `z` sort ordinally, preserving exact values. Fingerprint/canonical output is insertion-order independent.
- `Constructor_RejectsInvalidIdentityInputs` theory: null/empty/whitespace provider, model, preprocessing and decoder; null asset/projection enumerable. Assert the specific argument exception and parameter where established.
- `Constructor_RejectsInvalidAssets` theory: zero assets, null/blank key, exact duplicate key, short/long/nonhex/uppercase hash. Distinct case-sensitive keys are valid.
- `Constructor_RejectsInvalidProjection` theory: null/blank entry and exact duplicate entry rejected. Empty projection is valid; case-sensitive validity of coordinates is checked by schema, not invented identity normalization.

**Independent golden**

Freeze the following compact, single-line literal as test data. Construct the equivalent objects separately, passing assets in reverse insertion order. Do not obtain the expected JSON from `CanonicalJson`, `FromCanonicalJson`, or another production serializer.

```json
{"schemaVersion":1,"numericRepresentation":"float64","precision":"HighPrecision","questions":[{"id":"bin","kind":"binary","instructions":" Decide ","trueDescription":null,"falseDescription":null},{"id":"pick","kind":"choice","instructions":" Pick ","candidates":[{"id":"z","description":"Zed"},{"id":"A","description":"Upper"},{"id":"a","description":"Lower"}]},{"id":"score","kind":"score","instructions":" Rate ","rubric":["low","middle","high"]}],"provider":" provider ","modelId":"Model","assetHashes":{"A":"0000000000000000000000000000000000000000000000000000000000000000","z":"1111111111111111111111111111111111111111111111111111111111111111"},"preprocessing":" prep ","decoder":" decoder ","featureProjection":["bin","pick[2]","score[0]"]}
```

- UTF-8 length: **746 bytes**, with no BOM or newline.
- Independently calculated lowercase SHA-256: **`51c5dcca8275938a792407753bf835fe95949f43a99a1281a649e9e4effebb12`**.
- `CanonicalJson_AndFingerprint_MatchIndependentGolden`: compare exact literal and fixed hash; independently hash the literal's UTF-8 bytes as a second check. Assert 64 lowercase hex characters, `Version == 1`, and `MetadataPrefix == "decision_arrow."`.
- `CanonicalJson_PreservesEscapedAndUntrimmedStrings`: use quotes, backslashes, newline, non-ASCII text, case differences and surrounding spaces in supported fields. Assert decoded strings against input literals, required explicit binary-description nulls, and ordered JSON properties. Include null-versus-string binary descriptions. Do not rely only on serializer round-tripping itself.

**Schema test names and cases**

- `Constructor_RejectsInvalidContracts` theory: null questions/identity; undefined precision; empty/null-containing questions; exact duplicate question ID; exact reserved `row_id`; unknown projection coordinate. Duplicate candidate IDs are prevented by the public immutable candidate/question construction used to supply schema fixtures; assert that constructor boundary without pretending invalid candidates reached Arrow.
- `Constructor_UsesOrdinalQuestionAndProjectionIdentities` theory: `ROW_ID` allowed; `q`/`Q` and candidate `A`/`a` distinct; projection coordinates use exact case and zero-based indices; empty/subset/full projections accepted; duplicate and unknown/out-of-width coordinates rejected. **No alias-rejection requirement** for IDs whose spelling creates coordinate aliases.
- `Constructor_SnapshotsQuestionCollections`: mutate original question list, candidate list and rubric list after their respective constructors; schema/definitions stay unchanged. Assert retained schema `Identity` and `Precision`.
- `FeatureFingerprint_ChangesForSemanticVariants` theory: independently mutate question ID, kind, order, instructions, binary true/false descriptions including null/string, candidate ID/order/description, rubric order/content, provider/model case and whitespace, asset key/hash, preprocessing, decoder, precision (all three), projection order/content/subset/empty. Each valid single semantic change yields a different canonical string and fingerprint.
- `Schema_HasExactPhysicalStructureAndMetadata` theory for all three precisions: assert the complete structure below, exact question order and metadata, and absence of extra/legacy keys.
- `Schema_ReturnsFreshEquivalentSchemas`: two `Schema` accesses are different instances but each matches the explicit physical/metadata contract.
- `FromCanonicalJson_RoundTripsCanonicalGolden` theory: golden HighPrecision plus explicitly specified four-/two-decimal variants; compare all reconstructed question/identity/projection values, canonical text, fingerprint and schema.
- `FromCanonicalJson_RejectsInvalidOrNoncanonicalContracts` theory: malformed JSON; missing/wrong-typed root or question properties; unsupported version, `float32`/other numeric representation, kind or precision; whitespace/property-order changes, omitted explicit null, unknown property, alternate equivalent encoding. Numeric representation is fixed to `float64`: **rejection of alternatives**, not an invented supported precision/representation.
- `ParseQuestions_ReconstructsAllKindsInOrder`: directly parse manually specified question JSON; inspect IDs, kinds, instructions, descriptions, ordered candidate IDs/descriptions and rubric.
- `ParseQuestions_RejectsInvalidQuestionJson` theory: unknown kind, malformed/missing/wrong-typed fields and invalid immutable question definitions. Keep native JSON/property/enum/constructor exception distinctions; `ParseQuestions` alone is not a canonicality checker.

**Full structural oracle**

| Position | Exact field/children | Type and nullability | Exact metadata |
|---|---|---|---|
| 0 | `row_id` | `Int64`, non-null | No owned entries |
| Declared binary position | Question ID | `Double`, non-null | `decision_arrow.definition` = independently specified compact binary JSON; `decision_arrow.precision` = exact enum name |
| Declared choice position | Question ID; children `selected_index`, `probabilities` | Non-null `Struct`; non-null `Int32`; non-null `FixedSizeList<Double>` of candidate count | Same two question entries; no child/item owned entries |
| Declared score position | Question ID; children `expected_index`, `expected_index_lower_bound`, `expected_index_upper_bound`, `native_score`, `probabilities` | Non-null `Struct`; first three `Double` non-null; `native_score` nullable `Double`; non-null fixed list of rubric count | Same two question entries; no child/item owned entries |
| Every list item | `item` | Non-null `Double` | No owned entries |

Schema metadata has **exactly** `decision_arrow.version="1"`, `decision_arrow.contract=golden canonical JSON`, and `decision_arrow.feature_fingerprint=golden hash`. Compare complete key/value sets, not metadata enumeration order. No `jevaro.*` or extension registration. Choice candidate JSON property order is `id`, `description`; binary includes explicit null descriptions; score preserves rubric order.

#### 2. DecisionArrowRow.cs — immutable row wrapper

- **Source**: `src/DecisionInference.Arrow/DecisionArrowRow.cs`
- **Test File**: `tests/DecisionInference.Arrow.Tests/DecisionArrowRowTests.cs`
- **Test Class**: `DecisionArrowRowTests`
- **Methods/API**: constructor, `RowId`, `Input`, `Result`.

**Tests**

- `Constructor_RetainsRowIdAndExactPayloadReferences` theory: `long.MinValue`, negative, zero, positive and `long.MaxValue`; `Assert.Same` for input/result.
- `Constructor_RejectsNullInputOrResult` theory: each null argument individually; exact `ArgumentNullException`.
- `Constructor_DefersAssociationValidationToWriter`: the row accepts a result valid against a different input; `WriteAsync_RejectsInconsistentRows` below proves Arrow rejects the association. Do not incorrectly require the row constructor to validate it.

#### 3. DecisionArrowManifest.cs — records and JSON envelope

- **Source**: `src/DecisionInference.Arrow/DecisionArrowManifest.cs`
- **Test File**: `tests/DecisionInference.Arrow.Tests/DecisionArrowManifestTests.cs`
- **Test Class**: `DecisionArrowManifestTests`
- **Methods/API**: `DecisionArrowPackageArtifact(...)`; provenance/manifest init properties, record copies/equality; `DecisionArrowManifest.JsonOptions`.

**Tests**

- `JsonOptions_SerializesExactEnvelopeAndRecordValues`: manually specified expected JSON property names and decoded values for every artifact/provenance/manifest property; camelCase and indented output; default `SchemaVersion=1`; complete/partial/failed/cancelled values; optional null fields and populated fields. Cover package ID/version/hash/size, repository/commit/input/questions hashes, machine/runtime/mode, native batch size/CPU threads/measurements. Compare property sets and important wire values, not just object equality.
- `Records_WithCopiesRetainOriginalEnvelopeAndProvenance`: positional artifact properties and record equality; manifest/provenance `with` variants do not replace original values. Do not assert deep immutability of caller-owned record dictionaries/lists where none is promised.
- `JsonOptions_RejectsUnknownAndMissingRequiredMembers` theory: unknown root/nested member, missing each required manifest/provenance member, malformed JSON. JSON `null` is the reader's `"Missing manifest."` case, not a serializer-required-member failure. No new provenance semantic/hash/source-file validation.

#### 4. DecisionArrowBatchBuilder.cs — public-path builder coverage

- **Source**: `src/DecisionInference.Arrow/DecisionArrowBatchBuilder.cs`
- **Test File**: `tests/DecisionInference.Arrow.Tests/DecisionArrowBatchBuilderTests.cs`
- **Test Class**: `DecisionArrowBatchBuilderTests`
- **Methods/API**: internal constructor, `Append`, `Build`, `Count`, exercised **only through `DecisionArrowWriter.WriteAsync`**.

**Tests**

- `WriteAsync_RejectsInconsistentRows` theory:
  - Model identity differs ordinally, including case: `DecisionProtocolException("Model identity changed within the dataset.")`.
  - Input question count/order/serialized definition differs: `DecisionProtocolException("Question definitions changed within the dataset.")`.
  - Result valid against another input has missing/extra answer IDs, wrong answer kind, differing choice candidate ID set, or differing score width: association reconstruction rejects with `DecisionProtocolException`.
  - Answer precision differs: `DecisionProtocolException("Precision changed within the dataset.")`.
  - Use only constructible valid immutable inputs/results; no invalid answer object bypass.
- `WriteAsync_RejectsNullRows`: generator yields `null!`; no placeholder batch or completed manifest, and failure record describes the original exception.
- `WriteAsync_RejectsDuplicateSourceIds` theory: duplicate within a batch and across a batch boundary, including negative/zero IDs; exact protocol failure and no completed manifest.
- `WriteAsync_ChecksVectorAllocationBeforeAllocating`: one choice with **4096 valid candidates**, batch size **65536**, zero source rows; checked multiplication raises `OverflowException` before giant numeric allocation; failed record, no completed manifest.
- `WriteAsync_ResetsBuilderAtFullBatchBoundary`: tiny custom bound and mixed rows over several batches; inspect lengths, IDs, values, order and final short batch. Complements the full writer-bound theory and exercises `Count`/`Build` without reflecting on internal state.

#### 5. DecisionArrowWriter.cs — publication, fidelity and lifecycle

- **Source**: `src/DecisionInference.Arrow/DecisionArrowWriter.cs`
- **Test Files**:
  - `tests/DecisionInference.Arrow.Tests/DecisionArrowWriterTests.cs`
  - `tests/DecisionInference.Arrow.Tests/DecisionArrowFidelityTests.cs`
- **Test Classes**: `DecisionArrowWriterTests`, `DecisionArrowFidelityTests`
- **Methods/API**: `DefaultBatchSize`, `WriteAsync` including default/custom batch sizes and token overload behavior.

**Writer tests**

- `WriteAsync_RejectsInvalidArgumentsBeforePublication` theory: blank/null directory, null contract/rows/provenance, batch sizes `0`, negative and `65537`. Guards do not need failure records; existing sentinel content is untouched.
- `WriteAsync_PreservesExistingOutputDirectoryContents` theory: existing file, existing subdirectory (with nested sentinel), previous complete dataset. Snapshot relative paths and file bytes before/after; expect `IOException` and byte-for-byte preservation, not merely “file still exists.”
- `WriteAsync_AcceptsExistingEmptyDirectory`: valid export into an already-created empty directory.
- `WriteAsync_EmitsBoundedBatchesAndExactManifest` theory:
  - Default `DefaultBatchSize == 256`: `(0, [])`, `(1, [1])`, `(256, [256])`, `(257, [256,1])`, `(775, [256,256,256,7])`.
  - Custom bound 2: `(5, [2,2,1])`; bound 1; accepted upper bound 65536 with a tiny dataset, not 65536 large rows.
  - Inspect real IPC; no empty placeholder batches. Empty data still has schema and exact EOS.
  - Assert returned/serialized completed contents, status `"complete"`, null error, filename `"decisions.arrow"`, exact contract/fingerprint/provenance/count/bound, independent lowercase SHA/size, and terminal bytes `FF FF FF FF 00 00 00 00`.
- `WriteAsync_WritesSchemaBeforeFirstProducerMoveNext`: throw a test-owned exception at the generator's first move before yielding. Once writer cleanup releases the partial file, read its Arrow schema and compare against the independent physical oracle. Do not attempt to read the exclusively opened partial file concurrently.
- `WriteAsync_PublishesOnlyPartialArtifactsWhileProducerIsActive`: before first yield, after a full batch, after a short batch and just before exhaustion, assert partial record/status and data-partial existence, no completed manifest/final published dataset. After success assert final data/manifest, matching hash/size/count, and removal of partial/temp publications. These yield-time checks must be deterministic, not watcher/sleep based.
- `WriteAsync_ProducerFailureRethrowsSameInstanceAndRecordsAcceptedCount` theory: fail before rows and after two accepted rows crossing a small batch boundary. Capture one sentinel exception, assert **same instance**, exact failed status/error string, accepted count, null data hash/size, disposal signal, no completed manifest or invented rows.
- `WriteAsync_ProducerCancellationRethrowsSameInstanceAndRecordsAcceptedCount` theory: inject a known `OperationCanceledException` instance, including one not caused by the supplied token; and cancel a source token deterministically after known accepted rows, capturing/rethrowing the producer-created exception. Assert same instance where injected/captured, `"cancelled"`, exact error/count, no completed manifest, enumerator disposal. No `CancelAfter`.
- `WriteAsync_PreCancelledTokenProducesCancelledRecord`: pre-cancelled token; `OperationCanceledException` with cancellation observed, cancelled record/count zero and no completed manifest. Do not demand identity with an exception that did not exist before invocation.
- `WriteAsync_RecordingFailurePreservesOriginalAsFirstAggregateInner`: while the producer is active, replace the test-local `run.partial.json` file with a directory at that same path, then throw the sentinel exception. Publication of the failed record must fail deterministically. Assert `AggregateException`, original **same instance at index 0**, recording error at index 1, and no completed manifest. Keep any filesystem handles outside enumerator disposal if a Windows sharing-lock alternative is used.
- `WriteAsync_MergesMeasurementsWithoutMutatingProvenance`: caller measurement dictionary survives unchanged; output contains its values plus exact keys `exportBeforeManifestPublicationMilliseconds` and `exportProcessAllocatedBytes`; both finite and nonnegative. Assert no exact timings/allocations or total constant-memory claim.
- `WriteAsync_CleanupFailureMustNotLeaveCompletedManifest`: **blocked acceptance/optional deterministic regression probe**, described under findings below. The desired assertion is no completed manifest on reported failure; do not assert the current buggy complete-plus-failure state as acceptable.

**Fidelity tests**

- `WriteRead_PreservesMixedQuestionOrderAndNonArgmaxCandidateId`: reverse answer enumeration; supply choice probability pairs out of candidate order; candidates `["z","A","a"]`, stored vector `[0.1,0.8,0.1]`, selected `"a"` gives stored index **2**, not argmax **1**. Inspect metadata and physical arrays, then open with expected IDs. Add one-candidate choice.
- `WriteRead_PreservesFloat64BitsAndDeclaredPrecision` theory:
  - HighPrecision binary `0.12345678901234567`.
  - FourDecimalPlaces binary `0.2718`.
  - TwoDecimalPlaces binary `0.27`.
  - Compare `BitConverter.DoubleToInt64Bits`, exact metadata declaration, `Double` field/item types, and decoded arrays. No float32 conversion or broad tolerance.
- `WriteRead_PreservesExactScoreFieldsAndNullableNative` theory uses the independent table below. Compare every stored probability, expected value and both bounds exactly/bitwise against explicit expected doubles. Test both listed native value and **null** for each precision.
- `WriteRead_PreservesFeasibleRoundedVectorsWithoutNormalization` theory: two-decimal `[0.33,0.33,0.33]`, four-decimal `[0.3333,0.3333,0.3333]` in choice and score vectors; sums remain observed, not renormalized. Additional feasible endpoint/half-step cases `[0,1]`, two-decimal `[0.5,0.49]`, four-decimal `[0.5,0.4999]`; high-precision tolerance-interior `[0.5,0.5000005]`. Native null avoids introducing a second validator.
- `WriteRead_PreservesAllLongIdsAndProducerOrder`: distinct `long.MinValue`, negative, zero, positive and `long.MaxValue` spread over batches; expected IDs enumerated in reverse/shuffled order. Exposed rows match producer order, not sorted-set order.
- `FeatureFingerprint_IsIndependentOfProvenanceAndTransport` theory: vary package ID/version/hash/size, repository/commit, input/questions provenance hashes, machine/runtime/execution mode, native batch size, CPU threads, measurements, row ID sets and output batch bound independently. Canonical contract/fingerprint stays the golden value while appropriate envelope/data fields change. No requirement to verify provenance against external files.

| Precision | Score probability vector | Exact expected index | Exact lower bound | Exact upper bound | Native cases |
|---|---|---|---|---|---|
| HighPrecision | `[0.25,0.5,0.25]` | `1.0` | `0.999998` | `1.000002` | `1.0`, null |
| FourDecimalPlaces | `[0,0.0001,1]` | `2.0001` | `1.99995` | `1.99995` | `2.0`, null |
| TwoDecimalPlaces | `[0,0.01,1]` | `2.01` | `1.995` | `1.995` | `2.0`, null |

The rounded expected indices intentionally exceed the rubric maximum. **Do not clamp, normalize, invent confidence intervals, or reject these valid observations.** Expected bounds must not be derived by constructing another `ScoreDecisionAnswer` inside the assertion.

#### 6. DecisionArrowDatasetReader.cs — eager validation and stream ownership

- **Source**: `src/DecisionInference.Arrow/DecisionArrowDatasetReader.cs`
- **Test Files**:
  - `tests/DecisionInference.Arrow.Tests/DecisionArrowDatasetReaderTests.cs`
  - `tests/DecisionInference.Arrow.Tests/DecisionArrowDatasetReaderCorruptionTests.cs`
- **Test Classes**: `DecisionArrowDatasetReaderTests`, `DecisionArrowDatasetReaderCorruptionTests`
- **Methods/API**: both `OpenAsync` overloads, `Manifest`, `Contract`, `Schema`, `ReadNextRecordBatchAsync`, `Dispose`, use as standard `Apache.Arrow.Ipc.IArrowArrayStream`.

**Reader success/argument/lifecycle tests**

- `OpenAsync_RejectsInvalidArgumentsAndExpectedIdEnumeration` theory: null contract/expected IDs, duplicate expected IDs including across enumerable positions, expected-ID enumerable throws a sentinel exception (same instance). Assert duplicate-ID detail and correct argument failures, not malformed-data failures.
- `OpenAsync_UsesManifestPathAndResolvesDataRelativeToIt`: valid manifest path and relative `decisions.arrow`; missing manifest/file path cases retain native file exceptions. No global working-directory changes.
- `OpenAsync_ValidatesUnorderedExpectedIdSets`: expected IDs are an order-independent exact set; stored IDs retain producer order. Include empty set with empty dataset.
- `OpenAsync_WithoutExpectedIdsReadsValidDataset`: exercise the public overload without IDs on zero/nonempty valid data; do not use this overload as a substitute for acceptance tests requiring expected IDs.
- `OpenAsync_PreCancelledTokenPropagatesCancellation`: pre-cancelled token; deterministic cancellation and no returned reader. Do not claim that every individual I/O stage was independently cancellation-injected.
- `ReadNextRecordBatchAsync_ReplaysFirstBatchThroughIArrowArrayStream`: obtain reader from fully validating `OpenAsync`, access its properties, upcast to `IArrowArrayStream`, read each batch and verify first IDs/data are replayed from batch zero, not left at EOS after validation.
- `ReadNextRecordBatchAsync_ReturnsNullAtEosIncludingEmptyDataset`: zero rows and after final nonempty batch; EOS is null despite non-nullable public signature.
- `Dispose_DoesNotDisposeCallerOwnedBatchesAndReleasesFile`: retain batch one while advancing to batch two, dispose stream, then verify both retained arrays still have expected data; caller disposes batches. Exclusively reopen/delete file afterwards. Optional deterministic Windows probe: sharing prevents write/delete while reader lives, then permits them after disposal; no timing probe.
- `OpenAsync_FailureOrCancellationReleasesFileHandles`: manifest integrity failure, later-batch semantic failure after opening IPC, and pre-cancellation; no reader returned and test can exclusively reopen/delete the data file afterwards.

### Adversarial Reader Matrix

Use a valid baseline per theory case and alter **one invariant**. Raw IPC semantic/schema cases refresh both manifest hash and size. Integrity cases intentionally leave the targeted integrity field wrong. Explicitly assert the independent refreshed values before invoking `OpenAsync` so a semantic rejection cannot accidentally be a stale-hash test.

Most envelope/schema inconsistencies are `InvalidDataException`; malformed JSON remains `JsonException`, missing files remain file exceptions, and reconstructed invalid answer distributions/grid/domain/native values can be `DecisionProtocolException`. For each row, encode the exact expected exception category from the researched branch; do not use a blanket `ThrowsAny<Exception>` or universally require `InvalidDataException`. Arrow construction/parser errors are fixture/parser findings, not proof of reaching a semantic validator.

| Exact planned test name | Theory cases / target invariant |
|---|---|
| `OpenAsync_RejectsMalformedManifestJson` | Missing manifest, malformed JSON, unknown member, missing required member, literal JSON null (`"Missing manifest."`). Also covers JSON required-member rejection independently of an IPC file. |
| `OpenAsync_RejectsInvalidCompletionEnvelope` | Unsupported schema version; status `partial`, `failed`, `cancelled`, case variant of `complete`; non-null Error; negative RowCount; bound 0/65537; null DataSize; absent/invalid-length/nonhex/uppercase DataSha256. No invented semantic provenance gate. |
| `OpenAsync_RejectsInvalidDataFilename` | Null/blank, wrong basename, relative path segments, rooted path, slash/backslash-bearing name; exact `decisions.arrow` is the positive control. |
| `OpenAsync_RejectsDataIntegrityCorruption` | Missing data file; wrong actual size; same-size byte flip with stale hash; fewer than eight bytes with refreshed integrity; removed/changed EOS with refreshed integrity. Split length/hash faults from marker faults explicitly. |
| `OpenAsync_RejectsFeatureIdentityMismatch` | Invalid/noncanonical ContractJson; manifest fingerprint differs from recomputation; internally coherent changed contract/hash/schema versus expected contract. Reuse the full semantic-variant cases: question/candidate/rubric order/content, provider/model, asset key/hash, preprocessing/decoder, all precisions, projection order/content. `float32` is unsupported, not a second valid schema. |
| `OpenAsync_RejectsStructuralSchemaAndMetadataCorruption` | Missing/extra/wrong schema metadata key/value; missing/extra/wrong question metadata; extra child/item/row metadata; legacy key. Field count/name/order/type/nullability; struct child count/name/order/type/nullability; list width; item name/type/nullability. Exercise every field/child role, especially nullable native versus non-null others. Preserve coherent raw arrays. Metadata insertion order and null-versus-empty dictionaries are positive equivalence controls, not corruptions. |
| `OpenAsync_RejectsInvalidBatchLayout` | Batch larger than declared bound; wrong column count; top-level array length mismatch; null row ID/binary/choice struct/score struct. Public IPC construction feasibility for column-count/length cases must be established as below. |
| `OpenAsync_RejectsInvalidRowIdSetsAndCounts` | Duplicate stored IDs within/across batches; unexpected ID; declared RowCount too small/large; expected set missing an observed ID or containing an unseen ID; nonempty expected set with zero rows. These are distinct from duplicate expected IDs rejected at the argument boundary. |
| `OpenAsync_RejectsBinaryCorruption` | Null, NaN, positive/negative infinity; finite below 0/above 1; off-grid two-/four-decimal observation. Positive controls include exact high-precision bits and all declared precisions. |
| `OpenAsync_RejectsChoiceCorruption` | Null probabilities list/element; NaN/infinities; selected null, -1, width; finite probability outside domain; high-precision sum outside absolute `1e-6`; rounded off-grid values and infeasible clipped half-step intervals. Examples: `[0.5,0.500002]` high precision and `[0.49,0.49]` two-decimal are distribution faults; choose equally explicit four-decimal infeasible values. Non-argmax selection remains valid. |
| `OpenAsync_RejectsScoreCorruption` | Choice-style vector/null/nonfinite/domain/grid/distribution faults; null/nonfinite expected/lower/upper fields; each finite stored expected/bound altered independently by a detectable amount/bit; native NaN/infinities, below/above rubric domain, incompatible with reconstructed bounds, rounded off-grid native within otherwise compatible bounds. Native null and both rounded beyond-range expected-index fixtures remain valid controls. |
| `OpenAsync_RejectsCorruptLaterBatchBeforeExposure` | Parameterize all representable batch-local faults above: duplicate/unexpected IDs, bound/layout/null faults, binary, selected/list/probability faults, expected/lower/upper/native faults. Write a valid first batch and corrupt later/final batch, refresh hash/size, then assert **OpenAsync itself throws** and exposes no reader/first batch. Include a valid three-batch replay control. |
| `OpenAsync_RejectsCorruptEmptyStreamSchema` | Schema-only stream plus EOS, zero rows/expected IDs, refreshed hash/size; reuse all representable structural/metadata mutations. Reject during OpenAsync without needing a batch. Include a valid schema-only control. |

For rounding corruption, use explicit grid/domain values and interval expectations, not random fuzzing. HighPrecision tolerance is absolute `1e-6`; rounded half steps are `0.00005`/`0.005`, clipped to [0,1] with inclusive ties. Native off-grid cases should be otherwise compatible (for example two-decimal near 1 with `[0.33,0.33,0.33]`, and four-decimal near 1 with `[0.3333,0.3333,0.3333]`) so a different native check does not mask the target.

**Fixture feasibility and honest validation limits**

- IPC has a stream-wide schema: schema/metadata faults are tested on nonempty and schema-only streams, not claimed to appear solely in a final batch. The “corrupt later batch” matrix applies to **batch-local representable content/layout faults**.
- If public `RecordBatch`/Arrow writing disallows wrong column counts or unequal top-level lengths, record those exact `OpenAsync_RejectsInvalidBatchLayout` cases, rejected construction API and exception as blocked. Do not use reflection/unsafe malformed memory, silently drop them, or mark them reader-tested. Other valid raw-array corruption cases must still be implemented.
- Do not require every token stage to be cancellation-injected without a seam. Pre-cancelled open, producer-triggered writer cancellation, eager later-batch failure and handle-release tests are deterministic public-API coverage.
- Optional `OpenAsync_TrailingContentProbeReportsParserBoundaryBehavior`: append extra content plus terminal EOS, refresh hash/size, and report pinned Arrow parser behavior. Research does not establish rejection; this is a diagnostic finding, not a silently invented acceptance requirement.

### Acceptance Traceability

Every research checklist item is assigned below. Names resolve to the test classes/files specified above. A theory must retain descriptive case IDs so reporting can distinguish precision, null location, semantic variant, schema role, and batch position.

| Checklist item | Exact planned tests / cases | Blocker or boundary |
|---|---|---|
| 1. Canonical metadata, ordered properties/questions/candidates/rubric, exact strings, asset order and lowercase UTF-8 hash | `CanonicalJson_AndFingerprint_MatchIndependentGolden`; `CanonicalJson_PreservesEscapedAndUntrimmedStrings`; `Constructor_SortsAssetsOrdinally`; `Schema_HasExactPhysicalStructureAndMetadata` (three precisions); `ParseQuestions_ReconstructsAllKindsInOrder` | Independent fixed JSON/hash above; metadata compares complete sets, not insertion order. |
| 2. Semantic identity versus provenance independence | `FeatureFingerprint_ChangesForSemanticVariants` (full variant list); `FromCanonicalJson_RejectsInvalidOrNoncanonicalContracts` (unsupported numeric representation); `FeatureFingerprint_IsIndependentOfProvenanceAndTransport`; `OpenAsync_RejectsFeatureIdentityMismatch` | Numeric representation is always float64; do not invent float32 support. No provenance source verification. |
| 3. Reserved/duplicate/ordinal identities and all long IDs | `Constructor_RejectsInvalidContracts`; `Constructor_UsesOrdinalQuestionAndProjectionIdentities`; `Constructor_RejectsInvalidAssets`; `Constructor_RejectsInvalidProjection`; `WriteAsync_RejectsDuplicateSourceIds`; `OpenAsync_RejectsInvalidArgumentsAndExpectedIdEnumeration`; `Constructor_RetainsRowIdAndExactPayloadReferences`; `WriteRead_PreservesAllLongIdsAndProducerOrder` | Duplicate candidate IDs are rejected at public question construction, not by fabricated invalid Arrow objects. No coordinate-alias rejection semantics. |
| 4. Exact float64 and selected-ID lookup, reordered probabilities/nonargmax | `WriteRead_PreservesFloat64BitsAndDeclaredPrecision`; `WriteRead_PreservesMixedQuestionOrderAndNonArgmaxCandidateId` | Assert selected index 2 versus argmax 1 and exact ordered vector, not merely presence. |
| 5. Exact score expectation/bounds/native/null, both beyond-range examples | `WriteRead_PreservesExactScoreFieldsAndNullableNative` (six table rows after native/null expansion); `WriteRead_PreservesFeasibleRoundedVectorsWithoutNormalization` | Explicit expected doubles; no clamp/normalization/confidence semantics. |
| 6. Three precision declarations and invalid numerical content | `Schema_HasExactPhysicalStructureAndMetadata`; `WriteRead_PreservesFloat64BitsAndDeclaredPrecision`; `WriteRead_PreservesExactScoreFieldsAndNullableNative`; `OpenAsync_RejectsBinaryCorruption`; `OpenAsync_RejectsChoiceCorruption`; `OpenAsync_RejectsScoreCorruption` | Invalid values come from public raw IPC, not invalid abstraction construction; preserve exception distinctions. |
| 7. Default/custom bounds, empty/single/full/short, allocation overflow | `WriteAsync_EmitsBoundedBatchesAndExactManifest` (0/1/256/257/775 and custom bounds); `WriteAsync_ResetsBuilderAtFullBatchBoundary`; `WriteAsync_ChecksVectorAllocationBeforeAllocating` | No giant allocation or total constant-memory claim; ID sets are O(total rows). |
| 8. Schema before enumeration and finalized data before completed publication | `WriteAsync_WritesSchemaBeforeFirstProducerMoveNext`; `WriteAsync_PublishesOnlyPartialArtifactsWhileProducerIsActive`; `WriteAsync_EmitsBoundedBatchesAndExactManifest` | Generator/final-state assertions cover observable ordering. Exact flush/hash/rename micro-order inside the unobservable terminal span is static research evidence, not a fabricated timing test. Cleanup-after-publication bug remains a parent finding. |
| 9. Original producer failure/cancellation/status/count, no completed success, recording aggregate | `WriteAsync_ProducerFailureRethrowsSameInstanceAndRecordsAcceptedCount`; `WriteAsync_ProducerCancellationRethrowsSameInstanceAndRecordsAcceptedCount`; `WriteAsync_PreCancelledTokenProducesCancelledRecord`; `WriteAsync_RecordingFailurePreservesOriginalAsFirstAggregateInner`; desired `WriteAsync_CleanupFailureMustNotLeaveCompletedManifest` | Ordinary producer cases must pass. **Unconditional no-completed-manifest-on-any-error is blocked by the static cleanup bug.** Report, do not bless or fix it. |
| 10. Reader envelope/integrity/schema/batch/IDs/identity/numeric corruption | Every `OpenAsync_Rejects...` adversarial matrix test, including `OpenAsync_RejectsInvalidRowIdSetsAndCounts` and `OpenAsync_RejectsFeatureIdentityMismatch` | Public-IPC column-count/length feasibility cases must be explicitly reported if blocked; integrity refresh is mandatory for semantic/schema cases. |
| 11. Eager validation including final corruption and empty schema | `OpenAsync_RejectsCorruptLaterBatchBeforeExposure`; `OpenAsync_RejectsCorruptEmptyStreamSchema`; `OpenAsync_ValidatesUnorderedExpectedIdSets` | Assert OpenAsync throws before exposing any reader. Stream-wide schema mutations cannot be confined to the final batch. |
| 12. Standard stream/replay/caller batches/handles | `ReadNextRecordBatchAsync_ReplaysFirstBatchThroughIArrowArrayStream`; `ReadNextRecordBatchAsync_ReturnsNullAtEosIncludingEmptyDataset`; `Dispose_DoesNotDisposeCallerOwnedBatchesAndReleasesFile`; `OpenAsync_FailureOrCancellationReleasesFileHandles` | Caller uses `using` and owns returned batches. Windows lifetime locking probe is optional if deterministic. |
| 13. Existing output protection and allowed empty directory | `WriteAsync_PreservesExistingOutputDirectoryContents` (file/subdirectory/completed dataset); `WriteAsync_AcceptsExistingEmptyDirectory`; `WriteAsync_RejectsInvalidArgumentsBeforePublication` | Compare path inventories and bytes before/after; no overwrite accepted. |
| 14. Immutable fixtures/snapshots and correct row construction | `Constructor_RetainsExactStringsAndSnapshotsCollections`; `Constructor_SnapshotsQuestionCollections`; `Constructor_RetainsRowIdAndExactPayloadReferences`; helper `CreateRow`; `WriteAsync_RejectsInconsistentRows` | No reflection, mutable questions, answer-only row constructor, or unrelated abstraction suite. |
| 15. External harness discovery and root gap | Selected-workspace discovery and final full workspace commands in Commands; record identities and expected count delta | Waiting for parent-owned project, restore arguments and agreed harness. Root does not register Arrow; do not edit solutions to hide this. |

### Parent Findings and Blockers — Do Not Fix in This Test-Only Work

1. **Known static cleanup-after-manifest bug**: research reports `DecisionArrowWriter.cs:81-82` publishes `manifest.json` and then deletes `run.partial.json` inside the same try. Cleanup failure enters failure handling after completion is already visible, without removing/invalidation of the completed manifest. Failure recording can also aggregate. This blocks checklist 9's unconditional no-completed-manifest-on-error acceptance.
   - Optional deterministic Windows `WriteAsync_CleanupFailureMustNotLeaveCompletedManifest` probe: acquire a test-owned handle to `run.partial.json` from a final producer callback with sharing that forbids deletion/replacement; hold it **outside the generator's finally** until WriteAsync settles. No sleeps or watcher. Alternatively, a test-local directory substituted at that path forces deterministic cleanup/recording failure.
   - The assertion must express the desired contract (failure must not leave a completed manifest). It is expected to expose the finding on the researched implementation. Do not write a passing test that calls complete-plus-failure correct, alter production, or silently mark acceptance complete. Parent decides how to carry the failing probe/explicit blocked case in the test-only PR; any omission/skip requires an explicit unresolved finding.
2. **Exporter measurement collisions**: supplying either generated measurement key currently raises `ArgumentException` after data rename, and the failure record may name the now-missing partial file. No documented reserved-key policy exists. Report the research finding; do not invent reserved-key validation, assert this behavior as a desirable contract, or change production.
3. **Harness/restore**: intended project was absent when researched; parent now owns its creation/references. Experimental Arrow package and isolated restore setup need the parent's arguments. All execution gates remain waiting. No installations or analyzer retry.
4. **Raw-IPC feasibility**: independent malformed batch layout cases may be blocked by public Arrow construction/serialization. Report exactly which case and API prevents it, while retaining the rest of the corruption matrix.
5. **Provenance and aliases are not missing acceptance requirements**: records have no researched semantic provenance validation; no external package/source hash verification is required. Empty projections are accepted and coordinate aliases are not assigned new rejection semantics.
6. **Optional trailing-content probe**: parser EOS/file-length behavior is unverified. Report observed behavior if probed; do not claim a guaranteed rejection or broaden acceptance without parent approval.

### Mandatory Quality and Execution Gates

These gates are recorded for implementation, **not executed during planning**. They are reviews performed directly, without agents/nesting or production edits.

1. Parent supplies the owned test project, restore arguments and agreed external harness; restore only under that authorization. Keep all later build/test commands `--no-restore`, and use `--no-build` only following a successful build.
2. Complete named tests and track every theory case/explicit blocker against the acceptance table. Record which reader cases actually reached their intended validator, including valid refreshed hash/size and late-batch eager rejection.
3. **Mandatory `test-gap-analysis` gate** after implementing the suite: assess omitted/error/edge cases and whether tests distinguish plausible transport/contract faults. Respect the bounded Arrow scope, public fixture limits and known parent findings. No find-untested retry, coverage tooling, or new production semantics.
4. **Mandatory `assertion-quality` gate**: review assertion strength/depth/variety across the new files. Require independent canonical/hash expectations, complete schema/metadata comparisons, exact scalar bits/score constants, ID sets and order, same exception instance/aggregate ordering, byte-for-byte protection, and caller ownership. Timing/allocation checks legitimately assert finite/nonnegative values rather than exact numbers. No assertion-free tests, self-derived expected values, or presence-only substitutes for content.
5. Address test-only findings from both gates; report blockers instead of patching production or weakening desired acceptance.
6. After parent authorization, run scoped fix cycles, then **final full selected-workspace build with `--no-incremental`**, discovery/count-delta check, and **full selected-workspace tests** using the Commands section. Keep build/test/discovery output and unresolved findings separate; a passing scoped project is not proof of external harness discovery.

### Success Criteria

- [ ] Parent-owned Arrow test project and supplied restore setup are confirmed without modifying forbidden files.
- [ ] Helper and nine test files are implemented in `tests/DecisionInference.Arrow.Tests`, namespace `DecisionInference.Tests`; all six target sources have their assigned coverage.
- [ ] Every acceptance checklist row has implemented named cases or an explicit, parent-visible blocker; no unknown coverage is mislabelled.
- [ ] Golden literal/hash, full structure/metadata, all precision fidelity and both rounded beyond-range score examples pass exact assertions.
- [ ] Raw IPC corruption reaches intended validators with refreshed integrity; representable late-batch corruption fails OpenAsync eagerly.
- [ ] Producer failure/cancellation identity, recording aggregate, output protection, replay and caller-owned batch/file lifecycle cases pass.
- [ ] Static cleanup-after-manifest finding is reported as **unresolved failed acceptance**, not corrected in production or blessed by tests.
- [ ] `test-gap-analysis` and `assertion-quality` gates are completed and their findings handled/reported.
- [ ] Scoped tests and final full selected-workspace build/tests pass after supplied arguments; any blocked regression case is explicitly reconciled by parent, not hidden.
- [ ] External harness discovery enumerates the new tests/count delta, or the root/default entry-point gap is explicitly reported.
- [ ] No production/config/solution/other-test/version-control changes, dependency installs, analyzer retries, or agents/nesting.
