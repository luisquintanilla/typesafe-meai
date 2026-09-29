# Arrow test generation — final status

**Scope:** `src/DecisionInference.Arrow` only. **Strategy:** one broad
Research → Plan → Implement phase, followed by direct quality-gate review and
final full selected-workspace validation.

## Results

| Metric | Final result |
|---|---:|
| New test methods | 51 |
| Discovered / executed cases | 174 |
| Passed | 174 |
| Failed / skipped | 0 / 0 |
| Test source files | 9 |
| Shared fixture files | 1 |
| Full selected-workspace build | 0 warnings, 0 errors |

Changes were confined to `tests/DecisionInference.Arrow.Tests` and the three
mandated `.testagent` documents. The caller-approved restore helper generated its
own artifact config/cache/verification receipt and lock files. No production,
props, `eng`, samples, other tests, solutions, ancestor/global config or version
control were edited by this test worker. Parent-owned production updates were
consumed as delivered, not reverted or reconstructed.

## Files and discovery evidence

All test files are under `tests/DecisionInference.Arrow.Tests`.

| File | Methods | Discovered cases |
|---|---:|---:|
| `DecisionArrowIdentityTests.cs` | 3 | 12 |
| `DecisionArrowSchemaTests.cs` | 6 | 36 |
| `DecisionArrowRowTests.cs` | 2 | 7 |
| `DecisionArrowManifestTests.cs` | 3 | 6 |
| `DecisionArrowBatchBuilderTests.cs` | 4 | 12 |
| `DecisionArrowWriterTests.cs` | 11 | 28 |
| `DecisionArrowFidelityTests.cs` | 7 | 17 |
| `DecisionArrowDatasetReaderTests.cs` | 4 | 8 |
| `DecisionArrowDatasetReaderCorruptionTests.cs` | 11 | 48 |
| `DecisionArrowTestSupport.cs` | Support | — |

`DecisionInference.Arrow.Tests.csproj` matches the existing SDK-style `net10.0`
xUnit v3 VSTest conventions: versionless central package references, `OutputType`
Exe, `IsTestProject=true`, `IsPackable=false`, Xunit global using, and only the
Arrow production project reference. No coverlet/framework upgrade/model package
was added. The existing `eng/DecisionArrowExperiments.slnx` already registers it.
Discovery increased from the initial seven newly scaffolded row cases to 174;
all 174 are new tests for this request. Root `TypeSafe.MEAI.slnx` omits Arrow and
is not the selected experiment harness; neither solution was changed here.

## Commands and outcomes

Every command below was run from the repository root. Each restore/build/test
PowerShell process used the parent-documented shell-local setting:

```powershell
$env:ImportDirectoryBuildTargets='false'
```

1. `dotnet new xunit --no-restore --framework net10.0 --name DecisionInference.Arrow.Tests --output tests\DecisionInference.Arrow.Tests`
   exited 0. Template v2 references were replaced with repo-standard versionless
   xUnit v3 references, and the template test became meaningful row-wrapper tests.
2. Approved restore:

   ```powershell
   .\eng\experiments\Restore-DecisionArrow.ps1 -ArrowFeed 'C:\Users\luquinta\.copilot\session-state\64dc898c-ad00-4d88-81ce-534017782670\files\decision-arrow-experiments\feeds\arrow-baseline' -ArtifactDirectory '.\artifacts\decision-arrow-tests-restore' -Project 'tests\DecisionInference.Arrow.Tests\DecisionInference.Arrow.Tests.csproj' -AllowUnsignedLocalPackages -UpdateLocks
   ```

   exited 0 with the workaround. The helper reported verified exact public/local
   dependency closure and isolated cache. The earlier invocation without the
   setting failed on external `C:\Dev\Directory.Build.targets` / Arcade MSB4236;
   no ancestor/global file was changed.
3. Scoped implementation:

   ```powershell
   $env:ImportDirectoryBuildTargets='false'; dotnet test tests\DecisionInference.Arrow.Tests\DecisionInference.Arrow.Tests.csproj --no-restore --verbosity quiet
   ```

   initially passed 167 cases, no skips. A later fresh full run found the exact
   manifest property-list assertion stale after parent added `AuxiliaryAssetHashes`.
   The test fixture/wire oracle was updated, not production. Final quality review
   added five cases for inclusive probability boundaries and prepublication
   reserved-measurement guards, then an opaque spaced/case-sensitive/Unicode ID
   round-trip with exact physical fields, assets, provider/model and projection
   assertions. A test-only nullable xUnit overload compile error in that last
   addition was corrected and immediately rerun.
4. **Final full selected-workspace build:**

   ```powershell
   $env:ImportDirectoryBuildTargets='false'; dotnet build eng\DecisionArrowExperiments.slnx --no-restore --no-incremental --verbosity quiet
   ```

   exited 0: **0 warnings, 0 errors**.
5. **Final full selected-workspace tests, with fresh build:**

   ```powershell
   $env:ImportDirectoryBuildTargets='false'; dotnet test eng\DecisionArrowExperiments.slnx --no-restore --verbosity quiet
   ```

   latest execution exited 0: **Failed 0, Passed 174, Skipped 0, Total 174**.
   No `--no-build` was used for final test validation.
6. **Final harness discovery:**

   ```powershell
   $env:ImportDirectoryBuildTargets='false'; dotnet test eng\DecisionArrowExperiments.slnx --no-build --no-restore --list-tests --verbosity quiet
   ```

   latest execution exited 0 and enumerated **174** cases. `--no-build` was used only for discovery
   after the clean build/test run. Class counts are recorded above.
7. **Final exact parent-requested project command:**

   ```powershell
   $env:ImportDirectoryBuildTargets='false'; dotnet test tests\DecisionInference.Arrow.Tests\DecisionInference.Arrow.Tests.csproj --no-restore --verbosity quiet
   ```

   latest project execution exited 0: **Failed 0, Passed 174, Skipped 0, Total 174**.

## Requirement coverage

Names below are exact generated test methods; theory/loop variants are part of
the cited method. Rows quote the caller's behavioral requirements.

| Requirement | Evidence |
|---|---|
| “canonical metadata/fingerprint order/exact strings/assets/provider/model/precision/projection (package version independent)” | `CanonicalJsonAndFingerprint_MatchIndependentGolden` checks the independent 746-byte literal and fixed `51c5...bb12` SHA-256; `Schema_HasExactPhysicalStructureAndMetadataAndParsesCanonicalContract` asserts complete recursive types/nullability/metadata for all three precisions; `CanonicalJson_PreservesEscapingDescriptionsAndCollectionSnapshots` checks case/spaces/quotes/backslashes/newline/Unicode and snapshots; `Constructor_PreservesExactStringsAndSnapshotsOrdinalCollections` checks ordinal assets and reversed insertion; `WriteRead_PreservesOpaqueWhitespaceCaseAndUnicodeIds` checks spaced question/candidate/asset IDs, case-distinct and Unicode-normalization-distinct IDs and exact provider/model/preprocessing/decoder/projection; `FeatureFingerprint_ChangesForSemanticVariants` checks 22 semantic variants; `FeatureFingerprint_IsIndependentOfPackageVersionsProvenanceRowsAndBatching` checks concrete different package versions, provenance, IDs and bounds with unchanged golden identity. |
| “reserved/duplicate IDs” | `Constructor_RejectsInvalidContractsAndOrdinalIdentities` checks reserved `row_id`, duplicate questions/candidates, allowed `ROW_ID` and case-distinct IDs, projection bounds; `Constructor_RejectsInvalidAssetsAndProjection` checks exact duplicate assets/projections; `WriteAsync_RejectsDuplicateIdsWithinAndAcrossBatches` checks negative/zero source IDs within/across batches; `OpenAsync_RejectsInvalidArgumentsBeforeReadingManifest` checks duplicate expected IDs; `OpenAsync_RejectsIdsCountsAndBatchBoundBeforeReturningReader` checks duplicate raw IPC IDs. |
| “float64 exact values and selected_index preserve nonargmax candidate IDs” | `WriteRead_PreservesFloat64BitsQuestionOrderAndNonargmaxSelection` asserts double bits for `.12345678901234567`, `.2718`, `.27`; probability pairs and answers arrive in different orders; candidate `"a"` stores ordinal **2** despite argmax `"A"` at ordinal **1**. `WriteRead_AcceptsSingleCandidateAndOrdinalZero` checks width-one selection. |
| “score bounds/native score preserved incl rounded expected outside range” | `WriteRead_PreservesIndependentScoreExpectationsBoundsAndNativeNull` checks exact bits for expected **1**, bounds **.999998/1.000002**, and native **1/null**; four-decimal expected **2.0001**, bounds **1.99995/1.99995**; two-decimal expected **2.01**, bounds **1.995/1.995**, native **2/null**. `WriteRead_RetainsFeasibleRoundedObservationsWithoutNormalization` checks retained `.33/.3333` vectors, nonunit sums and independent expectations `.99/.9999`. |
| “all three precision declarations” | `Schema_HasExactPhysicalStructureAndMetadataAndParsesCanonicalContract`, `WriteRead_PreservesFloat64BitsQuestionOrderAndNonargmaxSelection`, `WriteRead_PreservesIndependentScoreExpectationsBoundsAndNativeNull` and `WriteRead_AcceptsBothInclusiveBinaryProbabilityBoundaries` each cover HighPrecision/FourDecimalPlaces/TwoDecimalPlaces. |
| “output 256 rows bounded final short zero/one/257” | `WriteAsync_EmitsBoundedBatchesAndFinalizedManifest` checks **0→[]**, **1→[1]**, **256→[256]**, **257→[256,1]**, **775→[256,256,256,7]**, custom bounds **1/2/65536**, row order and concrete observations. `WriteAsync_ChecksVectorCapacityBeforeGiantAllocationOrEnumeration` exercises checked multiplication with 4096 candidates and bound 65536, not giant allocation. |
| “completed manifest published last” | `WriteAsync_OnlyPartialArtifactsExistThroughoutProducerLifetime` checks schema/contract/partial record before first yield, throughout enumeration and exhaustion; finalized hash/size/files after success. `WriteAsync_CleanupFailureMustNotPublishCompletedManifest` uses deterministic test-local sharing lock to prove cleanup failure leaves no completed manifest. |
| “failed/cancelled records propagate original failures and no completed manifest” | `WriteAsync_ProducerErrorsPreserveInstanceAcceptedCountAndDisposal` checks same sentinel exception instance, failed/cancelled status, accepted counts **0/2**, exact error string, disposal and no completed manifest; `WriteAsync_DeterministicTokenCancellationRetainsProducerException` checks token and same original exception at count 1; `WriteAsync_PreCancelledTokenRecordsCancellationWithoutCompletedManifest`; `WriteAsync_RecordingFailureKeepsOriginalAsFirstAggregateInner` checks original exception retained first when record publication also fails. |
| “reject incomplete/hash/schema/metadata/count/ID sets/feature-contract/vector/null/nonfinite corruption” | `OpenAsync_RejectsInvalidCompletionEnvelope`; `OpenAsync_RejectsMalformedMissingAndNullManifestJsonWithNativeCategories`; `OpenAsync_RejectsIndependentIntegrityAndCompletionMarkerFaults`; `OpenAsync_RejectsRealIpcStructuralContractsIncludingSchemaOnlyStreams`; `OpenAsync_RejectsIdsCountsAndBatchBoundBeforeReturningReader`; `OpenAsync_RejectsFeatureContractsEvenWithIndependentCorrectIpcIntegrity`; `OpenAsync_EagerlyRejectsLaterBatchNumericalFaultsWithRefreshedIntegrity`. Raw public-API IPC fixtures independently refresh SHA-256/size and assert expected exception type/message, including nested widths/nullability, bad selected indices, grids/domain/distributions, null/nonfinite scalar/vector/score/native observations. |
| “ownership standard IArrowArrayStream caller-owned batches” | `ReadNext_ReplaysAllBatchesInProducerOrderAndCallerOwnsTheirLifetime` consumes through `IArrowArrayStream`, retains **[2,2,1]** batches, checks IDs from `long.MinValue` through `long.MaxValue`, exact observations after stream advancement/disposal, caller disposal and released file handle. |
| “existing output directory protected” | `WriteAsync_ProtectsAllExistingArtifactsByteForByte` snapshots files, nested directory paths and existing completed dataset bytes; attempted writes throw without changing any path/byte. `WriteAsync_OnlyPartialArtifactsExistThroughoutProducerLifetime` permits an existing empty directory. |
| “Build meaningful adversarial Arrow IPC/manifest fixtures by public APIs to exercise checks, not only wrong hash.” | `DecisionArrowTestSupport.WriteRawAsync` uses `ArrowStreamWriter`, `RecordBatch`, typed arrays, public nested constructors and validity bitmaps. Schema and content corruption tests update both SHA/size and verify messages demonstrating the intended reader guard, not integrity short-circuiting. |
| “Caller rows new DecisionArrowRow(id,input,result). Immutable question definitions existing abstraction.” | `Row` helper and `Constructor_RetainsRowIdAndExactPayloadReferences`; `WriteAsync_RevalidatesImmutableRowAssociationAndDatasetContract` supplies only constructible immutable input/results, then checks model/question/order/kind/candidate/rubric/precision mismatch via public writer. No mutable IDs, reflection or bypassed answer construction. |
| “Reader OpenAsync(path,expectedContract,expectedIds,token) fully validates before expose.” | `OpenAsync_EagerlyRejectsLaterBatchNumericalFaultsWithRefreshedIntegrity` puts faults in **second/third** batches after validated controls, asserts failure of **OpenAsync itself** and exclusive handle reopening; schema-only streams also validate. `ReadNext_ReplaysAllBatchesInProducerOrderAndCallerOwnsTheirLifetime` proves validation rewinds to the first batch. |
| “Use temp dirs with exact scoped cleanup, no models/downloads.” | `DecisionArrowTestSupport.TempDirectory` creates a unique `DecisionArrowTests-<guid>` temp root and deletes only that owned root; all filesystem probes stay beneath it. No model/network/port/timing/sleep/skip markers, provider references, inference or downloads in test source. |
| “mandated .testagent\{research,plan,status}.md” | All three documents retained. `research.md` records bounded target inventory/conventions/checklist/analyzer result, `plan.md` preceded broad implementation, this file records final gates/validation/evidence. |

## Mandatory quality gates

Both **`test-gap-analysis`** and **`assertion-quality`** were invoked and executed
as direct reviews against all six production source files, all nine final test
files and shared fixture code. Their .NET extension was read directly after the
`test-analysis-extensions` skill was unavailable. Reviews were repeated against
the final modified methods before this status was written.

### Pseudo-mutation gate

**Static reasoning only:** the caller forbids production edits. No mutation was
applied, no empirical mutation score or “confirmed survivor” is claimed.

Reviewed substantive mutation groups and concrete detecting tests:

| Hypothetical mutation group | Detecting evidence |
|---|---|
| Change canonical property/question/candidate/rubric/asset order; trim/case-normalize strings; omit a semantic fingerprint input | Independent golden/escaping/schema tests and 22 `FeatureFingerprint_ChangesForSemanticVariants` variants; asset order reversal must remain equal. Review gap for trimming/normalizing opaque IDs closed by `WriteRead_PreservesOpaqueWhitespaceCaseAndUnicodeIds`, with `" q "`, `" a "`, `"A"`, composed/decomposed Unicode and `" Assets/model "` assertions. |
| Remove reserved/duplicate/projection/precision/identity/hash guards | Schema/identity invalid-input theories, exact exception category/parameter/message, plus valid ordinal-case and three-precision controls. |
| Lose double precision, reorder probabilities by caller enumeration, replace selection with argmax, normalize/clamp observations, omit score/native/bounds | Exact-bit and independent score/vector fidelity tests, native null/non-null cases and selected ordinal 2 vs argmax 1. |
| Flip inclusive binary probability boundaries | Review gap closed by `WriteRead_AcceptsBothInclusiveBinaryProbabilityBoundaries` for both 0/1 under each precision; corrupt-domain cases also assert rejection. |
| Remove writer model/question/association/precision/duplicate-ID checks | `WriteAsync_RevalidatesImmutableRowAssociationAndDatasetContract` and duplicate-ID tests pin exact protocol errors and secondary failure-record state. |
| Emit an empty batch, mishandle exact/default/custom bounds, lose final short rows, skip allocation overflow check | 0/1/256/257/775 and custom-bound sequences, exact row IDs/values; checked-capacity test verifies no enumeration. |
| Publish completed manifest before cleanup or swallow/replace producer error/cancellation | Producer checkpoints and cleanup-lock regression; exact original exception identity/count/status/error/disposal/no-manifest; aggregate original retained first. |
| Remove new reserved exporter-measurement guard or mutate caller measurements | Review gap closed by `WriteAsync_RejectsReservedMeasurementNamesBeforeAnyPublication` for each name; directory remains absent, dictionary unchanged. Merge test asserts independent finite/nonnegative metrics and retained caller value. |
| Skip integrity/envelope/filename/recursive schema metadata/type/nullability/list-width validation | Real IPC/manifest corruption, refreshed integrity, schema-only and populated streams, exact error category/message. |
| Remove the consumed-offset check after the first parsed IPC EOS | `OpenAsync_RejectsTrailingAndConcatenatedStreamsEvenWithRefreshedIntegrityAndNoExpectedIds` rejects two concatenated valid streams and junk plus a second EOS, with independently refreshed hash/size and no expected-ID check masking the fault. |
| Skip later row/batch content, score expected/bound comparisons, selected-index/null/nonfinite/distribution/grid/native validation | Valid precision-specific controls then faults in batches 2/3; eager OpenAsync errors and released handles, not merely a deferred read failure. |
| Drop source/expected ID set or row/batch counts | Duplicates within/across raw batches, unexpected/missing IDs, over/under counts, excess batch bound, empty/nonempty expectations. |
| Fail to rewind, dispose returned batches, leak validation/reader files | Producer-order replay, retained caller batches across stream disposal, exclusive reopening/delete checks. |

No remaining **enumerated acceptance-scope** behavior gap was found by the final
static review. This is not a claim of total line/branch coverage or mutation score.
Parent post-handoff review added `OpenAsync_RejectsTrailingAndConcatenatedStreamsEvenWithRefreshedIntegrityAndNoExpectedIds`:
valid streams with concatenated data or trailing bytes plus a final EOS have
independently refreshed hash/size, yet must fail even without expected IDs.
The pinned FileStream implementation reads exact IPC buffers, so the validated
logical position must equal physical file length. This closes the coordinator's
whole-file-consumption finding; it is not a hash-mismatch proxy.

Arbitrary custom question subclasses,
undocumented coordinate-alias/provenance validation and exhaustive JSON type
permutations were not expanded beyond the requested behavior.

### Assertion-depth gate

- After the parent extent supplement: **51 test methods**, **370 direct assertion sites in test-method bodies** (7.25 per
  method), plus 32 assertion sites in shared/non-test helper methods.
- **0 assertion-free**, **0 trivial-presence-only**, **0 identified tautological
  tests**. Round-trip tests additionally assert independently specified physical
  values/schema/metadata, not just input-to-output identity.
- **29/51** methods have explicit exception assertions; **36/51** have explicit
  negative/exception assertions. Categories include concrete equality, Boolean,
  null, exception, type, string, collection, comparison, negative, side effects
  and recursive structure. Approximate comparisons are deliberately unnecessary:
  fidelity checks require exact double bits.
- Writer tests pair results/errors with output files, manifest state, row counts,
  dictionary ownership or producer disposal. Reader failures pair exact errors
  with independent integrity and released handles. Positive reader tests inspect
  IDs, values, schema and batch lifetime.
- No skips, timing waits, external URLs or model invocations were found.

### Prompt-scenario gate / coverage-gap review

Every caller checklist item was inspected against actual final test bodies and
mapped above. All feasible enumerated behavioral requirements are covered.
Theory cases count as distinct discovered executions, while corruption loops
exercise additional fixtures without inflating discovery count. No measurable
coverage target was requested; no coverage tool, package or percentage was added.

## Explicit limitations and findings

1. The mandated `find-untested-sources` skill was invoked once; its bounded analyzer
   execution occurred once and exited 2 for missing `tree-sitter-language-pack`.
   No installation/retry/manual substitute pairing result was invented. Static
   pairing remains unavailable and would not prove line/branch coverage anyway.
2. `RawIpcFixture_MissingColumnsCannotReachReaderViaPublicWriter` records a fixture
   boundary: the public Arrow writer refuses an omitted-column `RecordBatch`
   with `ArgumentOutOfRangeException("index")`. It does **not** claim reader
   column-count guard coverage. The array-length/schema-count corruption cases
   do reach reader validation. No private/byte-patching seam was introduced.
3. Initial static cleanup-after-manifest and measurement-collision findings were
   addressed by concurrent parent production updates. Deterministic regression
   tests pass; no production change was made by this worker.
4. The first full fresh test run caught an added public provenance field missing
   from the test oracle. Only tests were corrected, and the final run is green.
5. Root/default solution discovery intentionally remains unchanged; use the
   verified selected experiment solution or the Arrow test project.
6. Retain `.testagent` as auditable evidence. Any later `.gitignore` decision is
   the caller's; no agent-state cleanup or repository-state repair was performed.

## Post-parent-green confirmation

After the parent confirmed production library/sample compilation and the pinned
restore/signature gates green, the test worker repeated the approved helper against
the same test project and owned config/cache, **without `-UpdateLocks`**, followed by
the exact project test command. Both ran in the same PowerShell process with
`$env:ImportDirectoryBuildTargets='false'`.

```powershell
$env:ImportDirectoryBuildTargets='false'
.\eng\experiments\Restore-DecisionArrow.ps1 -ArrowFeed 'C:\Users\luquinta\.copilot\session-state\64dc898c-ad00-4d88-81ce-534017782670\files\decision-arrow-experiments\feeds\arrow-baseline' -ArtifactDirectory '.\artifacts\decision-arrow-tests-restore' -Project 'tests\DecisionInference.Arrow.Tests\DecisionInference.Arrow.Tests.csproj' -AllowUnsignedLocalPackages
dotnet test tests\DecisionInference.Arrow.Tests\DecisionInference.Arrow.Tests.csproj --no-restore --verbosity quiet
```

Exit 0. Helper: **locked public/local closure verified**, using its own
`NuGet.Experiment.config` and isolated `packages` cache. Tests:
**Failed 0, Passed 173, Skipped 0, Total 173**. No test code changed after the
final quality gates. No root scope expansion or SMS asset/corpus reads occurred.

## Authoritative ownership / final 174-case refresh

The parent confirmed exclusive Arrow-test/artifact ownership and no independent
duplicate Arrow worker. This worker created three synchronous RPI children; all
are now **idle**, not executing:

- Research: `987c9285-03ea-437b-b2df-5d3b6add6c3c`.
- Plan: `8f48d1d6-9eb1-4a58-a689-1937f7900045`.
- Implementation: `3a1572d2-dcd3-4acd-b147-3d94a32f8e5d`, phase completed.

No further child/pipeline was launched. Tool follow-up messaging is unsupported
for synchronous children; no worker was restarted to work around that limit.

An on-disk stat at the exact worktree confirmed all nine test files, support,
project and lock file. The parent-added extent regression was reviewed against
the parent-owned reader's `file.Position != file.Length` guard. Both quality
reviews were refreshed against this final delta; assertion metrics above reflect
**51 methods / 174 cases**.

Latest project test: **174 passed, 0 failed, 0 skipped**. Then the same recorded
selected-workspace non-incremental build, fresh-build test and discovery commands
were rerun: **0 build warnings/errors; 174 passed; 174 discovered; exit 0**.
The formerly unverified trailing-EOS observation is now resolved and covered.
No remaining confirmed production blocker is reported.

## Parent compile-log reconciliation

The saved parent log
`1790721667811-copilot-tool-output-26616-d7bd2412-e989-475d-a944-f98ef061a8c0.txt`
contains the implementation's earlier CS8631/xUnit1051 diagnostics. Current
working-tree corrections already include `Path.GetFileName(p)!` in the filename
sequence assertion and the shared `Token => TestContext.Current.CancellationToken`
passed at test async call sites. No `NoWarn` or warning-disable pragma was added.

To rule out incremental-build masking, the worker freshly reran:

```powershell
$env:ImportDirectoryBuildTargets='false'
dotnet build tests\DecisionInference.Arrow.Tests\DecisionInference.Arrow.Tests.csproj --no-restore --no-incremental --verbosity quiet
dotnet test tests\DecisionInference.Arrow.Tests\DecisionInference.Arrow.Tests.csproj --no-restore --verbosity quiet
```

Exit 0: **0 warnings, 0 errors; 174 passed, 0 failed, 0 skipped**. The diagnostics
are resolved in the current files, with no production/props/suppression edits.
