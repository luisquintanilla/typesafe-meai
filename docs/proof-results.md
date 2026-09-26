# Proof results

## Buffering and ownership simplification

The focused cleanup preserves the public APIs and all prior capabilities. The
settled `eng\verify.ps1 -All -Offline` run passed on Windows: **622 core +
28 evaluation = 650 tests passed, 0 failed, 0 skipped**. This retains the previous
627 cases and adds 23 regression cases in nine methods; no tests were removed.
The nonincremental Release build had **0 warnings/errors**. All 81
source/test/sample/consumer/configuration inputs fingerprinted before the final
gate were unchanged during execution.

All eight scenarios passed, as did the actual three-scenario aieval HTML/JSON
reports and fresh package-only consumer: **five decorated calls / five synthetic
HTTP sends**, **35 signed/hash-verified external archives + two local unsigned
proof packages** under the unchanged isolated consumer exception.
Fresh package receipt: `artifacts\package-proof\31ef1103d5ff45dda0b3791c39b87537`.
Final log: `artifacts\simplification-all-verification.log`.

Runtime C# changed from **26 files / 1816 physical lines** at `26bc421` to
**25 files / 1796 lines** (1833 including the two unchanged project files, versus
1853 before). The benefit is fewer conversions and one fewer middleware
responsibility, not a dramatic LOC reduction:

- Registered requests still call the body guard at both boundaries, but the
  second guard checks owned bytes rather than rebuffering. Success transport
  also returns those same bytes instead of copying via ReadAsByteArrayAsync.
- Per registered cache miss, NativeRequest reconstruction falls from two to
  one, and whole option snapshots from three to one. Raw protocol validation
  and public cache-hit correlation checks intentionally remain.
- Per-call evidence retains an owned byte copy and deep metadata/annotation
  isolation, but reuses immutable parsed native views instead of serializing
  the entire envelope to a data URI and reparsing it.

The attempted .NET 10 LoadIntoBufferAsync substitution exposed a real legacy
derived-StreamContent cancellation incompatibility. It was discarded, not
papered over by changing fixtures or abandoning background copies. The original
token-per-read loop and original cancellation fixtures remain. See
[transport rationale](protocol-contract.md#transport).

| Requirement | Evidence |
|---|---|
| "Eliminate repeated rebuffer/copy on registered path while retaining defensive direct-client safety." | `Buffering_ReusesOwnedContentAndBytesThroughRepeatedGuardsAndTransport` asserts one owned content/array across both guards and transport, with unchanged source-read count; `Buffering_UntrustedBclBufferedContentCannotClaimOwnedLimitValidation` |
| "Must preserve success/error body size limits (declared AND chunked/unknown/misleading lengths), exact native bytes/strictUTF8, header/media handling, cancellation/body-read deadlines covering attempts+total, correctly disposed/retryable failed responses, nonretryable oversized-body errors, and sanitized existing exception surface." | `Buffering_OwnedBytesRecheckActualLimitDespiteConsumedStreamAndMisleadingHeader`; `Buffering_PrebufferedContentStillChecksDeclaredLimitAndCancellation`; retained `BodyLimit_EnforcesExactSuccessAndErrorBoundariesWithAndWithoutContentLength`, `DirectClient_CancellationOrCallerDeadlineDisposesStalledBodyWithoutRetry`, `BodyRead_StallIsInsideAttemptBudgetAndRetryDisposesFirstResponse`, `BodyRead_RepeatedStallsCannotEscapeTotalOperationBudget`, `BodyRead_ExhaustedFailuresRemainHttpErrorsAndDisposeEveryResponse`, and exact-byte/invalid-UTF8 matrices |
| "Consolidate redundant normalization and representations using existing primitives and simplest explicit ownership." | Deleted internal normalization middleware; one raw `NativeRequest.Create` call site, one public `SnapshotOptions` call site; `RequestSnapshot_SerializesCallerMetadataOnceAndLeavesCallerOptionsUnchanged`; `Adapter_SnapshotsExtraOptionsAndResolvesEffectiveModelBeforeDecoratedEntry` |
| "Preserve validation BEFORE cache, key shape/tenant/model/endpoint semantics, correct validation of raw/direct entry, cache deserialization safety and correlation, native cache BELOW projection/thresholds/tool IDs, per-call independent metadata/evidence, original raw evidence byte fidelity, mutable caller data snapshot guarantees and concurrency." | `RawNativeEntry_ValidatesUntrustedRequestsWithoutPublicNormalization`; `CacheHit_RevalidatesNativeEnvelopeAndQuestionCorrelation`; `Copy_ReusesImmutableNativeViewsButOwnsBytesAndDeepMutableMetadata`; `Batch_ClonesSharedEvidenceForIndependentConcurrentResponses`; retained pre-cache option, cache-key mutation, threshold reuse, fresh tool-ID and 32-call isolation tests |
| "finish existing eng\verify.ps1 -All -Offline on settled source" | Exit 0; 650/650 passing, eight scenarios/reports and fresh package receipt above. Narrow gates: 25 cancellation/buffering, 255 transport/provider/cache/content, 85 typed/tool cases passed. |

No dependencies or tooling were added. No push, PR edit, merge, publishing, live
inference, or trust/visibility changes were performed for this cleanup. These
are local offline results, not live API/model-quality/calibration assurance.
Linux CI was not executed for this unpushed change.

## Prior MEAI-first architectural revision at 26bc421

Status: **all six approved revision tasks complete; settled offline/package proof
passed on Windows**. `eng\verify.ps1 -All -Offline` exited 0 with **599 core +
28 evaluation tests = 627 passed, 0 failed, 0 skipped**. The nonincremental
Release solution build had zero warnings/errors. All 83 source/test/sample/
consumer/configuration inputs checked before and after the final gate were
unchanged during execution.

All eight revised scenarios passed: native, pipeline, evaluate, tool, schema,
route, select-tool and dispatch. Actual aieval JSON/HTML reports include
`fixed-draft`, `jev-primary`, and `visible-child-failure`. Primary evidence
readback requires exactly one result with the current invocation's unique
proof marker; neither empty enumeration nor a stale report can pass.

The fresh package-only consumer passed native evidence/serialization, real
generic DTO and primitive output, evaluation, and standard one-shot tools:
**five decorated calls, five synthetic HTTP sends**, with no real service call.
Package receipt: `artifacts\package-proof\ea7efaa350c74f609f7a1b413dfb90d8`.
All **35 external archives** retained lock hashes and verified signatures; only
the **two local unsigned proof artifacts** used the isolated filesystem-only
consumer exception. Repository/global trust policy was not weakened.
Final log: `artifacts\revision-all-verification.log`.

### Revised acceptance matrix

These are current test names, not the historical baseline names below.

| Approved requirement | Settled evidence |
|---|---|
| "A consumer holding only `IChatClient` can use actual `GetResponseAsync<bool>` and a normal finite DTO" | `TypedApi_PrimitiveBooleanUsesRealMeaiDataWrapper`; `TypedApi_NormalOpenOptionalDtoUsesClientPolicyAndStandardDescriptions` |
| "Native question/answer correlation is checked before response emission. Consumers read typed `JevResultContent` without supplying question snapshots." | `GetResponse_RejectsCorrelationAndProtocolViolations`; `Constructor_OwnsExactUtf8AndImmutableAnswersWithoutQuestionSnapshot` |
| "Real JSON roundtrips through `AIContent`, `DataContent`, `ChatResponse`, and the configured distributed cache retain the custom runtime type, native JSON, typed answers, and additive fields." | `RegisteredJson_RoundtripPreservesSubtypeExactPayloadAnnotationsAndAdditiveFields`; `CacheHit_RestoresEveryNativeFieldThroughRealSerialization`; `Json_DefaultOptionsRemainUnregisteredAndFailExplicitly` |
| "Standard typed deserialization still works with adjacent native content; primitive wrapping and last-message text rules are respected." | `TypedApi_PrimitiveBooleanUsesRealMeaiDataWrapper`; `TypedApi_NormalOpenOptionalDtoUsesClientPolicyAndStandardDescriptions`; `BufferedStreaming_ReconstructsProjectionEvidenceAndCountsUsageOnce` |
| "Ordinary MEAI schema descriptions and function schemas work without a second reflection pass. Unsupported shapes fail before any native request." | `TypedApi_RootAndNestedDescriptionAttributesReachOnlyDescendantQuestions`; `SchemaDescriptions_ObjectReferenceOverlaysPreserveAncestorsAndLeafOverrides`; `StandardToolModes_UseOrdinaryFactorySchemasAndBoundNativeBatches`; `Schema_RejectsEveryUnlistedKeywordBeforeDecoratedHttp` |
| "Policy changes reuse native inference where appropriate, but reapply acceptance." | `RegisteredNativeCache_ReappliesChangedClientPolicyAndOwnsPerCallMetadata`; `SchemaDescriptions_OnlyDescriptionChangesMissNativeCacheWhileThresholdsReuse`; `Decision_AbstainsWithoutDefaultTypedValueAndExceptionRetainsSameEvidence` |
| "Normal `AIFunctionFactory.Create` plus `ChatOptions.Tools` yields a standard proposed call. Without invocation middleware the function executes zero times." | `StandardToolModes_UseOrdinaryFactorySchemasAndBoundNativeBatches`; `Select_InfersOnlySelectedArgumentsAndSnapshotsOptionsOnceAcrossBothBatches` |
| "Configured standard invocation middleware executes an authorized/approved function exactly once, preserves outer decorators, makes no post-execution Jev call, and never retries side effects on errors." | `Invoke_AuthorizedSelectionPreservesOriginalOuterWrappersApprovalAndHostContext`; `Invoke_FunctionExceptionPropagatesOnceWithoutAnotherModelOrToolLoop`; real `dispatch` scenario |
| "Approval denial/cancellation, invalid or changed arguments, unknown calls, no-match, and uncertainty execute zero functions." | `ApprovalResume_HostRejectsTamperUnknownCorrelationAndReplayBeforeMiddleware`; `Invoke_DeniedAuthorizationOrMissingSeparateApprovalCannotBypassNestedMarker`; `Invoke_CancellationAtAnyHostGatePreventsSideEffects`; `Select_NoMatchOrLowConfidenceStopsAfterSelectionWithoutExecutableCall` |
| "Real evaluation reports preserve a portable, lossless snapshot of custom evidence, without mutating the live response or replacing MEAI reporting." | `Reporting_JevPrimaryResponseUsesLosslessPortableHostSnapshotWithoutMutatingLiveContent`; `Evaluate_JevInputSnapshotsPlainDataWithMetadataAndAnnotationsAndProjectsActualJson`; real persisted `jev-primary` readback and aieval artifacts |
| "Existing exact-number, reference, UTF-8, cache isolation, transport, cancellation, routing, and privacy regressions still hold at the new public boundary." | `Schema_RejectsWholeScoreRangeBeyondExactMaximumBeforeHttp`; `FixedArguments_EnforceExactMinimumBeyondDoublePrecision`; `FixedIntegralJson_InvokesRealApprovedLongFunctionWithoutRounding`; `ResponseBytes_InvalidUtf8RejectsWithoutRetryEvidenceOrCacheEntry`; `ResponseCancellation_AfterNativeCompletionPreventsEvidenceAndTypedProjection`; retained transport, 32-call isolation, routing and telemetry matrices in the 599-case core run |
| "A fresh package-only consumer exercises native content, ordinary typed output, and evaluation. The complete settled offline proof is rerun before completion." | `eng\PackageConsumer\Program.cs`, package-only `.csproj`, signed provenance and fresh isolated consumer receipt above; successful settled `eng\verify.ps1 -All -Offline` |

The pinned default serializer **throws for unregistered AIContent/ChatResponse**.
Statically declared, unregistered DataContent instead preserves bytes/media but
can lose the runtime subtype; tests document that distinction rather than
claiming it throws. Registered roundtrips preserve the type through both bases.
Reporting deliberately uses the portable base snapshot and explicit reconstruction.

Review regressions also cover root and nested `DescriptionAttribute`,
response-format `SchemaDescription`, escaped object-ref use-site context,
explicit leaf precedence, unchanged schemas, sibling isolation, and independent
description-only cache misses. Threshold-only changes still reuse native evidence.
Successful HTTP bytes are retained exactly despite misleading charset headers;
invalid UTF-8 cannot become replacement-decoded evidence. Evaluation again
rejects unsupported top-level JSON Boolean/numeric context before the judge.

**Limits:** Linux CI and live compatibility/calibration remain **NOT RUN**.
The live project compiles but was not executed. No new push, PR, publishing,
visibility change, license choice or global auth/trust change was performed.
Historical counts below are **not** the revised result.

## Historical completed proof at a2b4877

The following records the superseded API proof and its then-current test names.
They are retained as historical evidence, not current public API instructions.

Status: **bounded offline proof implemented at the baseline commit**. Offline evidence does not prove live
compatibility, model quality, calibration, or production fitness.

| Gate | Status |
|---|---|
| P0 dependency restore, solution Release build, published routing runtime probe | PASS on Windows |
| A native and pipeline | PASS on Windows: 293/293 deterministic cases, 0 skipped |
| B evaluator/reporting/function middleware | PASS on Windows: 300 core + 26 evaluation cases, actual aieval reports |
| C schema/typed outputs/routing/selection/invocation | PASS on Windows: final core suite 551 cases, including all settled review regressions |
| D1 package-only consumer | PASS on Windows: two local packages, 35 verified external archives, three public package paths |
| Linux CI | NOT RUN locally |
| Live service contract / quality | NOT RUN; requires separate approval |

P0: SDK10.0.401; normal repository locked restore and aieval10.10.0 executable
confirmed. `BootstrapTests.RoutingChatClient_Create_SelectsAndInvokesDownstreamExactlyOnce`
passed using the restored public experimental RoutingChatClient and VSTest.
Build had 0 warnings/errors. Package signature enforcement remains enabled.

## Stage A

`pwsh -NoProfile -File .\eng\verify.ps1 -Stage A -Offline` completed with exit 0:
locked restore, tool restore, Release build (0 warnings/errors), 293 tests passed,
native and real distributed-cache sample scenarios passed. The cache scenario
observed two calls / one HTTP send. Live project separately reports **0 passed,
3 skipped**, not live verification. Linux CI has not run.

| Required measurable result | Exact test evidence |
|---|---|
| Three question types / one HTTP attempt / complete evidence | `NativeChatClientTests.GetResponse_MixedBatchPreservesCanonicalEnvelope` |
| Object/array/null versus literal text; malformed UTF-8 rejected | `NativeChatClientTests.GetResponse_PreservesJsonStateVersusLiteralText`; `NativeInputRegressionTests.GetResponse_RejectsInvalidUtf8BeforeSend` |
| Missing/duplicate/unknown/wrong answers fail | `NativeResponseContractTests.GetResponse_RejectsCorrelationAndProtocolViolations` |
| Probabilities/confidences/fractions and inclusive tolerance preserved | `NativeResponseContractTests.Decode_EnforcesAbsoluteSumToleranceWithoutRepair` |
| Native helper passes through decorated client | `NativeChatClientTests.GetTypeSafeResponse_UsesDecoratedClientAndStandardContent` |
| Real cache serialization, no RawRepresentation | `NativeCachingTests.CacheHit_RestoresEveryNativeFieldThroughRealSerialization` |
| State/question/rubric/order/model/endpoint/tenant cache sensitivity | `NativeCachingTests.CacheKey_ChangesForEachInferenceMutationAcrossSharedKeyedClients` |
| Primed cache cannot bypass capability validation | `NativeCapabilityTests.UnsupportedOptions_RejectBeforeHttpAndBeforePrimedRealCache` |
| Bounded status/connection retries and permanent errors | `TypeSafeHttpTransportTests.RetryPolicy_BoundsTwoRetriesAndPreservesTerminalStatus`; `Errors_PermanentStatusesNeverRetryAndDoNotLeakDiagnosticBodies` |
| Retry-After delta/date and total budget | `TypeSafeHttpTransportTests.RetryAfter_RespectsDeltaOrDateBeforeStartingNextAttempt`; `RetryAfter_LongerThanTotalBudgetNeverStartsLateAttempt` |
| Send/backoff/body/enumeration cancellation | `TypeSafeHttpTransportTests.Cancellation_DuringSendBackoffOrBodyStopsFurtherAttempts`; `BufferedStreamingTests.GetStreamingResponse_CancellationDuringEnumerationStopsUpdates` |
| Attempt/total stalled-body bounds and disposal | `TypeSafeHttpTransportTests.BodyRead_StallIsInsideAttemptBudgetAndRetryDisposesFirstResponse`; `BodyRead_RepeatedStallsCannotEscapeTotalOperationBudget` |
| Known/unknown-length body limits | `TypeSafeHttpTransportTests.BodyLimit_EnforcesExactSuccessAndErrorBoundariesWithAndWithoutContentLength`; `DirectClientBufferingTests.DirectClient_EnforcesUnknownLengthBodyBoundary` |
| Standard update aggregation, usage once | `BufferedStreamingTests.GetStreamingResponse_ReconstructsNormalSemanticsAndUsageOnceWithMeaiAggregator` |
| 32 controlled simultaneous independent calls | `NativeConcurrencyTests.GetTypeSafeResponse_ThirtyTwoControlledCallsRemainIsolatedAndOwnCallerSnapshots` |
| Models GET and shared transport | `TypeSafeModelAndOwnershipTests.GetModels_DecodesWrapperUsingSharedGetBearerAndOrigin` |

Regression fixes: strict UTF-8 instead of replacement fallback; reject unknown
top-level question fields; reuse bounded body reads for borrowed clients; decimal
aggregate validation at both tolerance endpoints; Retry-After date calculation
uses the injected pipeline clock. No assertions were weakened or tests skipped.

## Stage B

`pwsh -NoProfile -File .\eng\verify.ps1 -Stage B -Offline` completed with exit 0:
300 core cases (A plus 7 function cases), 26 evaluator/reporting cases, all
passed without skips; Release build 0 warnings/errors. The `evaluate` and `tool`
samples passed. Existing aieval 10.10.0 generated HTML and JSON reports; the gate
checked persisted types, fractional values, native evidence, primary model and
visible CompositeEvaluator error diagnostics.

| Required measurable result | Exact test evidence |
|---|---|
| Multiple metric definitions / one native inference / exact numeric-string kinds | `Evaluate_BatchesMetricsThroughDecoratedClientAndPreservesStateAndEvidence` |
| No fabricated interpretation; explicit host interpretation | `Evaluate_UsesOnlyExplicitHostInterpretation` |
| Full metadata, contexts and primary response survive actual disk reporting | `Reporting_DisposalPersistsAndReloadsCompleteMetricsContextAndPrimaryResponse` |
| Composite child errors cannot count as success | `Composite_FailingChildPersistsErrorDiagnosticAndCannotPass` |
| Structured JSON reaches judge as data rather than opaque base64 | `Evaluate_ProjectsStructuredValuesWithoutEncodingAndKeepsReportingContents` |
| Invalid UTF-8 fails before decorated judge or HTTP | `Evaluate_RejectsInvalidUtf8BeforeCallingJudge` |
| Concurrent evaluator snapshots and distinct batch IDs | `Evaluate_ConcurrentCallsOwnInputSnapshotsAndSeparateBatchIds` |
| Real UseFunctionInvocation, decorated judge, full native result | `FunctionInvocation_TwoTurnsReturnFullNativeEnvelopeThroughDecoratedClient` |
| Cancellation never enters judge; failures remain failures | `FunctionInvocation_CancellationDoesNotInvokeNativeJudge`; `FunctionInvocation_ServiceFailureRemainsVisibleInsteadOfSuccess` |

The tool closure and native helper now check cancellation before invoking the
decorated judge. Native evaluator state projection is distinct from standard
report serialization, avoiding DataContent's opaque data URI representation.

## Stage C

Actual `schema`, `route`, `select-tool`, and `dispatch` sample scenarios execute
successfully offline. The schema sample verifies the generic MEAI object and
Boolean-root paths and one cached distribution across two acceptance thresholds.
Tool selection records exactly two batches/zero invocations; dispatch then invokes
the required-approval preview once with no further model call.

Final deterministic core suite: **551 passed, 0 failed, 0 skipped** (A/B/C and
bootstrap); separate evaluation suite **26 passed, 0 failed, 0 skipped**.
This supersedes the earlier 243-case C snapshot: the settled suite includes
the exact comparison regressions and the real integer-parameter invocation fix.
Full orchestration is `pwsh -NoProfile -File .\eng\verify.ps1 -All -Offline`.
It performs locked restore, nonincremental Release build, all offline tests,
eight scenarios, actual aieval reports and package-only verification.

| Requirement | Exact schema/tool evidence |
|---|---|
| "Every unsupported schema fails with a JSON pointer before any HTTP send" | `Schema_RejectsEveryUnlistedKeywordBeforeDecoratedHttp`; `Schema_RejectsInvalidShapesAnnotationsAndConstraintsBeforeHttp`; `Schema_RejectsMissingSemanticsAndConflictingBindingsBeforeHttp` |
| "Object/nested object/enum/bool/explicit Score schemas and MEAI root wrappers exercise actual GetResponseAsync<T>" | `TypedApi_ProjectsRequiredNestedDtoWithoutMutatingCallerOrOwningNativeDefault`; `TypedApi_PrimitiveBooleanUsesRealMeaiDataWrapper` |
| Depth 32 / leaves 256 exact bounds and escaped local references | `Compile_EnforcesExactObjectTraversalDepth`; `Compile_EnforcesExactReferenceTraversalDepth`; `Compile_EnforcesExactDecisionLeafLimit`; `Compile_EscapedPointersAndSharedLocalReferencesRemainDistinct` |
| "Below/at false bound, abstention interval, at/above true bound behave exactly as specified" | `BooleanPolicy_InclusiveBoundsAndOpenIntervalNeverUseSchemaDefault` |
| "Below/at acceptance threshold tested; server confidence is not replaced by winning probability" | `ConfidencePolicy_UsesInclusiveServerConfidenceNotDistribution` |
| "No incomplete/default T; explicit result or documented exception retains native evidence" | `Decision_AbstainsWithoutDefaultTypedValueAndExceptionRetainsSameEvidence`; `Decision_LeafOrApplicationValidationAbstainsWithExplicitReason` |
| "Change only a projection threshold -> reuse native response; decision may change while probabilities do not" | `NativeCache_ReusesThresholdChangesButMissesRubricAndBothInstructionKinds` |
| "Probabilities/confidences/fractions" preserved after projection serialization/stream aggregation | `ProjectedResponse_RoundtripsJsonElementEvidenceSeparatelyFromTypedJson`; `BufferedStreaming_ReconstructsProjectionEvidenceAndCountsUsageOnce` |
| Exact Score constraint (maximum `1.9999999999999999` cannot admit `[0,2]`) | `Schema_RejectsWholeScoreRangeBeyondExactMaximumBeforeHttp` |
| Exact integer constraint above `2^53`, below-minimum rejection and valid boundary | `FixedArguments_EnforceExactMinimumBeyondDoublePrecision` |
| Reference-use-site annotations preserved without source mutation; explicit instructions win | `Schema_ReferenceDescriptionsOverlayThroughChainsWithoutMutatingDefinitions`; `Schema_ExplicitBindingInstructionsOverrideReferenceDescriptionWithoutChangingSchema`; `OptionalArguments_ReferenceSiblingDefaultOverridesResolvedAnnotation` |
| Actual approved `long` invocation for `1.0`, `1e2`, plain and decimal-notation large integers | `FixedIntegralJson_InvokesRealApprovedLongFunctionWithoutRounding` |
| "One selection call plus zero/one selected-argument call; all function invocation counters remain zero" | `Select_InfersOnlySelectedArgumentsAndSnapshotsOptionsOnceAcrossBothBatches`; `Select_ZeroParameterFactoryFunctionUsesOneNativeBatch`; `Select_NoMatchOrLowConfidenceStopsAfterSelectionWithoutExecutableCall` |
| All candidates prevalidated, optional/fixed strategy explicit, unsupported assertions still rejected | `Configuration_CompilesEvenUnselectedCandidatesBeforeHttp`; `OptionalArguments_RequireExplicitFixedOrDeclaredDefaultOmission`; `FixedArguments_DoNotWaiveUnsupportedAssertions` |
| "Authorized accepted call invokes once; denied/uncertain/invalid/canceled call invokes zero times" | `Invoke_AuthorizedSelectionPreservesOriginalOuterWrappersApprovalAndHostContext`; `Invoke_NonacceptedSelectionCannotReachAuthorizationOrFunction`; `Invoke_RechecksDeclarationArgumentsAndApplicationPolicyAfterAuthorization`; `Invoke_CancellationAtAnyHostGatePreventsSideEffects` |
| "Approval-required functions cannot be unwrapped or silently executed" | `Invoke_DeniedAuthorizationOrMissingSeparateApprovalCannotBypassNestedMarker` |
| No side-effect retry/model loop on function error | `Invoke_FunctionExceptionPropagatesOnceWithoutAnotherModelOrToolLoop` |

| Requirement | Exact routing evidence |
|---|---|
| "One classifier call and one selected-target call; all others zero; original downstream options unchanged" | `Route_UsesDecoratedClassifierOnceAndDelegatesUnchangedRequestAndUsage` (normal and streaming, decorated spy, one-shot enumerable) |
| "Low confidence follows explicit policy; HTTP/protocol failure propagates, not treated as abstention" | `Route_LowServerConfidenceAbstainsOrUsesOnlyExplicitFallback`; `Route_HttpAndProtocolErrorsNeverBecomeConfidenceFallback` |
| "At least 32 controlled simultaneous independent calls preserve their own states/answers and do not mutate shared options" | `Route_ThirtyTwoControlledRequestsKeepDecisionsOptionsAndTargetsIsolated` |
| Stateful handles and unsupported default modalities fail before classification | `Route_HandlesAndUnsupportedDefaultModalitiesRejectBeforeClassifier_CustomProjectionIsExplicit` |
| Borrowed ownership and cancellation before delegation | `Route_CancellationAfterObservationPreventsTarget_AndDisposalBorrowsAllClients` |
| Bounded route configuration | `Route_RejectsInvalidConfigurationWithoutCalls` |

Compiler review fixes were retained as regressions, not hidden by altered
thresholds: reference-use-site descriptions/defaults survive resolution;
invalid native entries retain schema pointers; integer/range validation uses
exact bounded arithmetic; integral fixed JSON is canonicalized before the real
MEAI function binder. The invocation regression first reproduced three failures
(`1.0`, `1e2`, `9007199254740993.0`) while the plain large integer passed; all four
now pass with one separately authorized/approved invocation each.

## D1 package and handoff

`eng\verify-packages.ps1` exits 0 after checking **35 external archives** against
the two library lockfiles with NuGet's canonical content-hash API, identity and
signature verification. It locally packs `TypeSafe.MEAI` and
`TypeSafe.MEAI.Evaluation` at `0.1.0-private.1`; archive inspection allows only
library outputs, README and NuGet metadata and verifies expected dependencies.
No source/test/sample/report artifacts are packaged.

| Requirement | Evidence |
|---|---|
| "Local packed libraries work in a minimal consumer that references the packages rather than project sources" | `eng\PackageConsumer\PackageConsumer.csproj` contains only two exact PackageReferences; script checks resolved assets contain no project references |
| Native, schema/typed APIs and separate evaluator all work from packages | `eng\PackageConsumer\Program.cs`: native full envelope, actual `GetResponseAsync<Decision>`, `IEvaluator`; exactly three decorated calls/three synthetic sends |
| Signed external provenance retained; unsigned exception only for local artifacts | `eng\verify-packages.ps1`: hash/identity/signature gates; fresh filesystem-only feed/config/cache; locked consumer restore; post-restore hashes checked against original closure |
| No publishing, licensing, certificate installation or global trust change | Local pack/run only; repository still has `signatureValidationMode=require` and all explicit repository signers use `allowUntrustedRoot=false` |

The strict fresh-package check exposed one previously cached dependency signed
by the official historical NuGet repository certificate ending `1477DF4`.
Its independently documented fingerprint was added to the repository's explicit
list, preserving signature/chain/timestamp validation. No external package was
accepted unsigned. Our two local artifacts are deliberately unsigned; only the
isolated consumer configuration permits them. See
[provenance and policy details](architecture.md#local-package-only-proof).

Successful package output/provenance is located via
`artifacts\package-proof\latest.txt`; each run has `feed`, `consumer`,
`provenance.json`, and `external-signatures.log`. The final orchestration log is
`artifacts\all-verification.log`. These synthetic generated artifacts are ignored,
not published.

**Remaining limits:** Linux CI is configured but NOT RUN. All live tests remain
NOT RUN (the isolated live runner previously reported three explicit skips).
SDK/API contract discrepancies remain unresolved without separately approved
synthetic live checks. Thresholds are not calibrated. This proof makes no
production-fitness, model-quality, public-publication or license claim.
