# Decision Arrow (experimental, local only)

The [contract and measurement scope](../../../docs/experiments/decision-arrow.md)
is additive to the decision comparison. No models/corpus are downloaded by
help, restore, build, tests, fixtures, preflight or export.

From this isolated repository worktree, explicitly opt into the verified feeds:

```powershell
$env:ImportDirectoryBuildTargets = 'false' # This host's documented external Arcade import workaround.
.\eng\experiments\Restore-DecisionArrow.ps1 `
  -ArrowFeed <immutable-arrow-baseline-feed> `
  -ArtifactDirectory artifacts\decision-arrow-restore `
  -AllowUnsignedLocalPackages -UpdateLocks
dotnet build eng\DecisionArrowExperiments.slnx --no-restore
dotnet test tests\DecisionInference.Arrow.Tests\DecisionInference.Arrow.Tests.csproj --no-restore
```

The shell-local build workaround changes no ancestor/global file. Omit it on
hosts without the unrelated external targets import. Remove `-UpdateLocks`
after the initial deliberate dependency update. All actual package lock files
are committed; root signing requirements remain unchanged.

```powershell
dotnet run --project samples\experiments\DecisionArrow --no-build --no-restore -- --help
dotnet run --project samples\experiments\DecisionArrow --no-build --no-restore -- fixture <empty-output> 257 HighPrecision <questions.v1.json>
dotnet run --project samples\experiments\DecisionArrow --no-build --no-restore -- summarize <manifest.json> <trusted-contract.json>
dotnet run --project samples\experiments\DecisionArrow --no-build --no-restore -- transport <empty-output> 65536 TwoDecimalPlaces <questions.v1.json>
```

Fixtures are synthetic, including a selected non-argmax choice and rounded
score expectation outside the rubric range. They support zero/one/257/larger
row counts and all three current precision declarations. The exact frozen
five-question file is owned by the ML.NET sample; an omitted question path
uses only small offline three-question smoke definitions.

```powershell
dotnet run --project samples\experiments\DecisionArrow --no-build --no-restore -- preflight <pinned-assets-directory> <questions.v1.json> <states.v1.jsonl> <empty-preflight-output>
dotnet run --project samples\experiments\DecisionArrow --no-build --no-restore -- export <pinned-assets-directory> <questions.v1.json> <states.v1.jsonl> <empty-output> <compiled-source-SHA> <immutable-package-receipt.json>
dotnet run --project samples\experiments\DecisionArrow --no-build --no-restore -- export <pinned-assets-directory> <questions.v1.json> <states.v1.jsonl> <empty-output> <compiled-source-SHA> <immutable-package-receipt.json> native 16 4
dotnet run --project samples\experiments\DecisionArrow --no-build --no-restore -- parity <pinned-assets-directory> <questions.v1.json> <synthetic-states.jsonl> <empty-output> 4
dotnet run --project samples\experiments\DecisionArrow --no-build --no-restore -- compare <scalar-manifest.json> <native-manifest.json> <trusted-contract.json> <new-report.json>
```

`export` always preflights all records again and stops on any failure; it also
verifies frozen inputs before publication. It requires the compiled exporter
commit and a clean producer package receipt. Native CPU ORT 1.23.2 and
Microsoft.ML.Tokenizers 2.0.0 stay unchanged; the default is one CPU thread.
Specify identical thread counts for scalar/native comparisons. Warmup is an
explicit separate first call, not a dropped dataset row; raw call metrics
include padded tensor shapes and inference GC deltas.
The [existing Laya asset prerequisites](../../quickstarts/Laya/README.md) apply.
An explicitly approved retained external asset directory can be read without
building or modifying its owning checkout.

Preflight-only output is marked `preflight-passed-not-scored`, not a dataset.
Failure/cancellation leaves partial artifacts and exits nonzero. Synthetic
transport results cannot satisfy retained-model parity or real predictor
quality gates.

After a clean coherent commit:

```powershell
.\eng\experiments\Publish-DecisionArrow.ps1 -Feed <new-immutable-adapter-feed> `
  -RestoreArtifactDirectory artifacts\decision-arrow-restore `
  -ArrowBaselineReceipt <arrow-baseline-feed\receipt.json>
```

No public package publishing occurs. The package-only consumer under
`eng\experiments\PackageConsumer` can restore with the optional `-DecisionFeed`
mapping and the receipt's exact `DecisionAdapterVersion`, then read the
completed 257-row fixture via the same public standard Arrow API.
