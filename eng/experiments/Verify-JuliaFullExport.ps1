[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $ExperimentDirectory,
    [Parameter(Mandatory)][string] $DatasetDirectory,
    [Parameter(Mandatory)][string] $SelectedProfilePath,
    [Parameter(Mandatory)][string] $ReaderOutputDirectory,
    [Parameter(Mandatory)][string] $HandoffDirectory
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $ExperimentDirectory).Path
$dataset = (Resolve-Path -LiteralPath $DatasetDirectory).Path
$readerOutput = [IO.Path]::GetFullPath($ReaderOutputDirectory)
$handoff = [IO.Path]::GetFullPath($HandoffDirectory)
foreach ($path in @($dataset, $readerOutput, $handoff)) {
    if (-not $path.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Full-export validation outputs must remain inside the experiment namespace.'
    }
}
if ((Test-Path -LiteralPath $readerOutput) -or (Test-Path -LiteralPath $handoff)) {
    throw 'Public-reader and handoff directories must be new.'
}
function Artifact([string] $path) {
    $resolved = (Resolve-Path -LiteralPath $path).Path
    if (-not $resolved.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Dataset evidence must remain inside the experiment namespace.'
    }
    return [ordered]@{
        path = [IO.Path]::GetRelativePath($root, $resolved)
        sizeBytes = (Get-Item -LiteralPath $resolved).Length
        sha256 = (Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
function WriteNewJson([string] $path, $value) {
    $file = [IO.File]::Open($path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        $bytes = [Text.UTF8Encoding]::new($false).GetBytes(($value | ConvertTo-Json -Depth 30))
        $file.Write($bytes, 0, $bytes.Length)
    }
    finally { $file.Dispose() }
}

$profile = Get-Content -LiteralPath $SelectedProfilePath -Raw | ConvertFrom-Json
$qualificationArtifact = Artifact $profile.qualificationReceiptPath
if ($profile.mode -cne 'scalar-cpu' -or $profile.status -cne 'SELECTED' -or
    $profile.intraThreads -ne 4 -or $profile.interThreads -ne 1 -or
    $qualificationArtifact.sha256 -cne $profile.qualificationReceiptSha256) {
    throw 'Selected CPU profile/qualification bytes changed.'
}
$qualification = Get-Content -LiteralPath $profile.qualificationReceiptPath -Raw | ConvertFrom-Json
$fingerprint = '78ed3dc0344f71f1d8515f5923dc2c1de3604e59e6b4a1c9cb89851e2b3cf05d'
if ($qualification.status -cne 'GO-READY' -or $qualification.featureFingerprint -cne $fingerprint -or
    $qualification.revisedAcceptance.oldGatePassed -ne $false -or
    $qualification.revisedAcceptance.status -cne 'RETIRED-AS-NONBLOCKING-BENCHMARK') {
    throw 'Explicit revised CPU authorization is absent.'
}
$manifestPath = Join-Path $dataset 'manifest.json'
$contractPath = Join-Path $dataset 'contract.json'
$runtimePath = Join-Path $dataset 'runtime.receipt.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$runtime = Get-Content -LiteralPath $runtimePath -Raw | ConvertFrom-Json
$preflightPath = $dataset + '.preflight\preflight.receipt.json'
$preflight = Get-Content -LiteralPath $preflightPath -Raw | ConvertFrom-Json
if ($manifest.status -cne 'complete' -or $manifest.rowCount -ne 5574 -or $manifest.outputBatchSize -ne 256 -or
    $manifest.featureFingerprint -cne $fingerprint -or $manifest.dataFile -cne 'decisions.arrow' -or
    $manifest.provenance.sourceCommit -cne 'fd14d3b1caf89886a5f6cc29fa6c01dcef199e01' -or
    $manifest.provenance.inputSha256 -cne 'c00636dc21f7a4e3bb099c508fbf24a407af546851e060c5852cb9aca54639fa' -or
    $manifest.provenance.questionsSha256 -cne '9e6e6c5c580934309cb1f9b4572c7b87a93ce22b32b50cb8b3af20caa20068ea' -or
    $manifest.provenance.executionMode -cne 'scalar-cpu' -or $manifest.provenance.cpuThreads -ne 4 -or
    $manifest.provenance.nativeStateBatchSize -ne 1 -or
    $runtime.status -cne 'FULL-EXPORT-VERIFIED' -or $runtime.full -ne $true -or $runtime.diagnostic -ne $false -or
    $runtime.rowCount -ne 5574 -or $runtime.featureFingerprint -cne $fingerprint -or
    $runtime.inputHash -cne $manifest.provenance.inputSha256 -or $runtime.precisionTolerance -ne 1e-6 -or
    $runtime.profile.mode -cne 'scalar-cpu' -or $runtime.profile.status -cne 'SELECTED' -or
    $runtime.profile.qualificationReceiptSha256 -cne $qualificationArtifact.sha256 -or
    $runtime.profile.intraThreads -ne 4 -or $runtime.profile.interThreads -ne 1 -or
    $runtime.runtime.nativeOrt -cne '1.23.2' -or $runtime.runtime.deviceIndex -ne $null -or
    $runtime.runtime.deviceUuid -ne $null -or $runtime.runtime.providerOptions -ne $null -or
    $runtime.runtime.arithmetic -cne 'ort1.23.2-cpu-fp32-intra4-inter1-default-sync-v1' -or
    $runtime.outputMemory.name -cne 'Cpu' -or $runtime.outputMemory.fp32 -ne $true -or
    $runtime.outputMemory.hostReadableBeforeSpan -ne $true -or
    $preflight.status -cne 'PASS' -or $preflight.rows -ne 5574 -or $preflight.full -ne $true -or
    $preflight.exactIds -ne $true -or $preflight.scoringOccurred -ne $false -or
    $preflight.truncation -ne $false -or @($preflight.failures).Count -ne 0) {
    throw 'Full manifest/runtime/preflight is not the exact approved completed CPU export.'
}
$dataArtifact = Artifact (Join-Path $dataset $manifest.dataFile)
$contractArtifact = Artifact $contractPath
$preparedArtifact = Artifact ($dataset + '.preflight\prepared.jsonl')
$originalPreparedArtifact = Artifact (Join-Path $root 'producer\original-preflight-v2\prepared.jsonl')
if ($dataArtifact.sha256 -cne $manifest.dataSha256 -or $dataArtifact.sizeBytes -ne $manifest.dataSize -or
    $contractArtifact.sha256 -cne $fingerprint -or [IO.File]::ReadAllText($contractPath) -cne $manifest.contractJson) {
    throw 'Completed data/contract bytes changed.'
}
if ($originalPreparedArtifact.sha256 -cne '8be3531e3196a9bf23658acec302244244429e6e36ddc040ef6392a0810e235c' -or
    $preparedArtifact.sha256 -cne $originalPreparedArtifact.sha256 -or
    $preparedArtifact.sizeBytes -ne $originalPreparedArtifact.sizeBytes -or
    $preflight.preparedSha256 -cne $preparedArtifact.sha256) {
    throw 'All5574 prepared IDs/shapes/tensor hashes differ from frozen ORIGINAL preflight bytes.'
}
foreach ($module in $runtime.runtime.modules) {
    if ((Get-FileHash -LiteralPath $module.path -Algorithm SHA256).Hash.ToLowerInvariant() -cne $module.sha256 -or
        -not @($qualification.cpu.runtime.modules | Where-Object sha256 -CEQ $module.sha256).Count) {
        throw 'Actual loaded CPU ORT module differs from its accepted closure.'
    }
}
foreach ($package in $runtime.packages) {
    $accepted = @($qualification.cpu.packages | Where-Object {
        $_.id -ceq $package.id -and $_.version -ceq $package.version -and
        $_.sha256 -ceq $package.sha256 -and $_.size -eq $package.size
    })
    if ($accepted.Count -ne 1) { throw 'Full runtime package differs from accepted isolated closure.' }
}
if ($runtime.packages.Count -ne $qualification.cpu.packages.Count) { throw 'Runtime package closure changed.' }

$repo = Split-Path (Split-Path $PSScriptRoot)
$readerProject = Join-Path $repo 'eng\experiments\PackageConsumer'
$assets = Get-Content -LiteralPath (Join-Path $readerProject 'obj\project.assets.json') -Raw | ConvertFrom-Json
$cache = @($assets.packageFolders.PSObject.Properties.Name)
if ($cache.Count -ne 1) { throw 'Public reader must use its isolated package cache.' }
$readerBin = Join-Path $readerProject 'bin\Release\net10.0'
$deps = Get-Content -LiteralPath (Join-Path $readerBin 'DecisionArrow.PackageConsumer.deps.json') -Raw | ConvertFrom-Json
$packages = @($assets.libraries.PSObject.Properties)
if ($packages.Count -ne 5 -or @($packages | Where-Object { $_.Value.type -cne 'package' }).Count -ne 0 -or
    @($deps.libraries.PSObject.Properties).Count -ne 6) {
    throw 'Public reader must have only five portable package dependencies and its own executable.'
}
$readerPackageArtifacts = @()
foreach ($library in $packages) {
    $parts = $library.Name.Split('/')
    $accepted = @($qualification.cpu.packages | Where-Object { $_.id -ceq $parts[0] -and $_.version -ceq $parts[1] })
    if ($accepted.Count -ne 1 -or $parts[0] -notin @('Apache.Arrow', 'Apache.Arrow.Scalars',
        'DecisionInference.Abstractions', 'DecisionInference.Arrow', 'Microsoft.Extensions.AI.Abstractions')) {
        throw 'Public reader dependency is not an exact portable pin.'
    }
    $folder = Join-Path $cache[0] $library.Value.path.Replace('/', '\')
    $nupkg = Join-Path $folder ($library.Name.Replace('/', '.').ToLowerInvariant() + '.nupkg')
    if ((Get-Item -LiteralPath $nupkg).Length -ne $accepted[0].size -or
        (Get-FileHash -LiteralPath $nupkg -Algorithm SHA256).Hash.ToLowerInvariant() -cne $accepted[0].sha256) {
        throw 'Public reader package bytes differ from accepted portable pins.'
    }
    $archive = [IO.Compression.ZipFile]::OpenRead($nupkg)
    try {
        $target = $assets.targets.'net10.0'.PSObject.Properties[$library.Name].Value
        foreach ($dll in $target.runtime.PSObject.Properties.Name) {
            $entry = $archive.GetEntry($dll)
            if ($null -eq $entry) { throw 'Expected public-reader DLL is absent from its exact package.' }
            $stream = $entry.Open()
            try { $hash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($stream)).ToLowerInvariant() }
            finally { $stream.Dispose() }
            if ((Get-FileHash -LiteralPath (Join-Path $readerBin ([IO.Path]::GetFileName($dll)))).Hash.ToLowerInvariant() -cne $hash) {
                throw 'Public reader runtime DLL differs from its package bytes.'
            }
        }
    }
    finally { $archive.Dispose() }
    $readerPackageArtifacts += $accepted[0]
}

New-Item -ItemType Directory -Path $readerOutput | Out-Null
$frozenReader = Join-Path $readerOutput 'runtime'
Copy-Item -LiteralPath $readerBin -Destination $frozenReader -Recurse
$timer = [Diagnostics.Stopwatch]::StartNew()
& dotnet (Join-Path $frozenReader 'DecisionArrow.PackageConsumer.dll') $manifestPath $contractPath 5574 2>&1 |
    Tee-Object -FilePath (Join-Path $readerOutput 'public-reader.console.log')
if ($LASTEXITCODE -ne 0) { throw "Public pinned reader failed with exit code $LASTEXITCODE; no full handoff published." }
$timer.Stop()
$readerReceiptPath = Join-Path $readerOutput 'public-reader.receipt.json'
WriteNewJson $readerReceiptPath ([ordered]@{
    schemaVersion = 1; status = 'PASS'; rowCount = 5574
    exactExpectedIdsAndOrder = 'Original1..5574, independent plan count, full sequential replay'
    validation = 'Public pinned package reader: complete manifest/hash/schema/nested values/IDs/fullEOS/trailing guards, followed by full replay'
    elapsedMilliseconds = $timer.Elapsed.TotalMilliseconds
    costScope = 'One completion/ABI verification invocation, not an allocation/performance rerun'
    readerSourcePath = 'eng\experiments\PackageConsumer\Program.cs'
    readerSourceSha256 = (Get-FileHash -LiteralPath (Join-Path $readerProject 'Program.cs')).Hash.ToLowerInvariant()
    dependencies = $readerPackageArtifacts
    frozenReaderClosure = @(Get-ChildItem -LiteralPath $frozenReader -File -Recurse | Sort-Object FullName | ForEach-Object { Artifact $_.FullName })
    manifest = Artifact $manifestPath; contract = $contractArtifact; data = $dataArtifact
})

$artifacts = [ordered]@{
    manifest = Artifact $manifestPath; contract = $contractArtifact; data = $dataArtifact
    runtime = Artifact $runtimePath; preflight = Artifact $preflightPath
    prepared = $preparedArtifact; originalPrepared = $originalPreparedArtifact
    selectedProfile = Artifact $SelectedProfilePath; qualificationReceipt = $qualificationArtifact
    priorQualificationReceipt = Artifact (Join-Path $root $qualification.priorQualification.path)
    portableReceipt = Artifact $profile.packageReceiptPath
    ownerAuthorization = Artifact (Join-Path $root $qualification.ownerAuthorization.path)
    publicReaderReceipt = Artifact $readerReceiptPath
}
New-Item -ItemType Directory -Path $handoff | Out-Null
WriteNewJson (Join-Path $handoff 'full-export.receipt.json') ([ordered]@{
    schemaVersion = 1; status = 'FULL-EXPORT-VERIFIED-PUBLIC-PACKAGE-READER'
    rowCount = 5574; datasetDirectory = $dataset; featureFingerprint = $fingerprint
    acceptedSources = [ordered]@{
        runtimeSourceCommit = $qualification.runtimeSourceCommit
        portableSourceCommit = $qualification.packageSourceCommit
        originalSourceCommit = $qualification.originalSourceCommit
    }
    expectedIds = [ordered]@{ first = 1; last = 5574; count = 5574; order = 'exact' }
    preparedTensorCount = 5574; preparedMatchesFrozenOriginalBytes = $true
    artifacts = $artifacts
    costs = [ordered]@{
        loadMilliseconds = $runtime.loadMilliseconds; warmupMilliseconds = $runtime.warmupMilliseconds
        extractionAndValidationMilliseconds = $runtime.extractionAndValidationMilliseconds
        scope = $runtime.timing; transfersMeasuredSeparately = $false
        repeatedPerformanceComparison = $false
    }
    revisedConsumerAllocationAcceptance = $qualification.revisedAcceptance
    cuda = 'REJECTED-UNCHANGED1e-6'; oldEvidenceOverwritten = $false
    consumerStudy = 'Authorized separate owner; full study/holdout artifacts not produced by this export'
    verifierSourceSha256 = (Get-FileHash -LiteralPath $PSCommandPath).Hash.ToLowerInvariant()
})
Write-Host 'Completed full5574 CPU export/public-package reader handoff published; no consumer study success inferred.'
