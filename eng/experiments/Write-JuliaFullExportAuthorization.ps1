[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $ExperimentDirectory,
    [Parameter(Mandatory)][string] $OwnerGoReceiptPath,
    [Parameter(Mandatory)][ValidatePattern('^[0-9a-f]{64}$')][string] $OwnerGoReceiptSha256,
    [Parameter(Mandatory)][string[]] $ConsumerEvidencePaths,
    [Parameter(Mandatory)][string] $OutputDirectory
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $ExperimentDirectory).Path
$output = [IO.Path]::GetFullPath($OutputDirectory)
if (-not $output.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'New authorization must remain inside the experiment namespace.'
}
if (Test-Path -LiteralPath $output) { throw 'Full-export authorization directory must be new.' }

function Artifact([string] $path) {
    $resolved = (Resolve-Path -LiteralPath $path).Path
    if (-not $resolved.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Authorization evidence must remain inside the experiment namespace.'
    }
    return [ordered]@{
        path = [IO.Path]::GetRelativePath($root, $resolved)
        sizeBytes = (Get-Item -LiteralPath $resolved).Length
        sha256 = (Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
function VerifyArtifact($artifact) {
    $actual = Artifact (Join-Path $root $artifact.path)
    if ($actual.sizeBytes -ne $artifact.sizeBytes -or $actual.sha256 -cne $artifact.sha256) {
        throw "Frozen evidence changed: $($artifact.path)"
    }
}
function WriteNewJson([string] $path, $value) {
    $file = [IO.File]::Open($path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        $bytes = [Text.UTF8Encoding]::new($false).GetBytes(($value | ConvertTo-Json -Depth 40))
        $file.Write($bytes, 0, $bytes.Length)
    }
    finally { $file.Dispose() }
}

$ownerArtifact = Artifact $OwnerGoReceiptPath
if ($ownerArtifact.sha256 -cne $OwnerGoReceiptSha256) { throw 'Explicit owner GO receipt hash changed.' }
$owner = Get-Content -LiteralPath $OwnerGoReceiptPath -Raw | ConvertFrom-Json
if ($owner.status -cne 'OWNER-GO-REVISED-ACCEPTANCE' -or $owner.authorizedMode -cne 'scalar-cpu' -or
    $owner.producerFullExportAuthorized -ne $true -or $owner.expectedExportRows -ne 5574 -or
    $owner.stateBatchSize -ne 1 -or $owner.outputBatchSize -ne 256 -or
    $owner.consumerAllocation.status -cne 'RETIRED-AS-NONBLOCKING-BENCHMARK' -or
    $owner.consumerAllocation.oldGatePassed -ne $false -or
    $owner.consumerAllocation.preserveAllV1ThroughV6Receipts -ne $true -or
    $owner.unchangedHardRequirements.numericTolerance -ne 1e-6 -or
    $owner.unchangedHardRequirements.exactSourceFeatureAndAssociationChecks -ne $true -or
    $owner.unchangedHardRequirements.cuda -cne 'REJECTED-HARD-NUMERICAL-GATE' -or
    $owner.unchangedHardRequirements.noNegativeEvidenceOverwrite -ne $true) {
    throw 'Receipt is not the explicit revised CPU-only full-export authorization.'
}

$priorPath = Join-Path $root 'selected-profile\cpu-fallback-v1\qualification.receipt.json'
$priorArtifact = Artifact $priorPath
if ($priorArtifact.sha256 -cne '6602c1f86a251855eb2f647dd700c6402870a0c1960c81d45d7484c2cfc50f07') {
    throw 'Original qualified CPU fallback receipt changed.'
}
$prior = Get-Content -LiteralPath $priorPath -Raw | ConvertFrom-Json
if ($prior.status -cne 'QUALIFIED-CPU-FALLBACK-FULL-GATED' -or $prior.selectedMode -cne 'scalar-cpu' -or
    $prior.featureFingerprint -cne '78ed3dc0344f71f1d8515f5923dc2c1de3604e59e6b4a1c9cb89851e2b3cf05d' -or
    $prior.runtimeSourceCommit -cne 'fd14d3b1caf89886a5f6cc29fa6c01dcef199e01' -or
    $prior.packageSourceCommit -cne '65880cdfd33580d23d0e833ac977f79acb18b1b0' -or
    $prior.cpu.controls8 -cne 'PASS' -or $prior.cpu.controls128 -cne 'PASS' -or
    $prior.cpu.maximumAbsoluteDifference -ne 0 -or $prior.cpu.preparedTensorsExact -ne $true -or
    $prior.cpu.exactIdsSelectionPrecisionNativeNull -ne $true -or $prior.cpu.nativeOrt -cne '1.23.2' -or
    $prior.cpu.intraThreads -ne 4 -or $prior.cpu.interThreads -ne 1 -or
    $prior.cpu.stateBatchSize -ne 1 -or $prior.cpu.outputBatchSize -ne 256 -or
    $prior.rejectedCuda.status -cne 'REJECTED-HARD-NUMERICAL-GATE' -or
    $prior.rejectedCuda.unchangedTolerance -ne 1e-6) {
    throw 'Exact CPU qualification/source/runtime policy changed.'
}
foreach ($artifact in @($prior.artifacts) + @($prior.frozenCpuClosure)) { VerifyArtifact $artifact }
$frozenFiles = @(Get-ChildItem -LiteralPath $prior.frozenCpuDirectory -File -Recurse)
if ($frozenFiles.Count -ne $prior.frozenCpuClosure.Count) { throw 'Frozen runtime closure file count changed.' }

$inputs = Get-Content -LiteralPath (Join-Path $root 'producer\inputs-v1\inputs.receipt.json') -Raw | ConvertFrom-Json
foreach ($file in $inputs.files) {
    if ((Get-Item -LiteralPath $file.path).Length -ne $file.sizeBytes -or
        (Get-FileHash -LiteralPath $file.path -Algorithm SHA256).Hash.ToLowerInvariant() -cne $file.sha256) {
        throw 'Original corpus/question/split/state/assets/history bytes changed.'
    }
}
$licensePath = Join-Path $root 'producer\inputs-v1\model-card.pinned.md'
if ((Get-FileHash -LiteralPath $licensePath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $inputs.license.sha256) {
    throw 'Retained pinned license evidence changed.'
}

$profilePath = Join-Path $root 'selected-profile\cpu-fallback-v1\selected-profile.json'
$profileArtifact = Artifact $profilePath
if ($profileArtifact.sha256 -cne 'a393ea3b2c93f97e19edbc15a6ffac25e49a58d2c9759dc7207a26a7a413f11f') {
    throw 'Original selected CPU profile changed.'
}
$profile = Get-Content -LiteralPath $profilePath -Raw | ConvertFrom-Json
$portableArtifact = Artifact $profile.packageReceiptPath
if ($portableArtifact.sha256 -cne '1baa85ce0b1be99d306713d8a19f287b6a8a599bcf5726169c965f842d94b03c') {
    throw 'Exact immutable portable package receipt changed.'
}
$portable = Get-Content -LiteralPath $profile.packageReceiptPath -Raw | ConvertFrom-Json
foreach ($package in $portable.packages) {
    $path = Join-Path (Split-Path $profile.packageReceiptPath) $package.file
    if ((Get-Item -LiteralPath $path).Length -ne $package.sizeBytes -or
        (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -cne $package.sha256) {
        throw "Portable package bytes changed: $($package.id)"
    }
}
$consumerArtifacts = @($ConsumerEvidencePaths | ForEach-Object { Artifact $_ })
if ($consumerArtifacts.Count -eq 0) { throw 'Completed consumer evidence references are required.' }

New-Item -ItemType Directory -Path $output | Out-Null
$receiptPath = Join-Path $output 'qualification.receipt.json'
$prior.status = 'GO-READY'
$prior | Add-Member priorQualification $priorArtifact
$prior | Add-Member ownerAuthorization $ownerArtifact
$prior | Add-Member consumerEvidence $consumerArtifacts
$prior | Add-Member revisedAcceptance $owner.consumerAllocation
$prior | Add-Member authorizationWriter ([ordered]@{
    repositoryPath = 'eng\experiments\Write-JuliaFullExportAuthorization.ps1'
    sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
    runtimeRebuilt = $false
})
$prior.fullExport = [ordered]@{
    status = 'AUTHORIZED-NOT-STARTED'
    completed = $false
    expectedRows = 5574
    originalExpectedIds = 'Contiguous original row IDs 1..5574, order exact'
    execution = 'One frozen optimized scalar-cpu export, state1/output256/intra4/inter1'
    publicPackageReaderRequiredBeforeConsumerHandoff = $true
    consumerStudyOwner = 'Separate ML.NET consumer; freeze-before-holdout and original study design unchanged'
    note = 'Explicit owner revised acceptance; old70% benchmark did not pass. Prior FULL-GATED profile and all failures retained.'
}
WriteNewJson $receiptPath $prior
$profile.qualificationReceiptPath = $receiptPath
$profile.qualificationReceiptSha256 = (Get-FileHash -LiteralPath $receiptPath -Algorithm SHA256).Hash.ToLowerInvariant()
WriteNewJson (Join-Path $output 'selected-profile.json') $profile
Write-Host "New immutable CPU GO-READY receipt=$receiptPath; frozen runtime reused without rebuild; previous receipt unchanged."
