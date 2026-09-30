[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $ExperimentDirectory,
    [Parameter(Mandatory)][string] $OutputDirectory
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $ExperimentDirectory).Path
$output = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $output) { throw 'Selected-profile receipt directory must be new.' }
$cpu8 = Get-Content -LiteralPath (Join-Path $root 'controls\readiness-v1\original-optimized-cpu-8.parity.json') -Raw | ConvertFrom-Json
$cpu128 = Get-Content -LiteralPath (Join-Path $root 'controls\cpu-fallback-v1\original-optimized-cpu-128.parity.json') -Raw | ConvertFrom-Json
$failure = Get-Content -LiteralPath (Join-Path $root 'controls\readiness-v1\cpu-cuda-8.parity.json.partial') -Raw | ConvertFrom-Json
$cuda = Get-Content -LiteralPath (Join-Path $root 'controls\readiness-v1\optimized-cuda-8\runtime.receipt.json') -Raw | ConvertFrom-Json
foreach ($pair in @(@($cpu8,8),@($cpu128,128))) {
    if ($pair[0].status -cne 'PASS' -or $pair[0].rows -ne $pair[1] -or
        $pair[0].tolerance -ne 1e-6 -or $pair[0].maximumAbsoluteDifference -gt 1e-6 -or
        $pair[0].exactIdsSelectionPrecisionNativeNull -ne $true) { throw 'CPU bounded numerical gate is not accepted.' }
}
if ($failure.status -cne 'FAILED' -or $failure.failedId -ne 1926 -or $failure.failedCoordinate -ne 0 -or
    $failure.tolerance -ne 1e-6 -or $failure.error -notlike '*difference=4.9192878088799574E-06*') {
    throw 'Expected unchanged CUDA hard numeric failure is absent.'
}
if ($cuda.placement.actualCudaScoring -ne $true -or $cuda.placement.label -cne 'HYBRID' -or
    $cuda.outputMemory.name -cne 'Cpu' -or $cuda.outputMemory.hostReadableBeforeSpan -ne $true) {
    throw 'CUDA readiness evidence changed.'
}
$expected = Get-Content -LiteralPath (Join-Path $root 'controls\cpu-fallback-v1\original-cpu-128\prepared-control.json') -Raw | ConvertFrom-Json
$actual = @(Get-Content -LiteralPath (Join-Path $root 'controls\cpu-fallback-v1\optimized-cpu-128.preflight\prepared.jsonl') |
    ForEach-Object { $_ | ConvertFrom-Json })
if ($expected.Count -ne 128 -or $actual.Count -ne 128) { throw 'Prepared control count changed.' }
for ($i = 0; $i -lt 128; $i++) {
    if ($expected[$i].rowId -ne $actual[$i].rowId -or $expected[$i].tensorSha256 -cne $actual[$i].tensorSha256 -or
        $expected[$i].SequenceLength -ne $actual[$i].SequenceLength -or
        $expected[$i].MarkerWidth -ne $actual[$i].MarkerWidth -or $actual[$i].BatchSize -ne 5) {
        throw "Prepared tensor differs at external control index$i."
    }
}
New-Item -ItemType Directory -Path $output | Out-Null
function Artifact([string] $path) {
    return [ordered]@{ path = [IO.Path]::GetRelativePath($root,$path); sizeBytes = (Get-Item -LiteralPath $path).Length;
        sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() }
}
$files = @(
    'reference\original-6dda59d-v2\freeze.receipt.json',
    'producer\original-preflight-v2\preflight.receipt.json',
    'producer\inputs-v1\inputs.receipt.json',
    'profile\paired-v1\paired.acceptance.json',
    'packages\portable-v1\receipt.json',
    'producer\static-handoff-v1\fixtures.receipt.json',
    'consumer\controls\selection.receipt.json',
    'controls\readiness-v1\original-optimized-cpu-8.parity.json',
    'controls\readiness-v1\cpu-cuda-8.parity.json.partial',
    'controls\readiness-v1\cuda-8-telemetry.csv',
    'controls\readiness-v1\optimized-cuda-8\manifest.json',
    'controls\readiness-v1\optimized-cuda-8\runtime.receipt.json',
    'controls\cpu-fallback-v1\original-optimized-cpu-128.parity.json',
    'controls\cpu-fallback-v1\original-cpu-128\prepared-control.json',
    'controls\cpu-fallback-v1\optimized-cpu-128.preflight\prepared.jsonl',
    'controls\cpu-fallback-v1\optimized-cpu-128\manifest.json',
    'controls\cpu-fallback-v1\optimized-cpu-128\runtime.receipt.json'
)
$cpu = Get-Content -LiteralPath (Join-Path $root 'controls\cpu-fallback-v1\optimized-cpu-128\runtime.receipt.json') -Raw | ConvertFrom-Json
$manifest = Get-Content -LiteralPath (Join-Path $root 'controls\cpu-fallback-v1\optimized-cpu-128\manifest.json') -Raw | ConvertFrom-Json
$repo = Split-Path (Split-Path $PSScriptRoot)
$runtime = Join-Path $repo 'samples\experiments\DecisionArrow\bin\Release\net10.0'
$frozen = Join-Path $output 'cpu-runtime'
Copy-Item -LiteralPath $runtime -Destination $frozen -Recurse
$closure = @(Get-ChildItem -LiteralPath $frozen -File -Recurse | Sort-Object FullName | ForEach-Object { Artifact $_.FullName })
$telemetry = @(Get-Content -LiteralPath (Join-Path $root 'controls\readiness-v1\cuda-8-telemetry.csv') |
    ConvertFrom-Csv -Header timestamp,index,uuid,usedMiB,freeMiB,temperatureC,powerW,smMHz,utilizationPercent)
$receiptPath = Join-Path $output 'qualification.receipt.json'
[ordered]@{
    schemaVersion = 1; status = 'QUALIFIED-CPU-FALLBACK-FULL-GATED'; selectedMode = 'scalar-cpu';
    featureFingerprint = $cpu.featureFingerprint; runtimeSourceCommit = $manifest.provenance.sourceCommit;
    packageSourceCommit = '65880cdfd33580d23d0e833ac977f79acb18b1b0';
    originalSourceCommit = '6dda59d19a60fdad2fc14152e3774d1d5ad9b345';
    cpu = @{ controls8 = 'PASS'; controls128 = 'PASS'; maximumAbsoluteDifference = $cpu128.maximumAbsoluteDifference;
        exactIdsSelectionPrecisionNativeNull = $true; preparedTensorCount = 128; preparedTensorsExact = $true;
        intraThreads = 4; interThreads = 1; stateBatchSize = 1; outputBatchSize = 256;
        nativeOrt = $cpu.runtime.nativeOrt; outputMemory = $cpu.outputMemory; runtime = $cpu.runtime; packages = $cpu.packages };
    rejectedCuda = @{ status = 'REJECTED-HARD-NUMERICAL-GATE'; featureFingerprint = $cuda.featureFingerprint;
        originalId = $failure.failedId; coordinate = $failure.failedCoordinate;
        firstFailingAbsoluteDifference = 4.9192878088799574e-6; unchangedTolerance = 1e-6;
        init = 'PASS'; placement = $cuda.placement.label; actualCudaScoring = $true;
        kernelEvents = @($cuda.placement.kernels | Group-Object provider | ForEach-Object { @{ provider = $_.Name; count = $_.Count } });
        cpuPlacedOperators = @($cuda.placement.kernels | Where-Object provider -CEQ 'CPUExecutionProvider' | Select-Object -ExpandProperty op -Unique);
        runtime = $cuda.runtime; outputMemory = $cuda.outputMemory; packages = $cuda.packages;
        telemetrySamples = $telemetry.Count; observedGlobalPeakVramMiB = ($telemetry.usedMiB | Measure-Object -Maximum).Maximum;
        controls128 = 'NOT-RUN-EARLY-REJECTED'; longestShapeMemoryStability = 'NOT-RUN-EARLY-REJECTED';
        pairedPerformanceSelection = 'NOT-RUN-EARLY-REJECTED'; toleranceWidened = $false; cpuRowsMixed = $false;
        note = 'Diagnostic8 timing is not paired clean performance evidence. First failure aborts comparison; v1 maximum0 reports only preceding checked values.' };
    artifacts = @($files | ForEach-Object { Artifact (Join-Path $root $_) });
    frozenCpuClosure = $closure; frozenCpuDirectory = $frozen;
    fullExport = @{ status = 'WAIT-JOINED-GATES-AND-COORDINATOR-GO'; completed = $false; expectedRows = 5574;
        note = 'Selected CPU profile is explicit; this qualification receipt is not GO-READY/full-export authorization.' }
} | ConvertTo-Json -Depth 24 | Set-Content -LiteralPath $receiptPath -Encoding utf8
$profile = Get-Content -LiteralPath (Join-Path $root 'producer\profiles\cpu-candidate-v2.json') -Raw | ConvertFrom-Json
$profile.status = 'SELECTED'
$profile | Add-Member qualificationReceiptPath $receiptPath
$profile | Add-Member qualificationReceiptSha256 (Get-FileHash -LiteralPath $receiptPath).Hash.ToLowerInvariant()
$profile | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $output 'selected-profile.json') -Encoding utf8
Write-Host "Explicit CPU fallback qualified; full export remains gated. Receipt=$receiptPath"
