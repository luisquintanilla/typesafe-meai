[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $OriginalProfileDirectory,
    [Parameter(Mandatory)][string] $CandidateProfileDirectory,
    [Parameter(Mandatory)][string] $Questions,
    [Parameter(Mandatory)][string] $LayaManifest,
    [Parameter(Mandatory)][string] $LayaContract,
    [Parameter(Mandatory)][string] $OutputDirectory
)
$ErrorActionPreference = "Stop"
$output = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $output) { throw "Paired measurement namespace must be new." }
$original = (Resolve-Path -LiteralPath $OriginalProfileDirectory).Path
$candidate = (Resolve-Path -LiteralPath $CandidateProfileDirectory).Path
New-Item -ItemType Directory -Path $output | Out-Null
$originalRuntime = Join-Path $output "original-runtime"
New-Item -ItemType Directory -Path $originalRuntime | Out-Null
Get-ChildItem -LiteralPath $original | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $originalRuntime -Recurse }
# Both arms execute identical harness bytes. ONLY the harness changes; original portable DLLs stay frozen.
foreach ($file in @("DecisionArrow.Profile.dll", "DecisionArrow.Profile.pdb")) {
    Copy-Item -LiteralPath (Join-Path $candidate $file) -Destination $originalRuntime -Force
}
foreach ($file in @("DecisionInference.Arrow.dll", "DecisionInference.Abstractions.dll", "Apache.Arrow.dll")) {
    if ((Get-FileHash (Join-Path $originalRuntime $file)).Hash -cne (Get-FileHash (Join-Path $original $file)).Hash) {
        throw "Original dependency was substituted."
    }
}
$closure = @{}
foreach ($arm in @("original", "candidate")) {
    $directory = if ($arm -eq "original") { $originalRuntime } else { $candidate }
    $closure[$arm] = @(Get-ChildItem -LiteralPath $directory -File -Recurse | Sort-Object FullName | ForEach-Object {
        @{ relativePath = [IO.Path]::GetRelativePath($directory, $_.FullName); sizeBytes = $_.Length;
           sha256 = (Get-FileHash -LiteralPath $_.FullName).Hash.ToLowerInvariant() }
    })
}
$runs = [Collections.Generic.List[object]]::new()
for ($pair = 0; $pair -lt 5; $pair++) {
    $order = if ($pair % 2 -eq 0) { @("original", "candidate") } else { @("candidate", "original") }
    foreach ($arm in $order) {
        $directory = if ($arm -eq "original") { $originalRuntime } else { $candidate }
        $runOutput = Join-Path $output "pair-$pair-$arm"
        & dotnet (Join-Path $directory "DecisionArrow.Profile.dll") profile $Questions $runOutput 1 $LayaManifest $LayaContract
        if ($LASTEXITCODE -ne 0) { throw "Paired profile failed: pair$pair/$arm; partial evidence retained." }
        $receiptPath = Join-Path $runOutput "profile.receipt.json"
        $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
        $runs.Add(@{ pair = $pair; arm = $arm; order = $order; path = $receiptPath; receipt = $receipt;
            receiptSha256 = (Get-FileHash -LiteralPath $receiptPath).Hash.ToLowerInvariant() })
    }
}
function Statistics([double[]] $values) {
    $sorted = @($values | Sort-Object)
    $mean = ($values | Measure-Object -Average).Average
    $variance = ($values | ForEach-Object { ($_ - $mean) * ($_ - $mean) } | Measure-Object -Average).Average
    return @{ median = $sorted[2]; mean = $mean; cvPercent = $(if ($mean -eq 0) { 0 } else { [Math]::Sqrt($variance) / $mean * 100 }) }
}
$gates = [Collections.Generic.List[object]]::new()
foreach ($precision in @("HighPrecision", "FourDecimalPlaces", "TwoDecimalPlaces")) {
    foreach ($count in @(1, 257, 65536)) {
        foreach ($scope in @("append", "validation")) {
            $arms = @{}
            foreach ($arm in @("original", "candidate")) {
                $samples = @($runs | Where-Object { $_.arm -ceq $arm } | ForEach-Object {
                    $_.receipt.samples | Where-Object { $_.precision -ceq $precision -and $_.count -eq $count -and [string]$_.repeat -ceq "0" }
                })
                if ($samples.Count -ne 5) { throw "Missing exact paired scope samples." }
                $arms[$arm] = @{
                    milliseconds = Statistics @($samples | ForEach-Object { $_.$scope.milliseconds })
                    allocatedBytes = Statistics @($samples | ForEach-Object { $_.$scope.allocatedBytes })
                }
            }
            $oldBytes = $arms.original.allocatedBytes.median
            $newBytes = $arms.candidate.allocatedBytes.median
            $reduction = if ($oldBytes -eq 0) { 0 } else { (1 - $newBytes / $oldBytes) * 100 }
            $oldTime = $arms.original.milliseconds.median
            $newTime = $arms.candidate.milliseconds.median
            $regression = if ($oldTime -eq 0) { 0 } else { ($newTime / $oldTime - 1) * 100 }
            $gates.Add(@{ precision = $precision; rows = $count; scope = $scope; arms = $arms;
                allocationReductionPercent = $reduction; medianTimeRegressionPercent = $regression;
                allocationTargetPercent = 50; maximumTimeRegressionPercent = 10;
                pass = $reduction -ge 50 -and $regression -le 10 })
        }
    }
}
$failed = @($gates | Where-Object { -not $_.pass })
$report = @{
    schemaVersion = 1; status = $(if ($failed.Count -eq 0) { "PASS" } else { "BLOCKED" }); gates = @($gates)
    scope = "5 fresh-process AB/BA paired runs; excluded in-process warmup; synthetic + retained Laya; no inference"
    order = @("AB", "BA", "AB", "BA", "AB"); closure = $closure
    harnessSha256 = (Get-FileHash -LiteralPath (Join-Path $candidate "DecisionArrow.Profile.dll")).Hash.ToLowerInvariant()
    questionsSha256 = (Get-FileHash -LiteralPath $Questions).Hash.ToLowerInvariant()
    layaManifestSha256 = (Get-FileHash -LiteralPath $LayaManifest).Hash.ToLowerInvariant()
    layaContractSha256 = (Get-FileHash -LiteralPath $LayaContract).Hash.ToLowerInvariant()
    rawRuns = @($runs | ForEach-Object { @{ pair = $_.pair; arm = $_.arm; order = $_.order; path = $_.path; sha256 = $_.receiptSha256 } })
    limitations = "Zero-row controls and schema/IPC/open/replay scopes remain in every raw run. Tiny-row timings/noise reported, not hidden. Warm OS cache, no flush; process memory is cumulative within one process. Managed bytes do not measure native memory."
}
$report | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath (Join-Path $output "paired.acceptance.json") -Encoding utf8
if ($failed.Count -gt 0) { throw "Producer core gates failed: $($failed.Count)/$($gates.Count); raw failures retained." }
Write-Host "Producer core PASS: $($gates.Count) allocation/time gates, five AB/BA pairs."
