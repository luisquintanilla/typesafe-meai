[CmdletBinding()]
param(
    [ValidateSet("A", "B", "C")][string]$Stage = "A",
    [switch]$All,
    [switch]$Offline
)
$ErrorActionPreference = "Stop"
if (-not $Offline) { throw "Only offline verification is authorized. Pass -Offline. Live requests require separate approval." }
$env:DOTNET_CLI_TELEMETRY_OPTOUT = "1"
$env:DOTNET_AIEVAL_TELEMETRY_OPTOUT = "1"
Push-Location (Join-Path $PSScriptRoot "..")
try {
    function Run-DotNet {
        & dotnet @args
        if ($LASTEXITCODE -ne 0) { throw "dotnet command failed with exit code $LASTEXITCODE." }
    }
    Run-DotNet restore --locked-mode --verbosity quiet
    Run-DotNet tool restore
    Run-DotNet build --configuration Release --no-restore --no-incremental --verbosity quiet
    $filter = "Stage=A|FullyQualifiedName~BootstrapTests"
    if ($All -or $Stage -in @("B", "C")) { $filter += "|Stage=B" }
    if ($All -or $Stage -eq "C") { $filter += "|Stage=C" }
    $coreTests = Join-Path "tests" "TypeSafe.MEAI.Tests" "TypeSafe.MEAI.Tests.csproj"
    $sample = Join-Path "samples" "TypeSafe.MEAI.Sample"
    Run-DotNet test $coreTests --configuration Release --no-build --no-restore --filter $filter
    foreach ($scenario in @("native", "pipeline")) {
        Run-DotNet run --project $sample --configuration Release --no-build --no-restore -- $scenario
    }
    if ($All -or $Stage -in @("B", "C")) {
        $evaluationTests = Join-Path "tests" "TypeSafe.MEAI.Evaluation.Tests" "TypeSafe.MEAI.Evaluation.Tests.csproj"
        Run-DotNet test $evaluationTests --configuration Release --no-build --no-restore --filter "Stage=B"
        $reportRoot = Join-Path "artifacts" "reports"
        foreach ($scenario in @("evaluate", "tool")) {
            Run-DotNet run --project $sample --configuration Release --no-build --no-restore -- $scenario
        }
        $htmlReport = Join-Path "artifacts" "support-report.html"
        $jsonReport = Join-Path "artifacts" "support-report.json"
        Run-DotNet aieval report --path $reportRoot --output $htmlReport --format html
        Run-DotNet aieval report --path $reportRoot --output $jsonReport --format json
        $report = Get-Content $jsonReport -Raw | ConvertFrom-Json
        $fixed = $report.scenarioRunResults | Where-Object scenarioName -eq "fixed-draft"
        if ($fixed.evaluationResult.metrics.quality.value -ne 1.05 -or
            $fixed.evaluationResult.metrics.destination.'$type' -ne "string" -or
            $fixed.evaluationResult.metrics.quality.metadata.'typesafe.v1.answer' -notmatch '"legend"' -or
            $fixed.modelResponse.modelId -ne "scripted-primary" -or
            (Get-Item $htmlReport).Length -eq 0) {
            throw "Persisted report fidelity check failed."
        }
        $failure = $report.scenarioRunResults | Where-Object scenarioName -eq "visible-child-failure"
        if (-not ($failure.evaluationResult.metrics.'intentional-failure'.diagnostics | Where-Object severity -eq "error")) {
            throw "Persisted CompositeEvaluator child failure was not visible as error diagnostics."
        }
    }
    if ($All -or $Stage -eq "C") {
        foreach ($scenario in @("schema", "route", "select-tool", "dispatch")) {
            Run-DotNet run --project $sample --configuration Release --no-build --no-restore -- $scenario
        }
    }
    if ($All) { & (Join-Path $PSScriptRoot "verify-packages.ps1") }
}
finally { Pop-Location }
