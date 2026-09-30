[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $OriginalReference,
    [Parameter(Mandatory)][string] $CandidateReference,
    [Parameter(Mandatory)][string] $OutputDirectory
)
$ErrorActionPreference = "Stop"
$output = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $output) { throw "Oracle result directory must be new." }
$original = (Resolve-Path -LiteralPath $OriginalReference).Path
$receipt = Get-Content -LiteralPath (Join-Path $original "freeze.receipt.json") -Raw | ConvertFrom-Json
if ($receipt.status -cne "FROZEN" -or $receipt.originalSourceCommit -cne "6dda59d19a60fdad2fc14152e3774d1d5ad9b345") {
    throw "Wrong original oracle."
}
foreach ($entry in $receipt.files) {
    $path = Join-Path $original $entry.relativePath
    if ((Get-Item -LiteralPath $path).Length -ne $entry.sizeBytes -or
        (Get-FileHash -LiteralPath $path).Hash.ToLowerInvariant() -cne $entry.sha256) { throw "Original binary closure changed." }
}
New-Item -ItemType Directory -Path $output | Out-Null
$cases = [Collections.Generic.List[object]]::new()
$invariant = [Globalization.CultureInfo]::InvariantCulture
foreach ($precision in @("HighPrecision", "FourDecimalPlaces", "TwoDecimalPlaces")) {
    $unit = if ($precision -eq "TwoDecimalPlaces") { 0.01 } else { 0.0001 }
    $vectors = @(
        @("0", "1"), @("-0", "1"), @("1", "-0"), @("0.5", "0.5"),
        @("0.25", "0.5", "0.25"), @("0.1", "0.8", "0.1"),
        @("0.33", "0.33", "0.34"), @("0.33", "0.33", "0.33"), @("0.34", "0.34", "0.34"),
        @("0", "0"), @("1", "1"), @("NaN", "1"), @("Infinity", "0"), @("-0.0001", "1"),
        @("0.499999", "0.5"), @("0.49999899999999997", "0.5"), @("0.500001", "0.5"),
        @("0.5000010000000001", "0.5"), @("0.5", "0.500001"),
        @("0.25", "0.5", "0.25000000000000006"),
        @("0.24999999999999997", "0.5", "0.25"),
        @("0.0001", "0.9999"), @("0.01", "0.99"), @("0.005", "0.995")
    )
    for ($i = 0; $i -le 100; $i++) {
        $p = $i / 100.0
        $vectors += ,@($p.ToString("R", $invariant), (1 - $p).ToString("R", $invariant))
    }
    foreach ($values in $vectors) {
        $cases.Add(@{ kind = "choice"; precision = $precision; selected = 0; values = @($values); native = $null })
        $cases.Add(@{ kind = "choice"; precision = $precision; selected = $values.Count - 1; values = @($values); native = $null })
        $cases.Add(@{ kind = "choice"; precision = $precision; selected = $values.Count; values = @($values); native = $null })
        foreach ($native in @($null, "0", "-0", "1", "2", "NaN", "Infinity", ($unit / 2).ToString("R", $invariant))) {
            $cases.Add(@{ kind = "score"; precision = $precision; values = @($values); native = $native })
        }
    }
    foreach ($value in @("-0", "0", "1", "0.5", "0.005", "0.00005", "NaN", "Infinity", "-0.0001",
        "0.49999999999999994", "0.5000000000000001", "1.0000000000000002")) {
        $cases.Add(@{ kind = "binary"; precision = $precision; values = @($value); native = $null })
    }
}
$input = Join-Path $output "cases.jsonl"
$writer = [IO.StreamWriter]::new([IO.FileStream]::new($input, [IO.FileMode]::CreateNew))
try { foreach ($case in $cases) { $writer.WriteLine(($case | ConvertTo-Json -Compress -Depth 5)) } } finally { $writer.Dispose() }
$originalOutput = Join-Path $output "original.jsonl"
$candidateOutput = Join-Path $output "candidate.jsonl"
& dotnet (Join-Path $original "bin\DecisionArrow.Reference.dll") oracle $input $originalOutput
if ($LASTEXITCODE -ne 0) { throw "Original oracle failed." }
& dotnet $CandidateReference oracle $input $candidateOutput
if ($LASTEXITCODE -ne 0) { throw "Candidate oracle failed." }
$first = [IO.File]::ReadAllLines($originalOutput)
$second = [IO.File]::ReadAllLines($candidateOutput)
$mismatches = [Collections.Generic.List[object]]::new()
if ($first.Length -ne $cases.Count -or $second.Length -ne $cases.Count) { throw "Oracle result count mismatch." }
for ($i = 0; $i -lt $first.Length; $i++) {
    if ($first[$i] -cne $second[$i]) { $mismatches.Add(@{ caseOrdinal = $i; original = $first[$i]; candidate = $second[$i] }) }
}
$evidence = @{
    schemaVersion = 1; status = $(if ($mismatches.Count -eq 0) { "PASS" } else { "BLOCKED" })
    originalSourceCommit = $receipt.originalSourceCommit; caseCount = $cases.Count; mismatches = @($mismatches)
    gate = "Original independent binaries vs candidate: exact accepted/error decisions and every probability/expectation/bound DOUBLE bit; original tolerance1e-6 unchanged."
    inputSha256 = (Get-FileHash -LiteralPath $input).Hash.ToLowerInvariant()
    originalResultsSha256 = (Get-FileHash -LiteralPath $originalOutput).Hash.ToLowerInvariant()
    candidateResultsSha256 = (Get-FileHash -LiteralPath $candidateOutput).Hash.ToLowerInvariant()
    originalFreezeReceiptSha256 = (Get-FileHash -LiteralPath (Join-Path $original "freeze.receipt.json")).Hash.ToLowerInvariant()
    candidateExecutableSha256 = (Get-FileHash -LiteralPath $CandidateReference).Hash.ToLowerInvariant()
}
$evidence | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $output "oracle.receipt.json") -Encoding utf8
if ($mismatches.Count -ne 0) { throw "Original numerical oracle differs in $($mismatches.Count) cases; counterexamples retained." }
Write-Host "Original independent numerical oracle PASS: $($cases.Count) cases, exact bits/errors, unchanged1e-6."
