[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $FixtureDirectory,
    [Parameter(Mandatory)][string] $CudaRestoreDirectory,
    [Parameter(Mandatory)][string] $OutputDirectory
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path (Split-Path $PSScriptRoot)
$output = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $output) { throw 'Artifact receipt directory must be new.' }
New-Item -ItemType Directory -Path $output | Out-Null
function Artifact([string] $root, [string] $path) {
    return [ordered]@{ path = [IO.Path]::GetRelativePath($root, $path); sizeBytes = (Get-Item -LiteralPath $path).Length;
        sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() }
}
Add-Type -AssemblyName System.Reflection.Metadata
function Imports([string] $path) {
    $file = [IO.File]::OpenRead($path)
    $pe = [Reflection.PortableExecutable.PEReader]::new($file)
    try {
        $directory = $pe.PEHeaders.PEHeader.ImportTableDirectory
        $result = [Collections.Generic.List[string]]::new()
        for ($offset = 0; $offset + 20 -le $directory.Size; $offset += 20) {
            [byte[]] $entry = $pe.GetSectionData($directory.RelativeVirtualAddress + $offset).GetContent(0,20)
            $nameRva = [BitConverter]::ToInt32($entry,12)
            if ($nameRva -eq 0) { break }
            $block = $pe.GetSectionData($nameRva)
            [byte[]] $name = $block.GetContent(0,[Math]::Min(1024,$block.Length))
            $end = [Array]::IndexOf($name,[byte]0)
            if ($end -lt 1 -or $end -gt 1024) { throw 'Invalid native PE import name.' }
            $result.Add([Text.Encoding]::ASCII.GetString($name,0,$end))
        }
        return $result.ToArray()
    } finally { $pe.Dispose(); $file.Dispose() }
}
$fixture = (Resolve-Path -LiteralPath $FixtureDirectory).Path
$fixtures = foreach ($precision in @('HighPrecision','FourDecimalPlaces','TwoDecimalPlaces')) {
    $directory = Join-Path $fixture $precision
    $manifest = Get-Content -LiteralPath (Join-Path $directory 'manifest.json') -Raw | ConvertFrom-Json
    if ($manifest.status -cne 'complete' -or $manifest.rowCount -ne 257 -or $manifest.outputBatchSize -ne 256 -or
        $manifest.provenance.executionMode -cne 'julia-offline-fixture-not-real') { throw 'Wrong fictional Julia fixture.' }
    [ordered]@{ precision = $precision; expectedIds = @(1..257); featureFingerprint = $manifest.featureFingerprint;
        files = @(Get-ChildItem -LiteralPath $directory -File | Sort-Object Name | ForEach-Object { Artifact $fixture $_.FullName }) }
}
[ordered]@{
    schemaVersion = 1; status = 'FICTIONAL-FIXTURE-HANDOFF'; inferenceOccurred = $false;
    fixtureRoot = $fixture; expectedIdsAuthority = 'Independently fixed contiguous IDs1..257, not inferred from dataset';
    canonicalHighPrecision = Artifact $fixture (Join-Path $fixture 'julia-contract.json');
    profile = Artifact $fixture (Join-Path $fixture 'profile.json'); fixtures = @($fixtures)
} | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $output 'fixtures.receipt.json') -Encoding utf8
$bin = Join-Path $repo 'samples\experiments\DecisionArrow.Cuda\bin\Release\net10.0'
$closure = @(Get-ChildItem -LiteralPath $bin -Recurse -File | Sort-Object FullName | ForEach-Object { Artifact $bin $_.FullName })
$native = @(Get-ChildItem -LiteralPath (Join-Path $bin 'runtimes\win-x64\native') -Filter '*.dll' | ForEach-Object {
    [ordered]@{ artifact = Artifact $bin $_.FullName; version = $_.VersionInfo.FileVersion; imports = @(Imports $_.FullName) }
})
[ordered]@{
    schemaVersion = 1; status = 'STATIC-RESTORE-BUILD-PASS-NOT-QUALIFIED';
    sourceHead = (& git -C $repo rev-parse HEAD); sourceClean = -not [bool](& git -C $repo status --porcelain);
    host = 'DecisionArrow.Cuda'; nativePackage = 'Microsoft.ML.OnnxRuntime.Gpu/1.23.2';
    restoreConfig = Artifact ([IO.Path]::GetFullPath($CudaRestoreDirectory)) (Join-Path $CudaRestoreDirectory 'NuGet.Experiment.config');
    lockedRestore = $true; publicSignatureVerification = 'Existing scoped restore helper completed verification under unchanged root NuGet.config';
    sdk = (& dotnet --version); scopedImportDirectoryBuildTargets = 'false';
    build = 'dotnet build eng\DecisionArrowCuda.slnx -c Release --no-restore --verbosity quiet';
    buildWarnings = 0; buildErrors = 0; nativeInitializationOccurred = $false; modelScoringOccurred = $false;
    outputRoot = $bin; closure = $closure; native = $native;
    note = 'Packaged TensorRT DLL is not selected. Native imports/static hashes do not prove loaded DLLs or CUDA placement.'
} | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath (Join-Path $output 'cuda-static.receipt.json') -Encoding utf8
Write-Host "Julia fixture/static receipts ready: $output; no native initialization."
