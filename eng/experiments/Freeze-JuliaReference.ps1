[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $ArrowFeed,
    [Parameter(Mandatory)][string] $ArtifactRoot,
    [string] $ReferenceDirectoryName = "original-6dda59d"
)
$ErrorActionPreference = "Stop"
$expected = "6dda59d19a60fdad2fc14152e3774d1d5ad9b345"
$repo = Split-Path (Split-Path $PSScriptRoot)
$root = [IO.Path]::GetFullPath($ArtifactRoot)
if ([IO.Path]::GetFileName($ReferenceDirectoryName) -cne $ReferenceDirectoryName) { throw "Reference directory name must be a single filename." }
$reference = Join-Path $root "reference\$ReferenceDirectoryName"
if (Test-Path -LiteralPath $reference) { throw "Original reference already exists; never replace a freeze." }
Push-Location $repo
try {
    if ((& git rev-parse HEAD) -cne $expected) { throw "Starting original source HEAD moved." }
    & git diff --quiet HEAD -- src samples/experiments/DecisionArrow
    if ($LASTEXITCODE -ne 0) { throw "Original provider/validator/sample source is modified; cannot freeze as original." }
    $env:ImportDirectoryBuildTargets = "false"
    & "$PSScriptRoot\Restore-DecisionArrow.ps1" -ArrowFeed $ArrowFeed -ArtifactDirectory (Join-Path $root "reference\restore") `
        -AllowUnsignedLocalPackages -Project "samples\experiments\DecisionArrow.Reference\DecisionArrow.Reference.csproj" -UpdateLocks
    New-Item -ItemType Directory -Path $reference | Out-Null
    & dotnet publish samples\experiments\DecisionArrow.Reference\DecisionArrow.Reference.csproj -c Release --no-restore `
        -o (Join-Path $reference "bin") --verbosity quiet
    if ($LASTEXITCODE -ne 0) { throw "Original reference publish failed." }
    & "$PSScriptRoot\Restore-DecisionArrow.ps1" -ArrowFeed $ArrowFeed -ArtifactDirectory (Join-Path $root "reference\restore") `
        -AllowUnsignedLocalPackages -Project "samples\experiments\DecisionArrow\DecisionArrow.csproj"
    & dotnet publish samples\experiments\DecisionArrow\DecisionArrow.csproj -c Release --no-restore `
        -o (Join-Path $reference "arrow-bin") --verbosity quiet
    if ($LASTEXITCODE -ne 0) { throw "Original Arrow host publish failed." }
    & git archive --format=zip "--output=$(Join-Path $reference 'original-source.zip')" $expected `
        src Directory.Build.props Directory.Packages.props global.json eng\experiments\dependencies.lock.json samples\experiments\DecisionArrow
    if ($LASTEXITCODE -ne 0) { throw "Original source archive failed." }
    $harness = Join-Path $reference "harness-source"
    New-Item -ItemType Directory -Path $harness | Out-Null
    Copy-Item samples\experiments\DecisionArrow.Reference\Program.cs,samples\experiments\DecisionArrow.Reference\DecisionArrow.Reference.csproj,`
        samples\experiments\DecisionArrow.Reference\packages.lock.json -Destination $harness
    Copy-Item eng\experiments\Freeze-JuliaReference.ps1 -Destination $harness
    $files = @(Get-ChildItem -LiteralPath $reference -File -Recurse | Sort-Object FullName | ForEach-Object {
        @{ relativePath = [IO.Path]::GetRelativePath($reference, $_.FullName); sizeBytes = $_.Length;
           sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
    })
    $receipt = @{
        schemaVersion = 1; status = "FROZEN"; originalSourceCommit = $expected; sdk = (& dotnet --version)
        scope = "Independent original Julia provider/Abstractions/Onnx/Arrow full executable closure; build only, no scoring."
        harnessSeparateFromOriginalSource = $true; frozenBeforeSharedRefactor = $true; files = $files
        cpuOrt = "1.23.2"; immutable = $true; createdUtc = [DateTimeOffset]::UtcNow.ToString("O")
    }
    $receipt | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $reference "freeze.receipt.json") -Encoding utf8
    Write-Host "FROZEN original reference: $reference"
} finally { Pop-Location }
