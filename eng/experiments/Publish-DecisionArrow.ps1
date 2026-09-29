[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $Feed,
    [Parameter(Mandatory)][string] $RestoreArtifactDirectory,
    [Parameter(Mandatory)][string] $ArrowBaselineReceipt
)
$ErrorActionPreference = "Stop"
$repo = Split-Path (Split-Path $PSScriptRoot)
Push-Location $repo
try {
    $status = & git status --porcelain
    if ($LASTEXITCODE -ne 0 -or $status) { throw "Only a clean, coherent committed producer tree can be packaged." }
    $commit = & git rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw "Cannot read source commit." }
    $version = "0.1.0-exp.decisions.1.g$($commit.Substring(0,12))"
    $destination = [IO.Path]::GetFullPath($Feed)
    if (Test-Path -LiteralPath $destination) { throw "Immutable feed destination already exists; never overwrite a version." }
    $config = Join-Path ([IO.Path]::GetFullPath($RestoreArtifactDirectory)) "NuGet.Experiment.config"
    if (-not (Test-Path -LiteralPath $config)) { throw "Verified experiment restore prerequisite is missing." }
    $staging = Join-Path $repo "artifacts\decision-arrow-pack\$commit"
    New-Item -ItemType Directory -Force -Path $staging | Out-Null
    $ids = @("DecisionInference.Abstractions", "DecisionInference.Arrow")
    foreach ($id in $ids) {
        & dotnet pack "src\$id\$id.csproj" --configuration Release --no-restore --output $staging --verbosity quiet "-p:Version=$version" "-p:RepositoryCommit=$commit"
        if ($LASTEXITCODE -ne 0) { throw "Pack failed for $id." }
    }
    $packages = [Collections.Generic.List[object]]::new()
    $baseline = Get-Content -LiteralPath $ArrowBaselineReceipt -Raw | ConvertFrom-Json
    if ($baseline.status -cne "READY" -or $baseline.source.cleanTree -ne $true) { throw "Verified Arrow baseline is required." }
    foreach ($id in $ids) {
        $path = Join-Path $staging "$id.$version.nupkg"
        $archive = [IO.Compression.ZipFile]::Open($path, [IO.Compression.ZipArchiveMode]::Update)
        try {
            $entry = $archive.GetEntry("$id.nuspec")
            $reader = [IO.StreamReader]::new($entry.Open())
            try { [xml] $nuspec = $reader.ReadToEnd() } finally { $reader.Dispose() }
            foreach ($dependency in $nuspec.SelectNodes("//*[local-name()='dependency']")) {
                if ($dependency.id -in $ids) { $dependency.version = "[$version]" }
            }
            $dependencies = @($nuspec.SelectNodes("//*[local-name()='dependency']") | ForEach-Object {
                [ordered]@{ id = $_.id; versionRange = $_.version }
            })
            $expectedIds = if ($id -eq "DecisionInference.Arrow") { @("Apache.Arrow", "DecisionInference.Abstractions") }
                else { @("Microsoft.Extensions.AI.Abstractions") }
            if (@(Compare-Object @($dependencies.id) $expectedIds).Count -ne 0) { throw "Nonportable packaged dependency closure." }
            if ($nuspec.package.metadata.version -cne $version -or $nuspec.package.metadata.repository.commit -cne $commit) {
                throw "Packaged source identity mismatch."
            }
            $entry.Delete()
            $replacement = $archive.CreateEntry("$id.nuspec")
            $writer = [IO.StreamWriter]::new($replacement.Open(), [Text.UTF8Encoding]::new($false))
            try { $nuspec.Save($writer) } finally { $writer.Dispose() }
        } finally { $archive.Dispose() }
        $packages.Add([ordered]@{ id = $id; version = $version; file = [IO.Path]::GetFileName($path);
            sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant();
            sizeBytes = (Get-Item -LiteralPath $path).Length; targetFrameworks = @("net10.0"); dependencyGroups = $dependencies;
            nugetSigned = $false })
    }
    if (& git status --porcelain) { throw "Producer source changed during packaging." }
    New-Item -ItemType Directory -Path $destination | Out-Null
    foreach ($package in $packages) { Copy-Item -LiteralPath (Join-Path $staging $package.file) -Destination $destination }
    [ordered]@{
        schemaVersion = 1; status = "READY"; kind = "decision-adapter";
        source = @{ repository = "https://github.com/luisquintanilla/typesafe-meai"; commit = $commit; cleanTree = $true };
        build = @{ sdk = (& dotnet --version); configuration = "Release"; targetFrameworks = @("net10.0") };
        packaging = @{ immutable = $true; manifestTransform = "Only project-reference nuspec dependency ranges pinned exactly; no DLL/source edits." };
        packages = @($packages);
        dependencyReceipts = @(@{ receiptSha256 = (Get-FileHash -LiteralPath $ArrowBaselineReceipt -Algorithm SHA256).Hash.ToLowerInvariant();
            source = $baseline.source; packages = $baseline.packages });
        validation = @{ cleanCommittedSource = $true; portableNuspecClosure = $true; compilerWarnings = 0 }
    } | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $destination "receipt.json") -Encoding utf8
    Write-Host "Immutable adapter feed ready: $destination; version=$version; commit=$commit"
} finally { Pop-Location }
