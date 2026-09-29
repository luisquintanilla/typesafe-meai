[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $ArrowFeed,
    [Parameter(Mandatory)][string] $ArtifactDirectory,
    [Parameter(Mandatory)][switch] $AllowUnsignedLocalPackages,
    [string] $DecisionFeed,
    [string] $Project = "eng\DecisionArrowExperiments.slnx",
    [switch] $UpdateLocks
)
$ErrorActionPreference = "Stop"
if (-not $AllowUnsignedLocalPackages) { throw "Explicit opt-in for designated unsigned experiment packages is required." }
$repo = Split-Path (Split-Path $PSScriptRoot)
$pin = Get-Content -LiteralPath (Join-Path $PSScriptRoot "dependencies.lock.json") -Raw | ConvertFrom-Json
$feed = (Resolve-Path -LiteralPath $ArrowFeed).Path
$receipt = Get-Content -LiteralPath (Join-Path $feed "receipt.json") -Raw | ConvertFrom-Json
if ($receipt.schemaVersion -ne 1 -or $receipt.status -cne "READY" -or $receipt.kind -cne "baseline" -or
    $receipt.source.commit -cne $pin.sourceCommit -or $receipt.source.cleanTree -ne $true -or
    $receipt.packaging.immutable -ne $true) { throw "Wrong, dirty or incomplete Arrow producer receipt." }
if (@($receipt.packages).Count -ne @($pin.packages).Count) { throw "Unexpected local package set." }
foreach ($expected in $pin.packages) {
    $package = @($receipt.packages | Where-Object id -CEQ $expected.id)
    if ($package.Count -ne 1 -or $package[0].version -cne $pin.version -or
        $package[0].sha256 -cne $expected.sha256 -or $package[0].sizeBytes -ne $expected.sizeBytes) {
        throw "Pinned artifact receipt mismatch for $($expected.id)."
    }
    $package = $package[0]
    if ([IO.Path]::GetFileName($package.file) -cne $package.file) { throw "Invalid artifact filename." }
    $path = Join-Path $feed $package.file
    if ((Get-Item -LiteralPath $path).Length -ne $package.sizeBytes -or
        (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -cne $package.sha256) {
        throw "Wrong package bytes for $($package.id)."
    }
    $archive = [IO.Compression.ZipFile]::OpenRead($path)
    try {
        $entry = $archive.GetEntry("$($package.id).nuspec")
        $reader = [IO.StreamReader]::new($entry.Open())
        try { [xml] $nuspec = $reader.ReadToEnd() } finally { $reader.Dispose() }
        if ($nuspec.package.metadata.id -cne $package.id -or $nuspec.package.metadata.version -cne $pin.version -or
            $nuspec.package.metadata.repository.commit -cne $pin.sourceCommit) { throw "Nuspec identity mismatch." }
        foreach ($dependency in $nuspec.SelectNodes("//*[local-name()='dependency']")) {
            if ($dependency.id -in @($pin.packages.id) -and $dependency.version -cne "[$($pin.version)]") {
                throw "Experimental project dependency is not exactly pinned."
            }
        }
    } finally { $archive.Dispose() }
}
$localArtifacts = @{}
foreach ($package in $receipt.packages) {
    $localArtifacts[$package.id.ToLowerInvariant()] = @{ source = $feed; version = $package.version; sha256 = $package.sha256 }
}
$decisionSource = ""
$decisionMap = ""
if ($DecisionFeed) {
    $decision = (Resolve-Path -LiteralPath $DecisionFeed).Path
    $adapter = Get-Content -LiteralPath (Join-Path $decision "receipt.json") -Raw | ConvertFrom-Json
    if ($adapter.status -cne "READY" -or $adapter.kind -cne "decision-adapter" -or
        $adapter.source.cleanTree -ne $true -or $adapter.packaging.immutable -ne $true -or
        $adapter.source.commit -cnotmatch '^[0-9a-f]{40}$' -or @($adapter.packages).Count -ne 2) {
        throw "Invalid adapter producer receipt."
    }
    foreach ($id in @("DecisionInference.Abstractions", "DecisionInference.Arrow")) {
        $package = @($adapter.packages | Where-Object id -CEQ $id)
        if ($package.Count -ne 1 -or $package[0].version -cne "0.1.0-exp.decisions.1.g$($adapter.source.commit.Substring(0,12))") {
            throw "Adapter version does not identify its clean source."
        }
        $package = $package[0]
        if ([IO.Path]::GetFileName($package.file) -cne $package.file) { throw "Invalid adapter filename." }
        $path = Join-Path $decision $package.file
        if ((Get-Item -LiteralPath $path).Length -ne $package.sizeBytes -or
            (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -cne $package.sha256) {
            throw "Adapter bytes do not match receipt."
        }
        $localArtifacts[$id.ToLowerInvariant()] = @{ source = $decision; version = $package.version; sha256 = $package.sha256 }
    }
    $escapedDecision = [Security.SecurityElement]::Escape($decision)
    $decisionSource = "<add key=`"decision-adapter`" value=`"$escapedDecision`"/>"
    $decisionMap = '<packageSource key="decision-adapter"><package pattern="DecisionInference.Arrow"/><package pattern="DecisionInference.Abstractions"/></packageSource>'
}
$artifacts = [IO.Path]::GetFullPath($ArtifactDirectory)
New-Item -ItemType Directory -Force -Path $artifacts | Out-Null
$config = Join-Path $artifacts "NuGet.Experiment.config"
$cache = Join-Path $artifacts "packages"
$escapedFeed = [Security.SecurityElement]::Escape($feed)
$xml = @"
<configuration>
  <packageSources><clear/><add key="arrow-baseline" value="$escapedFeed"/>$decisionSource<add key="nuget.org" value="https://www.nuget.org/api/v2/" protocolVersion="2"/></packageSources>
  <disabledPackageSources><clear/></disabledPackageSources>
  <packageSourceMapping><clear/>
    <packageSource key="arrow-baseline"><package pattern="Apache.Arrow"/><package pattern="Apache.Arrow.Compute"/><package pattern="Apache.Arrow.Scalars"/></packageSource>
    $decisionMap
    <packageSource key="nuget.org"><package pattern="*"/></packageSource>
  </packageSourceMapping>
  <config><add key="signatureValidationMode" value="accept"/></config>
</configuration>
"@
$xml | Set-Content -LiteralPath $config -Encoding utf8
Push-Location $repo
try {
    $arguments = @("restore", $Project, "--configfile", $config, "--packages", $cache, "--verbosity", "quiet")
    if ($DecisionFeed) { $arguments += "-p:DecisionAdapterVersion=$($adapter.packages[0].version)" }
    if ($UpdateLocks) { $arguments += "--force-evaluate" } else { $arguments += "--locked-mode" }
    & dotnet @arguments
    if ($LASTEXITCODE -ne 0) { throw "Experiment restore failed." }
    if ($UpdateLocks) {
        $locked = @("restore", $Project, "--configfile", $config, "--packages", $cache, "--locked-mode", "--verbosity", "quiet")
        if ($DecisionFeed) { $locked += "-p:DecisionAdapterVersion=$($adapter.packages[0].version)" }
        & dotnet @locked
        if ($LASTEXITCODE -ne 0) { throw "Locked restore failed." }
    }
    $public = [Collections.Generic.List[string]]::new()
    foreach ($package in Get-ChildItem -LiteralPath $cache -Recurse -Filter "*.nupkg") {
        $metadata = Get-Content -LiteralPath (Join-Path $package.DirectoryName ".nupkg.metadata") -Raw | ConvertFrom-Json
        if ($localArtifacts.ContainsKey($package.Directory.Parent.Name)) {
            $expected = $localArtifacts[$package.Directory.Parent.Name]
            if ($metadata.source -cne $expected.source) { throw "Local package escaped its exact source mapping." }
            if ($package.Directory.Name -cne $expected.version -or
                (Get-FileHash -LiteralPath $package.FullName -Algorithm SHA256).Hash.ToLowerInvariant() -cne $expected.sha256) {
                throw "Wrong experimental package in isolated cache."
            }
        } else {
            if ($metadata.source -cne "https://www.nuget.org/api/v2/") { throw "Public dependency escaped approved source." }
            $public.Add($package.FullName)
        }
    }
    if ($public.Count -gt 0) {
        & dotnet nuget verify @public --all --configfile (Join-Path $repo "NuGet.config") --verbosity quiet
        if ($LASTEXITCODE -ne 0) { throw "Public package signatures failed verification." }
    }
    Write-Host "Verified restore: config=$config; isolatedCache=$cache; locked public/local closure verified."
} finally { Pop-Location }
