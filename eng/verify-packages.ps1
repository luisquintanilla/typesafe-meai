[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot
Push-Location $repo
try {
    function Run-DotNet {
        & dotnet @args
        if ($LASTEXITCODE -ne 0) { throw "Package proof dotnet command failed: exit $LASTEXITCODE." }
    }
    $proof = Join-Path $repo "artifacts" "package-proof" ([Guid]::NewGuid().ToString("N"))
    $feed = Join-Path $proof "feed"
    $consumer = Join-Path $proof "consumer"
    New-Item -ItemType Directory -Path $feed, $consumer | Out-Null
    $libraries = @("TypeSafe.MEAI", "TypeSafe.MEAI.Evaluation")
    $provenance = [Collections.Generic.List[object]]::new()
    $external = [Collections.Generic.List[string]]::new()
    $seen = @{}

    # Use the pinned SDK's public NuGet hash API: signed-package lock hashes are not raw-file SHA512.
    $sdkVersion = & dotnet --version
    if ($LASTEXITCODE -ne 0) { throw "Cannot identify SDK." }
    $sdkLine = & dotnet --list-sdks | Where-Object { $_.StartsWith("$sdkVersion [", [StringComparison]::Ordinal) }
    if ($LASTEXITCODE -ne 0 -or $sdkLine -notmatch '\[(.+)\]$') { throw "Cannot locate pinned SDK." }
    $sdk = Join-Path $Matches[1] $sdkVersion
    foreach ($assembly in @("NuGet.Common", "NuGet.Frameworks", "NuGet.Versioning", "NuGet.Packaging")) {
        [void][Reflection.Assembly]::LoadFrom((Join-Path $sdk "$assembly.dll"))
    }
    foreach ($id in $libraries) {
        $directory = Join-Path $repo "src" $id
        $assets = Get-Content (Join-Path $directory "obj" "project.assets.json") -Raw | ConvertFrom-Json
        $lock = Get-Content (Join-Path $directory "packages.lock.json") -Raw | ConvertFrom-Json
        foreach ($entry in $lock.dependencies.'net10.0'.PSObject.Properties) {
            if ($entry.Value.type -eq "Project") { continue }
            $name = $entry.Name
            $version = $entry.Value.resolved
            $expected = $entry.Value.contentHash
            $key = "$name/$version"
            if ($seen.ContainsKey($key)) {
                if ($seen[$key] -cne $expected) { throw "Conflicting pinned hashes: $key." }
                continue
            }
            $seen[$key] = $expected
            $packageDirectory = Join-Path $assets.project.restore.packagesPath $name.ToLowerInvariant() $version
            $path = Join-Path $packageDirectory "$($name.ToLowerInvariant()).$version.nupkg"
            $reader = [NuGet.Packaging.PackageArchiveReader]::new($path)
            try {
                $actual = $reader.GetContentHash([Threading.CancellationToken]::None, $null)
                $identity = $reader.GetIdentity()
                if ($actual -cne $expected -or $identity.Id -ine $name -or $identity.Version.ToNormalizedString() -cne $version) {
                    throw "Cached archive does not match locked identity/hash: $key."
                }
            }
            finally { $reader.Dispose() }
            $destination = Join-Path $feed (Split-Path $path -Leaf)
            Copy-Item -LiteralPath $path -Destination $destination
            $external.Add($destination)
            $metadata = Get-Content (Join-Path $packageDirectory ".nupkg.metadata") -Raw | ConvertFrom-Json
            $provenance.Add([ordered]@{ id = $name; version = $version; lockContentHash = $actual;
                archiveSha512 = (Get-FileHash -LiteralPath $destination -Algorithm SHA512).Hash; cachedSource = $metadata.source; localUnsigned = $false })
        }
    }
    $verificationLog = Join-Path $proof "external-signatures.log"
    & dotnet nuget verify @external --all --configfile (Join-Path $repo "NuGet.config") --verbosity normal *> $verificationLog
    if ($LASTEXITCODE -ne 0) { throw "External signature verification failed; see $verificationLog." }
    Write-Host "Verified $($external.Count) exact external archives against library lock hashes and signatures."

    foreach ($id in $libraries) {
        Run-DotNet pack (Join-Path $repo "src" $id "$id.csproj") --configuration Release --no-build --no-restore --output $feed --verbosity quiet
        $localPackages = @(Get-ChildItem -LiteralPath $feed -Filter "$id.*.nupkg" |
            Where-Object Name -Match "^$([Regex]::Escape($id))\.[0-9]")
        if ($localPackages.Count -ne 1) { throw "Expected one local package for $id." }
        $package = $localPackages[0]
        $archive = [IO.Compression.ZipFile]::OpenRead($package.FullName)
        try {
            $allowed = @("[Content_Types].xml", "_rels/.rels", "$id.nuspec", "README.md",
                "lib/net10.0/$id.dll", "lib/net10.0/$id.xml")
            foreach ($entry in $archive.Entries) {
                if ($entry.FullName -notin $allowed -and $entry.FullName -notmatch '^package/services/metadata/core-properties/[^/]+\.psmdcp$') {
                    throw "Unexpected local package content: $($entry.FullName)."
                }
            }
            if ("lib/net10.0/$id.dll" -notin $archive.Entries.FullName) { throw "Missing library output in $id." }
            $nuspecReader = [IO.StreamReader]::new($archive.GetEntry("$id.nuspec").Open())
            try { [xml]$nuspec = $nuspecReader.ReadToEnd() } finally { $nuspecReader.Dispose() }
            $dependencies = @($nuspec.package.metadata.dependencies.group.dependency)
            $expectedDependencies = if ($id -eq "TypeSafe.MEAI") { @("Microsoft.Extensions.AI", "Microsoft.Extensions.Http.Resilience") }
                else { @("TypeSafe.MEAI", "Microsoft.Extensions.AI.Evaluation") }
            if (@(Compare-Object $expectedDependencies @($dependencies.id)).Count -ne 0) { throw "Unexpected packaged dependencies for $id." }
            $provenance.Add([ordered]@{ id = $id; version = $nuspec.package.metadata.version;
                archiveSha512 = (Get-FileHash -LiteralPath $package.FullName -Algorithm SHA512).Hash; localUnsigned = $true;
                entries = @($archive.Entries.FullName); dependencies = @($dependencies | ForEach-Object { "$($_.id) $($_.version)" }) })
        }
        finally { $archive.Dispose() }
    }
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "PackageConsumer" "PackageConsumer.csproj"), (Join-Path $PSScriptRoot "PackageConsumer" "Program.cs") -Destination $consumer
    $config = Join-Path $consumer "NuGet.config"
    $escapedFeed = [Security.SecurityElement]::Escape($feed)
    @"
<configuration>
  <packageSources><clear/><add key="verified-offline-artifacts" value="$escapedFeed"/></packageSources>
  <disabledPackageSources><clear/></disabledPackageSources>
  <packageSourceMapping><clear/></packageSourceMapping>
  <auditSources><clear/></auditSources>
  <config><add key="signatureValidationMode" value="accept"/></config>
</configuration>
"@ | Set-Content -LiteralPath $config -Encoding utf8
    $project = Join-Path $consumer "PackageConsumer.csproj"
    $packages = Join-Path $proof "packages"
    Run-DotNet restore $project --configfile $config --packages $packages --force-evaluate --verbosity quiet
    Run-DotNet restore $project --configfile $config --packages $packages --locked-mode --verbosity quiet
    $consumerAssets = Get-Content (Join-Path $consumer "obj" "project.assets.json") -Raw | ConvertFrom-Json
    if (@($consumerAssets.libraries.PSObject.Properties | Where-Object { $_.Value.type -eq "project" }).Count -ne 0) {
        throw "Package consumer unexpectedly references project sources."
    }
    if (@($consumerAssets.project.restore.sources.PSObject.Properties.Name | Where-Object { $_ -cne $feed }).Count -ne 0) {
        throw "Package consumer used a nonlocal source."
    }
    $consumerLock = Get-Content (Join-Path $consumer "packages.lock.json") -Raw | ConvertFrom-Json
    foreach ($entry in $consumerLock.dependencies.'net10.0'.PSObject.Properties) {
        if ($entry.Name -in $libraries) { continue }
        $key = "$($entry.Name)/$($entry.Value.resolved)"
        if (-not $seen.ContainsKey($key) -or $seen[$key] -cne $entry.Value.contentHash) {
            throw "Consumer external dependency escaped the verified library closure: $key."
        }
    }
    Run-DotNet run --project $project --configuration Release --no-restore
    $provenance | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $proof "provenance.json") -Encoding utf8
    $proof | Set-Content -LiteralPath (Join-Path $repo "artifacts" "package-proof" "latest.txt") -Encoding utf8
    Write-Host "Package-only proof and inspection passed: $proof"
}
finally { Pop-Location }
