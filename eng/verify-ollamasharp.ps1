[CmdletBinding()]
param(
    [switch]$Offline,
    [switch]$Packages,
    [string]$ApiBaseline,
    [string]$RetainedTestBaseline,
    [string]$OutputDirectory
)
$ErrorActionPreference = "Stop"
if (-not $Offline) { throw "Pass -Offline. This entry never installs/starts Ollama, acquires models or calls inference services." }
$repo = Split-Path $PSScriptRoot
$baseline = "04afdaee7892d362893502bea0a3b165d3b1b318"
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $repo "artifacts\ollamasharp-proof" ([Guid]::NewGuid().ToString("N")) }
$output = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path $output) {
    if (@(Get-ChildItem -LiteralPath $output -Force).Count) { throw "Use a fresh, empty output directory." }
}
New-Item -ItemType Directory -Force $output | Out-Null
$commands = [Collections.Generic.List[object]]::new()
$oldCache = $env:NUGET_PACKAGES
$oldHttpCache = $env:NUGET_HTTP_CACHE_PATH
$cache = Join-Path $output "validation-cache"
if ($oldCache) { $cache = $oldCache }
$env:NUGET_PACKAGES = $cache
$env:NUGET_HTTP_CACHE_PATH = Join-Path $output "http-cache"
Push-Location $repo
try {
    function Run-DotNet {
        param([string[]]$Arguments, [int]$ExpectedExit = 0)
        $started = [DateTimeOffset]::UtcNow.ToString("o")
        & dotnet @Arguments 2>&1 | Tee-Object -FilePath (Join-Path $output "commands.log") -Append | Write-Host
        $exit = $LASTEXITCODE
        $commands.Add([ordered]@{ command = "dotnet"; arguments = $Arguments; started = $started; exit = $exit; expectedExit = $ExpectedExit })
        if ($exit -ne $ExpectedExit) { throw "dotnet command failed: exit $exit, expected $ExpectedExit." }
    }
    function Assert-Preserved {
        $allowed = @("Directory.Packages.props", "README.md", "samples/README.md", "docs/decision-generator-design.md",
            "src/DecisionInference.SystemOne/DecisionInference.SystemOne.csproj",
            "src/DecisionInference.Ollama/OllamaDecisionGenerator.cs")
        $tracked = @(git ls-tree -r --name-only $baseline)
        if ($LASTEXITCODE -ne 0) { throw "Exact baseline unavailable." }
        $protected = @($tracked | Where-Object { $_ -notin $allowed })
        & git diff --quiet $baseline -- @protected
        if ($LASTEXITCODE -ne 0) { throw "Protected legacy source, test, graph or lockfile changed." }
        [xml]$before = (git show "${baseline}:Directory.Packages.props") -join "`n"
        if ($LASTEXITCODE -ne 0) { throw "Baseline package pins unavailable." }
        [xml]$after = Get-Content Directory.Packages.props -Raw
        foreach ($pin in $before.Project.ItemGroup.PackageVersion) {
            $now = @($after.Project.ItemGroup.PackageVersion | Where-Object Include -eq $pin.Include)
            if ($now.Count -ne 1 -or $now[0].Version -cne $pin.Version) { throw "Legacy package version changed: $($pin.Include)." }
        }
        $sdk = @($after.Project.ItemGroup.PackageVersion | Where-Object Include -eq "OllamaSharp")
        if ($sdk.Count -ne 1 -or $sdk[0].Version -cne "5.5.0") { throw "SDK pin changed." }
    }
    function Read-TestCounts([string]$path) {
        [xml]$trx = Get-Content -LiteralPath $path -Raw
        $counters = $trx.TestRun.ResultSummary.Counters
        if ([int]$counters.total -le 0 -or [int]$counters.failed -ne 0 -or [int]$counters.passed -ne [int]$counters.total) {
            throw "Unmet test gate: $path."
        }
        $names = [string[]]@($trx.TestRun.Results.UnitTestResult | ForEach-Object testName)
        [Array]::Sort($names, [StringComparer]::Ordinal)
        if ($names.Count -ne [int]$counters.total) { throw "Incomplete TRX results: $path." }
        $inventory = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData(
            [Text.Encoding]::UTF8.GetBytes([string]::Join("`n", $names))))
        return [ordered]@{ total = [int]$counters.total; passed = [int]$counters.passed;
            inventorySha256 = $inventory; sha256 = (Get-FileHash $path).Hash }
    }
    function Stage-Assemblies([string]$source, [string]$destination) {
        New-Item -ItemType Directory -Force $destination | Out-Null
        Get-ChildItem -LiteralPath $source -Filter *.dll |
            Where-Object Name -NotLike "*.Tests.dll" | ForEach-Object {
                $target = Join-Path $destination $_.Name
                if (-not (Test-Path -LiteralPath $target)) { Copy-Item -LiteralPath $_.FullName -Destination $target }
            }
    }
    Assert-Preserved
    Run-DotNet -Arguments @("restore", "eng\OllamaSharpProof.slnx", "--locked-mode", "--verbosity", "quiet")
    Run-DotNet -Arguments @("build", "eng\OllamaSharpProof.slnx", "--configuration", "Release", "--no-restore", "--no-incremental", "--verbosity", "quiet")
    $results = Join-Path $output "test-results"
    Run-DotNet -Arguments @("test", "tests\DecisionInference.OllamaSharp.Tests\DecisionInference.OllamaSharp.Tests.csproj",
        "--configuration", "Release", "--no-build", "--no-restore", "--logger", "trx;LogFileName=ollamasharp.trx",
        "--results-directory", $results, "--blame-hang-timeout", "2min", "--verbosity", "quiet")
    $newTests = Read-TestCounts (Join-Path $results "ollamasharp.trx")
    Run-DotNet -Arguments @("restore", "eng\DecisionComparison.slnx", "--locked-mode", "--verbosity", "quiet")
    Run-DotNet -Arguments @("build", "eng\DecisionComparison.slnx", "--configuration", "Release", "--no-restore", "--no-incremental", "--verbosity", "quiet")
    Run-DotNet -Arguments @("test", "tests\DecisionInference.Tests\DecisionInference.Tests.csproj",
        "--configuration", "Release", "--no-build", "--no-restore", "--logger", "trx;LogFileName=retained.trx",
        "--results-directory", $results, "--blame-hang-timeout", "2min", "--verbosity", "quiet")
    $retainedTests = Read-TestCounts (Join-Path $results "retained.trx")

    # Compile the same reflection algorithm used by the package consumer, without SDK/source references.
    $apiTool = Join-Path $output "api-tool"
    New-Item -ItemType Directory $apiTool | Out-Null
    $consumerSource = Get-Content eng\OllamaSharpPackageConsumer\Program.cs -Raw
    $start = $consumerSource.IndexOf("static void SnapshotApis(", [StringComparison]::Ordinal)
    $end = $consumerSource.IndexOf("internal sealed class FixtureHandler", [StringComparison]::Ordinal)
    if ($start -lt 0 -or $end -le $start) { throw "API algorithm source markers missing." }
    ("using System.Reflection;`nusing System.Runtime.Loader;`nusing System.Text.Json;`nSnapshotApis(args[0], args[1]);`n" +
        $consumerSource.Substring($start, $end - $start)) | Set-Content (Join-Path $apiTool "Program.cs")
    @'
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup><TargetFramework>net10.0</TargetFramework><OutputType>Exe</OutputType><Nullable>enable</Nullable><ImplicitUsings>enable</ImplicitUsings><ManagePackageVersionsCentrally>false</ManagePackageVersionsCentrally><RestorePackagesWithLockFile>false</RestorePackagesWithLockFile></PropertyGroup>
</Project>
'@ | Set-Content (Join-Path $apiTool "ApiSnapshot.csproj")
    Run-DotNet -Arguments @("build", (Join-Path $apiTool "ApiSnapshot.csproj"), "--configuration", "Release", "--verbosity", "quiet")
    $apiDll = Join-Path $apiTool "bin\Release\net10.0\ApiSnapshot.dll"
    if (-not $ApiBaseline -or -not $RetainedTestBaseline) {
        $zip = Join-Path $output "baseline.zip"
        & git archive --format=zip --output=$zip $baseline
        if ($LASTEXITCODE -ne 0) { throw "Read-only baseline archive failed." }
        $baselineRoot = Join-Path $output "baseline"
        [IO.Compression.ZipFile]::ExtractToDirectory($zip, $baselineRoot)
        Run-DotNet -Arguments @("restore", (Join-Path $baselineRoot "eng\DecisionComparison.slnx"), "--locked-mode", "--configfile", (Join-Path $repo "NuGet.config"), "--verbosity", "quiet")
        Run-DotNet -Arguments @("build", (Join-Path $baselineRoot "eng\DecisionComparison.slnx"), "--configuration", "Release", "--no-restore", "--verbosity", "quiet")
    }
    if (-not $RetainedTestBaseline) {
        $baselineResults = Join-Path $output "baseline-test-results"
        Run-DotNet -Arguments @("test", (Join-Path $baselineRoot "tests\DecisionInference.Tests\DecisionInference.Tests.csproj"),
            "--configuration", "Release", "--no-build", "--no-restore", "--logger", "trx;LogFileName=baseline.trx",
            "--results-directory", $baselineResults, "--verbosity", "quiet")
        $RetainedTestBaseline = Join-Path $baselineResults "baseline.trx"
    }
    $baselineTests = Read-TestCounts $RetainedTestBaseline
    if ($retainedTests.total -ne $baselineTests.total -or $retainedTests.inventorySha256 -cne $baselineTests.inventorySha256) {
        throw "Retained test cases differ from the measured exact-baseline inventory."
    }
    if ($ApiBaseline) {
        $ApiBaseline = (Resolve-Path -LiteralPath $ApiBaseline).Path
        if ((Get-FileHash $ApiBaseline).Hash -cne "ACFAE5597E7894129AFDA63A4841BEFEE0DA5B341CCA914D71DCE6223F4C620B") {
            throw "Supplied API baseline is not the accepted immutable 480-entry artifact."
        }
    }
    else {
        $baselineStage = Join-Path $output "api-baseline-dlls"
        Stage-Assemblies (Join-Path $baselineRoot "eng\DecisionProof\bin\Release\net10.0") $baselineStage
        $ApiBaseline = Join-Path $output "api-baseline.json"
        Run-DotNet -Arguments @($apiDll, $baselineStage, $ApiBaseline)
    }
    $currentStage = Join-Path $output "api-current-dlls"
    Stage-Assemblies (Join-Path $repo "eng\DecisionProof\bin\Release\net10.0") $currentStage
    Stage-Assemblies (Join-Path $repo "tests\DecisionInference.OllamaSharp.Tests\bin\Release\net10.0") $currentStage
    $apiAfter = Join-Path $output "api-current.json"
    Run-DotNet -Arguments @($apiDll, $currentStage, $apiAfter)
    $beforeApi = Get-Content $ApiBaseline -Raw | ConvertFrom-Json -AsHashtable
    $afterApi = Get-Content $apiAfter -Raw | ConvertFrom-Json -AsHashtable
    $oldCount = 0
    foreach ($assembly in $beforeApi.Keys) {
        if ([string]::Join("`n", $beforeApi[$assembly]) -cne [string]::Join("`n", $afterApi[$assembly])) {
            throw "Legacy API signatures/attributes changed: $assembly."
        }
        $oldCount += $beforeApi[$assembly].Count
    }
    if ($oldCount -ne 480 -or $afterApi["DecisionInference.SystemOne"].Count -or $afterApi["DecisionInference.Onnx"].Count) {
        throw "API baseline/completeness or internal exports changed."
    }
    $addedAssemblies = @($afterApi.Keys | Where-Object { -not $beforeApi.ContainsKey($_) })
    if ($addedAssemblies.Count -ne 1 -or $addedAssemblies[0] -cne "DecisionInference.OllamaSharp") {
        throw "New API escaped the optional SDK assembly."
    }
    $apis = [ordered]@{ retained = $oldCount; added = $afterApi["DecisionInference.OllamaSharp"].Count;
        beforeSha256 = (Get-FileHash $ApiBaseline).Hash; afterSha256 = (Get-FileHash $apiAfter).Hash }

    $sample = "samples\quickstarts\OllamaSharp\OllamaSharp.cs"
    Run-DotNet -Arguments @("run", "--file", $sample, "--", "--help")
    Run-DotNet -Arguments @("run", "--file", $sample, "--", "--check-config", "http://127.0.0.1:11434", "caller-installed-model")
    foreach ($invalid in @(@("--check-config", "http://example.invalid", "model"),
        @("--check-config", "https://user@example.invalid", "model"),
        @("--check-config", "https://example.invalid/path", "model"), @("--check-config"), @("--unexpected"))) {
        Run-DotNet -Arguments (@("run", "--file", $sample, "--") + $invalid) -ExpectedExit 2
    }
    foreach ($document in @("README.md", "samples\README.md", "docs\decision-generator-design.md",
        "docs\design\ollamasharp-adapter.md", "samples\quickstarts\OllamaSharp\README.md")) {
        $fullPath = Join-Path $repo $document
        foreach ($link in [regex]::Matches((Get-Content $fullPath -Raw), '\[[^\]]+\]\(([^)]+)\)')) {
            $target = $link.Groups[1].Value
            if ($target -match '^(https?://|#)') { continue }
            $relative = ($target -split "#", 2)[0].Replace("/", "\")
            if (-not (Test-Path -LiteralPath (Join-Path (Split-Path $fullPath) $relative))) { throw "Broken link in ${document}: $target." }
        }
    }
    $legacyPackageRoot = Join-Path $repo "artifacts\decision-proof\packages"
    $priorLegacyRuns = @()
    if (Test-Path $legacyPackageRoot) { $priorLegacyRuns = @(Get-ChildItem $legacyPackageRoot -Directory | ForEach-Object Name) }
    $legacyArguments = @{ Offline = $true }
    if ($Packages) {
        $legacyArguments.Packages = $true
        # The legacy decision solution references chat but does not list its project; pack requires its explicit Release output.
        Run-DotNet -Arguments @("build", "src\TypeSafe.MEAI\TypeSafe.MEAI.csproj", "--configuration", "Release",
            "--no-restore", "--no-incremental", "--verbosity", "quiet")
    }
    & (Join-Path $PSScriptRoot "verify-decisions.ps1") @legacyArguments 2>&1 |
        Tee-Object -FilePath (Join-Path $output "legacy-gate.log") | Write-Host
    $commands.Add([ordered]@{ command = "eng\verify-decisions.ps1"; offline = $true; packages = [bool]$Packages; exit = 0 })

    $packageReceipt = $null
    if ($Packages) {
        $legacyRun = @(Get-ChildItem $legacyPackageRoot -Directory | Where-Object Name -NotIn $priorLegacyRuns)
        if ($legacyRun.Count -ne 1) { throw "Cannot identify this invocation's legacy package proof." }
        $legacyAssetsPath = Join-Path $legacyRun[0].FullName "ollama-only\obj\project.assets.json"
        $legacyAssets = Get-Content $legacyAssetsPath -Raw | ConvertFrom-Json -AsHashtable
        $legacyClosure = @($legacyAssets.libraries.Keys | Sort-Object -CaseSensitive)
        $expectedLegacy = @("DecisionInference.Abstractions/0.1.0-private.1", "DecisionInference.Ollama/0.1.0-private.1",
            "DecisionInference.SystemOne/0.1.0-private.1", "Microsoft.Extensions.AI.Abstractions/10.10.0")
        if ([string]::Join("`n", $legacyClosure) -cne [string]::Join("`n", ($expectedLegacy | Sort-Object -CaseSensitive))) {
            throw "Original Ollama four-package closure changed."
        }
        $feed = Join-Path $output "feed"
        New-Item -ItemType Directory $feed | Out-Null
        $assets = Get-Content src\DecisionInference.OllamaSharp\obj\project.assets.json -Raw | ConvertFrom-Json
        $lock = Get-Content src\DecisionInference.OllamaSharp\packages.lock.json -Raw | ConvertFrom-Json
        $sdkVersion = & dotnet --version
        $sdkLine = & dotnet --list-sdks | Where-Object { $_.StartsWith("$sdkVersion [", [StringComparison]::Ordinal) }
        if ($sdkLine -notmatch '\[(.+)\]$') { throw "Cannot locate pinned SDK NuGet hash API." }
        $sdkPath = Join-Path $Matches[1] $sdkVersion
        foreach ($assembly in @("NuGet.Common", "NuGet.Frameworks", "NuGet.Versioning", "NuGet.Packaging")) {
            [void][Reflection.Assembly]::LoadFrom((Join-Path $sdkPath "$assembly.dll"))
        }
        $external = [Collections.Generic.List[string]]::new()
        $archives = @()
        foreach ($entry in $lock.dependencies.'net10.0'.PSObject.Properties) {
            if ($entry.Value.type -eq "Project") { continue }
            $id = $entry.Name.ToLowerInvariant()
            $version = $entry.Value.resolved
            $archive = Join-Path $assets.project.restore.packagesPath "$id\$version\$id.$version.nupkg"
            $reader = [NuGet.Packaging.PackageArchiveReader]::new($archive)
            try {
                if ($reader.GetContentHash([Threading.CancellationToken]::None, $null) -cne $entry.Value.contentHash) {
                    throw "External archive differs from lock: $id/$version."
                }
            }
            finally { $reader.Dispose() }
            Copy-Item -LiteralPath $archive -Destination $feed
            $external.Add($archive)
            $archives += [ordered]@{ id = $id; version = $version; contentHash = $entry.Value.contentHash;
                sha256 = (Get-FileHash $archive).Hash; length = (Get-Item $archive).Length }
        }
        Run-DotNet -Arguments (@("nuget", "verify") + $external.ToArray() + @("--all", "--configfile", "NuGet.config", "--verbosity", "quiet"))
        foreach ($name in @("DecisionInference.Abstractions", "DecisionInference.SystemOne", "DecisionInference.OllamaSharp")) {
            Run-DotNet -Arguments @("pack", "src\$name\$name.csproj", "--configuration", "Release", "--no-build", "--no-restore",
                "--output", $feed, "--verbosity", "quiet")
        }
        $config = Join-Path $output "NuGet.local.config"
        $escapedFeed = [Security.SecurityElement]::Escape($feed)
        # Unsigned private packages require local accept; external archives were individually hash/signature verified above.
        "<configuration><packageSources><clear/><add key=`"verified-local`" value=`"$escapedFeed`"/></packageSources><disabledPackageSources><clear/></disabledPackageSources><packageSourceMapping><clear/></packageSourceMapping><auditSources><clear/></auditSources><config><add key=`"signatureValidationMode`" value=`"accept`"/></config></configuration>" |
            Set-Content $config
        $consumerRoot = Join-Path $output "package-consumer"
        New-Item -ItemType Directory $consumerRoot | Out-Null
        Copy-Item eng\OllamaSharpPackageConsumer\OllamaSharpPackageConsumer.csproj, eng\OllamaSharpPackageConsumer\Program.cs,
            Directory.Build.props -Destination $consumerRoot
        $consumer = Join-Path $consumerRoot "OllamaSharpPackageConsumer.csproj"
        $freshCache = Join-Path $output "consumer-cache"
        # Repacked unsigned private versions have new archive hashes; never reuse another run's private-package lock.
        Run-DotNet -Arguments @("restore", $consumer, "--configfile", $config, "--packages", $freshCache, "--verbosity", "quiet")
        Run-DotNet -Arguments @("restore", $consumer, "--configfile", $config, "--packages", $freshCache, "--locked-mode", "--verbosity", "quiet")
        $consumerAssets = Get-Content (Join-Path $consumerRoot "obj\project.assets.json") -Raw | ConvertFrom-Json -AsHashtable
        $closure = @($consumerAssets.libraries.Keys | Sort-Object -CaseSensitive)
        $expected = @("DecisionInference.Abstractions/0.1.0-private.1", "DecisionInference.SystemOne/0.1.0-private.1",
            "DecisionInference.OllamaSharp/0.1.0-private.1", "OllamaSharp/5.5.0", "Microsoft.Extensions.AI/10.10.0",
            "Microsoft.Extensions.AI.Abstractions/10.10.0", "Microsoft.Extensions.Caching.Abstractions/10.0.12",
            "Microsoft.Extensions.DependencyInjection.Abstractions/10.0.12", "Microsoft.Extensions.Logging.Abstractions/10.0.12",
            "Microsoft.Extensions.Primitives/10.0.12", "System.Numerics.Tensors/10.0.12") | Sort-Object -CaseSensitive
        if ([string]::Join("`n", $closure) -cne [string]::Join("`n", $expected) -or
            @($consumerAssets.libraries.Values | Where-Object type -eq "project").Count -or
            @($consumerAssets.project.restore.sources.Keys | Where-Object { $_ -cne $feed }).Count -or
            @($consumerAssets.packageFolders.Keys | Where-Object { $_.TrimEnd("\") -cne $freshCache.TrimEnd("\") }).Count) {
            throw "Package-only consumer escaped the verified local closure/cache or includes source projects."
        }
        $consumerLockPath = Join-Path $consumerRoot "packages.lock.json"
        $consumerLock = Get-Content $consumerLockPath -Raw | ConvertFrom-Json -AsHashtable
        foreach ($archive in $archives) {
            $dependency = $consumerLock.dependencies.'net10.0'[$archive.id]
            if (-not $dependency) {
                $key = @($consumerLock.dependencies.'net10.0'.Keys | Where-Object { $_.ToLowerInvariant() -ceq $archive.id })
                $dependency = $consumerLock.dependencies.'net10.0'[$key[0]]
            }
            if ($dependency.resolved -cne $archive.version -or $dependency.contentHash -cne $archive.contentHash) {
                throw "Consumer external archive identity differs: $($archive.id)."
            }
        }
        Run-DotNet -Arguments @("build", $consumer, "--configuration", "Release", "--no-restore", "--no-incremental", "--verbosity", "quiet")
        Run-DotNet -Arguments @("run", "--project", $consumer, "--configuration", "Release", "--no-build", "--no-restore")
        $consumerOutput = Join-Path $consumerRoot "bin\Release\net10.0"
        $testedDll = Join-Path $repo "tests\DecisionInference.OllamaSharp.Tests\bin\Release\net10.0\DecisionInference.OllamaSharp.dll"
        if ((Get-FileHash $testedDll).Hash -cne (Get-FileHash (Join-Path $consumerOutput "DecisionInference.OllamaSharp.dll")).Hash) {
            throw "Consumer did not use the exact tested adapter DLL."
        }
        Copy-Item -LiteralPath $consumerLockPath -Destination eng\OllamaSharpPackageConsumer\packages.lock.json
        $privateArchives = @(Get-ChildItem $feed -Filter "decisioninference.*.nupkg" | ForEach-Object {
            [ordered]@{ name = $_.Name; length = $_.Length; sha256 = (Get-FileHash $_.FullName).Hash }
        })
        $packageReceipt = [ordered]@{ closure = $closure; legacyClosure = $legacyClosure; externalArchives = $archives;
            privateArchives = $privateArchives; testedAdapterDllSha256 = (Get-FileHash $testedDll).Hash;
            consumerDllSha256 = (Get-FileHash (Join-Path $consumerOutput "OllamaSharpPackageConsumer.dll")).Hash;
            freshCache = $freshCache; consumerProject = $consumer; consumerLockSha256 = (Get-FileHash $consumerLockPath).Hash;
            signaturesVerified = $true; projectReferences = 0 }
    }
    Assert-Preserved
    [ordered]@{ status = "passed"; baseline = $baseline; head = (& git rev-parse HEAD);
        newTests = $newTests; retainedTests = $retainedTests; measuredBaselineTests = $baselineTests;
        uniqueTotal = $newTests.total + $retainedTests.total;
        api = $apis; packages = $packageReceipt; sampleSafePaths = "passed"; documentationLinks = "passed";
        sdkTestDllSha256 = (Get-FileHash "tests\DecisionInference.OllamaSharp.Tests\bin\Release\net10.0\DecisionInference.OllamaSharp.Tests.dll").Hash;
        retainedTestDllSha256 = (Get-FileHash "tests\DecisionInference.Tests\bin\Release\net10.0\DecisionInference.Tests.dll").Hash;
        commands = $commands.ToArray(); noLiveInference = $true; noPublication = $true } |
        ConvertTo-Json -Depth 12 | Set-Content (Join-Path $output "verification.json")
    Write-Host "PASS: optional SDK proof, $($newTests.total) new + $($retainedTests.total) retained cases, 480 retained API entries. Receipt: $output"
}
catch {
    [ordered]@{ status = "blocked"; gateError = $_.Exception.Message; commands = $commands.ToArray() } |
        ConvertTo-Json -Depth 8 | Set-Content (Join-Path $output "blocked.json")
    throw
}
finally {
    Pop-Location
    $env:NUGET_PACKAGES = $oldCache
    $env:NUGET_HTTP_CACHE_PATH = $oldHttpCache
}
