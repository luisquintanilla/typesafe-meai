[CmdletBinding()]
param(
    [switch]$Offline,
    [switch]$IncludeBaseline,
    [switch]$Packages,
    [string]$LayaDirectory,
    [string]$QwenPath,
    [switch]$IndependentReference
)
$ErrorActionPreference = "Stop"
if (-not $Offline) { throw "Pass -Offline. This entry never downloads models or invokes live services." }
$repo = Split-Path $PSScriptRoot
Push-Location $repo
try {
    function Run-DotNet {
        & dotnet @args
        if ($LASTEXITCODE -ne 0) { throw "Decision verification dotnet command failed: $LASTEXITCODE." }
    }
    $baseline = "bc2f307ed2b7e8c5ea0eac5e95dabeb4868b7c19"
    $preserved = @(git ls-tree -r --name-only $baseline | Where-Object { $_ -ne "Directory.Packages.props" })
    if ($LASTEXITCODE -ne 0) { throw "Baseline commit unavailable." }
    & git diff --quiet $baseline -- @preserved
    if ($LASTEXITCODE -ne 0) { throw "A protected baseline file changed." }
    [xml]$oldPackages = (git show "${baseline}:Directory.Packages.props") -join "`n"
    [xml]$newPackages = Get-Content Directory.Packages.props -Raw
    foreach ($p in $oldPackages.Project.ItemGroup.PackageVersion) {
        $now = @($newPackages.Project.ItemGroup.PackageVersion | Where-Object Include -eq $p.Include)
        if ($now.Count -ne 1 -or $now[0].Version -ne $p.Version) { throw "Baseline package version changed: $($p.Include)." }
    }
    if ($IncludeBaseline) { & (Join-Path $PSScriptRoot "verify.ps1") -All -Offline }
    $solution = Join-Path $PSScriptRoot "DecisionComparison.slnx"
    Run-DotNet restore $solution --locked-mode --verbosity quiet
    Run-DotNet build $solution --configuration Release --no-restore --no-incremental --verbosity quiet
    Run-DotNet test tests\DecisionInference.Tests\DecisionInference.Tests.csproj --configuration Release --no-build --no-restore --blame-hang-timeout 2min
    $hostProject = "samples\DecisionComparison\DecisionComparison.csproj"
    Run-DotNet run --project $hostProject --configuration Release --no-build --no-restore -- compare
    Run-DotNet run --project $hostProject --configuration Release --no-build --no-restore -- inventory

    $libraries = @("DecisionInference.Abstractions", "DecisionInference.TypeSafe", "DecisionInference.Laya", "DecisionInference.TensorSharp")
    $graphs = @()
    foreach ($name in $libraries) {
        $lock = Get-Content "src\$name\packages.lock.json" -Raw | ConvertFrom-Json
        $deps = @($lock.dependencies.'net10.0'.PSObject.Properties | Where-Object { $_.Value.type -ne "Project" })
        if ($deps.Name -contains "Microsoft.ML" -or $deps.Name -contains "MLNet.TextInference.Onnx" -or
            $deps.Name -contains "TypeSafe.MEAI") { throw "Forbidden dependency in $name." }
        if ($name -in @("DecisionInference.Abstractions", "DecisionInference.TypeSafe") -and
            @($deps.Name | Where-Object { $_ -like "*OnnxRuntime*" -or $_ -like "TensorSharp*" }).Count) {
            throw "Local-runtime dependency leaked into $name."
        }
        $sources = @(Get-ChildItem "src\$name" -Filter *.cs -Recurse | Where-Object FullName -NotMatch '[\\/](obj|bin)[\\/]')
        $graphs += [ordered]@{ project = $name; runtimeFiles = $sources.Count;
            physicalLines = ($sources | ForEach-Object { [IO.File]::ReadAllLines($_.FullName).Length } | Measure-Object -Sum).Sum;
            packages = @($deps | ForEach-Object { "$($_.Name)/$($_.Value.resolved)" }) }
    }
    New-Item -ItemType Directory -Force artifacts\decision-proof | Out-Null
    $graphs | ConvertTo-Json -Depth 6 | Set-Content artifacts\decision-proof\dependency-graphs.json

    if ($LayaDirectory) {
        $LayaDirectory = (Resolve-Path $LayaDirectory).Path
        $expected = @{
            "laya.onnx" = "a874eb254b58b0fcb1e7ad56fbb188c29d64e08c9a46b689433e1f52c66dba1e"
            "laya.onnx.data" = "487746363a8da57bcadb4345352997d22a0fb90d70aa22c6856668d023242aba"
            "laya_config.json" = "5049005dc6ae3ca5e82cc7d85c421357d5c543817300c8e8c5281ddbc69bb561"
            "tokenizer\tokenizer.json" = "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30"
            "tokenizer\tokenizer_config.json" = "50044de60daaa73df97d262e15a40d4faf0160e7d742df64b377877a1320dd12"
        }
        foreach ($entry in $expected.GetEnumerator()) {
            if ((Get-FileHash (Join-Path $LayaDirectory $entry.Key) -Algorithm SHA256).Hash -ine $entry.Value) {
                throw "Laya asset hash mismatch: $($entry.Key)."
            }
        }
        $model = Join-Path $LayaDirectory "laya.onnx"
        $tokenizer = Join-Path $LayaDirectory "tokenizer\tokenizer.json"
        $profile = Join-Path $LayaDirectory "laya_config.json"
        Run-DotNet run --project $hostProject --configuration Release --no-build --no-restore -- stages $model $tokenizer $profile artifacts\decision-proof\laya-managed.json
        if ($IndependentReference) {
            & python (Join-Path $PSScriptRoot "verify-laya-reference.py") $LayaDirectory artifacts\decision-proof\laya-managed.json
            if ($LASTEXITCODE -ne 0) { throw "Independent Laya reference failed." }
        }
    }
    elseif ($IndependentReference) { throw "-IndependentReference requires -LayaDirectory." }
    else { Write-Host "Laya real-model proof NOT RUN: no explicit local directory." }
    if ($QwenPath) {
        $QwenPath = (Resolve-Path $QwenPath).Path
        Run-DotNet run --project $hostProject --configuration Release --no-build --no-restore -- provider qwen $QwenPath
    }
    else { Write-Host "Qwen real-model proof NOT RUN: no explicit local GGUF." }

    if ($Packages) {
        $proof = Join-Path $repo "artifacts\decision-proof\packages" ([Guid]::NewGuid().ToString("N"))
        $feed = Join-Path $proof "feed"
        $consumer = Join-Path $proof "consumer"
        New-Item -ItemType Directory -Path $feed, $consumer | Out-Null
        $assets = Get-Content samples\DecisionComparison\obj\project.assets.json -Raw | ConvertFrom-Json
        $hostLock = Get-Content samples\DecisionComparison\packages.lock.json -Raw | ConvertFrom-Json
        $sdkVersion = & dotnet --version
        $sdkLine = & dotnet --list-sdks | Where-Object { $_.StartsWith("$sdkVersion [", [StringComparison]::Ordinal) }
        if ($sdkLine -notmatch '\[(.+)\]$') { throw "Cannot locate SDK for public NuGet hash verification." }
        $sdk = Join-Path $Matches[1] $sdkVersion
        foreach ($assembly in @("NuGet.Common", "NuGet.Frameworks", "NuGet.Versioning", "NuGet.Packaging")) {
            [void][Reflection.Assembly]::LoadFrom((Join-Path $sdk "$assembly.dll"))
        }
        $external = @()
        $provenance = @()
        $entries = @($hostLock.dependencies.'net10.0'.PSObject.Properties)
        foreach ($name in $libraries) {
            $libraryLock = Get-Content "src\$name\packages.lock.json" -Raw | ConvertFrom-Json
            $entries += @($libraryLock.dependencies.'net10.0'.PSObject.Properties)
        }
        $seen = @{}
        foreach ($entry in $entries) {
            if ($entry.Value.type -eq "Project") { continue }
            $id = $entry.Name.ToLowerInvariant()
            $version = $entry.Value.resolved
            $key = "$id/$version"
            if ($seen.ContainsKey($key)) {
                if ($seen[$key] -cne $entry.Value.contentHash) { throw "Conflicting locked hashes: $key." }
                continue
            }
            $seen[$key] = $entry.Value.contentHash
            $archive = Join-Path $assets.project.restore.packagesPath "$id\$version\$id.$version.nupkg"
            $reader = [NuGet.Packaging.PackageArchiveReader]::new($archive)
            try {
                if ($reader.GetContentHash([Threading.CancellationToken]::None, $null) -cne $entry.Value.contentHash) {
                    throw "Archive differs from host lock: $id/$version."
                }
            }
            finally { $reader.Dispose() }
            $destination = Join-Path $feed (Split-Path $archive -Leaf)
            Copy-Item -LiteralPath $archive -Destination $destination
            $external += $destination
            $provenance += @{ id = $id; version = $version; contentHash = $entry.Value.contentHash; archiveSha256 = (Get-FileHash $archive).Hash }
        }
        & dotnet nuget verify @external --all --configfile NuGet.config --verbosity quiet *> (Join-Path $proof "signatures.log")
        if ($LASTEXITCODE -ne 0) { throw "External package signature verification failed." }
        foreach ($name in @($libraries) + @("TypeSafe.MEAI")) {
            Run-DotNet pack "src\$name\$name.csproj" --configuration Release --no-build --no-restore --output $feed --verbosity quiet
        }
        Copy-Item samples\DecisionComparison\Program.cs, samples\DecisionComparison\RecordedJevHandler.cs $consumer
        # Generated throwaway copy of the same host: only PackageReferences, never source ProjectReferences.
        @'
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable><ManagePackageVersionsCentrally>false</ManagePackageVersionsCentrally><DisableImplicitLibraryPacksFolder>true</DisableImplicitLibraryPacksFolder><DisableImplicitNuGetFallbackFolder>true</DisableImplicitNuGetFallbackFolder><NuGetAudit>false</NuGetAudit></PropertyGroup>
  <ItemGroup>
    <PackageReference Include="DecisionInference.TypeSafe" Version="0.1.0-private.1" />
    <PackageReference Include="DecisionInference.Laya" Version="0.1.0-private.1" />
    <PackageReference Include="DecisionInference.TensorSharp" Version="0.1.0-private.1" />
    <PackageReference Include="TypeSafe.MEAI" Version="0.1.0-private.1" />
    <PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.23.2" />
  </ItemGroup>
</Project>
'@ | Set-Content (Join-Path $consumer "DecisionPackageConsumer.csproj")
        $escapedFeed = [Security.SecurityElement]::Escape($feed)
        "<configuration><packageSources><clear/><add key=`"verified-local`" value=`"$escapedFeed`"/></packageSources><disabledPackageSources><clear/></disabledPackageSources><packageSourceMapping><clear/></packageSourceMapping><auditSources><clear/></auditSources><config><add key=`"signatureValidationMode`" value=`"accept`"/></config></configuration>" |
            Set-Content (Join-Path $consumer "NuGet.config")
        $project = Join-Path $consumer "DecisionPackageConsumer.csproj"
        Run-DotNet restore $project --configfile (Join-Path $consumer "NuGet.config") --packages (Join-Path $proof "cache") --verbosity quiet
        $resolved = Get-Content (Join-Path $consumer "obj\project.assets.json") -Raw | ConvertFrom-Json
        if (@($resolved.libraries.PSObject.Properties | Where-Object { $_.Value.type -eq "project" }).Count) { throw "Source project leaked into package consumer." }
        if (@($resolved.project.restore.sources.PSObject.Properties.Name | Where-Object { $_ -cne $feed }).Count) { throw "Nonlocal package source used." }
        $consumerLock = Get-Content (Join-Path $consumer "packages.lock.json") -Raw | ConvertFrom-Json
        foreach ($entry in $consumerLock.dependencies.'net10.0'.PSObject.Properties) {
            if ($entry.Name -in @($libraries) + @("TypeSafe.MEAI")) { continue }
            $key = "$($entry.Name.ToLowerInvariant())/$($entry.Value.resolved)"
            if (-not $seen.ContainsKey($key) -or $seen[$key] -cne $entry.Value.contentHash) { throw "Package consumer escaped the verified closure: $key." }
        }
        Run-DotNet restore $project --configfile (Join-Path $consumer "NuGet.config") --packages (Join-Path $proof "cache") --locked-mode --verbosity quiet
        Run-DotNet run --project $project --configuration Release --no-restore -- compare
        if ($LayaDirectory) { Run-DotNet run --project $project --configuration Release --no-build --no-restore -- stages $model $tokenizer $profile }
        if ($QwenPath) { Run-DotNet run --project $project --configuration Release --no-build --no-restore -- provider qwen $QwenPath }
        $standalone = Join-Path $proof "laya-only"
        New-Item -ItemType Directory -Path $standalone | Out-Null
        Copy-Item (Join-Path $consumer "NuGet.config") $standalone
        @'
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable><ManagePackageVersionsCentrally>false</ManagePackageVersionsCentrally><DisableImplicitLibraryPacksFolder>true</DisableImplicitLibraryPacksFolder><DisableImplicitNuGetFallbackFolder>true</DisableImplicitNuGetFallbackFolder><NuGetAudit>false</NuGetAudit></PropertyGroup>
  <ItemGroup><PackageReference Include="DecisionInference.Laya" Version="0.1.0-private.1"/><PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.23.2"/></ItemGroup>
</Project>
'@ | Set-Content (Join-Path $standalone "DecisionLayaConsumer.csproj")
        @'
using System.Text.Json;
using DecisionInference;
using DecisionInference.Laya;
var profile = LayaProfile.Load(args[2]);
var (tokenizer, metadata) = LayaTokenizer.Load(args[1]);
var prepare = new PrepareDecisionInputs(profile, tokenizer, metadata);
var decode = new DecodeDecisions(profile);
using var scorer = ScoreOnnxDecisionModel.Load(args[0]);
using IDecisionClient client = new LayaDecisionClient(prepare, scorer, decode);
var request = new DecisionRequest(JsonSerializer.SerializeToElement("A password request."),
    [new ChoiceDecisionQuestion(" kind ", "Choose a category.", [new(" Safe ", "Safe"), new("unsafe", "Unsafe")])]);
var batch = prepare.Prepare(request);
var direct = decode.Decode(batch, scorer.Score(batch));
var wrapped = await client.GetResponseAsync(request);
if (!JsonElement.DeepEquals(direct.Answers[0].ProviderEvidence, wrapped.Answers[0].ProviderEvidence))
    throw new InvalidOperationException("Standalone stage/client evidence differs.");
var answer = (ChoiceDecisionAnswer)wrapped.Answers[0];
if (answer.QuestionId != " kind " || !answer.Probabilities.ContainsKey(" Safe "))
    throw new InvalidOperationException("Opaque IDs changed.");
Console.WriteLine("Laya-only package consumer: public stage/client parity, no chat or ML.NET types.");
'@ | Set-Content (Join-Path $standalone "Program.cs")
        $standaloneProject = Join-Path $standalone "DecisionLayaConsumer.csproj"
        Run-DotNet restore $standaloneProject --configfile (Join-Path $standalone "NuGet.config") --packages (Join-Path $proof "cache") --verbosity quiet
        $standaloneAssets = Get-Content (Join-Path $standalone "obj\project.assets.json") -Raw | ConvertFrom-Json
        if (@($standaloneAssets.libraries.PSObject.Properties.Name | Where-Object {
            $_ -like "TypeSafe.MEAI/*" -or $_ -like "Microsoft.ML/*" -or $_ -like "TensorSharp*" -or $_ -like "MLNet.*"
        }).Count) { throw "Unrelated runtime leaked into Laya-only consumer." }
        Run-DotNet build $standaloneProject --configuration Release --no-restore --verbosity quiet
        if ($LayaDirectory) { Run-DotNet run --project $standaloneProject --configuration Release --no-build --no-restore -- $model $tokenizer $profile }
        $provenance | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $proof "provenance.json")
        Write-Host "Package-only host proof passed; $($external.Count) external archives hash/signature verified: $proof"
    }
    & git diff --quiet $baseline -- @preserved
    if ($LASTEXITCODE -ne 0) { throw "Verification changed a protected baseline file." }
    Write-Host "Decision proof complete. Jev evidence is synthetic only; no calibrated quality claim."
}
finally { Pop-Location }
