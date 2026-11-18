[CmdletBinding()]
param(
    # dotnet/extensions checkout of the typed decision layer (dotnet/extensions#7796).
    [Parameter(Mandatory)][string]$TypedLayerCheckout,
    # dotnet/extensions checkout of the OpenAI decision adapter.
    [Parameter(Mandatory)][string]$OpenAIAdapterCheckout,
    [string]$Version = "11.0.0-decisions.1",
    # Optional SDK root to use instead of each checkout's restored .dotnet directory.
    [string]$DotnetRoot
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$feed = Join-Path $root "artifacts\official-feed"
$abstractions = "src/Libraries/Microsoft.Extensions.AI.Abstractions"

function Get-Tree([string]$checkout, [string]$path) {
    $tree = & git -C $checkout rev-parse "HEAD:$path"
    if ($LASTEXITCODE -ne 0) { throw "Cannot read $path in $checkout." }
    return $tree
}

function Invoke-Pack([string]$checkout, [string]$project) {
    $sdkRoot = if ($DotnetRoot) { $DotnetRoot } else { Join-Path $checkout ".dotnet" }
    $dotnet = Join-Path $sdkRoot "dotnet.exe"
    if (-not (Test-Path -LiteralPath $dotnet)) {
        throw "Restore $checkout first (restore.cmd) so its pinned SDK exists at .dotnet, or pass -DotnetRoot."
    }
    $previousRoot = $env:DOTNET_ROOT
    try {
        $env:DOTNET_ROOT = Split-Path $dotnet -Parent
        # The dotnet host selects the SDK from global.json in the working directory.
        Push-Location -LiteralPath $checkout
        & $dotnet pack (Join-Path $checkout $project) -c Release --nologo -v q -o $feed "-p:PackageVersion=$Version"
        if ($LASTEXITCODE -ne 0) { throw "Packing $project failed." }
    } finally {
        Pop-Location
        $env:DOTNET_ROOT = $previousRoot
    }
}

foreach ($checkout in $TypedLayerCheckout, $OpenAIAdapterCheckout) {
    if (& git -C $checkout status --porcelain) { throw "$checkout has uncommitted changes." }
}
# Both packages depend on the same Abstractions package, so both checkouts must share its sources.
$typedTree = Get-Tree $TypedLayerCheckout $abstractions
$adapterTree = Get-Tree $OpenAIAdapterCheckout $abstractions
& git -C $TypedLayerCheckout diff --quiet $typedTree $adapterTree -- . ':(exclude)*.md'
if ($LASTEXITCODE -ne 0) {
    throw "The two checkouts contain different Microsoft.Extensions.AI.Abstractions sources. Rebase them onto the same foundation commit."
}

$cache = if ($env:NUGET_PACKAGES) { $env:NUGET_PACKAGES } else { Join-Path $HOME ".nuget\packages" }
foreach ($id in "microsoft.extensions.ai.abstractions", "microsoft.extensions.ai", "microsoft.extensions.ai.openai") {
    $cached = Join-Path $cache "$id\$Version"
    if (Test-Path -LiteralPath $cached) { Remove-Item -LiteralPath $cached -Recurse -Force }
}
if (Test-Path -LiteralPath $feed) { Remove-Item -LiteralPath $feed -Recurse -Force }
New-Item -ItemType Directory -Path $feed | Out-Null

Invoke-Pack $TypedLayerCheckout "$abstractions/Microsoft.Extensions.AI.Abstractions.csproj"
Invoke-Pack $TypedLayerCheckout "src/Libraries/Microsoft.Extensions.AI/Microsoft.Extensions.AI.csproj"
Invoke-Pack $OpenAIAdapterCheckout "src/Libraries/Microsoft.Extensions.AI.OpenAI/Microsoft.Extensions.AI.OpenAI.csproj"
Get-ChildItem -LiteralPath $feed -Filter *.symbols.nupkg | Remove-Item

Write-Host "Typed layer: $(& git -C $TypedLayerCheckout rev-parse HEAD)"
Write-Host "OpenAI adapter: $(& git -C $OpenAIAdapterCheckout rev-parse HEAD)"
Get-ChildItem -LiteralPath $feed | ForEach-Object { Write-Host "Packed $($_.Name)" }
