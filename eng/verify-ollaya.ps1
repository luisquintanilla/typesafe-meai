[CmdletBinding()]
param(
    [Parameter(Mandatory)][uri]$Endpoint,
    [Parameter(Mandatory)][string]$Model,
    [string]$Receipt = "artifacts\ollaya-proof\managed-live.json"
)
$ErrorActionPreference = "Stop"
Push-Location (Split-Path $PSScriptRoot)
try {
    New-Item -ItemType Directory -Force (Split-Path $Receipt) | Out-Null
    & dotnet run --project eng\DecisionProof\DecisionProof.csproj --configuration Release --no-build --no-restore -- provider ollaya $Endpoint.AbsoluteUri $Model $Receipt
    if ($LASTEXITCODE -ne 0) { throw "Live Ollaya contract proof failed." }
}
finally { Pop-Location }
