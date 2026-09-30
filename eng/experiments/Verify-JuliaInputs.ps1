[CmdletBinding()]
param(
    [Parameter(Mandatory)][string] $ExperimentRoot,
    [Parameter(Mandatory)][string] $AssetsDirectory,
    [Parameter(Mandatory)][string] $OutputDirectory
)
$ErrorActionPreference = "Stop"
if (Test-Path -LiteralPath $OutputDirectory) { throw "Input evidence directory must be new." }
$expected = [ordered]@{
    "corpus\uci-sms-v1\SMSSpamCollection" = "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d"
    "corpus\frozen-v1\questions.v1.json" = "9e6e6c5c580934309cb1f9b4572c7b87a93ce22b32b50cb8b3af20caa20068ea"
    "corpus\frozen-v1\split.v1.json" = "1f5967c9e6d228fda4322845a77ee1464dff6ec87ea578abafa16f679127cf6f"
    "corpus\frozen-v1\states.v1.jsonl" = "c00636dc21f7a4e3bb099c508fbf24a407af546851e060c5852cb9aca54639fa"
    "contracts\decision-arrow\final-handoff.v1.json" = "33d86835b7e43fb9c7763f644cb2ba38f2b7f4fad9e2d05b3b9f80032081cdd3"
}
$assets = [ordered]@{
    "model.onnx" = "97141d0cfb1da6204e9f8f24d581af72eaeb82cda21149d83eaa6df7160fbcd9"
    "model.onnx.data" = "fd915be810d7ebfb80fb05a48dd33c9484d17ae1b6bcb9e1f544cbaaa913ded1"
    "tokenizer.json" = "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f"
    "asset-receipt.json" = "bf9fb5c32b912fd07ea4cfba27ab0a07819cc6c23d68e0fc4480f3b129fae031"
}
$files = [Collections.Generic.List[object]]::new()
foreach ($set in @(@{ root = $ExperimentRoot; hashes = $expected }, @{ root = $AssetsDirectory; hashes = $assets })) {
    foreach ($entry in $set.hashes.GetEnumerator()) {
        $path = Join-Path $set.root $entry.Key
        $hash = (Get-FileHash -LiteralPath $path).Hash.ToLowerInvariant()
        if ($hash -cne $entry.Value) { throw "Pinned Julia input mismatch: $($entry.Key)" }
        $files.Add(@{ path = $path; sha256 = $hash; sizeBytes = (Get-Item -LiteralPath $path).Length })
    }
}
New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
$licenseUrl = "https://huggingface.co/SupersonicLabs/Julia-1-ONNX/raw/82a2fadf8fccfccdc5fd4e1009ba8f1a265eb7a8/README.md"
$modelCard = Join-Path $OutputDirectory "model-card.pinned.md"
Invoke-WebRequest -Uri $licenseUrl -OutFile $modelCard
if (-not ([IO.File]::ReadAllText($modelCard) -match '(?m)^license: apache-2\.0$')) { throw "Pinned license/model-card evidence not found." }
@{
    schemaVersion = 1; status = "VERIFIED"; files = @($files)
    license = @{ evidenceUrl = $licenseUrl; file = "model-card.pinned.md"; declaredLicense = "Apache-2.0";
        sha256 = (Get-FileHash -LiteralPath $modelCard).Hash.ToLowerInvariant() }
    unusedInputs = @("julia_config.json", "tokenizer_config.json")
    preserved = "Original5574 corpus/5102groups/seed1729/splits3344+1116+1114; labels/splits NEVER passed to Julia."
} | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath (Join-Path $OutputDirectory "inputs.receipt.json") -Encoding utf8
