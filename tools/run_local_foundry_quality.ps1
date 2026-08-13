# 공식 Kimi K3 shard를 두 slot으로 내려받아 quality K3X fragment로 제조합니다.
param(
    [int]$StartIndex = 3,
    [int]$EndIndex = 96
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$manifestPath = Join-Path $repositoryRoot "artifacts\m37-local-foundry\source-manifest.json"
$configManifestPath = Join-Path $repositoryRoot "artifacts\m26-official\live\source-manifest.json"
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$revision = $manifest.revision
$stagingRoot = "D:\K3X-staging"
$destination = "C:\K3X\shards"
$ledger = "C:\K3X\immortal-ledger-quality.json"
$progress = "C:\K3X\foundry-progress.jsonl"
$hf = (Get-Command hf).Source

if ($StartIndex -lt 1 -or $EndIndex -gt $manifest.shards.Count -or $StartIndex -gt $EndIndex) {
    throw "invalid shard range"
}
if ((hf auth whoami --format json | ConvertFrom-Json).user -ne "rsb1813") {
    throw "unexpected Hugging Face account"
}

$env:HF_XET_HIGH_PERFORMANCE = "1"
$env:HF_HUB_DISABLE_XET = "0"
$env:HF_XET_CACHE = Join-Path $stagingRoot ".xet-cache"
Remove-Item Env:HF_HOME -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $destination,(Join-Path $stagingRoot "logs") | Out-Null

function Get-SlotPath([int]$index) {
    $slot = ($index - 1) % 2
    return Join-Path $stagingRoot "slot-$slot"
}

function Start-NextDownload([int]$index) {
    if ($index -gt $EndIndex) {
        return
    }
    $shard = $manifest.shards[$index - 1]
    $slotPath = Get-SlotPath $index
    $target = Join-Path $slotPath $shard.filename
    if (Test-Path -LiteralPath $target) {
        return
    }
    New-Item -ItemType Directory -Force -Path $slotPath | Out-Null
    $logBase = Join-Path $stagingRoot ("logs\download-" + $shard.filename)
    Start-Process -FilePath $hf -ArgumentList @(
        "download", $manifest.repository, $shard.filename,
        "--revision", $revision, "--local-dir", $slotPath, "--quiet"
    ) -RedirectStandardOutput ($logBase + ".stdout.log") `
      -RedirectStandardError ($logBase + ".stderr.log") `
      -WindowStyle Hidden | Out-Null
}

for ($index = $StartIndex; $index -le $EndIndex; $index++) {
    $shard = $manifest.shards[$index - 1]
    $slotPath = Get-SlotPath $index
    New-Item -ItemType Directory -Force -Path $slotPath | Out-Null

    & $hf download $manifest.repository $shard.filename `
        --revision $revision --local-dir $slotPath --quiet
    if ($LASTEXITCODE -ne 0) {
        throw "download failed for $($shard.filename)"
    }
    Start-NextDownload ($index + 1)

    $sourceLinux = "/mnt/d/K3X-staging/slot-$((($index - 1) % 2))/$($shard.filename)"
    $command = @(
        "cd /mnt/c/Users/jolib/Documents/project-k3x/.worktrees/milestone-twenty-four-cuda-graph-cache",
        "PYTHONPATH=converter:reference /home/jolib/.venvs/k3x-m1/bin/python tools/convert_local_shard.py",
        "--manifest artifacts/m37-local-foundry/source-manifest.json",
        "--config-manifest artifacts/m26-official/live/source-manifest.json",
        "--source $sourceLinux",
        "--destination /mnt/c/K3X/shards",
        "--ledger /mnt/c/K3X/immortal-ledger-quality.json",
        "--output-budget-bytes 1510500000000",
        "--delete-source"
    ) -join " "
    $result = & wsl -e bash -lc $command
    if ($LASTEXITCODE -ne 0) {
        throw "conversion failed for $($shard.filename)"
    }
    Add-Content -LiteralPath $progress -Value $result -Encoding utf8
}
