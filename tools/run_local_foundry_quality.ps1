# 공식 Kimi K3 shard를 두 slot으로 내려받아 quality K3X fragment로 제조합니다.
param(
    [int]$StartIndex = 3,
    [int]$EndIndex = 96,
    [string]$StagingRoot = "D:\K3X-staging",
    [string]$Ledger = "C:\K3X\immortal-ledger-quality.json",
    [string]$Progress = "C:\K3X\foundry-progress.jsonl",
    [string]$TemporaryDirectory = "",
    [string]$StagingLock = "C:\K3X\foundry-ram-stage.lock",
    [string]$OutputAuditLock = "C:\K3X\foundry-output-audit.lock",
    [int]$DownloadSlots = 2,
    [switch]$Finalize
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$manifestPath = Join-Path $repositoryRoot "artifacts\m37-local-foundry\source-manifest.json"
$configManifestPath = Join-Path $repositoryRoot "artifacts\m26-official\live\source-manifest.json"
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$revision = $manifest.revision
$destination = "C:\K3X\shards"
$hf = (Get-Command hf).Source

if ($StartIndex -lt 1 -or $EndIndex -gt $manifest.shards.Count -or $StartIndex -gt $EndIndex) {
    throw "invalid shard range"
}
if ($DownloadSlots -lt 1) {
    throw "invalid download slot count"
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

function Invoke-BoundedDownload(
    [string]$filename,
    [string]$slotPath
) {
    $semaphore = [System.Threading.Semaphore]::new(
        $DownloadSlots, $DownloadSlots, "K3XFoundryDownloads"
    )
    $null = $semaphore.WaitOne()
    try {
        & $hf download $manifest.repository $filename `
            --revision $revision --local-dir $slotPath --quiet
        if ($LASTEXITCODE -ne 0) {
            throw "download failed for $filename"
        }
    }
    finally {
        $null = $semaphore.Release()
        $semaphore.Dispose()
    }
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

function Start-NextDownloadAfterMarker([int]$index, [string]$markerPath) {
    if ($index -gt $EndIndex) {
        return $null
    }
    $shard = $manifest.shards[$index - 1]
    $slotPath = Get-SlotPath $index
    $target = Join-Path $slotPath $shard.filename
    if (Test-Path -LiteralPath $target) {
        return $null
    }
    New-Item -ItemType Directory -Force -Path $slotPath | Out-Null
    $logBase = Join-Path $stagingRoot ("logs\download-" + $shard.filename)
    return Start-Job -ScriptBlock {
        param($MarkerPath, $HfPath, $Repository, $Filename, $Revision, $SlotPath, $LogBase, $DownloadSlots)
        while (-not (Test-Path -LiteralPath $MarkerPath)) {
            Start-Sleep -Milliseconds 250
        }
        $semaphore = [System.Threading.Semaphore]::new(
            $DownloadSlots, $DownloadSlots, "K3XFoundryDownloads"
        )
        $null = $semaphore.WaitOne()
        try {
            $process = Start-Process -FilePath $HfPath -ArgumentList @(
                "download", $Repository, $Filename,
                "--revision", $Revision, "--local-dir", $SlotPath, "--quiet"
            ) -RedirectStandardOutput ($LogBase + ".stdout.log") `
              -RedirectStandardError ($LogBase + ".stderr.log") `
              -WindowStyle Hidden -Wait -PassThru
            if ($process.ExitCode -ne 0) {
                throw "download failed for $Filename"
            }
        }
        finally {
            $null = $semaphore.Release()
            $semaphore.Dispose()
        }
    } -ArgumentList @(
        $markerPath, $hf, $manifest.repository, $shard.filename,
        $revision, $slotPath, $logBase, $DownloadSlots
    )
}

$prefetchJob = $null

for ($index = $StartIndex; $index -le $EndIndex; $index++) {
    if ($null -ne $prefetchJob) {
        Wait-Job -Job $prefetchJob | Out-Null
        Receive-Job -Job $prefetchJob -ErrorAction Stop | Out-Null
        Remove-Job -Job $prefetchJob
        $prefetchJob = $null
    }
    $shard = $manifest.shards[$index - 1]
    $slotPath = Get-SlotPath $index
    $target = Join-Path $slotPath $shard.filename
    New-Item -ItemType Directory -Force -Path $slotPath | Out-Null

    Invoke-BoundedDownload $shard.filename $slotPath
    $sourceLinux = (& wsl -e wslpath -a -u $target).Trim()
    $temporaryArgument = ""
    $stagingReadyArgument = ""
    $stagingLockArgument = ""
    $outputAuditLockArgument = ""
    if ($TemporaryDirectory) {
        $temporaryArgument = "--temporary-directory $TemporaryDirectory"
        $stagingReady = $target + ".ram-ready"
        Remove-Item -LiteralPath $stagingReady -Force -ErrorAction SilentlyContinue
        $stagingReadyLinux = (& wsl -e wslpath -a -u $stagingReady).Trim()
        $stagingReadyArgument = "--staging-ready-file $stagingReadyLinux"
        $stagingLockLinux = (& wsl -e wslpath -a -u $StagingLock).Trim()
        $stagingLockArgument = "--staging-lock-file $stagingLockLinux"
        $outputAuditLockLinux = (& wsl -e wslpath -a -u $OutputAuditLock).Trim()
        $outputAuditLockArgument = "--output-audit-lock-file $outputAuditLockLinux"
        $prefetchJob = Start-NextDownloadAfterMarker ($index + 1) $stagingReady
    }
    else {
        Start-NextDownload ($index + 1)
    }
    $command = @(
        "cd /mnt/c/Users/jolib/Documents/project-k3x/.worktrees/milestone-twenty-four-cuda-graph-cache &&",
        "PYTHONPATH=converter:reference /home/jolib/.venvs/k3x-m1/bin/python tools/convert_local_shard.py",
        "--manifest artifacts/m37-local-foundry/source-manifest.json",
        "--config-manifest artifacts/m26-official/live/source-manifest.json",
        "--source $sourceLinux",
        "--destination /mnt/c/K3X/shards",
        "--ledger $((& wsl -e wslpath -a -u $Ledger).Trim())",
        $temporaryArgument,
        $stagingReadyArgument,
        $stagingLockArgument,
        $outputAuditLockArgument,
        "--output-budget-bytes 1510500000000",
        "--delete-source"
    ) -join " "
    $result = & wsl -e bash -lc $command
    if ($LASTEXITCODE -ne 0) {
        throw "conversion failed for $($shard.filename)"
    }
    if ($TemporaryDirectory) {
        Remove-Item -LiteralPath $stagingReady -Force -ErrorAction SilentlyContinue
    }
    Add-Content -LiteralPath $progress -Value $result -Encoding utf8
}

if (-not $Finalize) {
    exit 0
}

$setCommand = @(
    "cd /mnt/c/Users/jolib/Documents/project-k3x/.worktrees/milestone-twenty-four-cuda-graph-cache &&",
    "PYTHONPATH=converter:reference /home/jolib/.venvs/k3x-m1/bin/python tools/write_fragment_set.py",
    "--manifest artifacts/m37-local-foundry/source-manifest.json",
    "--destination /mnt/c/K3X/shards",
    "--ledger $((& wsl -e wslpath -a -u $Ledger).Trim())",
    "--output /mnt/c/K3X/shards/model.k3xset",
    "--output-budget-bytes 1510500000000"
) -join " "
$setResult = & wsl -e bash -lc $setCommand
if ($LASTEXITCODE -ne 0) {
    throw "fragment set finalization failed"
}
Add-Content -LiteralPath $progress -Value $setResult -Encoding utf8
