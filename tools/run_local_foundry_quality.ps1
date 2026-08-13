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
    [int]$DownloadTimeoutSeconds = 600,
    [int]$DownloadMaxAttempts = 3,
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
$downloadHelper = Join-Path $PSScriptRoot "invoke_hf_download.ps1"

if ($StartIndex -lt 1 -or $EndIndex -gt $manifest.shards.Count -or $StartIndex -gt $EndIndex) {
    throw "invalid shard range"
}
if ($DownloadSlots -lt 1 -or $DownloadTimeoutSeconds -lt 1 -or $DownloadMaxAttempts -lt 1) {
    throw "invalid download configuration"
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

function Enter-DownloadSlot {
    while ($true) {
        for ($slot = 0; $slot -lt $DownloadSlots; $slot++) {
            $mutex = [System.Threading.Mutex]::new(
                $false, "K3XFoundryDownloadSlot$slot"
            )
            try {
                if ($mutex.WaitOne(0)) {
                    return $mutex
                }
            }
            catch [System.Threading.AbandonedMutexException] {
                return $mutex
            }
            $mutex.Dispose()
        }
        Start-Sleep -Milliseconds 250
    }
}

function Invoke-BoundedDownload(
    [string]$filename,
    [string]$slotPath
) {
    $slotMutex = Enter-DownloadSlot
    try {
        & $downloadHelper -HfPath $hf -Repository $manifest.repository `
            -Filename $filename -Revision $revision -SlotPath $slotPath `
            -LogBase (Join-Path $stagingRoot ("logs\download-" + $filename)) `
            -TimeoutSeconds $DownloadTimeoutSeconds `
            -MaxAttempts $DownloadMaxAttempts
    }
    finally {
        $slotMutex.ReleaseMutex()
        $slotMutex.Dispose()
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
        param($MarkerPath, $DownloadHelper, $HfPath, $Repository, $Filename, $Revision, $SlotPath, $LogBase, $XetCache, $DownloadSlots, $TimeoutSeconds, $MaxAttempts)
        $env:HF_XET_HIGH_PERFORMANCE = "1"
        $env:HF_HUB_DISABLE_XET = "0"
        $env:HF_XET_CACHE = $XetCache
        Remove-Item Env:HF_HOME -ErrorAction SilentlyContinue
        while (-not (Test-Path -LiteralPath $MarkerPath)) {
            Start-Sleep -Milliseconds 250
        }
        $slotMutex = $null
        while ($null -eq $slotMutex) {
            for ($slot = 0; $slot -lt $DownloadSlots; $slot++) {
                $candidate = [System.Threading.Mutex]::new(
                    $false, "K3XFoundryDownloadSlot$slot"
                )
                try {
                    if ($candidate.WaitOne(0)) {
                        $slotMutex = $candidate
                        break
                    }
                }
                catch [System.Threading.AbandonedMutexException] {
                    $slotMutex = $candidate
                    break
                }
                $candidate.Dispose()
            }
            if ($null -eq $slotMutex) {
                Start-Sleep -Milliseconds 250
            }
        }
        try {
            & $DownloadHelper -HfPath $HfPath -Repository $Repository `
                -Filename $Filename -Revision $Revision -SlotPath $SlotPath `
                -LogBase $LogBase -TimeoutSeconds $TimeoutSeconds `
                -MaxAttempts $MaxAttempts
        }
        finally {
            $slotMutex.ReleaseMutex()
            $slotMutex.Dispose()
        }
    } -ArgumentList @(
        $markerPath, $downloadHelper, $hf, $manifest.repository,
        $shard.filename, $revision, $slotPath, $logBase,
        (Join-Path $stagingRoot ".xet-cache"), $DownloadSlots,
        $DownloadTimeoutSeconds, $DownloadMaxAttempts
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
