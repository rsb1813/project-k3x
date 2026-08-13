# RAM staging 이후 별도 프로세스에서 제한된 HF Xet prefetch를 실행합니다.
param(
    [Parameter(Mandatory = $true)][string]$MarkerPath,
    [Parameter(Mandatory = $true)][string]$DownloadHelper,
    [Parameter(Mandatory = $true)][string]$HfPath,
    [Parameter(Mandatory = $true)][string]$Repository,
    [Parameter(Mandatory = $true)][string]$Filename,
    [Parameter(Mandatory = $true)][string]$Revision,
    [Parameter(Mandatory = $true)][string]$SlotPath,
    [Parameter(Mandatory = $true)][string]$LogBase,
    [Parameter(Mandatory = $true)][string]$XetCache,
    [int]$DownloadSlots = 2,
    [int]$TimeoutSeconds = 600,
    [int]$MaxAttempts = 3
)

$ErrorActionPreference = "Stop"
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
