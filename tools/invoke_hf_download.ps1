# HF Xet shard 다운로드를 시간 제한과 부분 재개 재시도로 실행합니다.
param(
    [Parameter(Mandatory = $true)][string]$HfPath,
    [Parameter(Mandatory = $true)][string]$Repository,
    [Parameter(Mandatory = $true)][string]$Filename,
    [Parameter(Mandatory = $true)][string]$Revision,
    [Parameter(Mandatory = $true)][string]$SlotPath,
    [Parameter(Mandatory = $true)][string]$LogBase,
    [int]$TimeoutSeconds = 600,
    [int]$MaxAttempts = 3
)

$ErrorActionPreference = "Stop"
if ($TimeoutSeconds -lt 1 -or $MaxAttempts -lt 1) {
    throw "invalid download retry configuration"
}

for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    $attemptLog = "$LogBase.attempt-$attempt"
    $process = Start-Process -FilePath $HfPath -ArgumentList @(
        "download", $Repository, $Filename,
        "--revision", $Revision, "--local-dir", $SlotPath, "--quiet"
    ) -RedirectStandardOutput ($attemptLog + ".stdout.log") `
      -RedirectStandardError ($attemptLog + ".stderr.log") `
      -WindowStyle Hidden -PassThru
    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        & taskkill.exe /PID $process.Id /T /F | Out-Null
        $process.WaitForExit()
        $process.Dispose()
        if ($attempt -eq $MaxAttempts) {
            throw "download timed out for $Filename after $MaxAttempts attempts"
        }
        continue
    }
    $process.WaitForExit()
    $process.Dispose()
    if (Test-Path -LiteralPath (Join-Path $SlotPath $Filename)) {
        return
    }
    if ($attempt -eq $MaxAttempts) {
        throw "download failed for $Filename after $MaxAttempts attempts without a completed target"
    }
}
