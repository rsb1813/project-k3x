# Shard 6 봉인 후 단일 Local Foundry를 세 개의 겹치지 않는 worker로 교체합니다.
param(
    [int]$ExpectedCompletedUnits = 6
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$ledger = "C:\K3X\immortal-ledger-quality.json"
$pidPath = "C:\K3X\foundry.pid"
$foundry = Join-Path $PSScriptRoot "run_local_foundry_quality.ps1"

while ($true) {
    $completed = (Get-Content -Raw -LiteralPath $ledger | ConvertFrom-Json).completed_units.Count
    if ($completed -ge $ExpectedCompletedUnits) {
        break
    }
    Start-Sleep -Seconds 5
}

$existingPid = [int](Get-Content -Raw -LiteralPath $pidPath)
$existing = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
if ($null -eq $existing -or $existing.ProcessName -ne "powershell") {
    throw "existing foundry conductor identity mismatch"
}
Stop-Process -Id $existingPid -Force
Wait-Process -Id $existingPid -ErrorAction SilentlyContinue
& wsl -e bash -lc "pkill -f 'convert_local_shard.py.*model-00007-of-000096' || true"

$workers = @(
    @{ Name = "a"; Start = 7; End = 36; Staging = "D:\K3X-staging" },
    @{ Name = "b"; Start = 37; End = 66; Staging = "D:\K3X-staging-b" },
    @{ Name = "c"; Start = 67; End = 96; Staging = "D:\K3X-staging-c" }
)
$launched = @()
foreach ($worker in $workers) {
    $stdout = "C:\K3X\foundry-$($worker.Name).stdout.log"
    $stderr = "C:\K3X\foundry-$($worker.Name).stderr.log"
    $progress = "C:\K3X\foundry-$($worker.Name)-progress.jsonl"
    $process = Start-Process -FilePath powershell.exe -ArgumentList @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $foundry,
        "-StartIndex", $worker.Start, "-EndIndex", $worker.End,
        "-StagingRoot", $worker.Staging, "-Ledger", $ledger,
        "-Progress", $progress
    ) -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
      -WindowStyle Hidden -PassThru
    $launched += @{
        name = $worker.Name
        pid = $process.Id
        start = $worker.Start
        end = $worker.End
        staging = $worker.Staging
    }
}
$launched | ConvertTo-Json | Set-Content -LiteralPath "C:\K3X\parallel-foundry-pids.json" -Encoding utf8
