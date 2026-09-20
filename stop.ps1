#Requires -Version 5.1
<#
.SYNOPSIS
    ticai-daban Windows 停止脚本（对应 stop.sh）

.DESCRIPTION
    两级清理，与 stop.sh 一致：
      1) pid 文件精确停止
      2) 命令行兜底扫描 —— 旧进程在 pid 文件被覆盖后杀不掉（20260828 事故根因），
         用命令行匹配把残留连同 rqalpha 派生的策略子进程一并清掉
    校验最多等 5 秒确认无残留。

.EXAMPLE
    .\stop.ps1
#>

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$LogDir = Join-Path $Root "logs"

# 匹配本项目服务；sim.py 派生的 rqalpha 子进程命令行同样含 apps/sim.py
$Patterns = 'apps[\\/](radar|poller|server|sim|ai_feed)\.py|collect[\\/]fetch_news\.py'

function Get-Stray {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -match $Patterns }
}

# ---- 1) pid 文件精确停止 ----
foreach ($name in @("poller", "server", "radar", "sim", "aifeed", "newsfeed")) {
    $f = Join-Path $LogDir "$name.pid"
    if (-not (Test-Path $f)) { continue }
    $procId = Get-Content $f -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($procId -and (Get-Process -Id $procId -ErrorAction SilentlyContinue)) {
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        Write-Host "已停止 $name (PID $procId)"
    }
    Remove-Item $f -Force -ErrorAction SilentlyContinue
}

# ---- 2) 兜底清理 ----
foreach ($s in @(Get-Stray)) {
    Stop-Process -Id $s.ProcessId -Force -ErrorAction SilentlyContinue
    Write-Host "兜底清理残留 PID $($s.ProcessId)"
}

# ---- 3) 校验 ----
for ($i = 0; $i -lt 5; $i++) {
    $left = @(Get-Stray)
    if ($left.Count -eq 0) {
        Write-Host "校验通过: 无残留进程"
        exit 0
    }
    Start-Sleep -Seconds 1
}
Write-Warning "仍有 $((@(Get-Stray)).Count) 个进程未退出"