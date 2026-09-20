#Requires -Version 5.1
<#
.SYNOPSIS
    ticai-daban Windows 启动脚本（PowerShell 原生，不依赖 Git Bash/WSL）

.DESCRIPTION
    与 macOS 的 start.sh 等价，拉起同一组服务。差异点：
      - 判活用 Get-CimInstance 读命令行做二次匹配（Windows 无 `ps -p -o args=`，
        且 PID 会被复用，只判存在会误判「已在运行」而跳过拉起）
      - 解释器路径是 .venv\Scripts\python.exe（非 .venv/bin/python）
      - stdout/stderr 分两个文件重定向（PowerShell 不允许同一文件吃两路）
      - 停止靠 Stop-Process（Windows 没有优雅 SIGTERM，radar 的内存轨迹
        来不及 flush —— 雷达每个 cycle 都会落盘 intraday_px，损失有限）

.PARAMETER Port
    看板端口，默认 8766（8765 已被 loongsuite-pilot dashboard 占用）

.PARAMETER WithSim
    额外拉起策略模拟（rqalpha）。默认不拉，先跑通看板/雷达时不需要它。

.EXAMPLE
    .\start.ps1
    .\start.ps1 -Port 8899
    .\start.ps1 -WithSim
#>
param(
    [int]$Port = 8766,
    [switch]$WithSim
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# ---- 解释器: 优先项目 venv ----
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Warning "未找到 .venv\Scripts\python.exe, 回退系统 python"
    Write-Warning "建议先执行: python -m venv .venv; .venv\Scripts\pip install -r requirements.txt"
    $Py = "python"
}
Write-Host "解释器: $Py"

function Get-ProcCmdLine([int]$procId) {
    try {
        $p = Get-CimInstance Win32_Process -Filter "ProcessId=$procId" -ErrorAction Stop
        return $p.CommandLine
    } catch {
        return $null
    }
}

function Test-Running([string]$PidFile, [string]$Pattern) {
    if (-not (Test-Path $PidFile)) { return $false }
    $procId = Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $procId) { return $false }
    if (-not (Get-Process -Id $procId -ErrorAction SilentlyContinue)) { return $false }
    $cmd = Get-ProcCmdLine $procId
    return [bool]($cmd -and $cmd -match $Pattern)
}

function Start-Svc([string]$Name, [string]$Pattern, [string[]]$SvcArgs) {
    $pidFile = Join-Path $LogDir "$Name.pid"
    if (Test-Running $pidFile $Pattern) {
        $old = Get-Content $pidFile | Select-Object -First 1
        Write-Host "$Name 已在运行 PID $old"
        return
    }
    $out = Join-Path $LogDir "$Name.log"
    $err = Join-Path $LogDir "$Name.err.log"
    $p = Start-Process -FilePath $Py -ArgumentList $SvcArgs `
        -WorkingDirectory $Root -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $out -RedirectStandardError $err
    $p.Id | Out-File -Encoding ascii $pidFile
    Write-Host "$Name 已启动 PID $($p.Id) (日志 logs\$Name.log)"
}

Start-Svc "poller"   "apps/poller.py"        @("-u", "apps/poller.py")
Start-Svc "server"   "apps/server.py"        @("-u", "apps/server.py", "$Port")
Start-Svc "radar"    "apps/radar.py"         @("-u", "apps/radar.py")
Start-Svc "aifeed"   "apps/ai_feed.py"       @("-u", "apps/ai_feed.py", "--loop")
Start-Svc "newsfeed" "collect/fetch_news.py" @("-u", "collect/fetch_news.py", "--loop")

if ($WithSim) {
    Start-Svc "sim" "apps/sim.py" @("-u", "apps/sim.py")
} else {
    Write-Host "策略模拟未拉起（需要时加 -WithSim）"
}

Write-Host ""
Write-Host "打开 http://localhost:$Port"