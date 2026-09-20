#Requires -Version 5.1
<#
.SYNOPSIS
    ticai-daban PC(Windows) 初始化自检 —— 首次 git pull 后跑一次

.DESCRIPTION
    对应 macOS 侧的 deploy/launchd 前置准备, 但 PC 是「本机 QMT + 原生 Windows」,
    检查项不同。四组检查各自独立判定, 单项失败不中断后续:

      1. 环境   Python 版本 / venv / 关键依赖能否 import
      2. 配置   .env 必填项 —— Windows 没有 ~/.zshrc 兜底, TUSHARE_TOKEN
                必须落在 .env 里才能被 config.py 读到
      3. 数据   data/ 关键数据集是否存在 + 各集日期水位(与 Mac 侧比对有无漏拷)
      4. QMT    本机 Redis / FormulaServer 端口连通性(BIGQMT_* 须指向 127.0.0.1)

    幂等: 可反复运行, 只读不写(除非显式加 -Install)。

.PARAMETER Install
    自动补齐环境: 缺 venv 则创建、缺依赖则 pip install、缺 .env 则从模板复制。
    不加时只报告不动环境。

.PARAMETER Start
    自检通过后直接调用 start.ps1 拉起服务。

.EXAMPLE
    .\init.ps1                  # 只体检
    .\init.ps1 -Install         # 体检并补齐环境
    .\init.ps1 -Install -Start  # 补齐后直接启动
#>
param(
    [switch]$Install,
    [switch]$Start
)

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$script:Issues = New-Object System.Collections.ArrayList
function Add-Issue([string]$Text, [string]$Hint) {
    [void]$script:Issues.Add([pscustomobject]@{ Text = $Text; Hint = $Hint })
}

function Write-Section([string]$Title) {
    Write-Host ""
    Write-Host "== $Title" -ForegroundColor Cyan
}

function Test-Port([string]$HostName, [int]$Port) {
    # 同步 Connect 即可: 本机回环不通时是立即 ConnectionRefused, 不会久等
    $c = New-Object System.Net.Sockets.TcpClient
    try {
        $c.Connect($HostName, $Port)
        return $true
    } catch {
        return $false
    } finally {
        $c.Close()
    }
}

# ---------------------------------------------------------------- 1. 环境
Write-Section "1/4 环境"

$Py = Join-Path $Root ".venv\Scripts\python.exe"
$SysPy = (Get-Command python -ErrorAction SilentlyContinue).Source

if (-not (Test-Path $Py)) {
    Write-Host "MISS  未找到 .venv\Scripts\python.exe"
    if ($Install -and $SysPy) {
        Write-Host "      正在创建 venv ..."
        & $SysPy -m venv .venv
        if (Test-Path $Py) { Write-Host "OK    venv 已创建" -ForegroundColor Green }
    } else {
        Add-Issue ".venv 不存在" "运行 .\init.ps1 -Install, 或手工: python -m venv .venv"
    }
} else {
    Write-Host "OK    venv: $Py" -ForegroundColor Green
}

if (Test-Path $Py) {
    $pyver = & $Py -c "import sys; print('%d.%d.%d' % sys.version_info[:3])" 2>$null
    $minor = & $Py -c "import sys; print(sys.version_info.minor)" 2>$null
    Write-Host "      Python $pyver"
    if ([int]$minor -lt 10 -or [int]$minor -gt 13) {
        Add-Issue "Python 版本 $pyver 不在建议区间 3.10~3.13" "rqalpha 要求 pandas<3.0, 版本偏差易踩依赖坑"
    }

    # 核心依赖(缺一就跑不起服务) / 可选依赖(rqalpha 策略模拟, langgraph AI 编排)
    $coreDeps = @("pandas", "numpy", "pyarrow", "tushare", "akshare", "redis", "requests")
    $optDeps = @("rqalpha", "langgraph")
    $missing = @()
    foreach ($m in $coreDeps) {
        & $Py -c "import $m" 2>$null
        if ($LASTEXITCODE -ne 0) { $missing += $m }
    }
    if ($missing.Count -gt 0) {
        Write-Host "MISS  核心依赖缺失: $($missing -join ', ')" -ForegroundColor Yellow
        if ($Install) {
            Write-Host "      正在 pip install -r requirements.txt ..."
            & $Py -m pip install -r (Join-Path $Root "requirements.txt")
        } else {
            Add-Issue "核心依赖缺失: $($missing -join ', ')" "运行 .\init.ps1 -Install"
        }
    } else {
        $pdver = & $Py -c "import pandas; print(pandas.__version__)" 2>$null
        Write-Host "OK    核心依赖齐备 (pandas $pdver)" -ForegroundColor Green
    }

    $missOpt = @()
    foreach ($m in $optDeps) {
        & $Py -c "import $m" 2>$null
        if ($LASTEXITCODE -ne 0) { $missOpt += $m }
    }
    if ($missOpt.Count -gt 0) {
        Write-Host "WARN  可选依赖缺失: $($missOpt -join ', ')（只影响策略模拟/AI 编排, 看板与雷达不受影响）" -ForegroundColor Yellow
    }
}

# ---------------------------------------------------------------- 2. 配置
Write-Section "2/4 配置 (.env)"

$EnvFile = Join-Path $Root ".env"
$EnvExample = Join-Path $Root ".env.example"
if (-not (Test-Path $EnvFile)) {
    Write-Host "MISS  .env 不存在"
    if ($Install -and (Test-Path $EnvExample)) {
        Copy-Item $EnvExample $EnvFile
        Write-Host "OK    已从 .env.example 复制, 请补齐 TUSHARE_TOKEN / BIGQMT_* 后再启动" -ForegroundColor Green
    }
    Add-Issue ".env 不存在" "执行 .\init.ps1 -Install 生成模板, 然后填写 TUSHARE_TOKEN"
} else {
    Write-Host "OK    .env 存在" -ForegroundColor Green
    $envText = Get-Content $EnvFile -Raw

    if ($envText -notmatch "(?m)^\s*TUSHARE_TOKEN\s*=\s*\S+" -or $envText -match "your_token_here") {
        Add-Issue "TUSHARE_TOKEN 未填写" "Windows 没有 ~/.zshrc 兜底, 必须写进 .env; 否则所有 collect/build 脚本取不到数据"
    } else {
        Write-Host "      TUSHARE_TOKEN 已填"
    }

    # QMT 源地址: PC 是本机, 应为回环; 若残留 Mac 侧的局域网 IP 会连不上
    foreach ($key in @("BIGQMT_REDIS_HOST", "BIGQMT_FORMULA_HOST")) {
        $m = [regex]::Match($envText, "(?m)^\s*$key\s*=\s*(\S+)")
        if (-not $m.Success) {
            Add-Issue "$key 未配置" "PC 是本机 QMT, 填 127.0.0.1"
        } else {
            $val = $m.Groups[1].Value.Trim('"').Trim("'")
            Write-Host "      $key = $val"
            if ($val -ne "127.0.0.1" -and $val -ne "localhost") {
                Add-Issue "$key = $val（非回环）" "PC 上 QMT 在本机, 应改成 127.0.0.1; 沿用局域网 IP 说明 .env 是从 Mac 直接拷来的"
            }
        }
    }

    $quote = [regex]::Match($envText, "(?m)^\s*QUOTE_SOURCE\s*=\s*(\S+)")
    if ($quote.Success) { Write-Host "      QUOTE_SOURCE = $($quote.Groups[1].Value)" }
}

# ---------------------------------------------------------------- 3. 数据
Write-Section "3/4 数据 (data/)"

if (-not (Test-Path (Join-Path $Root "data"))) {
    Write-Host "MISS  data/ 目录不存在" -ForegroundColor Yellow
    Add-Issue "data/ 缺失" "从 Mac 侧 tar 拷贝(见对话中的排除清单), 或跑 collect/ 脚本用 tushare 重建"
} elseif (Test-Path $Py) {
    $pyCode = @'
import os, sys
sys.path.insert(0, os.getcwd())
try:
    from datastore import path_of
    import pyarrow.parquet as pq
except Exception as e:
    print("SKIP  无法加载 datastore/pyarrow: %s" % e); sys.exit(0)
# 第三项是该表的日期列名(各域不统一: trade_cal 用 cal_date,
# kpl 静态快照用 snapshot_date, 其余为 trade_date)
keys = [
    ("meta.trade_cal", "交易日历", "cal_date"),
    ("limitup.events", "涨停事件", "trade_date"),
    ("limitup.events_enriched", "事件富化", "trade_date"),
    ("limitup.ths_limit", "同花顺榜单", "trade_date"),
    ("limitup.kpl_events", "开盘啦事件", "trade_date"),
    ("market.daily_panel", "全A日线面板", "trade_date"),
    ("market.index_panel", "指数日线", "trade_date"),
    ("theme.attribution", "题材归属", "trade_date"),
    ("theme.day", "题材日度快照", "trade_date"),
    ("theme.kpl_concepts", "kpl题材列表", "snapshot_date"),
    ("factor.longtou", "龙头因子", "trade_date"),
    ("hmlist.detail", "龙虎榜明细", "trade_date"),
]
missing = []
for k, label, col in keys:
    try:
        p = path_of(k)
    except Exception as e:
        print("ERR   %-14s 路径解析失败 %s" % (label, e)); continue
    if not p.exists():
        print("MISS  %-14s %s" % (label, p))
        missing.append(label); continue
    try:
        t = pq.read_table(p, columns=[col])
        s = t.column(col).to_pandas().astype(str)
        print("OK    %-14s %s ~ %s  (%d 行)" % (label, s.min(), s.max(), len(s)))
    except Exception as e:
        print("WARN  %-14s 存在但读取失败: %s" % (label, e))
print("__MISSING__ %d" % len(missing))
'@
    $out = & $Py -c $pyCode 2>&1
    $out | Where-Object { $_ -notmatch "^__MISSING__" } | ForEach-Object {
        $line = $_
        if ($line -like "OK*") { Write-Host "      $line" -ForegroundColor Green }
        elseif ($line -like "MISS*") { Write-Host "      $line" -ForegroundColor Yellow }
        else { Write-Host "      $line" }
    }
    $missLine = $out | Where-Object { $_ -match "^__MISSING__" }
    if ($missLine -and [int]($missLine -replace "__MISSING__ ", "") -gt 0) {
        Add-Issue "data/ 有 $($missLine -replace '__MISSING__ ', '') 个关键数据集缺失" "从 Mac 侧补齐(排除清单: *.bak / radar_log_*.jsonl)"
    }
}

# 策略净值(看板策略页要用; 缺了不影响雷达)
$runsDir = Join-Path $Root "data\sim\runs"
if (Test-Path $runsDir) {
    $n = (Get-ChildItem $runsDir -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "*__main" }).Count
    Write-Host "OK    策略 run 目录 $n 个" -ForegroundColor Green
} else {
    Write-Host "WARN  data\sim\runs 缺失（看板策略页会空）" -ForegroundColor Yellow
}

# ---------------------------------------------------------------- 4. QMT
Write-Section "4/4 QMT 本机连通性"

$redisHost = "127.0.0.1"; $redisPort = 6379; $formulaHost = "127.0.0.1"; $formulaPort = 58600
if (Test-Path $EnvFile) {
    $envText2 = Get-Content $EnvFile -Raw
    foreach ($pair in @(@("BIGQMT_REDIS_HOST", "redisHost"), @("BIGQMT_REDIS_PORT", "redisPort"),
            @("BIGQMT_FORMULA_HOST", "formulaHost"))) {
        $m = [regex]::Match($envText2, "(?m)^\s*$($pair[0])\s*=\s*(\S+)")
        if ($m.Success) { Set-Variable -Name $pair[1] -Value $m.Groups[1].Value.Trim('"').Trim("'") }
    }
}

if (Test-Port $redisHost ([int]$redisPort)) {
    Write-Host "OK    Redis $redisHost`:$redisPort 可达" -ForegroundColor Green
} else {
    Write-Host "MISS  Redis $redisHost`:$redisPort 不可达" -ForegroundColor Yellow
    Add-Issue "Redis 端口不通" "确认 QMT 客户端已启动、xtquant_big_convert 桥接服务在跑; 否则雷达拿不到实时行情(可临时 QUOTE_SOURCE=tx 降级)"
}
if (Test-Port $formulaHost $formulaPort) {
    Write-Host "OK    FormulaServer $formulaHost`:$formulaPort 可达" -ForegroundColor Green
} else {
    Write-Host "WARN  FormulaServer $formulaHost`:$formulaPort 不可达（盘前快路径降级, 盘中推送不受影响）" -ForegroundColor Yellow
}

# ---------------------------------------------------------------- 汇总
Write-Section "汇总"
if ($script:Issues.Count -eq 0) {
    Write-Host "全部检查通过" -ForegroundColor Green
} else {
    Write-Host "发现 $($script:Issues.Count) 个待处理项:" -ForegroundColor Yellow
    $i = 1
    foreach ($it in $script:Issues) {
        Write-Host "  $i) $($it.Text)"
        Write-Host "     → $($it.Hint)"
        $i++
    }
}

Write-Host ""
Write-Host "下一步: .\start.ps1        # 拉起 poller/看板(8766)/雷达/aifeed/newsfeed"
Write-Host "         .\start.ps1 -WithSim  # 如需策略模拟(rqalpha)"

if ($Start -and $script:Issues.Count -eq 0) {
    Write-Host ""
    & (Join-Path $Root "start.ps1")
}