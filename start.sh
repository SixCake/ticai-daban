#!/bin/bash
# 题材打板监控平台 一键启动
# 用法: bash start.sh [port]
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  # 优先用项目 venv: rqalpha 要求 pandas<3.0, 而 Homebrew 系统 Python 是
  # externally-managed(PEP 668) 且装的是 pandas 3.0.x, 无法降级。
  # 全部服务统一跑 .venv, 单一环境(见 docs/adr/0002)。
  if [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
  elif [ -x /opt/homebrew/bin/python3.12 ]; then
    PY=/opt/homebrew/bin/python3.12
    echo "警告: 未找到 .venv, 回退系统 Python; rqalpha 策略模拟不可用"
    echo "      请执行: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  else
    PY=python3
  fi
fi
PORT=${1:-8766}
mkdir -p logs
echo "解释器: $PY ($($PY -V 2>&1))"

# 判活: PID 存在 且 命令行匹配预期脚本。
# 为何不能只用 kill -0: 进程退出后 OS 会把该 PID 复用给别的进程,
# kill -0 仍成功 → 误判"已在运行"而跳过拉起(实测踩坑: sim.pid 残留
# 导致某日 live 模拟根本没启动, 看板数据停在上一交易日)。
is_running() {
  local pidf="$1" pat="$2" pid
  [ -f "$pidf" ] || return 1
  pid=$(cat "$pidf" 2>/dev/null)
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || return 1
  ps -p "$pid" -o args= 2>/dev/null | grep -q "$pat"
}

if is_running logs/poller.pid "apps/poller.py"; then
  echo "轮询引擎已在运行 PID $(cat logs/poller.pid)"
else
  nohup "$PY" -u apps/poller.py >> logs/poller.log 2>&1 &
  echo $! > logs/poller.pid
  echo "轮询引擎已启动 PID $! (日志 logs/poller.log)"
fi

if is_running logs/server.pid "apps/server.py"; then
  echo "看板服务已在运行 PID $(cat logs/server.pid)"
else
  nohup "$PY" -u apps/server.py "$PORT" >> logs/server.log 2>&1 &
  echo $! > logs/server.pid
  echo "看板服务已启动 PID $! (日志 logs/server.log)"
fi

if is_running logs/radar.pid "apps/radar.py"; then
  echo "预警雷达已在运行 PID $(cat logs/radar.pid)"
else
  nohup "$PY" -u apps/radar.py >> logs/radar.log 2>&1 &
  echo $! > logs/radar.pid
  echo "预警雷达已启动 PID $! (日志 logs/radar.log)"
fi

# AI Feed 生产者(ADR-0003): 盘中持续产出 feed 文件供策略 ai_feed() 订阅。
# 追加式产出 + 时间戳闸门, 回测读同一批文件 → 确定性可重放。
if [ -f apps/ai_feed.py ]; then
  if is_running logs/aifeed.pid "apps/ai_feed.py"; then
    echo "AI Feed 生产者已在运行 PID $(cat logs/aifeed.pid)"
  else
    nohup "$PY" -u apps/ai_feed.py --loop >> logs/aifeed.log 2>&1 &
    echo $! > logs/aifeed.pid
    echo "AI Feed 生产者已启动 PID $! (日志 logs/aifeed.log)"
  fi
fi

# 新闻数据源采集器: 轮询订阅的新闻/RSS流落盘, 供 LLM 线索生产者分析(输入层)。
if [ -f collect/fetch_news.py ]; then
  if is_running logs/newsfeed.pid "collect/fetch_news.py"; then
    echo "新闻采集器已在运行 PID $(cat logs/newsfeed.pid)"
  else
    nohup "$PY" -u collect/fetch_news.py --loop >> logs/newsfeed.log 2>&1 &
    echo $! > logs/newsfeed.pid
    echo "新闻采集器已启动 PID $! (日志 logs/newsfeed.log)"
  fi
fi

# 策略模拟(rqalpha): 按 strategies/strategies.yaml 启用清单拉起 N 个策略进程。
# 仅在 apps/sim.py 存在且 rqalpha 可导入时启动, 否则跳过(不阻塞主链路)。
if [ -f apps/sim.py ] && "$PY" -c "import rqalpha" 2>/dev/null; then
  if is_running logs/sim.pid "apps/sim.py"; then
    echo "策略模拟已在运行 PID $(cat logs/sim.pid)"
  else
    nohup "$PY" -u apps/sim.py >> logs/sim.log 2>&1 &
    echo $! > logs/sim.pid
    echo "策略模拟已启动 PID $! (日志 logs/sim.log)"
  fi
else
  echo "策略模拟跳过(apps/sim.py 缺失或 rqalpha 未装)"
fi

echo "打开 http://localhost:$PORT"
