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
PORT=${1:-8765}
mkdir -p logs
echo "解释器: $PY ($($PY -V 2>&1))"

if [ -f logs/poller.pid ] && kill -0 "$(cat logs/poller.pid)" 2>/dev/null; then
  echo "轮询引擎已在运行 PID $(cat logs/poller.pid)"
else
  nohup "$PY" -u apps/poller.py >> logs/poller.log 2>&1 &
  echo $! > logs/poller.pid
  echo "轮询引擎已启动 PID $! (日志 logs/poller.log)"
fi

if [ -f logs/server.pid ] && kill -0 "$(cat logs/server.pid)" 2>/dev/null; then
  echo "看板服务已在运行 PID $(cat logs/server.pid)"
else
  nohup "$PY" -u apps/server.py "$PORT" >> logs/server.log 2>&1 &
  echo $! > logs/server.pid
  echo "看板服务已启动 PID $! (日志 logs/server.log)"
fi

if [ -f logs/radar.pid ] && kill -0 "$(cat logs/radar.pid)" 2>/dev/null; then
  echo "预警雷达已在运行 PID $(cat logs/radar.pid)"
else
  nohup "$PY" -u apps/radar.py >> logs/radar.log 2>&1 &
  echo $! > logs/radar.pid
  echo "预警雷达已启动 PID $! (日志 logs/radar.log)"
fi

# 策略模拟(rqalpha): 按 strategies/strategies.yaml 启用清单拉起 N 个策略进程。
# 仅在 apps/sim.py 存在且 rqalpha 可导入时启动, 否则跳过(不阻塞主链路)。
if [ -f apps/sim.py ] && "$PY" -c "import rqalpha" 2>/dev/null; then
  if [ -f logs/sim.pid ] && kill -0 "$(cat logs/sim.pid)" 2>/dev/null; then
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
