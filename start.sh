#!/bin/bash
# 题材打板监控平台 一键启动
# 用法: bash start.sh [port]
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  if [ -x /opt/homebrew/bin/python3.12 ]; then PY=/opt/homebrew/bin/python3.12; else PY=python3; fi
fi
PORT=${1:-8765}
mkdir -p logs

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

echo "打开 http://localhost:$PORT"
