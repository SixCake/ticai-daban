#!/bin/bash
# 停止监控平台（pid文件 + 进程名兜底，确保无残留）
cd "$(dirname "$0")"
for f in logs/poller.pid logs/server.pid logs/radar.pid logs/sim.pid logs/aifeed.pid; do
  if [ -f "$f" ]; then
    pid=$(cat "$f")
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" && echo "已停止 $f (PID $pid)"
    fi
    rm -f "$f"
  fi
done

# 兜底: 按进程名杀残留(旧进程在pid文件被新进程覆盖后杀不掉, 20260828事故根因)
# sim.py 会派生 rqalpha 策略子进程, 一并纳入匹配范围
pkill -f "apps/(radar|poller|server|sim)\.py" 2>/dev/null

# 校验: 等最多5秒确认无残留, 否则强杀
for i in 1 2 3 4 5; do
  left=$(pgrep -f "apps/(radar|poller|server|sim)\.py" | wc -l | tr -d ' ')
  if [ "$left" = "0" ]; then
    echo "校验通过: 无残留进程"
    exit 0
  fi
  sleep 1
done
pkill -9 -f "apps/(radar|poller|server|sim)\.py" 2>/dev/null
echo "警告: 有进程拒绝退出, 已强制杀死"
