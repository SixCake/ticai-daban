#!/bin/bash
# 停止监控平台
cd "$(dirname "$0")"
for f in logs/poller.pid logs/server.pid logs/radar.pid; do
  if [ -f "$f" ]; then
    pid=$(cat "$f")
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" && echo "已停止 $f (PID $pid)"
    fi
    rm -f "$f"
  fi
done
