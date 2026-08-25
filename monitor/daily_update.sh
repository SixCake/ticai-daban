#!/bin/bash
# 收盘后更新事件库与题材归属（增量），供次日轮询与复盘使用
# 用法: bash monitor/daily_update.sh
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  if [ -x /opt/homebrew/bin/python3.12 ]; then PY=/opt/homebrew/bin/python3.12; else PY=python3; fi
fi
echo "==> 增量拉取涨停事件"
"$PY" collect/fetch_limit_events.py
echo "==> 重建事件富化(一字板/T+1收益)"
"$PY" build/enrich_events.py
echo "==> 重建题材归属"
"$PY" build/attribute.py
echo "==> 重建题材日度快照"
"$PY" build/theme_daily.py
echo "==> 雷达轨迹标注(挂涨停结果/首封时间)"
"$PY" monitor/label_radar.py
echo "==> 完成"
