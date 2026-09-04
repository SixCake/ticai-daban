#!/bin/bash
# 收盘后更新事件库与题材归属（增量），供次日轮询与复盘使用
# 用法: bash daily_update.sh
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  # 优先用项目 venv(与 start.sh 一致): rqalpha 要求 pandas<3.0,
  # 而 Homebrew 系统 Python 是 PEP 668 externally-managed 且装的是 pandas 3.0.x
  if [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
  elif [ -x /opt/homebrew/bin/python3.12 ]; then
    PY=/opt/homebrew/bin/python3.12
  else
    PY=python3
  fi
fi
echo "解释器: $PY ($($PY -V 2>&1))"
echo "==> 增量拉取涨停事件"
"$PY" collect/fetch_limit_events.py
echo "==> 增量拉取开盘啦事件(kpl题材标注, T+1)"
"$PY" collect/fetch_kpl_events.py || echo "kpl事件拉取失败(归属降级延续法)"
echo "==> 开盘啦题材板块+成分快照(kpl宇宙)"
"$PY" collect/fetch_kpl_concepts.py || echo "kpl成分拉取失败(雷达宇宙沿用旧快照)"
echo "==> 增量拉取同花顺涨停池榜单(涨停原因/板型/封板状态, 当日16点后可得)"
"$PY" collect/fetch_ths_limit.py || echo "同花顺榜单拉取失败(复盘涨停原因降级为开盘啦兜底)"
echo "==> 全A日线面板补尾(tushare, 先于富化让昨日事件可算T+1)"
"$PY" collect/fetch_daily_panel.py || echo "面板补尾失败(富化/因子沿用旧面板)"
echo "==> 指数日线补尾(供策略模拟的宽基基准; 失败不影响主流程)"
"$PY" collect/fetch_index_panel.py || echo "指数补尾失败(策略模拟只能用自建打板基准 DBBNCH)"
echo "==> 竞价数据T+1官方校正(stk_auction, 只补official对照不覆盖盘中值)"
"$PY" collect/fetch_auction.py --days 3 || echo "竞价校正失败(竞价影子字段无官方对照)"
echo "==> 重建事件富化(一字板/T+1收益)"
"$PY" build/enrich_events.py
echo "==> 重建题材归属"
"$PY" build/attribute.py
echo "==> 重建题材日度快照"
"$PY" build/theme_daily.py
echo "==> 申万行业分类映射(一级/二级, 供雷达分级聚合)"
"$PY" collect/fetch_sw.py || echo "申万映射失败(雷达申万聚合沿用旧sw_map)"
echo "==> 龙头因子日表构建(研究22/23: qscore/sscore/环境)"
"$PY" collect/factor_longtou.py || echo "龙头因子构建失败(看板因子列降级为空)"
echo "==> 雷达轨迹标注(挂涨停结果/首封时间)"
"$PY" apps/label_radar.py
echo "==> 复盘快照生成(最新交易日)"
"$PY" apps/review.py || echo "复盘快照生成失败(服务端可现场构建)"
echo "==> 涨停/触板标的1分钟线采集(东财当日深度, 供研究06)"
"$PY" collect/fetch_zt_minute.py --max-days 2 || echo "分钟线采集失败(不影响主流程)"
echo "==> 完成"
