# -*- coding: utf-8 -*-
"""收盘后数据完整性自检: 把「静默缺口」变成显式报错。

为什么需要: daily_update.sh 各步骤都用 `|| echo 降级` 容错, 任一步失败主流程
照常退出 0; 且 sim 的就绪闸在雷达未及时落盘盘中快照时会等待 30 分钟然后静默
放弃拉起 —— 策略当日净值凭空缺失却无人察觉(实测 20260915~0917 漏跑,
v5_daban 净值停在 0914, 直到 0918 被「继承 0911 持仓」的错误链掩盖)。

期望覆盖日:
  当日类(涨停事件/富化/同花顺榜单/日线面板/题材归属/题材日快照/席位明细/
  盘中快照/策略净值) = 最近交易日(≤今日)
  开盘啦事件 = 前一交易日 —— kpl_list 是 T+1 数据, 当日标注次日才可拉

用法: python build/check_daily.py
退出码: 0=全部齐备; 1=存在缺口(缺口清单已打印)
"""
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from datastore import path_of  # noqa: E402


def max_date(p: Path, col: str = "trade_date") -> str | None:
    """只读一列取最大日期(归一成 YYYYMMDD)。

    为何不用 load(): longtou 等表 300MB+, 自检只需日期列, 整表载入纯浪费。
    date 列在不同数据集里有时是 20260918 有时是 2026-09-18, 统一去掉分隔符。
    """
    if not p.exists():
        return None
    for colname in (col, "date"):
        try:
            import pyarrow.parquet as pq
            t = pq.read_table(p, columns=[colname])
            s = (t.column(colname).to_pandas().astype(str)
                 .str.replace("-", "", regex=False).str[:8])
            if len(s):
                return str(s.max())
        except Exception:
            continue
    return None


def recent_trade_days(n: int = 2) -> list[str]:
    """最近 n 个已过去的交易日(升序)。交易日历含未来日期, 必须按今日截断。"""
    today = datetime.now().strftime("%Y%m%d")
    cal = pd.read_parquet(path_of("meta.trade_cal"))
    d = cal[cal["is_open"] == 1]["cal_date"].astype(str)
    return d[d <= today].sort_values().tolist()[-n:]


def enabled_strategies() -> list[str]:
    """strategies.yaml 里 enabled=true 的策略名"""
    import yaml
    f = ROOT / "strategies/strategies.yaml"
    if not f.exists():
        return []
    reg = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    return [r["name"] for r in (reg.get("strategies") or []) if r.get("enabled")]


def main() -> int:
    days = recent_trade_days(2)
    if not days:
        print("[check] 交易日历为空, 无法自检")
        return 1
    cur = days[-1]                        # 最近交易日
    prev = days[0] if len(days) > 1 else cur
    if cur != datetime.now().strftime("%Y%m%d"):
        print(f"[check] 今日非交易日, 按最近交易日 {cur} 校验")

    # (名称, 数据集键, 期望覆盖日)
    specs = [
        ("涨停事件", "limitup.events", cur),
        ("事件富化", "limitup.events_enriched", cur),
        ("同花顺榜单", "limitup.ths_limit", cur),
        ("全A日线面板", "market.daily_panel", cur),
        ("题材归属", "theme.attribution", cur),
        ("题材日快照", "theme.day", cur),
        ("席位明细", "hmlist.detail", cur),
        ("开盘啦事件(T+1)", "limitup.kpl_events", prev),
    ]
    rows = [(label, max_date(path_of(key)), want)
            for label, key, want in specs]

    # 盘中快照: sim live 回放/继承所依赖
    ipx = ROOT / f"data/live/intraday_px_{cur}.json"
    rows.append(("盘中快照", cur if ipx.exists() else None, cur))

    # 策略净值: 每个启用的策略都应有目标交易日的结算行
    for name in enabled_strategies():
        p = ROOT / f"data/sim/runs/{name}__main/equity.parquet"
        rows.append((f"策略净值·{name}", max_date(p), cur))

    print(f"[check] 目标交易日 {cur}(开盘啦按 T+1 期望 {prev})")
    misses = []
    for label, got, want in rows:
        ok = got is not None and got >= want
        mark = "OK  " if ok else "MISS"
        detail = f"{got}" if got else "无数据"
        extra = "" if ok else f"  < 期望 {want}"
        print(f"  {mark} {label:<22} {detail}{extra}")
        if not ok:
            misses.append(label)

    if misses:
        print(f"[check] 发现 {len(misses)} 项缺口: {'、'.join(misses)}")
        print("[check] 请检查对应采集/构建步骤(见 logs/daily_update_launchd.log)")
        return 1
    print(f"[check] 全部 {len(rows)} 项齐备")
    return 0


if __name__ == "__main__":
    sys.exit(main())