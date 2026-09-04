# -*- coding: utf-8 -*-
"""AI Feed 生产者 — 独立定时任务, 产出策略可订阅的 feed 文件

设计依据(ADR-0003): AI 不进策略代码。策略只读 feed 文件, 生产者随时产出。
这样做的理由是 LLM 输出不可重放 —— 若策略盘中直接调 LLM, 回测时 AI 看到的
是"当天真实新闻", 属于未来信息, 会系统性高估胜率。落成文件后回测读同一批
文件 → 确定性可重放。

产出位置(见 rqalpha_mod_ticai/feeds.py):
  共享 feed   data/sim/ai_feeds/{feed_name}/{date}.json
  私有 feed   data/sim/ai_feeds/private/{strategy}/{feed_name}/{date}.json

条目结构:
  {"ts": epoch, "t": "HH:MM:SS", "topic": str, "score": float|None,
   "text": str, "src": str, "extra": {...}}
  ts 是时间戳闸门的依据: 策略只能看到 ts <= 当前模拟时刻的条目。

内置生产者 theme_narrative(题材叙事强度):
  当前实现是【规则基线】, 不是 LLM —— 用项目已有的题材热度/连板高度/
  涨停家数算叙事强度, 作为 LLM 接入前的可用占位与对照基准。接入真实
  LLM 时只需替换 _theme_narrative_entries 的实现(或在 PRODUCERS 里新增
  一个生产者), feed 契约与下游策略都不用改。

用法:
  python apps/ai_feed.py                          # 产出当日全部 feed
  python apps/ai_feed.py --date 20260903          # 指定日期
  python apps/ai_feed.py --feed theme_narrative   # 只产出一个 feed
  python apps/ai_feed.py --loop                   # 盘中持续产出(默认 300s)
  python apps/ai_feed.py --list                   # 列出可用生产者
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA  # noqa: E402

from rqalpha_mod_ticai import feeds  # noqa: E402

LIVE = DATA / "live"
LOOP_INTERVAL = 300          # 盘中持续产出间隔(秒)
DEFAULT_PREAMARKET = "09:00:00"   # 非当日补产时的默认产出时刻


def _ts_for(day: str, at: str | None = None) -> float:
    """条目时间戳 —— 时间戳闸门的依据, 必须与模拟时刻同口径。

    当日产出用墙钟; 补产历史日则必须用【那一天的时刻】—— 否则
    回放该日时 ts 全是未来的墙钟, 闸门会把条目全部滤掉(实测踩坑)。
    at 可显式指定 HH:MM:SS(默认盘前 09:00)。"""
    if at:
        return datetime.strptime(f"{day} {at}", "%Y%m%d %H:%M:%S").timestamp()
    if day == datetime.now().strftime("%Y%m%d"):
        return time.time()
    return datetime.strptime(f"{day} {DEFAULT_PREAMARKET}",
                             "%Y%m%d %H:%M:%S").timestamp()


# ---------- 生产者 ----------

def _theme_narrative_entries(day: str, ts: float | None = None) -> list:
    """题材叙事强度 — 规则基线实现(非 LLM)。

    口径: 用雷达 radar.json 的题材热度快照算叙事强度
      score = 归一化热度 × 涨停家数加成 × 头部超额加成
    接入真实 LLM 时替换本函数即可(读新闻/公告 → 输出同结构条目),
    feed 契约与下游策略不需要改。

    数据缺失时返回空列表(不伪造分数)。
    """
    f = LIVE / "radar.json"
    if not f.exists():
        return []
    import json
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return []
    themes = d.get("themes") or []
    if not themes:
        return []
    heats = [float(t.get("heat") or 0) for t in themes]
    hmax = max(heats) if heats else 0.0
    if hmax <= 0:
        return []
    out = []
    now = ts if ts is not None else time.time()
    for t in themes[:30]:
        heat = float(t.get("heat") or 0)
        zt = int(t.get("zt") or 0)
        headx = float(t.get("headx") or 0)
        # 归一热度(0~1) + 涨停家数加成 + 头部超额加成
        score = round(min(1.0, heat / hmax
                          + min(zt, 10) / 40.0
                          + min(max(headx, 0), 5) / 20.0), 4)
        out.append({
            "ts": now,
            "topic": t.get("name"),
            "score": score,
            "text": (f"{t.get('name')} 热度{heat:.1f} 涨停{zt}家 "
                     f"头部超额{headx:+.2f} 成分{t.get('nmem')}只"),
            "src": "rule_baseline",      # 接入 LLM 后改为 llm 标识
            "extra": {"heat": heat, "zt": zt, "headx": headx,
                      "nmem": t.get("nmem"),
                      "concept_code": t.get("concept_code")},
        })
    out.sort(key=lambda e: -(e["score"] or 0))
    return out


def _market_risk_entries(day: str, ts: float | None = None) -> list:
    """市场风险标注 — 规则基线(炸板率/最高连板/涨停家数)。"""
    f = LIVE / "latest.json"
    if not f.exists():
        return []
    import json
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return []
    sent = d.get("sentiment") or {}
    if not sent:
        return []
    broken = float(sent.get("broken_rate") or 0)
    zt = int(sent.get("zt_count") or 0)
    height = int(sent.get("max_height") or 0)
    # 炸板率高分歧 → 风险高; 涨停家数少 → 情绪弱
    risk = round(min(1.0, broken * 1.2 + max(0, (20 - zt)) / 40.0), 4)
    return [{
        "ts": ts if ts is not None else time.time(),
        "topic": "market",
        "score": risk,
        "text": (f"炸板率{broken:.1%} 涨停{zt}家 最高{height}板 "
                 f"→ 分歧度{risk:.2f}"),
        "src": "rule_baseline",
        "extra": {"broken_rate": broken, "zt_count": zt,
                  "max_height": height,
                  "divg": sent.get("divg"), "stage": sent.get("stage")},
    }]


PRODUCERS = {
    "theme_narrative": (_theme_narrative_entries,
                        "题材叙事强度(规则基线; 接入 LLM 时替换实现)"),
    "market_risk": (_market_risk_entries,
                    "市场风险标注(炸板率/涨停家数/连板高度)"),
}


# ---------- 产出 ----------

def produce(name: str, day: str, strategy: str | None = None,
            append: bool = True, at: str | None = None) -> int:
    """跑一个生产者并落盘; 返回条目数"""
    if name not in PRODUCERS:
        print(f"[ai_feed] 未知生产者 {name}, 可选: {sorted(PRODUCERS)}")
        return 0
    fn, _ = PRODUCERS[name]
    ts = _ts_for(day, at)
    try:
        entries = fn(day, ts)
    except Exception as e:
        print(f"[ai_feed] {name} 产出失败: {type(e).__name__}: {e}")
        return 0
    if not entries:
        print(f"[ai_feed] {name} {day}: 无数据(不伪造条目)")
        return 0
    # 盘中持续产出用 append(保留历史条目, 时间戳闸门才有意义);
    # 单次产出用 write(覆盖)
    if append:
        p = feeds.append_feed(name, day, entries, strategy=strategy)
    else:
        p = feeds.write_feed(name, day, entries, strategy=strategy)
    tag = f"private/{strategy}" if strategy else "shared"
    print(f"[ai_feed] {name}({tag}) {day}: {len(entries)} 条 → {p}")
    return len(entries)


def produce_all(day: str, names: list | None = None,
                append: bool = True, at: str | None = None) -> int:
    total = 0
    for name in (names or sorted(PRODUCERS)):
        total += produce(name, day, append=append, at=at)
    return total


def cli() -> int:
    ap = argparse.ArgumentParser(description="AI Feed 生产者")
    ap.add_argument("--date", help="目标日期 YYYYMMDD(默认今天)")
    ap.add_argument("--feed", help="只产出指定 feed")
    ap.add_argument("--strategy", help="写入某策略的私有 feed 目录")
    ap.add_argument("--at", help="产出时刻 HH:MM:SS(补产历史日时用; "
                                "默认当日=墙钟, 历史日=09:00)")
    ap.add_argument("--once", action="store_true",
                    help="覆盖式产出(默认盘中用追加式)")
    ap.add_argument("--loop", action="store_true",
                    help="盘中持续产出(每 LOOP_INTERVAL 秒)")
    ap.add_argument("--interval", type=int, default=LOOP_INTERVAL)
    ap.add_argument("--list", action="store_true", help="列出可用生产者")
    args = ap.parse_args()

    if args.list:
        for name, (_, desc) in sorted(PRODUCERS.items()):
            print(f"{name:<20} {desc}")
        return 0

    day = args.date or datetime.now().strftime("%Y%m%d")
    names = [args.feed] if args.feed else None

    if not args.loop:
        n = produce_all(day, names, append=not args.once, at=args.at)
        print(f"[ai_feed] 完成: 共 {n} 条")
        return 0

    print(f"[ai_feed] 持续产出模式, 间隔 {args.interval}s, Ctrl-C 退出")
    while True:
        produce_all(day, names, append=True, at=args.at)
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(cli())
