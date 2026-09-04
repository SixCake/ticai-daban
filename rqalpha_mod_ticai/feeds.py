# -*- coding: utf-8 -*-
"""AI Feed 层 — 发布/订阅存储 + 时间戳闸门(领域逻辑唯一出处)

设计依据(ADR-0003): AI 不进策略代码。LLM 输出不可重放, 若策略盘中直接
调 LLM, 回测时 AI 看到的是"当天真实新闻" → 未来信息污染, 胜率被高估。
故 AI 产物一律落盘为 feed, 策略只读文件; 回测读同一批文件 → 确定性可重放。

订阅模型:
  共享 feed   data/sim/ai_feeds/{feed_name}/{date}.json
  私有 feed   data/sim/ai_feeds/private/{strategy}/{date}.json
私有 feed 不与其它策略共享(用户诉求: 策略可为自己定制 AI 信息源)。
"哪些 feed 可被某策略读取"的鉴权在 api.py 做(按 config.yaml 声明),
本模块只负责存储与时间戳闸门, 不做鉴权。

时间戳闸门(防未来信息):
  每条 entry 带 ts(epoch 秒) + t(HH:MM:SS 便于人读);
  read_feed(cutoff=...) 只返回 ts <= cutoff 的条目。
  盘中模拟 cutoff=当前墙钟; 回放 cutoff=回放时刻的 epoch。
  cutoff=None 表示不加闸门(仅供离线分析/生产者自检, 策略路径禁用)。

条目结构(生产者约定, 消费者只读不改):
  {"ts": epoch, "t": "HH:MM:SS", "topic": str, "score": float|None,
   "text": str, "src": str, "extra": {...}}
"""
import json
import time
from datetime import datetime
from pathlib import Path

from config import DATA

FEED_ROOT = DATA / "sim" / "ai_feeds"
PRIVATE_DIR = FEED_ROOT / "private"
MAX_ENTRY_TEXT = 4000          # 单条文本上限(防单文件膨胀)


def _ensure(dirpath: Path) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath


def feed_path(name: str, date: str, strategy: str | None = None) -> Path:
    """feed 文件路径; strategy 非空则为该策略私有 feed"""
    if strategy:
        return PRIVATE_DIR / strategy / name / f"{date}.json"
    return FEED_ROOT / name / f"{date}.json"


def list_feeds(date: str, strategy: str | None = None) -> list:
    """当日存在的 feed 名(升序); 只扫目录, 不解析内容"""
    root = (PRIVATE_DIR / strategy) if strategy else FEED_ROOT
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name == "private":
            continue
        if (d / f"{date}.json").exists():
            out.append(d.name)
    return out


def list_private_strategies() -> list:
    """有私有 feed 目录的策略名"""
    if not PRIVATE_DIR.exists():
        return []
    return sorted(d.name for d in PRIVATE_DIR.iterdir() if d.is_dir())


def write_feed(name: str, date: str, entries: list,
               strategy: str | None = None) -> Path:
    """生产者写出 feed(覆盖式: 同一 feed 同一天以最新产出为准)。
    entries 缺 ts/t 时自动补齐(以写入时刻为准)。"""
    p = feed_path(name, date, strategy)
    _ensure(p.parent)
    now = time.time()
    clean = []
    for e in entries:
        d = dict(e)
        if "ts" not in d:
            d["ts"] = now
        if "t" not in d:
            d["t"] = datetime.fromtimestamp(d["ts"]).strftime("%H:%M:%S")
        if isinstance(d.get("text"), str) and len(d["text"]) > MAX_ENTRY_TEXT:
            d["text"] = d["text"][:MAX_ENTRY_TEXT]
        clean.append(d)
    clean.sort(key=lambda d: d["ts"])
    p.write_text(json.dumps({"date": date, "name": name,
                             "strategy": strategy, "entries": clean},
                            ensure_ascii=False), encoding="utf-8")
    return p


def append_feed(name: str, date: str, entries: list,
                strategy: str | None = None) -> Path:
    """增量追加(AI 随时产出场景): 已有条目保留, 新条目按 ts 归位去重。
    去重键 = (ts 取整到秒, topic)。"""
    p = feed_path(name, date, strategy)
    old = []
    if p.exists():
        try:
            old = json.loads(p.read_text(encoding="utf-8")).get("entries", [])
        except Exception:
            old = []
    seen = {(int(e["ts"]), e.get("topic")) for e in old if "ts" in e}
    now = time.time()
    add = []
    for e in entries:
        d = dict(e)
        if "ts" not in d:
            d["ts"] = now
        if "t" not in d:
            d["t"] = datetime.fromtimestamp(d["ts"]).strftime("%H:%M:%S")
        k = (int(d["ts"]), d.get("topic"))
        if k in seen:
            continue
        seen.add(k)
        add.append(d)
    merged = sorted(old + add, key=lambda d: d.get("ts", 0))
    _ensure(p.parent)
    p.write_text(json.dumps({"date": date, "name": name,
                             "strategy": strategy, "entries": merged},
                            ensure_ascii=False), encoding="utf-8")
    return p


def read_feed(name: str, date: str, cutoff: float | None = None,
              strategy: str | None = None) -> list:
    """读 feed 条目(按时间戳闸门过滤, 时间升序)。
    cutoff 为 epoch 秒: 只返回 ts <= cutoff 的条目。
    文件缺失/损坏返回空列表(不抛异常 — 策略不应因 AI 缺产出而崩)。"""
    p = feed_path(name, date, strategy)
    if not p.exists():
        return []
    try:
        entries = json.loads(p.read_text(encoding="utf-8")).get("entries", [])
    except Exception:
        return []
    if cutoff is None:
        return entries
    return [e for e in entries if e.get("ts", 0) <= cutoff]


def latest_entry(name: str, date: str, cutoff: float | None = None,
                 strategy: str | None = None, topic: str | None = None):
    """取闸门内最后一条(可按 topic 过滤); 无则 None"""
    es = read_feed(name, date, cutoff, strategy)
    if topic:
        es = [e for e in es if e.get("topic") == topic]
    return es[-1] if es else None


def cutoff_now() -> float:
    """盘中模拟用的闸门值(当前墙钟)"""
    return time.time()


def cutoff_of(date: str, hhmmss: str) -> float:
    """回放用的闸门值: 由 YYYYMMDD + HH:MM:SS 构造 epoch"""
    return datetime.strptime(f"{date} {hhmmss}", "%Y%m%d %H:%M:%S").timestamp()
