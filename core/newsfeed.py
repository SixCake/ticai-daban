# -*- coding: utf-8 -*-
"""数据源 feed 订阅层 — LLM 分析的信息输入(与 ai_feeds 输出方向相反)

设计: 与 rqalpha_mod_ticai/feeds.py 对称的发布/订阅 + 时间戳闸门, 但方向相反:
  feeds.py    = AI 产出 → 策略消费(输出)
  newsfeed.py = 新闻源采集 → LLM 消费(输入)

订阅模型:
  源注册表   data/live/news/sources.json          可订阅的新闻/RSS流清单
  源条目     data/live/news/{source_id}/{date}.json  采集落盘的原始新闻

时间戳闸门(与 feeds.py 同口径, 防未来信息):
  每条 item 带 ts(epoch); read_items(cutoff=...) 只返回 ts<=cutoff 的条目。
  LLM 生产者盘中 cutoff=墙钟; 补产历史日 cutoff=那天时刻(否则回放全被滤掉)。

条目结构(采集器约定, LLM 只读不改):
  {"ts": epoch, "t": "HH:MM:SS", "source": str, "title": str,
   "text": str, "url": str, "extra": {...}}
"""
import json
import time
from datetime import datetime
from pathlib import Path

from config import DATA

NEWS_ROOT = DATA / "live" / "news"
SOURCES_FILE = NEWS_ROOT / "sources.json"
MAX_ITEM_TEXT = 4000          # 单条正文上限(防单文件膨胀)

# 内置可订阅源(type 决定采集方式, 见 collect/fetch_news.py)。
# 首次 load_sources() 返回本默认清单(不写盘), 由 save_sources() 落盘。
DEFAULT_SOURCES = [
    {"id": "cls_telegraph", "name": "财联社电报", "type": "cls",
     "enabled": True, "poll_sec": 300, "url": ""},
    {"id": "em_global", "name": "东财全球快讯", "type": "em",
     "enabled": False, "poll_sec": 300, "url": ""},
    {"id": "ths_global", "name": "同花顺快讯", "type": "ths",
     "enabled": False, "poll_sec": 300, "url": ""},
    {"id": "cninfo_announce", "name": "巨潮公司公告", "type": "announce",
     "enabled": False, "poll_sec": 300, "url": ""},
]


def _ensure(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------- 源注册表 ----------

def load_sources() -> list:
    """源注册表; 文件缺失时返回内置默认源(不写盘)。
    文件存在时, 自动补齐「注册表里缺失的内置源」(默认禁用) —— 否则
    老用户的 sources.json 永远看不到后续新增的内置源(如 cninfo_announce)。"""
    if not SOURCES_FILE.exists():
        return [dict(s) for s in DEFAULT_SOURCES]
    try:
        saved = json.loads(SOURCES_FILE.read_text(
            encoding="utf-8")).get("sources", [])
    except Exception:
        saved = []
    have = {s.get("id") for s in saved}
    merged = [dict(s) for s in saved]
    for d in DEFAULT_SOURCES:
        if d["id"] not in have:
            merged.append(dict(d, enabled=False))
    return merged


def save_sources(sources: list) -> Path:
    """覆写源注册表(规整: 只留 id/name/type/enabled/poll_sec/url)"""
    clean = {"sources": [
        {"id": s.get("id"), "name": s.get("name") or s.get("id"),
         "type": s.get("type", "rss"), "enabled": bool(s.get("enabled", True)),
         "poll_sec": int(s.get("poll_sec", 300) or 300),
         "url": s.get("url", "")}
        for s in sources if s.get("id")],
        "updated": datetime.now().strftime("%Y%m%d %H:%M:%S")}
    _ensure(NEWS_ROOT)
    SOURCES_FILE.write_text(json.dumps(clean, ensure_ascii=False),
                            encoding="utf-8")
    return SOURCES_FILE


def enabled_sources() -> list:
    """仅启用中的源(采集器据此轮询)"""
    return [s for s in load_sources() if s.get("enabled")]


# ---------- 源条目 ----------

def items_path(source_id: str, date: str) -> Path:
    return NEWS_ROOT / source_id / f"{date}.json"


def append_items(source_id: str, date: str, items: list) -> Path:
    """增量追加原始新闻(去重键=(ts取整秒, title))。缺 ts/t 自动补齐。
    盘中持续采集用本函数(保留历史条目, 时间戳闸门才有意义)。"""
    p = items_path(source_id, date)
    old = []
    if p.exists():
        try:
            old = json.loads(p.read_text(encoding="utf-8")).get("items", [])
        except Exception:
            old = []
    seen = {(int(e["ts"]), e.get("title")) for e in old if "ts" in e}
    now = time.time()
    add = []
    for e in items:
        d = dict(e)
        if "ts" not in d:
            d["ts"] = now
        if "t" not in d:
            d["t"] = datetime.fromtimestamp(d["ts"]).strftime("%H:%M:%S")
        if isinstance(d.get("text"), str) and len(d["text"]) > MAX_ITEM_TEXT:
            d["text"] = d["text"][:MAX_ITEM_TEXT]
        d.setdefault("source", source_id)
        k = (int(d["ts"]), d.get("title"))
        if k in seen:
            continue
        seen.add(k)
        add.append(d)
    merged = sorted(old + add, key=lambda d: d.get("ts", 0))
    _ensure(p.parent)
    p.write_text(json.dumps({"date": date, "source": source_id,
                             "items": merged}, ensure_ascii=False),
                 encoding="utf-8")
    return p


def read_items(source_id: str, date: str, cutoff: float | None = None) -> list:
    """读某源条目(时间戳闸门过滤, 时间升序)。缺失/损坏返回空(不抛异常 —
    LLM 生产者不应因某源缺采集而崩)。"""
    p = items_path(source_id, date)
    if not p.exists():
        return []
    try:
        items = json.loads(p.read_text(encoding="utf-8")).get("items", [])
    except Exception:
        return []
    if cutoff is None:
        return items
    return [e for e in items if e.get("ts", 0) <= cutoff]


def read_subscribed(source_ids: list, date: str, cutoff: float | None = None,
                    limit: int = 60, per_limit: int = 12) -> list:
    """合并多个订阅源的条目(按 ts 降序, 供 LLM 分析)。
    per_limit=每源最多取最近N条 —— 避免单一高产源(如东财一日200条)
    把整体 limit 占满、淹没其它源; source_ids 为空返回空(不伪造)。"""
    out = []
    for sid in source_ids or []:
        its = read_items(sid, date, cutoff)
        its.sort(key=lambda e: e.get("ts", 0), reverse=True)
        out.extend(its[:per_limit])
    out.sort(key=lambda e: e.get("ts", 0), reverse=True)
    return out[:limit]
