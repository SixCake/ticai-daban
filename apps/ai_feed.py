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
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA  # noqa: E402

from rqalpha_mod_ticai import feeds  # noqa: E402
from core import llm, newsfeed, theme_kb  # noqa: E402
from core.attribute import load_con2stock  # noqa: E402
from core.codes import ts_code_of  # noqa: E402

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


# ---------- 投资线索(板块/股票识别) ----------

IMPACTS = ["严重利好", "轻微利好", "间接利好", "中性",
           "轻微利空", "间接利空", "严重利空"]
GRADES = ["S", "A", "B", "C"]
CLUE_TYPES = ["sector", "stock", "fund"]
MAX_SRC_TEXT = 500            # 线索 extra.sources 单条正文截断(展示用)

LLM_CFG_PATH = DATA / "sim" / "ai_feeds" / "_llm" / "config.json"
_SECTOR_FULL: list | None = None
_STOCK_MAP: dict | None = None

_LLM_SYSTEM = (
    "你是A股题材投研分析师。阅读给定的新闻快讯, 识别中国A股可交易的投资线索。\n"
    "每条线索必须给出: title(一句话标题), type(sector板块类|stock股票类|fund基金类), "
    "grade(S|A|B|C 重要性), heat(0-10综合热度), sentiment(bullish|bearish|neutral), "
    "summary(2-3句加工解读), bullish(利好点列表), bearish(利空点列表), "
    "sectors(关联板块[{name,impact}]), stocks(关联个股[{name,impact}]), "
    "events(相关事件列表), "
    "worth_dig(true/false 是否值得题材级深挖), worth_reason(判断理由)。\n"
    f"impact 只能取: {'/'.join(IMPACTS)}。\n"
    "板块 name 尽量使用候选板块词表中的名称; 个股 name 用股票简称或6位代码。\n"
    "worth_dig 判据: 值得深挖=消息可能改变题材整体方向(政策催化/产业拐点/"
    "龙头异动/板块联动等题材级影响); 不值得深挖=仅影响单股的事件"
    "(个股财务造假/退市/处罚/减持等不改变题材方向的消息)。\n"
    "只依据给定新闻, 不得编造新闻中没有的板块/个股/事件; 无有效线索时返回 "
    "{\"clues\": []}。\n"
    "每条线索必须给出 refs: 该线索所依据的新闻编号列表(对应输入中的 [i] 编号), "
    "用于溯源, 不得编造编号。\n"
    "只输出 JSON, 形如 {\"clues\": [...]}。"
)


def _llm_config() -> dict:
    """LLM 生产者配置; 缺失返回缺省(启用+订阅全部启用源)"""
    if LLM_CFG_PATH.exists():
        try:
            return json.loads(LLM_CFG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"llm_clue": {"enabled": True, "model": None,
                         "sources": [s["id"] for s in newsfeed.enabled_sources()],
                         "prompt": None}}


def save_llm_config(cfg: dict) -> Path:
    LLM_CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LLM_CFG_PATH.write_text(json.dumps(cfg, ensure_ascii=False),
                            encoding="utf-8")
    return LLM_CFG_PATH


LLM_STATE_PATH = DATA / "sim" / "ai_feeds" / "_llm" / "state.json"


def _llm_state() -> dict:
    """生产者运行态(增量水位): {feed: {date: 已消费新闻的最大ts}}"""
    if LLM_STATE_PATH.exists():
        try:
            return json.loads(LLM_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_llm_state(st: dict) -> Path:
    LLM_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LLM_STATE_PATH.write_text(json.dumps(st, ensure_ascii=False),
                              encoding="utf-8")
    return LLM_STATE_PATH


def _norm_title(s) -> str:
    return re.sub(r"\W+", "", s or "")


def _title_sim(a, b) -> float:
    """标题字符二元组 Jaccard 相似度(跨源同义新闻去重用)"""
    a, b = _norm_title(a), _norm_title(b)
    if len(a) < 2 or len(b) < 2:
        return 1.0 if a and a == b else 0.0
    A = {a[i:i + 2] for i in range(len(a) - 1)}
    B = {b[i:i + 2] for i in range(len(b) - 1)}
    return len(A & B) / len(A | B)


def _dedup_items(items: list, th: float = 0.6) -> list:
    """跨源/同义新闻聚类, 每簇保留最早一条(避免同一新闻多源重复喂LLM)"""
    keep = []
    for it in sorted(items, key=lambda e: e.get("ts") or 0):
        if not any(_title_sim(it.get("title"), k.get("title")) >= th
                   for k in keep):
            keep.append(it)
    return keep


def _sector_full() -> list:
    """板块归一匹配全集: 题材/板块名(con2stock键) + 申万L1/L2"""
    global _SECTOR_FULL
    if _SECTOR_FULL is not None:
        return _SECTOR_FULL
    names = []
    try:
        names += list(load_con2stock().keys())
    except Exception:
        pass
    fsw = DATA / "meta" / "sw_map.json"
    if fsw.exists():
        try:
            for v in json.loads(fsw.read_text(encoding="utf-8")).values():
                if isinstance(v, dict):
                    names += [v.get("l1"), v.get("l2")]
        except Exception:
            pass
    seen, out = set(), []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    _SECTOR_FULL = out
    return out


def _sector_vocab() -> list:
    """注入 prompt 的候选板块词表(小): radar热题前40 + 申万L1"""
    names = []
    f = LIVE / "radar.json"
    if f.exists():
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            names += [t.get("name") for t in (d.get("themes") or [])[:40]]
        except Exception:
            pass
    fsw = DATA / "meta" / "sw_map.json"
    if fsw.exists():
        try:
            for v in json.loads(fsw.read_text(encoding="utf-8")).values():
                if isinstance(v, dict) and v.get("l1"):
                    names.append(v["l1"])
        except Exception:
            pass
    seen, out = set(), []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _stock_map() -> dict:
    """股票简称→ts_code(用 kpl_events 活跃票名表)"""
    global _STOCK_MAP
    if _STOCK_MAP is not None:
        return _STOCK_MAP
    from datastore import load as _ds_load
    try:
        ev = _ds_load("limitup.kpl_events", columns=["ts_code", "name"])
        _STOCK_MAP = {n: c for c, n in zip(ev["ts_code"], ev["name"]) if n}
    except Exception:
        _STOCK_MAP = {}
    return _STOCK_MAP


def _match_sector(raw: str, vocab: list) -> dict | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    for v in vocab:
        if v == raw:
            return {"name": v, "raw": raw, "matched": True, "code": None}
    for v in sorted(vocab, key=len, reverse=True):
        if len(v) >= 2 and (v in raw or raw in v):
            return {"name": v, "raw": raw, "matched": True, "code": None}
    return {"name": raw, "raw": raw, "matched": False, "code": None}


def _match_stock(raw: str) -> dict | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    code = None
    if re.fullmatch(r"\d{6}", raw):
        code = ts_code_of(raw)
    else:
        code = _stock_map().get(raw)
    return {"code": code, "name": raw, "matched": bool(code)}


def _norm_impact(v) -> str:
    return v if v in IMPACTS else "中性"


def _norm_grade(v) -> str:
    return v if v in GRADES else "B"


def _norm_type(v) -> str:
    return v if v in CLUE_TYPES else "sector"


def _src_refs(items: list, limit: int = 12) -> list:
    """线索 extra.sources: 本次分析输入源条目(截断正文, 展示用)"""
    return [{"source": it.get("source"), "ts": it.get("ts"),
             "t": it.get("t"), "title": it.get("title"),
             "text": (it.get("text") or "")[:MAX_SRC_TEXT]}
            for it in items[:limit]]


def _build_llm_user(items: list, vocab: list, custom: str | None,
                    item_text: int = 120, vocab_n: int = 30) -> str:
    lines = ["候选板块词表(优先使用): " + "、".join(vocab[:vocab_n]), "",
             "新闻快讯(按时间倒序):"]
    for i, it in enumerate(items, 1):
        lines.append(f"[{i}] {it.get('t','')} {it.get('title','')}"
                     f" | {(it.get('text') or '')[:item_text]}")
    if custom:
        lines += ["", "附加要求: " + custom]
    lines += ["", "请输出 JSON: {\"clues\": [...]}"]
    return "\n".join(lines)


def _llm_clue_entries(day: str, ts: float | None = None) -> list:
    """投资线索 — LLM 识别板块/股票/基金线索 + 受益受损方向。
    无 LLM 密钥/无新闻/调用失败 → 返回 [](不伪造)。"""
    cfg = _llm_config().get("llm_clue", {})
    if not cfg.get("enabled", True):
        return []
    if not llm.available():
        return []
    sources = cfg.get("sources") or [s["id"] for s in newsfeed.enabled_sources()]
    cutoff = ts if ts is not None else time.time()
    # 攒批合并(省token): 新增新闻不足 batch_min 且距上次调用不足
    # batch_max_sec 时本轮跳过(不推进水位, 新闻攒到下一轮一次处理);
    # 显式传 ts(回测补产) bypass 攒批保证回放完整
    batch_min = int(cfg.get("batch_min", 6))
    batch_max_sec = int(cfg.get("batch_max_sec", 900))
    item_text = int(cfg.get("item_text", 120))
    lim = int(cfg.get("limit", 24))
    plim = int(cfg.get("per_limit", 8))
    vocab_n = int(cfg.get("vocab_n", 30))
    # 增量水位: 只分析「上次产出之后」的新新闻, 防止每轮重复分析同批新闻
    # 产生换措辞的重复线索
    st = _llm_state()
    wm = float((st.get("llm_clue") or {}).get(day, 0) or 0)
    raw = newsfeed.read_subscribed(sources, day, cutoff, limit=lim,
                                   per_limit=plim)
    items = _dedup_items([it for it in raw if (it.get("ts") or 0) > wm])
    if not items:
        return []
    last_call = float((st.get("llm_clue") or {}).get("_last_call", 0) or 0)
    if (ts is None and len(items) < batch_min
            and (time.time() - last_call) < batch_max_sec):
        print(f"[ai_feed] llm_clue 攒批: 新{len(items)}条<{batch_min} 且距上次"
              f"调用<{batch_max_sec}s, 本轮跳过(合并到下轮)")
        return []
    existing = {e.get("topic") for e in feeds.read_feed("llm_clue", day)}
    user = _build_llm_user(items, _sector_vocab(), cfg.get("prompt"),
                           item_text=item_text, vocab_n=vocab_n)
    res = llm.chat_json(_LLM_SYSTEM, user, model=cfg.get("model"))
    if res is not None:
        # 本批新闻已消费: 推进水位 + 记录调用时刻(失败不推进, 下轮重试)
        st.setdefault("llm_clue", {})[day] = max(
            (it.get("ts") or 0) for it in items)
        st["llm_clue"]["_last_call"] = time.time()
        _save_llm_state(st)
    if not res:
        return []
    full = _sector_full()
    out = []
    for c in res.get("clues") or []:
        title = (c.get("title") or "").strip()
        if not title or title in existing:
            continue
        existing.add(title)
        secs = []
        for s in c.get("sectors") or []:
            m = _match_sector(s.get("name"), full)
            if m:
                m["impact"] = _norm_impact(s.get("impact"))
                secs.append(m)
        stks = []
        for s in c.get("stocks") or []:
            m = _match_stock(s.get("name"))
            if m:
                m["impact"] = _norm_impact(s.get("impact"))
                stks.append(m)
        heat = max(0.0, min(10.0, float(c.get("heat") or 0)))
        # 溯源: 按 LLM 回填的 refs 定位真实源头新闻(而非整批输入)
        refs = [int(r) for r in (c.get("refs") or [])
                if isinstance(r, (int, float)) and 1 <= int(r) <= len(items)]
        ref_items = [items[r - 1] for r in dict.fromkeys(refs)]
        if not ref_items:
            # LLM 未回填 refs: 退化为「与线索标题最相关的新闻」,
            # 仍只嵌关联内容而非整批输入
            ref_items = [it for it in items
                         if _title_sim(title, it.get("title")) >= 0.35][:3]
        srcs = _src_refs(ref_items)
        src0 = ref_items[0].get("source") if ref_items else None
        cts = (min((it.get("ts") or cutoff) for it in ref_items)
               if ref_items else cutoff)
        out.append({
            "ts": cts,
            "topic": title,
            "score": round(heat / 10.0, 4),
            "text": (c.get("summary") or title),
            "src": "llm",
            "extra": {
                "clue_type": _norm_type(c.get("type")),
                "grade": _norm_grade(c.get("grade")),
                "heat": round(heat, 1),
                "sentiment": c.get("sentiment") or "neutral",
                "bullish": list(c.get("bullish") or []),
                "bearish": list(c.get("bearish") or []),
                "sectors": secs, "stocks": stks,
                "events": list(c.get("events") or []),
                "sources": srcs, "source": src0,
                "model": llm.model_name(cfg.get("model")),
                "worth_dig": bool(c.get("worth_dig")),
                "worth_reason": c.get("worth_reason") or "",
            },
        })
    out.sort(key=lambda e: -(e["extra"]["heat"]))
    return out


_BULL = ("利好", "提振", "中标", "突破", "增长", "上涨", "签约", "合作",
         "获批", "增产", "提价", "超预期")
_BEAR = ("召回", "处罚", "禁令", "下滑", "亏损", "下跌", "违规", "立案",
         "减持", "停产", "调查")


def _clue_rule_entries(day: str, ts: float | None = None) -> list:
    """投资线索 — 规则基线对照(无需 LLM): 标题命中板块词表即板块类线索,
    impact 用情感词表判档。供无 LLM 时跑通全链路 + 研究侧 A/B 对照。"""
    cutoff = ts if ts is not None else time.time()
    sources = [s["id"] for s in newsfeed.enabled_sources()]
    items = newsfeed.read_subscribed(sources, day, cutoff, limit=40)
    if not items:
        return []
    existing = {e.get("topic") for e in feeds.read_feed("clue_rule", day)}
    full = _sector_full()
    out = []
    for it in items:
        title = (it.get("title") or "").strip()
        if not title or title in existing:
            continue
        secs = []
        for v in full:
            if len(v) >= 2 and v in title:
                secs.append({"name": v, "raw": v, "matched": True,
                             "code": None})
        if not secs:
            continue
        existing.add(title)
        bull = any(w in title for w in _BULL)
        bear = any(w in title for w in _BEAR)
        impact = ("轻微利好" if bull and not bear
                  else "轻微利空" if bear and not bull else "中性")
        for s in secs:
            s["impact"] = impact
        sentiment = ("bullish" if bull and not bear
                     else "bearish" if bear and not bull else "neutral")
        out.append({
            "ts": it.get("ts") or cutoff,
            "topic": title,
            "score": 0.5,
            "text": title,
            "src": "rule_baseline",
            "extra": {
                "clue_type": "sector", "grade": "B", "heat": 5.0,
                "sentiment": sentiment,
                "bullish": [title] if sentiment == "bullish" else [],
                "bearish": [title] if sentiment == "bearish" else [],
                "sectors": secs, "stocks": [], "events": [],
                "sources": _src_refs([it], 1), "source": it.get("source"),
                "model": None,
            },
        })
    return out


# ---------- 题材催化归因(第一催化/驱动类型/受益环节) ----------

_CATALYST_SYSTEM = (
    "你是A股题材催化归因分析师。阅读给定的新闻快讯, 识别正在发酵的题材及其第一催化。\n"
    "每条催化必须给出: theme(题材名), driver_type(政策|产业|事件|技术|周期|海外映射), "
    "first_catalyst(第一催化一句话, 尽量含时间点), catalyst_conf(可确认|中|低), "
    "benefit_nodes(受益产业链环节列表, 如上游设备/中游设计/下游分销), "
    "turning_factor(拐点因子: 题材从横盘/下跌转为启动的边际变化信号), "
    "persistence_factor(持续性因子: 趋势能否延续的依据), heat(0-10催化强度), "
    "refs(该催化所依据的新闻编号列表, 对应输入[i]编号)。\n"
    "拐点因子与持续性因子必须分开给, 不可混用。\n"
    "catalyst_conf 判据: 政策原文/公司公告/可日频跟踪的价格信号=可确认; "
    "多源新闻一致=中; 单一传闻/推测=低。\n"
    "只依据给定新闻, 不编造新闻中没有的题材/环节/事件; 无有效催化返回 "
    "{\"catalysts\": []}。只输出 JSON。"
)

# 无LLM时的驱动类型关键词规则基线(降级用, catalyst_conf=低)
_DRIVER_KW = {
    "政策": ("政策", "国务院", "部委", "补贴", "试点", "规划", "意见",
           "通知", "发改委", "实施方案"),
    "海外映射": ("美股", "纳斯达克", "英伟达", "台积电", "海外", "隔夜",
             "费城", "美光"),
    "周期": ("涨价", "价格上涨", "现货价", "报价", "提价", "景气",
           "去库存", "供不应求"),
    "事件": ("冲突", "制裁", "事故", "突发", "中标", "签约", "并购", "重组"),
    "技术": ("突破", "首发", "量产", "研发成功", "新技术", "专利"),
    "产业": ("扩产", "产能", "订单", "出货", "渗透率", "资本开支", "投产"),
}


def _driver_by_kw(text: str) -> str:
    """关键词判驱动类型(规则基线/LLM回退校验用); 无命中归产业"""
    for dt, kws in _DRIVER_KW.items():
        if any(w in text for w in kws):
            return dt
    return "产业"


def _build_catalyst_user(items: list, vocab: list, item_text: int = 140,
                         vocab_n: int = 30) -> str:
    lines = ["候选题材词表(优先使用): " + "、".join(vocab[:vocab_n]), "",
             "新闻快讯(按时间倒序):"]
    for i, it in enumerate(items, 1):
        lines.append(f"[{i}] {it.get('t','')} {it.get('title','')}"
                     f" | {(it.get('text') or '')[:item_text]}")
    lines += ["", "请输出 JSON: {\"catalysts\": [...]}"]
    return "\n".join(lines)


def _kb_crossref(theme_name: str) -> dict:
    """与已入库剧本交叉验证: 命中则回填 playbook_ref/driver_type(供D/E消费)"""
    try:
        kb = theme_kb.load_kb()
    except Exception:
        return {}
    th = kb["theme"]
    if th.empty or not theme_name:
        return {}
    row = None
    # 先按 name 或 concept_code 精确匹配(radar/LLM 常产出原始题材名=concept_code)
    hit = th[(th["name"] == theme_name) | (th["concept_code"] == theme_name)]
    if len(hit):
        row = hit.iloc[0]
    else:
        for _, r in th.iterrows():
            if (theme_kb._name_sim(theme_name, r.get("name")) >= 0.5
                    or theme_kb._name_sim(theme_name,
                                          str(r.get("concept_code"))) >= 0.5):
                row = r
                break
    if row is None:
        return {}
    return {"kb_concept_code": row.get("concept_code"),
            "kb_name": row.get("name"),
            "kb_driver_type": row.get("driver_type"),
            "kb_playbook_ref": row.get("playbook_ref")}


def _theme_catalyst_entries(day: str, ts: float | None = None) -> list:
    """题材催化归因 — LLM 抽第一催化/驱动类型/受益环节 + 拐点/持续性双因子。

    无 LLM 密钥/调用失败 → 降级关键词规则基线(src=rule_baseline,
    catalyst_conf=低), 保证全链路可跑通且不伪造 LLM 结论。无新闻 → []。
    回放安全: 走 newsfeed 时间戳闸门(cutoff=ts)。
    """
    cfg = _llm_config().get("theme_catalyst", {})
    if not cfg.get("enabled", True):
        return []
    cutoff = ts if ts is not None else time.time()
    sources = cfg.get("sources") or [s["id"] for s in newsfeed.enabled_sources()]
    items = _dedup_items(newsfeed.read_subscribed(
        sources, day, cutoff, limit=int(cfg.get("limit", 24)),
        per_limit=int(cfg.get("per_limit", 8))))
    if not items:
        return []
    existing = {e.get("topic") for e in feeds.read_feed("theme_catalyst", day)}
    res = None
    if llm.available():
        user = _build_catalyst_user(items, _sector_vocab(),
                                    item_text=int(cfg.get("item_text", 140)))
        res = llm.chat_json(_CATALYST_SYSTEM, user, model=cfg.get("model"))
    out = []
    if res:
        for c in res.get("catalysts") or []:
            theme = (c.get("theme") or "").strip()
            if not theme or theme in existing:
                continue
            existing.add(theme)
            fc = c.get("first_catalyst") or ""
            dt = c.get("driver_type")
            dt = dt if dt in theme_kb.DRIVER_TYPES \
                else _driver_by_kw(theme + fc)
            conf = c.get("catalyst_conf")
            conf = conf if conf in theme_kb.CATALYST_CONF else "低"
            heat = max(0.0, min(10.0, float(c.get("heat") or 0)))
            refs = [int(r) for r in (c.get("refs") or [])
                    if isinstance(r, (int, float)) and 1 <= int(r) <= len(items)]
            ref_items = [items[r - 1] for r in dict.fromkeys(refs)]
            if not ref_items:
                ref_items = [it for it in items
                             if _title_sim(theme, it.get("title")) >= 0.3][:3]
            cts = (min((it.get("ts") or cutoff) for it in ref_items)
                   if ref_items else cutoff)
            extra = {
                "driver_type": dt, "first_catalyst": fc,
                "catalyst_conf": conf,
                "benefit_nodes": list(c.get("benefit_nodes") or []),
                "turning_factor": c.get("turning_factor"),
                "persistence_factor": c.get("persistence_factor"),
                "sources": _src_refs(ref_items),
                "source": (ref_items[0].get("source") if ref_items else None),
                "model": llm.model_name(cfg.get("model")),
            }
            extra.update(_kb_crossref(theme))
            out.append({"ts": cts, "topic": theme,
                        "score": round(heat / 10.0, 4),
                        "text": fc or theme, "src": "llm", "extra": extra})
    else:
        # 规则基线: 标题命中板块词表即候选题材, 驱动类型按关键词判, 置信度低
        full = _sector_full()
        for it in items:
            title = (it.get("title") or "").strip()
            if not title:
                continue
            theme = next((v for v in sorted(full, key=len, reverse=True)
                          if len(v) >= 2 and v in title), None)
            if not theme or theme in existing:
                continue
            existing.add(theme)
            body = title + (it.get("text") or "")[:120]
            extra = {
                "driver_type": _driver_by_kw(body), "first_catalyst": title,
                "catalyst_conf": "低", "benefit_nodes": [],
                "turning_factor": None, "persistence_factor": None,
                "sources": _src_refs([it], 1), "source": it.get("source"),
                "model": None,
            }
            extra.update(_kb_crossref(theme))
            out.append({"ts": it.get("ts") or cutoff, "topic": theme,
                        "score": 0.5, "text": title,
                        "src": "rule_baseline", "extra": extra})
    out.sort(key=lambda e: -(e["score"] or 0))
    return out


# ---------- 题材智能体(LangGraph 多角色 → 题材级因子) ----------

def _hot_themes(day: str, topn: int) -> list:
    """当日热门题材名(topn): 今日用 radar 热度排序, 历史日用 theme.day
    关联家数排序(无未来信息)。"""
    today = datetime.now().strftime("%Y%m%d")
    if day == today:
        f = LIVE / "radar.json"
        if f.exists():
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                th = [(t.get("name"), float(t.get("heat") or 0))
                      for t in (d.get("themes") or []) if t.get("name")]
                th.sort(key=lambda x: -x[1])
                return [n for n, _ in th[:topn]]
            except Exception:
                pass
        return []
    try:
        from datastore import load
        td = load("theme.day", columns=["trade_date", "concept_code", "zt_all"])
        row = td[td["trade_date"] == day].sort_values("zt_all", ascending=False)
        return list(row["concept_code"].head(topn))
    except Exception:
        return []


_GRADE_W = {"S": 4, "A": 3, "B": 2, "C": 1}


_PROCESSED_DIR = DATA / "sim" / "ai_feeds" / "_theme_processed"


def _load_processed(day: str) -> set:
    """今日已处理过的题材(已深挖或已判不值得), 防重复跑同一条消息催化。"""
    p = _PROCESSED_DIR / f"{day}.json"
    if p.exists():
        try:
            return set(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            return set()
    return set()


def _mark_processed(day: str, themes) -> None:
    if not themes:
        return
    p = _PROCESSED_DIR / f"{day}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    cur = _load_processed(day) | set(themes)
    p.write_text(json.dumps(sorted(cur), ensure_ascii=False), encoding="utf-8")


def _clue_themes(day: str, topn: int, min_grade=("S", "A", "B"),
                 min_heat: float = 6.0) -> list:
    """从高价值AI线索提取题材 —— 预测性触发(正确方向)。

    为何不用 radar 热度(旧逻辑本末倒置): radar 热度前N 是已经爆发的题材
    (结果), 拿它们跑智能体是"用结果查原因"、无预测效应。正确链路是:
    消息→AI线索发现高价值线索(早于题材发酵)→智能体深挖→因子。

    线索源: llm_clue(grade/heat/sectors) + theme_catalyst(催化归因可确认/中)。
    按线索价值(grade权重+heat)排序取 topn 个题材; 无高价值线索则返回空
    (不深挖, 不降级用 radar 热度事后追认)。"""
    cand: dict = {}
    for e in feeds.read_feed("llm_clue", day):
        ex = e.get("extra") or {}
        grade = ex.get("grade") or "C"
        heat = float(ex.get("heat") or 0)
        if grade not in min_grade and heat < min_heat:
            continue                       # 低价值线索不触发深挖
        # 排除: 单股利空消息(只影响个股不影响题材, 不值得题材级深挖)
        # 例: "*ST卓然涉财务造假" → clue_type=stock + bearish + 仅1股关联
        # 这类消息不会改变题材整体方向, 深挖无意义且浪费算力
        if (ex.get("clue_type") == "stock"
                and len(ex.get("stocks") or []) <= 1
                and (ex.get("sentiment") == "bearish"
                     or any("利空" in (s.get("impact") or "")
                            for s in (ex.get("sectors") or [])))):
            continue
        # LLM 已判不值得深挖 → 不纳入题材候选(旧数据缺失此字段则默认放行)
        if ex.get("worth_dig") is False:
            continue
        val = _GRADE_W.get(grade, 1) * 2 + heat
        secs = ex.get("sectors") or []
        if secs:
            # 去重: 同一条新闻只取主板块(首个), 不同板块不重复跑
            nm = secs[0].get("name")
            if nm:
                cand[nm] = max(cand.get(nm, 0), val)
    for e in feeds.read_feed("theme_catalyst", day):
        ex = e.get("extra") or {}
        if ex.get("catalyst_conf") in ("可确认", "中"):
            nm = e.get("topic")
            if nm:
                cand[nm] = max(cand.get(nm, 0),
                               6 + float(e.get("score") or 0) * 4)
    ranked = sorted(cand.items(), key=lambda kv: -kv[1])
    return [t for t, _ in ranked[:topn]]


def _theme_news(theme: str, day: str, all_news: list, limit: int = 8) -> list:
    """过滤出与题材相关的新闻(而非全部泛快讯) —— 修2。

    旧逻辑把全部订阅新闻(含纽约联储/泥石流等无关宏观)喂给每个题材,
    催化分析师只能"无一条涉及本题材"。改为只拿该题材相关新闻:
      ① llm_clue/theme_catalyst 中该题材线索的溯源新闻(extra.sources);
      ② 标题命中题材名(去'概念'后缀)。"""
    rel, seen = [], set()
    t_core = theme.replace("概念", "").strip()
    for fn in ("llm_clue", "theme_catalyst"):
        for e in feeds.read_feed(fn, day):
            ex = e.get("extra") or {}
            secs = ([s.get("name") for s in (ex.get("sectors") or [])]
                    if fn == "llm_clue" else [e.get("topic")])
            if not any(theme == s or (t_core and t_core in (s or ""))
                       for s in secs):
                continue
            for src in (ex.get("sources") or []):
                ti = src.get("title")
                if ti and ti not in seen:
                    seen.add(ti)
                    rel.append({"t": src.get("t"), "title": ti,
                                "source": src.get("source"),
                                "text": src.get("text")})
    for it in all_news:
        ti = it.get("title") or ""
        if ti not in seen and t_core and t_core in ti:
            seen.add(ti)
            rel.append({"t": it.get("t"), "title": ti,
                        "source": it.get("source"), "text": it.get("text")})
    return rel[:limit]


def _theme_agent_entries(day: str, ts: float | None = None) -> list:
    """题材智能体 — LangGraph 多角色深挖高价值AI线索题材, 落成题材级因子。

    触发(预测性): 由 _clue_themes 从 llm_clue/theme_catalyst 取高价值线索题材
    (早于题材发酵), 而非 radar 热度前N(已爆发、事后追认、无预测效应)。

    每个题材跑一次 run_graph(催化/阶段/龙头/情绪 4 分析师 + 多空辩论 +
    裁判), 过程写运行态文件(data/live/ai_agent_run.json, 供 SSE 可视化),
    裁判结论落成 theme_factor feed 条目(带 ts → 可回放)。
    无 LLM 密钥: run_graph 内部降级为规则基线(src=rule_baseline), 不伪造。
    回放安全: ts 由 produce() 的 _ts_for(day, at) 传入(补产历史日必传 --at)。"""
    cfg = _llm_config().get("theme_agent", {})
    if not cfg.get("enabled", True):
        return []
    cutoff = ts if ts is not None else time.time()
    topn = int(cfg.get("top_n", 3))
    # 预测性触发: 深挖高价值AI线索题材(非 radar 热度事后追认)
    themes = _clue_themes(day, topn,
                          min_grade=tuple(cfg.get("clue_grades",
                                                  ["S", "A", "B"])),
                          min_heat=float(cfg.get("clue_min_heat", 6.0)))
    # 去重: 今日已处理过的题材(已深挖或已判不值得)不重复跑同一条消息催化
    themes = [th for th in themes if th not in _load_processed(day)]
    if not themes:
        return []   # 无高价值线索 或 今日均已处理 → 不深挖
    try:
        from core.ai_agent_graph import RunTrace, run_graph
    except Exception as e:
        print(f"[ai_feed] theme_agent 不可用(langgraph 未装?): {e}")
        return []
    sources = cfg.get("sources") or [s["id"] for s in newsfeed.enabled_sources()]
    news = newsfeed.read_subscribed(sources, day, cutoff,
                                    limit=int(cfg.get("limit", 20)))
    srcs = [{"t": it.get("t"), "title": it.get("title"),
             "source": it.get("source")} for it in (news or [])[:8]]
    out = []
    from core.ai_agent_graph import judge_worth
    for th in themes:
        th_news = _theme_news(th, day, news)
        # 深挖价值判断闸门: 不值得深挖则跳过(不浪费算力)
        worth = judge_worth(th, th_news, model=cfg.get("model"))
        if not worth.get("worth"):
            _mark_processed(day, [th])   # 判不值得也算已处理, 今日不重复判
            print(f"[ai_feed] theme_agent {th} 不值得深挖, 跳过: "
                  f"{worth.get('reason')}")
            continue
        trace = RunTrace(th, day)
        try:
            res = run_graph(th, day=day, cutoff=cutoff,
                            news_items=th_news,
                            emit_cb=trace, model=cfg.get("model"))
        except Exception as e:
            print(f"[ai_feed] theme_agent {th} 失败: {type(e).__name__}: {e}")
            continue   # 失败不标记, 允许下轮重试
        trace.finish(res)
        _mark_processed(day, [th])   # 深挖成功, 今日不再重复跑同一题材
        f = res.get("factor") or {}
        d = res.get("debate") or {}
        out.append({
            "ts": cutoff, "topic": th,
            "score": round(float(f.get("confidence")
                                 or f.get("sustainability") or 0), 4),
            "text": res.get("conclusion") or th,
            "src": res.get("src") or "rule_baseline",
            "extra": {
                "factor": f,
                "roles": {k: res.get(f"{k}_report")
                          for k in ("catalyst", "chain", "stage", "leader",
                                    "sentiment")},
                "debate": {"bull": d.get("bull"), "bear": d.get("bear"),
                           "judge": res.get("conclusion")},
                "sources": srcs, "model": f.get("model"),
            },
        })
    return out


PRODUCERS = {
    "theme_narrative": (_theme_narrative_entries,
                        "题材叙事强度(规则基线; 接入 LLM 时替换实现)"),
    "theme_factor": (_theme_agent_entries,
                     "题材智能体因子(LangGraph多角色→题材级因子feed; "
                     "无LLM降级规则基线)"),
    "theme_catalyst": (_theme_catalyst_entries,
                       "题材催化归因(LLM抽第一催化/驱动类型/受益环节; "
                       "无LLM降级关键词规则基线)"),
    "market_risk": (_market_risk_entries,
                    "市场风险标注(炸板率/涨停家数/连板高度)"),
    "llm_clue": (_llm_clue_entries,
                 "投资线索(LLM识别板块/股票/基金线索+受益受损)"),
    "clue_rule": (_clue_rule_entries,
                  "投资线索(规则基线对照, 无需LLM)"),
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
        # 每轮重算日期(除非显式 --date 补产历史日): 常驻进程跨日后自动
        # 切到当日, 否则 day 冻结在启动那天 → 今日 feed 永不产出(实测踩坑:
        # 周一启动的进程周二仍只产周一的 feed, AI线索页今日空白)。
        # 与 collect/fetch_news.py --loop 同口径。
        day = args.date or datetime.now().strftime("%Y%m%d")
        produce_all(day, names, append=True, at=args.at)
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(cli())
