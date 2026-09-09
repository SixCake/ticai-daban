# -*- coding: utf-8 -*-
"""题材知识库领域逻辑（唯一出处）— 认知/复盘资产, 非买卖信号

定位: 把历史题材的「发酵剧本」结构化沉淀, 新题材盘中异动时检索最相似
剧本, 给出阶段预案与退潮/切换信号。**只展示与给预案, 不接入任何买卖
拦截**(与 core/cycle.py、core/shortboard.py、V5 影子层同原则)。

存储(见 datastore.py 注册表, 全部 static parquet):
  theme.kb_theme       题材档案结构化
  theme.kb_stock       个股四维画像
  theme.kb_similarity  题材相似边
  theme.kb_signal      退潮/切换信号清单
  data/theme/playbooks/{playbook_ref}.md   剧本正文(markdown, 不入库)

本模块是 KB 的 schema 与计算唯一出处: build_theme_kb / review / ai_feed /
radar 一律 import 本模块的列常量, 禁止各自硬编码列名。

无未来信息: match/response_chain 只吃「当前盘中题材状态」与「已入库的
历史剧本」, 不读任何当日尚未发生的量价。infer_stage 为启发式 v0(校准中)。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from config import DATA
from core.cycle import theme_mode
from core.heat import HOT_THRESHOLD

PLAYBOOK_DIR = DATA / "theme" / "playbooks"

# ---------- 枚举(唯一出处, 下游校验用) ----------
DRIVER_TYPES = ["政策", "产业", "事件", "技术", "周期", "海外映射"]
STAGES = ["潜伏", "启动", "发酵", "高潮", "分歧", "退潮"]
ROLES = ["龙头", "中军", "补涨", "影子"]
CATALYST_CONF = ["可确认", "中", "低"]       # 第一催化置信度分级
SIGNAL_TYPES = ["退潮", "切换"]

# ---------- 列 schema(唯一出处) ----------
KB_THEME_COLS = [
    "concept_code", "name", "aliases", "driver_type", "parent_theme",
    "first_catalyst", "catalyst_conf", "stage_axis",
    "falsification", "env_precondition", "playbook_ref",
]
KB_STOCK_COLS = [
    "concept_code", "ts_code", "name", "chain_node", "role",
    "moat", "realize_cycle", "competition", "ceiling",
    "benefit_purity", "four_dim_score",
]
KB_SIM_COLS = ["a_code", "a_name", "b_code", "b_name",
               "sim_score", "sim_driver", "sim_chain", "sim_flow"]
KB_SIGNAL_COLS = ["playbook_ref", "phase", "signal", "type", "confidence"]


def _empty(cols: list) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})


def _load_opt(name: str, cols: list) -> pd.DataFrame:
    """读注册数据集; 文件缺失/损坏返回带正确列的空表(不抛, 不伪造)。"""
    try:
        from datastore import load
        df = load(name)
    except Exception:
        return _empty(cols)
    # 补齐缺失列(向前兼容旧库), 保持列序
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def load_kb() -> dict:
    """一次性载入 KB 全部结构化表 + 剧本索引(供 match 复用, 避免逐题读盘)。

    返回 {"theme": df, "stock": df, "similarity": df, "signal": df,
          "playbooks": {ref: markdown_text}}
    """
    theme = _load_opt("theme.kb_theme", KB_THEME_COLS)
    stock = _load_opt("theme.kb_stock", KB_STOCK_COLS)
    sim = _load_opt("theme.kb_similarity", KB_SIM_COLS)
    sig = _load_opt("theme.kb_signal", KB_SIGNAL_COLS)
    plays: dict = {}
    if PLAYBOOK_DIR.exists():
        for f in PLAYBOOK_DIR.glob("*.md"):
            try:
                plays[f.stem] = f.read_text(encoding="utf-8")
            except Exception:
                continue
    return {"theme": theme, "stock": stock, "similarity": sim,
            "signal": sig, "playbooks": plays}


def load_playbook(ref: str) -> str | None:
    """按 playbook_ref 读剧本正文 markdown; 缺失返回 None。"""
    if not ref:
        return None
    p = PLAYBOOK_DIR / f"{ref}.md"
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return None


# ---------- 阶段判定(启发式 v0) ----------

def infer_stage(row: dict, age: int | None = None,
                stage: str | None = None) -> str:
    """盘中题材当前生命周期阶段粗判(潜伏/启动/发酵/高潮/分歧/退潮)。

    row:   theme_heat 产出的题材行(heat/zt/nmem/dens5...);
    stage: 题材阶段(研究39 定稿, core.cycle.theme_stage 的
           爆发/主升/鱼尾/无), **优先使用**;
    age:   旧口径波龄(theme.day.theme_age), 仅当 stage 缺失时降级使用
           —— 研究36/37 已证 age 是持续性而非阶段且方向反置。
    启发式 v0, 仅展示参照, 校准中 —— 与 cycle.stage_of 同原则不接买卖。
    """
    heat = float(row.get("heat") or 0)
    zt = int(row.get("zt") or 0)
    nmem = int(row.get("nmem") or 0)
    dens = (zt / nmem) if nmem else 0.0
    hot = heat >= HOT_THRESHOLD
    mode = stage if stage in ("爆发", "主升", "鱼尾") else (
        theme_mode(age) if age is not None else None)
    if mode is not None:
        if mode == "爆发":
            return "发酵" if hot else "启动"
        if mode == "主升":
            return "高潮" if (dens >= 0.15 or heat >= HOT_THRESHOLD * 2) else "发酵"
        # 鱼尾: 补涨性质, 热在=分歧, 已冷=退潮
        return "分歧" if hot else "退潮"
    if not hot:
        return "启动"
    if dens >= 0.15 or heat >= HOT_THRESHOLD * 2:
        return "高潮"
    return "发酵"


# ---------- 相似度 ----------

def _name_sim(a: str, b: str) -> float:
    """题材名字符二元组 Jaccard 相似度(名称线索匹配用)。"""
    a = (a or "").strip()
    b = (b or "").strip()
    if len(a) < 2 or len(b) < 2:
        return 1.0 if a and a == b else 0.0
    A = {a[i:i + 2] for i in range(len(a) - 1)}
    B = {b[i:i + 2] for i in range(len(b) - 1)}
    return len(A & B) / len(A | B)


def _parse_aliases(v) -> list:
    """aliases 字段(JSON字符串或list) → 别名列表"""
    if isinstance(v, list):
        return [str(x) for x in v if x]
    if isinstance(v, str) and v.strip():
        try:
            a = json.loads(v)
            return [str(x) for x in a if x] if isinstance(a, list) else []
        except Exception:
            return [x for x in re.split(r"[、,，/]", v) if x.strip()]
    return []


def _theme_keys(kb_row) -> list:
    """一个 KB 档案行的全部可匹配名: name + concept_code + aliases"""
    keys = []
    for f in ("name", "concept_code"):
        v = kb_row.get(f)
        if v and str(v) not in keys:
            keys.append(str(v))
    for a in _parse_aliases(kb_row.get("aliases")):
        if a not in keys:
            keys.append(a)
    return keys


def _match_theme(row: dict, kb: dict) -> tuple[pd.Series | None, float, str]:
    """把盘中题材行/新闻题材名匹配到 KB 档案: 返回 (档案行, 相似分, 匹配方式)。

    优先级: concept_code/name/alias 精确 > 名称/别名模糊(≥0.5)。
    别名解决"存储芯片/存储器/DRAM"归一到同一档案(新闻催化与 KB 桥接的关键)。
    无匹配返回 (None, 0, 'none')。"""
    theme = kb["theme"]
    if theme.empty:
        return None, 0.0, "none"
    code = str(row.get("concept_code") or "")
    name = str(row.get("name") or "")
    probe = {code, name} - {""}
    # 精确: concept_code / name / alias 命中
    for _, r in theme.iterrows():
        if probe & set(_theme_keys(r)):
            return r, 1.0, "exact"
    # 模糊: 名称或别名二元组相似≥0.5
    best, bs = None, 0.0
    for _, r in theme.iterrows():
        for k in _theme_keys(r):
            s = max(_name_sim(name, k), _name_sim(code, k))
            if s > bs:
                best, bs = r, s
    if best is not None and bs >= 0.5:
        return best, round(bs, 3), "fuzzy"
    return None, 0.0, "none"


def _similar_playbooks(kb_row: pd.Series, kb: dict, top_n: int) -> list:
    """经 kb_similarity 找与档案题材相似的其他题材剧本(按相似度降序)。"""
    sim = kb["similarity"]
    theme = kb["theme"]
    if sim.empty or kb_row is None:
        return []
    code = kb_row.get("concept_code")
    edges = sim[(sim["a_code"] == code) | (sim["b_code"] == code)]
    out = []
    for _, e in edges.iterrows():
        other = e["b_code"] if e["a_code"] == code else e["a_code"]
        o_name = e["b_name"] if e["a_code"] == code else e["a_name"]
        ref = None
        hit = theme[theme["concept_code"] == other]
        if len(hit):
            ref = hit.iloc[0].get("playbook_ref")
            o_name = o_name or hit.iloc[0].get("name")
        out.append({"concept_code": other, "name": o_name,
                    "sim_score": round(float(e.get("sim_score") or 0), 3),
                    "sim_driver": e.get("sim_driver"),
                    "sim_chain": e.get("sim_chain"),
                    "sim_flow": e.get("sim_flow"),
                    "playbook_ref": ref})
    out.sort(key=lambda d: -d["sim_score"])
    return out[:top_n]


# ---------- 响应链 ----------

def response_chain(kb: dict, kb_row: pd.Series, stage: str) -> dict:
    """给定匹配到的历史档案与当前阶段, 输出响应链预案。

    内容: 驱动类型 / 剧本正文 / 当前阶段与相邻阶段的退潮·切换信号 /
    龙头候选(按受益纯度降序) / 证伪条款 / 环境前提。
    """
    ref = kb_row.get("playbook_ref")
    stock = kb["stock"]
    sig = kb["signal"]
    code = kb_row.get("concept_code")
    # 龙头候选: 该题材下 role∈(龙头,中军) 按受益纯度降序
    leaders = []
    if not stock.empty:
        sub = stock[(stock["concept_code"] == code)
                    & (stock["role"].isin(["龙头", "中军"]))].copy()
        sub["_p"] = pd.to_numeric(sub["benefit_purity"], errors="coerce")
        sub = sub.sort_values("_p", ascending=False)
        for _, r in sub.head(8).iterrows():
            leaders.append({"ts_code": r.get("ts_code"),
                            "name": r.get("name"),
                            "role": r.get("role"),
                            "chain_node": r.get("chain_node"),
                            "benefit_purity": r.get("benefit_purity")})
    # 信号: 当前阶段 + 下一阶段(前瞻退潮), 无阶段匹配则给全部退潮信号
    signals = []
    if not sig.empty and ref:
        sub = sig[sig["playbook_ref"] == ref]
        try:
            si = STAGES.index(stage)
            phases = {stage, STAGES[min(si + 1, len(STAGES) - 1)]}
        except ValueError:
            phases = {stage}
        hit = sub[sub["phase"].isin(phases)]
        if hit.empty:
            hit = sub[sub["type"] == "退潮"]
        for _, r in hit.iterrows():
            signals.append({"phase": r.get("phase"), "signal": r.get("signal"),
                            "type": r.get("type"),
                            "confidence": r.get("confidence")})
    return {
        "driver_type": kb_row.get("driver_type"),
        "playbook_ref": ref,
        "playbook": (kb.get("playbooks") or {}).get(ref) if ref else None,
        "stage": stage,
        "leaders": leaders,
        "signals": signals,
        "falsification": kb_row.get("falsification"),
        "env_precondition": kb_row.get("env_precondition"),
        "first_catalyst": kb_row.get("first_catalyst"),
        "catalyst_conf": kb_row.get("catalyst_conf"),
    }


def match(cur_themes: list, kb: dict | None = None, top_n: int = 3,
          age_by: dict | None = None, stage_by: dict | None = None) -> list:
    """对盘中热点题材逐个匹配历史剧本并生成响应链。

    cur_themes: theme_heat 产出的题材行列表(通常已按 heat 降序);
    kb:       预载的知识库(load_kb()), 缺省内部载入一次;
    stage_by: {concept_code: 题材阶段} 研究39 定稿口径(波次驱动), **优先**;
    age_by:   {concept_code: theme_age} 旧口径持续性日龄, 仅降级用;
    返回: 只含匹配到剧本(match_way!='none')的题材响应, 按 heat 降序。
    """
    if kb is None:
        kb = load_kb()
    if kb["theme"].empty:
        return []
    age_by = age_by or {}
    stage_by = stage_by or {}
    out = []
    for row in cur_themes:
        if float(row.get("heat") or 0) < HOT_THRESHOLD:
            continue
        kb_row, score, way = _match_theme(row, kb)
        if kb_row is None:
            continue
        stage = infer_stage(row, age_by.get(row.get("concept_code")),
                            stage_by.get(row.get("concept_code")))
        resp = response_chain(kb, kb_row, stage)
        out.append({
            "concept_code": row.get("concept_code"),
            "name": row.get("name"),
            "heat": row.get("heat"),
            "zt": row.get("zt"),
            "stage": stage,
            "match_way": way,
            "match_score": score,
            "matched_theme": kb_row.get("name"),
            "similar": _similar_playbooks(kb_row, kb, top_n),
            "response": resp,
        })
    return out


# ---------- 数据访问: 股票池 / 篮子净值 / 价格位置 ----------

def kpl_theme_pool(concept_names: list) -> list:
    """从 theme.kpl_members 取若干 kpl 概念成分的并集 —— 题材股票池唯一出处。

    用 curated 概念成分(kpl_concept_cons), 不用事件级直标(con2stock[原始名]),
    后者会把养元饮品/合肥城建/券商等无关票混入(实测污染)。剔除 .BJ;
    按 hot_num 降序。concept_names 用概念名匹配。
    """
    if not concept_names:
        return []
    try:
        from datastore import load
        km = load("theme.kpl_members")
    except Exception:
        return []
    km = km[km["concept_name"].isin(concept_names)].copy()
    if km.empty:
        return []
    km["_h"] = pd.to_numeric(km["hot_num"], errors="coerce").fillna(0)
    km = km.sort_values("_h", ascending=False)
    out, seen = [], set()
    for _, r in km.iterrows():
        c = r.get("con_code")
        if c and c not in seen and not str(c).endswith(".BJ"):
            seen.add(c)
            out.append(c)
    return out


def basket_curve(codes: list, start: str, end: str,
                 panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """等权篮子净值曲线(各股以其窗口首个有效 close 归一)。

    返回 DataFrame[date, mean_nv, med_nv, n]; med_nv(中位数)抗单只大牛股扭曲,
    用于阶段/位置判定。panel 可传入已加载面板避免重复读盘。无数据返回空表。
    """
    cols = ["date", "mean_nv", "med_nv", "n"]
    if not codes:
        return pd.DataFrame(columns=cols)
    try:
        if panel is not None:
            pn = panel
        else:
            from datastore import load
            pn = load("market.daily_panel",
                      columns=["trade_date", "ts_code", "close"])
    except Exception:
        return pd.DataFrame(columns=cols)
    sub = pn[(pn["ts_code"].isin(codes)) & (pn["trade_date"] >= start)
             & (pn["trade_date"] <= end)]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    piv = sub.pivot_table(index="trade_date", columns="ts_code", values="close")
    base = piv.apply(lambda c: c.dropna().iloc[0] if c.notna().any()
                     else float("nan"))
    norm = piv / base
    df = pd.DataFrame({"date": norm.index.astype(str),
                       "mean_nv": norm.mean(axis=1).values,
                       "med_nv": norm.median(axis=1).values,
                       "n": norm.notna().sum(axis=1).values})
    return df.dropna(subset=["med_nv"]).reset_index(drop=True)


def price_position(codes: list, asof: str | None = None, lookback: int = 500,
                   panel: pd.DataFrame | None = None) -> dict | None:
    """题材篮子当前价格位置(数据驱动, 无未来信息: 只用 <=asof 的数据)。

    用中位净值在 trailing 窗口的相对位置判低位/中位/高位 —— 点火探测的
    “价格位置”维度(区分拐点与顶部叙事的关键)。返回:
      {asof, med_nv, trail_max, trail_min, pos_ratio(0低-1高), position,
       pct_from_peak, turning_up(近20日上行), n_days}; 数据不足返回 None。
    """
    if not codes:
        return None
    if panel is None:
        try:
            from datastore import load
            panel = load("market.daily_panel",
                         columns=["trade_date", "ts_code", "close"])
        except Exception:
            return None
    dates = sorted(str(d) for d in panel["trade_date"].unique())
    if asof:
        dates = [d for d in dates if d <= str(asof)]
    if len(dates) < 20:
        return None
    end = dates[-1]
    start = dates[-lookback] if len(dates) >= lookback else dates[0]
    bc = basket_curve(codes, start, end, panel=panel)
    if len(bc) < 20:
        return None
    nv = bc["med_nv"]
    cur = float(nv.iloc[-1])
    tmax, tmin = float(nv.max()), float(nv.min())
    pos_ratio = (cur - tmin) / (tmax - tmin) if tmax > tmin else 0.5
    position = "低位" if pos_ratio < 0.35 else "中位" if pos_ratio < 0.70 else "高位"
    turning_up = bool(len(nv) >= 20 and cur > float(nv.iloc[-20]))
    return {"asof": end, "med_nv": round(cur, 3), "trail_max": round(tmax, 3),
            "trail_min": round(tmin, 3), "pos_ratio": round(pos_ratio, 3),
            "position": position,
            "pct_from_peak": round((cur / tmax - 1) * 100, 1) if tmax else None,
            "turning_up": turning_up, "n_days": len(bc)}
