# -*- coding: utf-8 -*-
"""题材知识库生成器（离线批处理）— 从人工 config + 现有量价数据落库

职责: 把一个题材的「人工校验定性字段(config)」与「项目已有量价数据
(theme.day / kpl_members)自动推导字段」合并, upsert 进 KB 四张表(见
core/theme_kb.py 的 schema 唯一出处)。config 文件同时充当复盘空白模板。

设计原则:
- 领域逻辑/列名一律 import core.theme_kb, 本文件不重复定义 schema。
- 写库统一走 datastore.save/load, 禁止硬编码 data/ 路径。
- 定性字段(第一催化/四维画像/证伪条款)由人工校验后写 config, 每条带
  catalyst_conf 置信度; 量价字段(阶段轴/角色初筛/成分池)自动推导。
- 认知资产: 本生成器产出只供展示与检索, 不接任何买卖决策。

用法:
  python collect/build_theme_kb.py --config data/theme/kb_configs/存储超级周期.json
  python collect/build_theme_kb.py --config <cfg> --emit-playbook   # 顺带写剧本md骨架
  python collect/build_theme_kb.py --template                       # 打印空白config模板
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datastore import load, save, path_of  # noqa: E402
from core import theme_kb as kb  # noqa: E402

CONFIG_DIR = ROOT / "data" / "theme" / "kb_configs"

# 空白模板(复盘新题材时复制此结构填写; --template 打印)
TEMPLATE = {
    "concept_code": "存储",          # 必须=radar实时题材键(kpl原始题材名)
    "name": "存储超级周期",
    "aliases": ["存储芯片", "存储器", "DRAM"],  # 别名: 供新闻催化/radar归一到本档案
    "driver_type": "周期",           # 政策/产业/事件/技术/周期/海外映射
    "parent_theme": "半导体周期",
    "first_catalyst": "……(第一催化一句话)",
    "catalyst_conf": "可确认",       # 可确认/中/低
    "window": ["20250401", "20260430"],
    "env_precondition": "……(剧本成立的市场环境前提)",
    "falsification": "……(证伪条款: 出现什么信号说明本剧本不适用)",
    "playbook_ref": "涨价周期型_存储超级周期",
    "member_sources": ["半导体存储器", "长鑫存储概念"],  # kpl概念名, 自动补全股池
    "stage_axis": None,              # None=从theme.day自动推导; 或手填[{stage,date,note}]
    "stocks": [                      # 核心人工四维画像(其余成分自动初筛)
        {"ts_code": "603986.SH", "name": "兆易创新", "chain_node": "中游设计",
         "role": "中军", "moat": "……", "realize_cycle": "……",
         "competition": "……", "ceiling": "……", "benefit_purity": 0.9,
         "four_dim_score": None},
    ],
    "signals": [                     # 退潮/切换信号清单
        {"phase": "退潮", "signal": "……", "type": "退潮", "confidence": "高"},
    ],
    "similar": [                     # 题材相似边(可指向尚未入库的同构题材)
        {"b_code": "MLCC涨价", "b_name": "2017 MLCC涨价", "sim_score": 0.8,
         "sim_driver": "高", "sim_chain": "中", "sim_flow": "高"},
    ],
}


# ---------- 统一股票池(唯一出处) ----------

def kpl_theme_pool(concept_names: list) -> list:
    """从 theme.kpl_members 取若干 kpl 概念成分的并集 —— 题材股票池唯一出处。

    用 curated 概念成分(kpl_concept_cons), 不用事件级直标(con2stock[原始名]),
    后者会把养元饮品/合肥城建/券商等无关票混入(实测污染)。剔除 .BJ;
    按 hot_num 降序(人气高在前)。concept_names 用概念名匹配。
    """
    if not concept_names:
        return []
    try:
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

    返回 DataFrame[date, mean_nv, med_nv, n]; med_nv(中位数)抗单只大牛股
    扭曲, 用于阶段判定。数据源 market.daily_panel.close(已核与 pct_chg
    一致, 无送转断点)。panel 可传入已加载面板避免重复读盘。无数据返回空表。
    """
    cols = ["date", "mean_nv", "med_nv", "n"]
    if not codes:
        return pd.DataFrame(columns=cols)
    try:
        pn = panel if panel is not None else load(
            "market.daily_panel", columns=["trade_date", "ts_code", "close"])
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


# ---------- 自动推导 ----------

def derive_stage_axis(concept_code: str, window: list, pool: list | None = None,
                      panel: pd.DataFrame | None = None) -> list:
    """生命周期阶段轴推导。

    趋势型(有成分池 pool)优先用【篮子中位净值曲线】拐点(数据驱动, 适用
    机构趋势行情); 无 pool 时降级用 theme.day 涨停数口径(仅适用连板驱动题材,
    对趋势型会误判 —— 实测存储 theme.day zt_cnt 峰值龙头是园林股)。
    """
    if pool:
        axis = derive_stage_axis_basket(pool, window, panel=panel)
        if axis:
            return axis
    return _derive_stage_axis_zt(concept_code, window)


def derive_stage_axis_basket(codes: list, window: list,
                             panel: pd.DataFrame | None = None) -> list:
    """从篮子中位净值曲线推阶段轴(数据驱动, 趋势型题材主口径)。

    潜伏=窗口起点(基线); 启动=基线段后中位净值首次突破基线上沿×1.05;
    发酵=启动与高潮中点; 高潮=中位净值峰值日; 退潮=峰值后首次回撤≥30%。
    阶段名取自 core.theme_kb.STAGES。窗口不足返回 []。
    """
    if not window:
        return []
    df = basket_curve(codes, str(window[0]), str(window[1]), panel=panel)
    if len(df) < 20:
        return []
    nv = df["med_nv"]
    dates = df["date"].tolist()
    q = max(20, len(df) // 4)                 # 前1/4作潜伏基线段
    baseline_hi = float(nv.iloc[:q].quantile(0.9))
    peak_i = int(nv.idxmax())
    peak_v = float(nv.iloc[peak_i])
    start_i = next((i for i in range(q, peak_i + 1)
                    if nv.iloc[i] > baseline_hi * 1.05), q)
    axis = [
        {"stage": "潜伏", "date": dates[0], "med_nv": round(float(nv.iloc[0]), 3)},
        {"stage": "启动", "date": dates[start_i],
         "med_nv": round(float(nv.iloc[start_i]), 3)},
    ]
    if peak_i > start_i:
        mid_i = start_i + (peak_i - start_i) // 2
        axis.append({"stage": "发酵", "date": dates[mid_i],
                     "med_nv": round(float(nv.iloc[mid_i]), 3)})
    axis.append({"stage": "高潮", "date": dates[peak_i],
                 "med_nv": round(peak_v, 3)})
    ebb_i = next((i for i in range(peak_i + 1, len(df))
                  if nv.iloc[i] < peak_v * 0.70), None)
    if ebb_i is not None:
        axis.append({"stage": "退潮", "date": dates[ebb_i],
                     "med_nv": round(float(nv.iloc[ebb_i]), 3)})
    return axis


def _derive_stage_axis_zt(concept_code: str, window: list) -> list:
    """从 theme.day 涨停数推阶段轴(降级口径, 仅适用连板驱动题材)。

    注: 对趋势型题材不可靠(涨停数低且事件级直标龙头可能被污染),
    仅在无成分池时降级使用。启动=首个 zt_cnt>=3; 高潮=zt_cnt最大;
    分歧=高潮后回落>=40%; 退潮=分歧后 zt_cnt<=1。
    """
    try:
        td = load("theme.day",
                  columns=["trade_date", "concept_code", "zt_cnt",
                           "max_height", "theme_age", "leader_name"])
    except Exception:
        return []
    sub = td[td["concept_code"] == concept_code].copy()
    if sub.empty:
        return []
    if window:
        lo, hi = str(window[0]), str(window[1])
        sub = sub[(sub["trade_date"] >= lo) & (sub["trade_date"] <= hi)]
    sub = sub.sort_values("trade_date").reset_index(drop=True)
    if sub.empty:
        return []
    zt = sub["zt_cnt"].tolist()
    axis = []

    def rec(stage, i):
        r = sub.iloc[i]
        axis.append({"stage": stage, "date": str(r["trade_date"]),
                     "zt_cnt": int(r["zt_cnt"]),
                     "max_height": int(r["max_height"] or 0),
                     "leader_name": r["leader_name"]})

    start = next((i for i, z in enumerate(zt) if z >= 3), None)
    if start is None:
        return axis
    rec("启动", start)
    peak = max(range(start, len(zt)),
               key=lambda i: (zt[i], sub.iloc[i]["max_height"]))
    if peak > start:
        mid = next((i for i in range(start, peak) if zt[i] >= 2), start)
        if mid != start:
            rec("发酵", mid)
    rec("高潮", peak)
    thr = zt[peak] * 0.6
    div = next((i for i in range(peak + 1, len(zt)) if zt[i] <= thr), None)
    if div is not None:
        rec("分歧", div)
        ebb = next((i for i in range(div, len(zt)) if zt[i] <= 1), None)
        if ebb is not None:
            rec("退潮", ebb)
    return axis


def autofill_stocks(cfg: dict, manual_codes: set) -> list:
    """补全股池(规则初筛, 两层制第二层) —— 走统一池 kpl_theme_pool。

    核心人工四维股(cfg.stocks)优先, 其余成分: role=跟随, chain_node=待核实,
    moat=开盘啦入选原因desc(捕获「这只票是做什么的」原始信息, 待人工核实),
    benefit_purity=None。池与 build_theme_replay/阶段轴同一出处(消除多池不一致)。
    """
    srcs = cfg.get("member_sources") or []
    pool = kpl_theme_pool(srcs)
    if not pool:
        return []
    info = {}
    try:
        km = load("theme.kpl_members")
        kms = km[km["concept_name"].isin(srcs)]
        for _, r in kms.iterrows():
            c = r.get("con_code")
            if c and c not in info:
                info[c] = {"name": r.get("con_name"), "desc": r.get("desc")}
    except Exception:
        pass
    out, seen = [], set(manual_codes)
    for code in pool:                    # 保持 kpl_theme_pool 的 hot_num 降序
        if code in seen:
            continue
        seen.add(code)
        d = info.get(code, {})
        out.append({
            "concept_code": cfg["concept_code"], "ts_code": code,
            "name": d.get("name"), "chain_node": "待核实",
            "role": "跟随", "moat": (d.get("desc") or "")[:200] or None,
            "realize_cycle": None, "competition": None, "ceiling": None,
            "benefit_purity": None, "four_dim_score": None,
        })
    return out


# ---------- 组装 ----------

def _validate(cfg: dict) -> None:
    dt = cfg.get("driver_type")
    if dt not in kb.DRIVER_TYPES:
        raise ValueError(f"driver_type '{dt}' 非法, 可选 {kb.DRIVER_TYPES}")
    cc = cfg.get("catalyst_conf")
    if cc not in kb.CATALYST_CONF:
        raise ValueError(f"catalyst_conf '{cc}' 非法, 可选 {kb.CATALYST_CONF}")
    if not cfg.get("concept_code") or not cfg.get("playbook_ref"):
        raise ValueError("concept_code / playbook_ref 必填")


def build_rows(cfg: dict) -> dict:
    """按 config 组装四张表的新增行(DataFrame)。"""
    _validate(cfg)
    code = cfg["concept_code"]
    pool = kpl_theme_pool(cfg.get("member_sources") or [])
    axis = cfg.get("stage_axis") or derive_stage_axis(
        code, cfg.get("window"), pool=pool)
    theme_row = {
        "concept_code": code, "name": cfg.get("name") or code,
        "aliases": json.dumps(cfg.get("aliases") or [], ensure_ascii=False),
        "driver_type": cfg["driver_type"],
        "parent_theme": cfg.get("parent_theme"),
        "first_catalyst": cfg.get("first_catalyst"),
        "catalyst_conf": cfg.get("catalyst_conf"),
        "stage_axis": json.dumps(axis, ensure_ascii=False),
        "falsification": cfg.get("falsification"),
        "env_precondition": cfg.get("env_precondition"),
        "playbook_ref": cfg.get("playbook_ref"),
    }
    manual = []
    for s in cfg.get("stocks") or []:
        role = s.get("role")
        if role and role not in kb.ROLES:
            raise ValueError(f"个股 {s.get('name')} role '{role}' 非法 {kb.ROLES}")
        manual.append({
            "concept_code": code, "ts_code": s.get("ts_code"),
            "name": s.get("name"), "chain_node": s.get("chain_node"),
            "role": role, "moat": s.get("moat"),
            "realize_cycle": s.get("realize_cycle"),
            "competition": s.get("competition"), "ceiling": s.get("ceiling"),
            "benefit_purity": s.get("benefit_purity"),
            "four_dim_score": s.get("four_dim_score"),
        })
    manual_codes = {s.get("ts_code") for s in manual if s.get("ts_code")}
    stocks = manual + autofill_stocks(cfg, manual_codes)

    ref = cfg["playbook_ref"]
    signals = []
    for g in cfg.get("signals") or []:
        st = g.get("type")
        if st not in kb.SIGNAL_TYPES:
            raise ValueError(f"信号 type '{st}' 非法 {kb.SIGNAL_TYPES}")
        if g.get("phase") and g["phase"] not in kb.STAGES:
            raise ValueError(f"信号 phase '{g['phase']}' 非法 {kb.STAGES}")
        signals.append({"playbook_ref": ref, "phase": g.get("phase"),
                        "signal": g.get("signal"), "type": st,
                        "confidence": g.get("confidence")})

    similar = []
    for e in cfg.get("similar") or []:
        similar.append({"a_code": code, "a_name": theme_row["name"],
                        "b_code": e.get("b_code"),
                        "b_name": e.get("b_name"),
                        "sim_score": e.get("sim_score"),
                        "sim_driver": e.get("sim_driver"),
                        "sim_chain": e.get("sim_chain"),
                        "sim_flow": e.get("sim_flow")})

    return {
        "theme": pd.DataFrame([theme_row], columns=kb.KB_THEME_COLS),
        "stock": pd.DataFrame(stocks, columns=kb.KB_STOCK_COLS),
        "signal": pd.DataFrame(signals, columns=kb.KB_SIGNAL_COLS),
        "similarity": pd.DataFrame(similar, columns=kb.KB_SIM_COLS),
    }


# ---------- upsert ----------

def _upsert(ds_name: str, new: pd.DataFrame, cols: list, keys: list) -> int:
    """按 keys 去重合并进现有数据集(旧记录被同键新记录替换), 写回。"""
    if new.empty:
        return 0
    try:
        old = load(ds_name)
        for c in cols:
            if c not in old.columns:
                old[c] = None
        old = old[cols]
    except Exception:
        old = kb._empty(cols)
    if old.empty:
        merged = new[cols].reset_index(drop=True)
    else:
        drop = old.merge(new[keys].drop_duplicates(), on=keys, how="inner")
        if len(drop):
            old = old.merge(drop[keys].drop_duplicates(), on=keys,
                            how="left", indicator=True)
            old = old[old["_merge"] == "left_only"].drop(columns="_merge")
        merged = pd.concat([old, new[cols]], ignore_index=True)
    save(ds_name, merged)
    return len(new)


def write_kb(cfg: dict) -> dict:
    rows = build_rows(cfg)
    code = cfg["concept_code"]
    ref = cfg["playbook_ref"]
    n = {}
    n["theme"] = _upsert("theme.kb_theme", rows["theme"],
                         kb.KB_THEME_COLS, ["concept_code"])
    n["stock"] = _upsert("theme.kb_stock", rows["stock"],
                         kb.KB_STOCK_COLS, ["concept_code"])
    n["signal"] = _upsert("theme.kb_signal", rows["signal"],
                          kb.KB_SIGNAL_COLS, ["playbook_ref"])
    n["similarity"] = _upsert("theme.kb_similarity", rows["similarity"],
                              kb.KB_SIM_COLS, ["a_code"])
    print(f"[build_theme_kb] {code}({cfg.get('name')}) 入库: "
          f"档案{n['theme']} 个股{n['stock']} 信号{n['signal']} 相似{n['similarity']}")
    return n


def emit_playbook_skeleton(cfg: dict) -> Path:
    """写剧本 markdown 骨架(若已存在则不覆盖, 保护人工正文)。"""
    kb.PLAYBOOK_DIR.mkdir(parents=True, exist_ok=True)
    p = kb.PLAYBOOK_DIR / f"{cfg['playbook_ref']}.md"
    if p.exists():
        print(f"[build_theme_kb] 剧本已存在, 跳过: {p}")
        return p
    axis = cfg.get("stage_axis") or derive_stage_axis(
        cfg["concept_code"], cfg.get("window"),
        pool=kpl_theme_pool(cfg.get("member_sources") or []))
    basket_axis = bool(axis and "med_nv" in axis[0])
    lines = [
        f"---",
        f"playbook_ref: {cfg['playbook_ref']}",
        f"concept_code: {cfg['concept_code']}",
        f"name: {cfg.get('name')}",
        f"driver_type: {cfg['driver_type']}",
        f"parent_theme: {cfg.get('parent_theme')}",
        f"catalyst_conf: {cfg.get('catalyst_conf')}",
        f"---",
        "",
        f"# {cfg.get('name')} · {cfg['driver_type']}型剧本",
        "",
        "## 第一催化",
        cfg.get("first_catalyst") or "_(待填)_",
        "",
        "## 生命周期阶段轴(数据驱动自动推导, 待人工校验)",
        "",
    ]
    if basket_axis:
        lines += ["口径: 篮子中位净值(kpl成分等权归一, 抗单只异常)。", "",
                  "| 阶段 | 日期 | 中位净值 |", "|---|---|---|"]
        for a in axis:
            lines.append(f"| {a['stage']} | {a['date']} | {a.get('med_nv')} |")
    else:
        lines += ["口径: theme.day 涨停数(连板驱动题材降级口径)。", "",
                  "| 阶段 | 日期 | 涨停数 | 最高板 | 龙头 |",
                  "|---|---|---|---|---|"]
        for a in axis:
            lines.append(f"| {a['stage']} | {a['date']} | {a.get('zt_cnt')} "
                         f"| {a.get('max_height')} | {a.get('leader_name')} |")
    lines += [
        "",
        "## 发酵路径", "_(待填: 谁先动→谁跟进→谁补涨→扩散)_", "",
        "## 退潮先行信号", "_(见 kb_signal)_", "",
        "## 环境前提", cfg.get("env_precondition") or "_(待填)_", "",
        "## 证伪条款", cfg.get("falsification") or "_(待填)_", "",
        "## 可复制性判定", "_(待填: 通用剧本 or 一次性特例)_", "",
    ]
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"[build_theme_kb] 剧本骨架写出: {p}")
    return p


def cli() -> int:
    ap = argparse.ArgumentParser(description="题材知识库生成器")
    ap.add_argument("--config", help="题材 config JSON 路径")
    ap.add_argument("--emit-playbook", action="store_true",
                    help="顺带写剧本 markdown 骨架(不覆盖已存在)")
    ap.add_argument("--template", action="store_true", help="打印空白config模板")
    args = ap.parse_args()
    if args.template:
        print(json.dumps(TEMPLATE, ensure_ascii=False, indent=2))
        return 0
    if not args.config:
        ap.error("需 --config 或 --template")
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    write_kb(cfg)
    if args.emit_playbook:
        emit_playbook_skeleton(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(cli())
