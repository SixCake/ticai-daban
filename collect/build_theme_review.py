# -*- coding: utf-8 -*-
"""题材复盘工作簿生成器（三层结合复盘法的执行脚本）

方法论见 docs/题材复盘方法论.md。本脚本是"数据层"自动化引擎 + 工作簿合成:
  数据层(自动): 全池/核心篮子净值、纯度溢价、阶段轴、各阶段涨幅、涨停梯队
  催化层(config): 第一催化/逐日事件/信源(WebSearch检索后填入config)
  深度层(config): 驱动链/产业链拆解/四维穿透/角色/证伪(人工/LLM填入config)

产出多 sheet xlsx 工作簿(对齐参考级复盘结构):
  题材档案卡 / 角色队列 / 逐日还原表 / 阶段统计 / 剧本提炼 / 全池行情 / 信源

用法:
  python collect/build_theme_review.py --config data/theme/review_configs/存储超级周期.json
  python collect/build_theme_review.py --config <cfg> --out <xlsx路径>

config 结构(定性字段人工/LLM填, 量价字段脚本自动算):
  concept_code/name/window/member_sources/core_stocks(核心股ts_code列表)
  driver_type/parent_theme/first_catalyst/catalyst_conf/driver_chain(驱动链)
  industry_chain(产业链拆解)/playbook(剧本字段)/sources(信源)/daily_events(逐日事件)
  roles(核心股四维穿透: 壁垒/兑现/格局/天花板)
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from datastore import load  # noqa: E402
from collect.build_theme_kb import (kpl_theme_pool, basket_curve,  # noqa: E402
                                    derive_stage_axis_basket)
from apps.review import build_theme_limit_replay  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "data" / "review"


def _norm_pct(curve: pd.DataFrame) -> float:
    """篮子全程涨幅%(末/首-1)"""
    if curve is None or not len(curve):
        return 0.0
    nv = curve["med_nv"].tolist()
    return round((nv[-1] / nv[0] - 1) * 100, 1) if nv[0] else 0.0


def _stage_returns(pool_curve, axis, codes, start, end):
    """各阶段涨幅: 每只股在每个阶段区间的涨幅(角色队列用)。"""
    if not axis:
        return {}
    panel = load("market.daily_panel", columns=["trade_date", "ts_code", "close"])
    panel = panel[(panel["ts_code"].isin(codes)) & (panel["trade_date"] >= start)
                  & (panel["trade_date"] <= end)]
    piv = panel.pivot_table(index="trade_date", columns="ts_code", values="close")
    bounds = [(a["stage"], a["date"]) for a in axis]
    out = {}
    for c in codes:
        if c not in piv.columns:
            continue
        s = piv[c].dropna()
        if len(s) < 2:
            continue
        rets = {}
        for i, (stage, d0) in enumerate(bounds):
            d1 = bounds[i + 1][1] if i + 1 < len(bounds) else end
            seg = s[(s.index >= d0) & (s.index < d1)]
            if len(seg) >= 2 and seg.iloc[0]:
                rets[stage] = round((seg.iloc[-1] / seg.iloc[0] - 1) * 100, 1)
        out[c] = rets
    return out


def build_review_workbook(cfg: dict) -> dict:
    """合成复盘工作簿(数据层自动算 + config定性字段)。"""
    code = cfg["concept_code"]
    start, end = cfg["window"]
    srcs = cfg.get("member_sources") or [code]
    pool = kpl_theme_pool(srcs)
    if not pool:
        return {"error": f"题材 '{code}' 无成分(kpl未命中)"}
    core = cfg.get("core_stocks") or []

    # 数据层: 篮子净值(全池 + 核心) + 纯度溢价
    pool_curve = basket_curve(pool, start, end)
    core_curve = basket_curve(core, start, end) if core else None
    pool_pct = _norm_pct(pool_curve)
    core_pct = _norm_pct(core_curve)
    purity_premium = round(core_pct - pool_pct, 1) if core else None

    # 数据层: 阶段轴(篮子中位净值拐点)
    axis = derive_stage_axis_basket(pool, [start, end])

    # 数据层: 各阶段涨幅(角色队列)
    stage_ret = _stage_returns(pool_curve, axis, core or pool, start, end)

    # 数据层: 涨停梯队(情绪型)
    limit = build_theme_limit_replay(code, start, end)
    limit_daily = limit.get("daily", []) if "error" not in limit else []

    # 全池行情(归一化净值表)
    pool_nv = []
    if len(pool_curve):
        for _, r in pool_curve.iterrows():
            pool_nv.append({"date": str(r["date"]),
                            "pool_med_nv": round(float(r["med_nv"]), 4),
                            "pool_mean_nv": round(float(r["mean_nv"]), 4),
                            "n": int(r["n"])})

    return {
        "concept_code": code, "name": cfg.get("name"), "window": [start, end],
        "pool_source": srcs, "n_pool": len(pool), "n_core": len(core),
        # 数据层
        "pool_pct": pool_pct, "core_pct": core_pct,
        "purity_premium": purity_premium,
        "stage_axis": axis, "stage_returns": stage_ret,
        "pool_nv": pool_nv, "limit_daily": limit_daily,
        "limit_leaders": limit.get("leaders", []) if "error" not in limit else [],
        # 催化层(config)
        "first_catalyst": cfg.get("first_catalyst"),
        "catalyst_conf": cfg.get("catalyst_conf"),
        "daily_events": cfg.get("daily_events") or [],
        "sources": cfg.get("sources") or [],
        # 深度层(config)
        "driver_type": cfg.get("driver_type"), "parent_theme": cfg.get("parent_theme"),
        "driver_chain": cfg.get("driver_chain"),
        "industry_chain": cfg.get("industry_chain"),
        "roles": cfg.get("roles") or [], "playbook": cfg.get("playbook") or {},
    }


def write_xlsx(wb: dict, path: Path):
    """写多 sheet xlsx 工作簿(对齐参考级复盘结构)。"""
    if "error" in wb:
        print(f"[ERROR] {wb['error']}")
        return
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        # 1 题材档案卡
        prof = [["题材名称", wb["name"]], ["题材分类", wb["driver_type"]],
                ["父题材", wb["parent_theme"]],
                ["股票池", f"kpl {wb['pool_source']} 共{wb['n_pool']}只; 核心{wb['n_core']}只"],
                ["第一催化", wb["first_catalyst"]], ["置信度", wb["catalyst_conf"]],
                ["核心驱动链", wb["driver_chain"]], ["产业链拆解", wb["industry_chain"]],
                ["生命周期", " → ".join(f"{a['stage']} {a['date']}" for a in wb["stage_axis"])],
                ["全程表现", f"核心{wb['core_pct']}% / 全池{wb['pool_pct']}% / 纯度溢价{wb['purity_premium']}pp"],
                ["可复制性", wb["playbook"].get("可复制性")
                 or wb["playbook"].get("replicability")],
                ["证伪条款", wb["playbook"].get("falsification")
                 or wb["playbook"].get("证伪条款")]]
        pd.DataFrame(prof, columns=["项", "内容"]).to_excel(xw, "题材档案卡", index=False)
        # 2 角色队列(四维穿透 + 各阶段涨幅)
        roles = []
        for r in wb["roles"]:
            sr = wb["stage_returns"].get(r["ts_code"], {})
            roles.append({"股票": r.get("name"), "角色": r.get("role"),
                          "产业链环节": r.get("chain_node"), "核心壁垒": r.get("moat"),
                          "业绩兑现周期": r.get("realize_cycle"),
                          "竞争格局": r.get("competition"), "天花板": r.get("ceiling"),
                          **{f"{k}%": v for k, v in sr.items()}})
        pd.DataFrame(roles).to_excel(xw, "角色队列", index=False)
        # 3 逐日还原表(数据层篮子 + 催化层事件)
        ev_map = {e["date"]: e for e in wb["daily_events"]}
        daily = []
        for d in wb["limit_daily"]:
            e = ev_map.get(d["date"], {})
            daily.append({"日期": d["date"], "涨停家数": d["zt_cnt"],
                          "最高连板": d["max_height"],
                          "领涨股": d["lead"]["name"] if d["lead"] else None,
                          "当日事件": e.get("event"), "当日叙事": e.get("narrative")})
        pd.DataFrame(daily).to_excel(xw, "逐日还原表", index=False)
        # 4 阶段统计
        pd.DataFrame(wb["stage_axis"]).to_excel(xw, "阶段统计", index=False)
        # 5 剧本提炼
        pb = [[k, v] for k, v in wb["playbook"].items()]
        pd.DataFrame(pb, columns=["项", "内容"]).to_excel(xw, "剧本提炼", index=False)
        # 6 全池行情
        pd.DataFrame(wb["pool_nv"]).to_excel(xw, "全池行情", index=False)
        # 7 信源
        pd.DataFrame(wb["sources"], columns=["信息项", "来源"]).to_excel(
            xw, "信源", index=False)
    print(f"[build_theme_review] 工作簿写出: {path}")
    print(f"  题材={wb['name']} 池={wb['n_pool']}只 核心={wb['n_core']}只 "
          f"纯度溢价={wb['purity_premium']}pp 阶段={len(wb['stage_axis'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8"))
    wb = build_review_workbook(cfg)
    if "error" in wb:
        print(f"[ERROR] {wb['error']}")
        return
    out = Path(a.out) if a.out else OUT / f"theme_review_{cfg['concept_code']}.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_xlsx(wb, out)


if __name__ == "__main__":
    main()
