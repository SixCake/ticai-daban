# -*- coding: utf-8 -*-
"""题材知识库单测（theme-research 知识库 Spec Test Plan）

覆盖:
  1. schema 枚举/列常量自洽（core/theme_kb.py 唯一出处）
  2. infer_stage 波龄驱动与热度驱动两条路径（启发式 v0）
  3. match: 单份剧本 KB 下命中/未命中/冷题材过滤
  4. response_chain: 龙头按受益纯度排序、当前+下一阶段信号、剧本正文
  5. build_theme_kb: driver_type 校验、derive_stage_axis 返回结构
  6. 集成: 真实 KB(存储样板)已入库时 match 命中且 radar 快照 JSON 无 NaN

运行: .venv/bin/python tests/test_theme_kb.py
不依赖 pytest（纯 assert + 退出码），与 test_ticai_data_source.py 同风格。
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import theme_kb as kb  # noqa: E402
from core.heat import HOT_THRESHOLD  # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    if cond:
        print(f"  [PASS] {name}")
    else:
        print(f"  [FAIL] {name} {detail}")
        FAIL.append(name)


def _synthetic_kb() -> dict:
    """构造单份剧本的内存 KB（不依赖磁盘, 隔离测试 match/response_chain）"""
    theme = pd.DataFrame([{
        "concept_code": "测试题材", "name": "测试题材", "driver_type": "周期",
        "parent_theme": "测试父题", "first_catalyst": "现货价上涨",
        "catalyst_conf": "可确认",
        "stage_axis": json.dumps([{"stage": "启动", "date": "20250101"}],
                                 ensure_ascii=False),
        "falsification": "价格见顶回落则不适用", "env_precondition": "寡头控价",
        "playbook_ref": "测试剧本",
    }], columns=kb.KB_THEME_COLS)
    stock = pd.DataFrame([
        {"concept_code": "测试题材", "ts_code": "A.SH", "name": "甲",
         "chain_node": "中游设计", "role": "中军", "moat": "x",
         "realize_cycle": "x", "competition": "x", "ceiling": "x",
         "benefit_purity": 0.9, "four_dim_score": None},
        {"concept_code": "测试题材", "ts_code": "B.SZ", "name": "乙",
         "chain_node": "下游分销", "role": "龙头", "moat": "x",
         "realize_cycle": "x", "competition": "x", "ceiling": "x",
         "benefit_purity": 0.6, "four_dim_score": None},
        {"concept_code": "测试题材", "ts_code": "C.SZ", "name": "丙",
         "chain_node": "上游设备", "role": "跟随", "moat": None,
         "realize_cycle": None, "competition": None, "ceiling": None,
         "benefit_purity": None, "four_dim_score": None},
    ], columns=kb.KB_STOCK_COLS)
    signal = pd.DataFrame([
        {"playbook_ref": "测试剧本", "phase": "高潮", "signal": "一波龙头见顶",
         "type": "退潮", "confidence": "高"},
        {"playbook_ref": "测试剧本", "phase": "分歧", "signal": "中军先杀",
         "type": "退潮", "confidence": "高"},
        {"playbook_ref": "测试剧本", "phase": "退潮", "signal": "产业资本减持",
         "type": "退潮", "confidence": "中"},
    ], columns=kb.KB_SIGNAL_COLS)
    sim = pd.DataFrame([{
        "a_code": "测试题材", "a_name": "测试题材", "b_code": "同构题材",
        "b_name": "历史同构", "sim_score": 0.8, "sim_driver": "高",
        "sim_chain": "中", "sim_flow": "高",
    }], columns=kb.KB_SIM_COLS)
    return {"theme": theme, "stock": stock, "similarity": sim,
            "signal": signal, "playbooks": {"测试剧本": "# 测试剧本正文"}}


def main() -> int:
    syn = _synthetic_kb()

    print("1) schema 枚举/列常量")
    check("DRIVER_TYPES 六类", len(kb.DRIVER_TYPES) == 6, kb.DRIVER_TYPES)
    check("STAGES 六阶段", kb.STAGES == ["潜伏", "启动", "发酵", "高潮", "分歧", "退潮"])
    check("ROLES 含龙头/中军/补涨/影子", set(kb.ROLES) == {"龙头", "中军", "补涨", "影子"})
    check("KB_THEME_COLS 含证伪条款", "falsification" in kb.KB_THEME_COLS)
    check("空表构造列齐全", list(kb._empty(kb.KB_STOCK_COLS).columns) == kb.KB_STOCK_COLS)

    print("2) infer_stage 阶段判定")
    hot = {"heat": HOT_THRESHOLD * 3, "zt": 20, "nmem": 100}
    cold = {"heat": 1, "zt": 0, "nmem": 100}
    check("波龄1+热 → 发酵", kb.infer_stage(hot, age=1) == "发酵")
    check("波龄1+冷 → 启动", kb.infer_stage(cold, age=1) == "启动",
          kb.infer_stage(cold, age=1))
    check("波龄5+冷 → 退潮", kb.infer_stage(cold, age=5) == "退潮")
    check("波龄5+热 → 分歧", kb.infer_stage(hot, age=5) == "分歧")
    check("无波龄+高密度 → 高潮",
          kb.infer_stage({"heat": HOT_THRESHOLD * 3, "zt": 30, "nmem": 100}) == "高潮")
    check("无波龄+冷 → 启动", kb.infer_stage(cold) == "启动")

    print("3) match 命中/未命中/冷题材")
    r = kb.match([{"concept_code": "测试题材", "name": "测试题材",
                   "heat": HOT_THRESHOLD * 2, "zt": 8, "nmem": 50}], syn)
    check("热题材命中单份剧本", len(r) == 1 and r[0]["match_way"] == "exact", r)
    check("冷题材被过滤(heat<阈值)",
          kb.match([{"concept_code": "测试题材", "name": "测试题材",
                     "heat": 1, "zt": 0, "nmem": 50}], syn) == [])
    check("未入库题材不命中",
          kb.match([{"concept_code": "无关", "name": "无关题材",
                     "heat": HOT_THRESHOLD * 2, "zt": 8, "nmem": 50}], syn) == [])
    check("空 KB 返回空", kb.match([{"concept_code": "x", "name": "x",
                                     "heat": 99, "zt": 9, "nmem": 50}],
                                    {"theme": kb._empty(kb.KB_THEME_COLS),
                                     "stock": syn["stock"],
                                     "similarity": syn["similarity"],
                                     "signal": syn["signal"],
                                     "playbooks": {}}) == [])

    print("4) response_chain 龙头排序/阶段信号/正文")
    row = syn["theme"].iloc[0]
    resp = kb.response_chain(syn, row, "高潮")
    check("龙头候选按受益纯度降序",
          [l["name"] for l in resp["leaders"]] == ["甲", "乙"], resp["leaders"])
    check("跟随股不入龙头候选",
          all(l["role"] in ("龙头", "中军") for l in resp["leaders"]))
    check("高潮阶段 → 含高潮+分歧信号(不含退潮)",
          {s["phase"] for s in resp["signals"]} == {"高潮", "分歧"},
          [s["phase"] for s in resp["signals"]])
    resp2 = kb.response_chain(syn, row, "潜伏")
    check("无匹配阶段 → 回退全部退潮信号",
          len(resp2["signals"]) == 3, resp2["signals"])
    check("剧本正文可读", resp["playbook"] == "# 测试剧本正文")
    check("证伪条款透传", resp["falsification"] == "价格见顶回落则不适用")

    print("5) build_theme_kb 校验/阶段轴")
    import collect.build_theme_kb as btk
    try:
        btk._validate({"concept_code": "x", "playbook_ref": "y",
                       "driver_type": "非法类型", "catalyst_conf": "可确认"})
        check("非法 driver_type 应报错", False)
    except ValueError:
        check("非法 driver_type 报错", True)
    try:
        btk._validate({"concept_code": "x", "playbook_ref": "y",
                       "driver_type": "周期", "catalyst_conf": "非法"})
        check("非法 catalyst_conf 应报错", False)
    except ValueError:
        check("非法 catalyst_conf 报错", True)
    axis = btk.derive_stage_axis("不存在的题材XYZ", ["20250101", "20250201"])
    check("未知题材阶段轴为空列表", axis == [], axis)
    check("TEMPLATE 含 stocks/signals/similar",
          all(k in btk.TEMPLATE for k in ("stocks", "signals", "similar")))

    print("6) 集成: 真实 KB(存储样板)")
    real = kb.load_kb()
    if real["theme"].empty:
        print("  [SKIP] 真实 KB 未入库(存储样板未跑 build_theme_kb)")
    else:
        rr = kb.match([{"concept_code": "存储", "name": "存储",
                        "heat": HOT_THRESHOLD * 3, "zt": 9, "nmem": 60}], real)
        check("存储样板命中", len(rr) == 1 and rr[0]["matched_theme"], rr)
        if rr:
            check("存储剧本含相似题材", len(rr[0]["similar"]) >= 1)
            check("存储龙头候选非空", len(rr[0]["response"]["leaders"]) >= 3)
        # radar 快照投影 JSON 无 NaN
        try:
            import apps.radar as R
            rad = R.Radar()
            snap = rad._kb_snapshot(
                [{"concept_code": "存储", "name": "存储",
                  "heat": HOT_THRESHOLD * 3, "zt": 9, "nmem": 60}], "20260904")
            json.dumps(snap, ensure_ascii=False, allow_nan=False)
            check("radar kb_response JSON 无 NaN", snap is not None)
            check("剧本正文不入 radar 快照(精简)",
                  all("playbook" not in s for s in (snap or [])))
        except Exception as e:
            check("radar kb_response 可构建", False, f"{type(e).__name__}: {e}")

    print("7) 修复验证: 纯净池/篮子净值/数据驱动阶段轴/role时序")
    try:
        import collect.build_theme_kb as btk2
        import apps.review as RV
        from datastore import load as _ld
        pool = btk2.kpl_theme_pool(["半导体存储器"])
        check("kpl_theme_pool 返回成分", len(pool) >= 20, len(pool))
        check("池剔除 .BJ", all(not str(c).endswith(".BJ") for c in pool))
        contam = {"603156.SH", "002208.SZ", "600999.SH"}  # 养元饮品/合肥城建/招商证券
        check("池无事件级直标污染票", not (set(pool) & contam), set(pool) & contam)
        panel = _ld("market.daily_panel", columns=["trade_date", "ts_code", "close"])
        bc = btk2.basket_curve(pool, "20240102", "20260903", panel=panel)
        check("篮子曲线有 mean/med 净值",
              len(bc) > 100 and {"mean_nv", "med_nv"} <= set(bc.columns))
        check("中位净值抗异常(峰值<均值峰值)",
              float(bc["med_nv"].max()) < float(bc["mean_nv"].max()))
        axis = btk2.derive_stage_axis_basket(pool, ["20240102", "20260903"],
                                             panel=panel)
        stages = [a["stage"] for a in axis]
        check("净值阶段轴含潜伏/启动/高潮/退潮",
              {"潜伏", "启动", "高潮", "退潮"} <= set(stages), stages)
        check("阶段轴带 med_nv 字段", all("med_nv" in a for a in axis))
        hi = next(a for a in axis if a["stage"] == "高潮")
        check("高潮=中位净值峰值日",
              hi["date"] == str(bc.loc[bc["med_nv"].idxmax(), "date"]), hi)
        wb = RV.build_theme_replay("存储", "20240102", "20260903",
                                   member_sources=["半导体存储器"])
        check("复盘工作簿无 error", "error" not in wb, wb.get("error"))
        check("池来源=kpl(非降级)", wb.get("pool_source") == "kpl",
              wb.get("pool_source"))
        check("股池带 peak_offset_days(时序去前视)",
              all("peak_offset_days" in r for r in wb["pool"]))
        check("龙头峰值贴近篮子峰(|offset|<=10)",
              all(abs(r["peak_offset_days"] or 99) <= 10
                  for r in wb["pool"] if r["role"] == "龙头"))
        check("影子=弱势(全程涨幅<30%)",
              all(r["total_pct"] < 30 for r in wb["pool"] if r["role"] == "影子"))
        kt = _ld("theme.kb_theme")
        row = kt[kt["concept_code"] == "存储"].iloc[0]
        check("存储档案 catalyst_conf=中(诚实非可确认)",
              row["catalyst_conf"] == "中", row["catalyst_conf"])
        ax = json.loads(row["stage_axis"])
        check("入库阶段轴为净值口径(med_nv)", bool(ax) and "med_nv" in ax[0])
    except Exception as e:
        check("修复验证组可运行", False, f"{type(e).__name__}: {e}")

    print("8) 情绪型复盘(review.py --limit, kpl涨停梯队)")
    try:
        import apps.review as RV2
        wb = RV2.build_theme_limit_replay("猪", "20190101", "20191231")
        check("情绪型复盘无error", "error" not in wb, wb.get("error"))
        check("口径=kpl情绪型", wb.get("caliber") == "kpl情绪型(涨停梯队)")
        check("龙头top1=新五丰(涨停频次)",
              bool(wb["leaders"]) and wb["leaders"][0]["name"] == "新五丰",
              wb["leaders"][0]["name"] if wb["leaders"] else None)
        check("股池无ST/退(剔除)",
              not any("ST" in r["name"] or "退" in r["name"] for r in wb["pool"]))
        check("龙头名称补全(无裸代码)",
              all(r["name"] != r["ts_code"] for r in wb["leaders"]))
        ax = wb["stage_axis"]
        stages = {a["stage"]: a["date"] for a in ax}
        check("阶段轴启动≠高潮(滚动窗口避免退化)",
              stages.get("启动") != stages.get("高潮"), stages)
        check("高潮带roll5_cnt(滚动累计)",
              any("roll5_cnt" in a for a in ax))
        err = RV2.build_theme_limit_replay("不存在题材XYZ", "20190101", "20191231")
        check("无匹配题材返回error", "error" in err)
    except Exception as e:
        check("情绪型复盘组可运行", False, f"{type(e).__name__}: {e}")

    print()
    if FAIL:
        print(f"FAILED: {len(FAIL)} 项 -> {FAIL}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
