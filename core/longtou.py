# -*- coding: utf-8 -*-
"""龙头因子复合评分（研究22/23前向验证产物）— 唯一出处

qscore 次日质量分(0-4): 次日胜率/盈亏比分层, test 72.7%→85.5%
  = (ldlr_prev<0.5) + (ind_rank>3.5) + (zb_cnt20≤1.5) + (0.55<y_volr5≤2.2)
sscore 封板概率分(0-5): 封板率分层, test 23.3%→60.2%
  = (zb_cnt20≥0.5) + (ind_ztdens≥0.03) + (ind_rank≤3.5)
    + (y_volr5<2.5) + (ind_breadth≥0.65)
env_status 市场环境: ldlr_prev≥0.5 → 环境降权(仅雷达预警层展示,
  研究23证实对主力规则池无增量, 不作硬闸门)

阈值冻结自 research/23_combo_gate.py L122-127/L157-163;
因子口径(近似涨跌停价/vol量比/events行业映射)见 collect/factor_longtou.py。
证伪项(禁用): 行业排名前3作次日质量加分(longtou龙头稀缺性在打板场景
方向相反)、ld_prev绝对家数、cycle_prev/adv_prev 单独作闸门。
"""

FACTOR_CONTRACT = "longtou-factor-r1-20260827"

# 逐档校准概率（research/out/23_combo_gate.md test段, 冻结不改）
# qscore → 次日胜率(主力规则池test); 同档盈亏比 1.91/1.92/2.26/2.47
QSCORE_NEXT_WIN = {1: 0.73, 2: 0.81, 3: 0.79, 4: 0.86}
# sscore → 封板率(决策时刻→收盘封死, test); 注意次日胜率随sscore降
# (0.84→0.67): 聚焦度换次日弹性——高封档宜日内溢价, 高质量档才隔夜
SSCORE_SEAL = {0: 0.28, 1: 0.23, 2: 0.32, 3: 0.41, 4: 0.50, 5: 0.60}


def qscore_of(row) -> int | None:
    """次日质量分0-4; 任一成分缺失返回None(不伪装为低分)"""
    try:
        ldlr = row["ldlr_prev"]
        rank = row["ind_rank"]
        zb = row["zb_cnt20"]
        vr = row["y_volr5"]
    except (KeyError, TypeError, IndexError):
        return None
    vals = (ldlr, rank, zb, vr)
    if any(v is None or v != v for v in vals):   # None 或 NaN
        return None
    return (int(ldlr < 0.5) + int(rank > 3.5) + int(zb <= 1.5)
            + int(0.55 < vr <= 2.2))


def sscore_of(row) -> int | None:
    """封板概率分0-5; 任一成分缺失返回None"""
    try:
        zb = row["zb_cnt20"]
        ztd = row["ind_ztdens"]
        rank = row["ind_rank"]
        vr = row["y_volr5"]
        brd = row["ind_breadth"]
    except (KeyError, TypeError, IndexError):
        return None
    vals = (zb, ztd, rank, vr, brd)
    if any(v is None or v != v for v in vals):
        return None
    return (int(zb >= 0.5) + int(ztd >= 0.03) + int(rank <= 3.5)
            + int(vr < 2.5) + int(brd >= 0.65))


def env_status(market: dict) -> dict:
    """市场环境判定。market 需含 zt_prev/ld_prev/ldlr_prev(决策日T可见的
    T-1全市场统计); 缺失返回 status='数据不足'。downweight 仅供展示降权,
    不是硬闸门(研究23: 主力规则池盘中条件已内生过滤坏日)。"""
    if not market or market.get("ldlr_prev") is None:
        return {"status": "数据不足", "downweight": False, "reason": "昨日市场生态字段缺失"}
    ldlr = market["ldlr_prev"]
    if ldlr >= 0.5:
        return {"status": "环境降权", "downweight": True,
                "reason": f"昨日跌停达涨停{ldlr:.0%}(≥50%), 雷达层降权展示"}
    return {"status": "正常", "downweight": False,
            "reason": f"昨日跌停/涨停比{ldlr:.2f}"}
