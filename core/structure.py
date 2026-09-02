# -*- coding: utf-8 -*-
"""结构层评分 V5（研究22/23/24/24b 定稿）— 影子输出, 不改触发逻辑

两层架构的结构层: g_chip 结构闸(≥0.6 合格) + V5 融合分排序
(0.5盘中 + 0.25生态 + 0.25筹码)。盘中组 r3/pathvol 取自 early_signal
轨迹(调用方传入), 其余为 T-1 日线结构因子, 盘前每日一次计算。

因子定义与阈值全部来自 research/24_fusion_ablation.py(train 拟合):
  g_chip = mean(ldlr_prev<0.5, ind_rank>3.5, zb_cnt20≤1.5,
                y_volr5∈(0.55,2.2], neg_streak≥2.5)
  g_eco  = mean(pct(ind_ztdens), 1-pct(ind_rank), pct(ind_breadth),
                pct(zb_cnt20), vol甜蜜区)
  g_intra= mean(pct(r3), pct(pathvol))
  V5     = 0.5*g_intra + 0.25*g_eco + 0.25*g_chip
分位网格 pct() 固化在 data/meta/struct_grids.json(train 200分位降采样),
盘中/实时不得重新拟合。行业映射用 events 最新口径(data/meta/industry_map.json),
ind_rank/breadth/ztdens 在雷达宇宙行业内计算(研究口径为全市场, 影子期近似)。
ldlr_prev 用昨日东财涨停池/跌停池家数比, 拉取失败时该项弃用并按
可用项重归一(不伪装为通过)。
"""
import json
from bisect import bisect_right
from pathlib import Path

from config import DATA

META = DATA / "meta"
CHIP_GATE = 0.6          # 结构闸阈值(研究24b V5)
ZB_RATE = {"30": 0.20, "68": 0.20}   # 涨停幅近似(未做ST修正, 同研究22口径)

_grids: dict | None = None
_ind_map: dict | None = None


def _load_grids() -> dict:
    global _grids
    if _grids is None:
        _grids = json.loads((META / "struct_grids.json").read_text())["grids"]
    return _grids


def load_industry_map() -> dict:
    """{ts_code: industry}, events 最新口径; 缺失返回空(结构分降级)"""
    global _ind_map
    if _ind_map is None:
        f = META / "industry_map.json"
        _ind_map = json.loads(f.read_text()) if f.exists() else {}
    return _ind_map


def pct(col: str, v: float, invert: bool = False) -> float:
    """train 固化分位(200点网格 searchsorted)"""
    grid = _load_grids()[col]
    p = bisect_right(grid, v) / len(grid)
    return 1.0 - p if invert else p


def _limit_rate(code: str) -> float:
    return ZB_RATE.get(str(code)[:2], 0.10)


def stock_struct(code: str, bars: list) -> dict | None:
    """单票 T-1 结构因子。bars=[(date, high, low, close, vol)] 升序,
    最后一根非今日 bar 为昨日。返回 None=数据不足(禁止补缺)"""
    today = None
    if bars:
        import datetime as _dt
        today = _dt.datetime.now().strftime("%Y%m%d")
    rows = [b for b in bars if b[0] != today]
    if len(rows) < 21:
        return None
    yest = rows[-1]
    if yest[3] <= 0 or yest[4] is None:
        return None
    # zb_cnt20: 近20日触板未封天数(涨停价=前收×(1+rate)近似)
    zb = 0
    for i in range(len(rows) - 20, len(rows)):
        d, hi, lo, cl, vol = rows[i]
        pre = rows[i - 1][3] if i > 0 else None
        if not pre or pre <= 0 or hi is None or cl is None:
            continue
        lp = round(pre * (1 + _limit_rate(code)), 2)
        if hi >= lp * 0.999 and cl < lp * 0.999:
            zb += 1
    # y_volr5: 昨日量 / 前5日均量
    vols = [r[4] for r in rows[-6:-1] if r[4] is not None and r[4] > 0]
    vma5 = sum(vols) / len(vols) if len(vols) >= 3 else None
    volr5 = yest[4] / vma5 if vma5 and yest[4] is not None else None
    # neg_streak: 截至昨日连续收跌天数
    streak = 0
    for i in range(len(rows) - 1, 0, -1):
        if rows[i][3] < rows[i - 1][3]:
            streak += 1
        else:
            break
    return {"zb_cnt20": zb, "y_volr5": volr5, "neg_streak": streak,
            "y_pct": (yest[3] / rows[-2][3] - 1) * 100 if rows[-2][3] > 0
            else 0.0}


def build_struct_scores(codes: list, bars_by_code: dict,
                        ldlr_prev: float | None) -> dict:
    """{code: {g_chip, gate, v5_base, comps...}}。
    v5_base = 0.25*g_eco + 0.25*g_chip(不含盘中组); 调用方叠加
    0.5*g_intra(r3/pathvol 分位)得完整 V5 融合分。
    ldlr_prev 为全市场常量无截面区分度(研究中是跨时间的坏日闸门),
    不进 g_chip, 仅随输出透传供调用方另行降权; 行业缺失项弃用重归一。"""
    ind_map = load_industry_map()
    # 行业昨日统计(雷达宇宙内, 影子期近似口径)
    ystats: dict = {}
    for c in codes:
        s = stock_struct(c, bars_by_code.get(c, []))
        if s is None:
            continue
        ind = ind_map.get(c)
        if ind:
            ystats.setdefault(ind, []).append((c, s["y_pct"]))
    ind_ctx = {}
    for ind, pairs in ystats.items():
        n = len(pairs)
        if n < 5:          # 成员<5 不参与(同 longtou 有效成员门槛)
            continue
        ups = sum(1 for _, p in pairs if p > 0)
        ranked = sorted(pairs, key=lambda x: -x[1])
        rank_of = {c: i + 1 for i, (c, _) in enumerate(ranked)}
        ind_ctx[ind] = {"n": n, "breadth": ups / n, "rank": rank_of,
                        "top_pct": ranked[0][1]}
    out = {}
    for c in codes:
        s = stock_struct(c, bars_by_code.get(c, []))
        if s is None:
            continue
        ind = ind_map.get(c)
        ctx = ind_ctx.get(ind) if ind else None
        sweet = s["y_volr5"] is not None and 0.55 < s["y_volr5"] <= 2.2
        # ---- g_chip(二元组, 缺失项弃用重归一; ldlr不进截面分) ----
        chip_items = []
        if ctx:
            chip_items.append(1.0 if ctx["rank"].get(c, 999) > 3.5 else 0.0)
        chip_items.append(1.0 if s["zb_cnt20"] <= 1.5 else 0.0)
        chip_items.append(1.0 if sweet else 0.0)
        chip_items.append(1.0 if s["neg_streak"] >= 2.5 else 0.0)
        g_chip = sum(chip_items) / len(chip_items) if chip_items else 0.0
        # ---- g_eco(分位组, 行业缺失时仅疤痕+甜蜜区) ----
        eco_items = [pct("zb_cnt20", s["zb_cnt20"]),
                     1.0 if sweet else 0.0]
        if ctx:
            eco_items += [pct("ind_breadth", ctx["breadth"]),
                          pct("ind_rank", ctx["rank"].get(c, 999),
                              invert=True)]
        g_eco = sum(eco_items) / len(eco_items)
        out[c] = {
            "g_chip": round(g_chip, 3),
            "gate": g_chip >= CHIP_GATE,
            "v5_base": round(0.25 * g_eco + 0.25 * g_chip, 4),
            "zb_cnt20": s["zb_cnt20"],
            "y_volr5": round(s["y_volr5"], 2) if s["y_volr5"] else None,
            "neg_streak": s["neg_streak"],
            "ind_rank": ctx["rank"].get(c) if ctx else None,
            "ind_breadth": round(ctx["breadth"], 2) if ctx else None,
            "ldlr_prev": ldlr_prev,
        }
    return out


def v5_full(base: float, r3: float | None, pathvol: float | None) -> float | None:
    """完整 V5 = v5_base + 0.5*g_intra; 盘中因子缺失返回 None(不猜测)"""
    if r3 is None or pathvol is None:
        return None
    g_intra = (pct("r3", r3) + pct("pathvol", pathvol)) / 2
    return round(base + 0.5 * g_intra, 4)


def fetch_ldlr_prev(yest: str) -> float | None:
    """指定交易日(从日bar推导的上一交易日)跌停/涨停家数比(东财池);
    失败/返回异常返回 None(调用方弃用该项不阻断)"""
    try:
        import akshare as ak
        zt = ak.stock_zt_pool_em(date=yest)
        if zt is None or len(zt) == 0:
            return None
        try:
            dt = ak.stock_zt_pool_dtgc_em(date=yest)
            n_dt = len(dt) if dt is not None else None
        except Exception:
            n_dt = None
        if n_dt is None:      # 跌停池拉取失败: 返回0会伪装成最优生态, 弃用
            print("[struct] 跌停池拉取失败, ldlr_prev 弃用")
            return None
        return round(n_dt / len(zt), 3)
    except Exception as e:
        print(f"[struct] ldlr_prev 拉取失败(弃用该项): {e}")
        return None
