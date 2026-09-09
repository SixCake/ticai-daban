# -*- coding: utf-8 -*-
"""研究37: K线形态/阻力点/突破点因子的单因子筛选

背景: 现有 15+ 个因子(core/structure 的 g_chip/g_eco/g_intra + 研究30 增量)
全是**统计聚合量**(触板未封天数/量比/连跌天数/涨速), 没有一个几何/形态维度。
本研究补这个缺口 —— 即用户说的「选票审美」。

方法论(沿用项目定稿口径):
  · 主判据是 T+1/T+2/T+3 胜率与盈亏比, **不是封板率** —— 研究30/31/08
    三次证明封板率与 EV 系统性反向, 用封板率筛因子会选出「容易涨停但
    次日亏钱」的票
  · 三段市况独立验证(牛/熊/震荡), 方向一致性 ≥2/3 才算通过
  · 市况分段用数据驱动三分位(同 research/31), 不设人为阈值
  · 盈亏比/期望复用 core/exit_rules.pl_stats, 不重写口径

两套持仓口径并列:
  主口径 无条件持有   所有触板股都算 T+1/T+2/T+3 原始收益, 6.8年全窗口
  副口径 定稿卖出规则 走 core/exit_rules.simulate_exit, **仅7个近期交易日**
                     (intraday_px 只有近7天, 无法回填) → 只作交叉校验

前视防护: 因子只用 T-1 及更早 bar; 标签只用 T+1 及之后。

产物: research/out/37_shape_factors.md
用法: python research/37_shape_factors.py [--quick]

因子定义的**唯一出处是 core/shape.py**(供 research/38 方案矩阵复用)。
本文件的本地副本作为已发布报告的冻结快照保留(同 research 01-36 惯例),
两处逻辑已核对等价 —— 后续新增/修改因子只改 core/shape.py。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA                                     # noqa: E402
from datastore import load, path_of                        # noqa: E402
from core.exit_rules import pl_stats, simulate_exit        # noqa: E402

LIVE = DATA / "live"      # config 不导出 LIVE, 同 apps/radar 的本地定义

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

# ---- 因子清单 ----
# 连续因子按5档分位切; 布尔因子按 0/1 两组对比
CONT = ["body_ratio", "upper_shadow", "dist_high_20", "dist_high_60",
        "dist_high_120", "trapped_ratio", "int_prox", "box_width_20",
        "flat_days", "ma_align", "vol_price_sync"]
BOOL = ["fake_yin", "n_shape", "doji_shrink", "new_high_20", "new_high_60",
        "new_high_120", "gap_up"]
GROUP = {
    "body_ratio": "A 形态", "upper_shadow": "A 形态", "fake_yin": "A 形态",
    "n_shape": "A 形态", "doji_shrink": "A 形态",
    "dist_high_20": "B 阻力", "dist_high_60": "B 阻力",
    "dist_high_120": "B 阻力", "trapped_ratio": "B 阻力",
    "int_prox": "B 阻力",
    "box_width_20": "C 突破", "flat_days": "C 突破",
    "new_high_20": "C 突破", "new_high_60": "C 突破",
    "new_high_120": "C 突破", "gap_up": "C 突破", "ma_align": "C 突破",
    "vol_price_sync": "D 量能",
}
# 天然推高入场价位的因子: 研究08 已证 6%以上入场数学上不可行,
# 这类因子必须与入场价位闸门联合评估才有意义
PRICE_RISK = {"new_high_20", "new_high_60", "new_high_120", "gap_up",
              "dist_high_20", "dist_high_60", "dist_high_120"}
LABELS = ["r1_open", "r1_close", "r2_close", "r3_close"]
NLABEL = {"r1_open": "T+1开盘", "r1_close": "T+1收盘",
          "r2_close": "T+2收盘", "r3_close": "T+3收盘"}

say = print


# ---------------------------------------------------------------- 数据装载
def build_bars(panel: pd.DataFrame) -> dict:
    """{code: {date/open/high/low/close/vol/pct 的 numpy 数组}} 升序"""
    out = {}
    for code, g in panel.groupby("ts_code", sort=False):
        g = g.sort_values("trade_date")
        out[code] = {
            "date": g["trade_date"].values,
            "open": g["open"].to_numpy(dtype=float),
            "high": g["high"].to_numpy(dtype=float),
            "low": g["low"].to_numpy(dtype=float),
            "close": g["close"].to_numpy(dtype=float),
            "vol": g["vol"].to_numpy(dtype=float),
            "pct": g["pct_chg"].to_numpy(dtype=float),
        }
    return out


# ---------------------------------------------------------------- 因子计算
def compute_factors(b: dict, i: int) -> dict:
    """信号日 T = b['date'][i]。**只用 i-1 及更早的 bar**(前视防护)。

    缺失不补缺: 数据不足或一字板(high==low)记 None, 不填 0 伪装。
    """
    o, h, l, c, v, p = (b["open"], b["high"], b["low"], b["close"],
                        b["vol"], b["pct"])
    if i < 1:
        return {}
    j = i - 1                      # T-1
    out = {}
    # ---- A 组 K线形态 (T-1 bar) ----
    rng = h[j] - l[j]
    if rng and rng > 0:
        out["body_ratio"] = (c[j] - o[j]) / rng
        out["upper_shadow"] = (h[j] - max(o[j], c[j])) / rng
        out["doji_shrink"] = None          # 需 vma5, 下面算
    else:
        out["body_ratio"] = None           # 一字板: 无实体可言, 不填0
        out["upper_shadow"] = None
        out["doji_shrink"] = None
    if j >= 1 and c[j - 1] > 0:
        out["fake_yin"] = int(c[j] < o[j] and c[j] > c[j - 1])
    else:
        out["fake_yin"] = None
    # vma5 = T-1 之前5日均量(不含 T-1)
    vs = v[max(0, j - 5):j]
    vs = vs[~np.isnan(vs)]
    vma5 = vs.mean() if len(vs) >= 3 else None
    if vma5 and vma5 > 0 and v[j] and not np.isnan(v[j]):
        shrink = v[j] < 0.7 * vma5
        if out["doji_shrink"] is None and rng and rng > 0:
            out["doji_shrink"] = int(abs(c[j] - o[j]) / rng < 0.2 and shrink)
        elif rng and rng > 0:
            out["doji_shrink"] = int(abs(c[j] - o[j]) / rng < 0.2 and shrink)
    # N字板: 近5日内 ∃ 放量阳→缩量阴→放量阳
    ns = 0
    if j >= 4 and vma5:
        for k in range(max(1, j - 4), j - 1):
            if (v[k] > 1.2 * vma5 and p[k] > 3.0
                    and v[k + 1] < 0.8 * v[k] and p[k + 1] < 0
                    and v[k + 2] > v[k + 1] and p[k + 2] > 0):
                ns = 1
                break
    out["n_shape"] = ns if j >= 4 and vma5 else None
    # ---- B 组 阻力点 ----
    for n, key in ((20, "dist_high_20"), (60, "dist_high_60"),
                   (120, "dist_high_120")):
        seg = h[max(0, j - n + 1):j + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg) >= min(n, 20) and c[j] > 0:
            mx = seg.max()
            out[key] = (c[j] - mx) / mx * 100
        else:
            out[key] = None
    seg = c[max(0, j - 119):j + 1]
    seg = seg[~np.isnan(seg)]
    out["trapped_ratio"] = ((seg > c[j]).mean() * 100
                            if len(seg) >= 60 and c[j] > 0 else None)
    out["int_prox"] = (abs(c[j] - round(c[j] / 5) * 5) / c[j] * 100
                       if c[j] > 0 else None)
    # ---- C 组 突破点 ----
    hs = h[max(0, j - 19):j + 1]
    ls = l[max(0, j - 19):j + 1]
    hs, ls = hs[~np.isnan(hs)], ls[~np.isnan(ls)]
    out["box_width_20"] = ((hs.max() - ls.min()) / ls.min() * 100
                           if len(hs) >= 15 and ls.min() > 0 else None)
    fd = 0
    for k in range(j, 0, -1):
        if abs(p[k]) < 2.0:
            fd += 1
        else:
            break
    out["flat_days"] = fd
    for n, key in ((20, "new_high_20"), (60, "new_high_60"),
                   (120, "new_high_120")):
        seg = h[max(0, j - n + 1):j]        # 不含 T-1 自身
        seg = seg[~np.isnan(seg)]
        out[key] = (int(h[j] >= seg.max())
                    if len(seg) >= min(n, 20) and not np.isnan(h[j]) else None)
    out["gap_up"] = (int(o[j] > h[j - 1])
                     if j >= 1 and not np.isnan(o[j]) and not np.isnan(h[j - 1])
                     else None)
    mas = {}
    for n in (5, 10, 20, 60):
        seg = c[max(0, j - n + 1):j + 1]
        seg = seg[~np.isnan(seg)]
        mas[n] = seg.mean() if len(seg) >= n else None
    if all(mas[n] is not None for n in (5, 10, 20, 60)):
        out["ma_align"] = (int(mas[5] > mas[10]) + int(mas[10] > mas[20])
                           + int(mas[20] > mas[60]))
    else:
        out["ma_align"] = None
    # ---- D 组 量能 ----
    ps = p[max(0, j - 19):j + 1]
    vs2 = v[max(0, j - 19):j + 1]
    ok = ~(np.isnan(ps) | np.isnan(vs2))
    if ok.sum() >= 15 and ps[ok].std() > 0 and vs2[ok].std() > 0:
        out["vol_price_sync"] = float(np.corrcoef(ps[ok], vs2[ok])[0, 1])
    else:
        out["vol_price_sync"] = None
    return out


def compute_labels(b: dict, i: int) -> dict:
    """标签只用 T+1 及之后(前视防护的另一侧)"""
    c, o = b["close"], b["open"]
    base = c[i]
    if not base or base <= 0 or np.isnan(base):
        return {}
    out = {}
    out["r1_open"] = ((o[i + 1] / base - 1) * 100
                      if i + 1 < len(c) and not np.isnan(o[i + 1]) else None)
    for k, key in ((1, "r1_close"), (2, "r2_close"), (3, "r3_close")):
        out[key] = ((c[i + k] / base - 1) * 100
                    if i + k < len(c) and not np.isnan(c[i + k]) else None)
    return out


# ---------------------------------------------------------------- 市况分段
def regimes(dates: pd.Series) -> dict:
    """按月均"全A中位涨幅"三分位切牛/熊/震荡(同 research/31, 数据驱动)"""
    return {}


def build_regimes(panel: pd.DataFrame) -> dict:
    mr = panel.groupby("trade_date")["pct_chg"].median().rename("mret")
    mon = mr.groupby(mr.index.str[:6]).mean().sort_values()
    n = len(mon)
    bear = set(mon.index[:n // 3])
    bull = set(mon.index[-n // 3:])
    return {"bull": bull, "bear": bear}


def reg_of(date: str, rg: dict) -> str:
    m = date[:6]
    return "熊" if m in rg["bear"] else "牛" if m in rg["bull"] else "震荡"


# ---------------------------------------------------------------- 评估
def quintile_table(d: pd.DataFrame, fac: str, lab: str) -> list:
    """连续因子5档分位表 → [(档位, 样本数, 胜率%, 平均盈利, 平均亏损, 盈亏比, 期望)]"""
    s = d[[fac, lab]].dropna()
    if len(s) < 500:
        return []
    try:
        s["q"] = pd.qcut(s[fac], 5, labels=False, duplicates="drop")
    except Exception:
        return []
    rows = []
    for q in sorted(s["q"].dropna().unique()):
        sub = s[s["q"] == q][lab]
        n, wr, aw, al, ratio, exp = pl_stats(list(sub))
        rows.append((int(q) + 1, n,
                     round(wr * 100, 2) if wr is not None else None,
                     round(aw, 3) if aw is not None else None,
                     round(al, 3) if al is not None else None,
                     round(ratio, 3) if ratio is not None else None,
                     round(exp, 3) if exp is not None else None))
    return rows


def bool_table(d: pd.DataFrame, fac: str, lab: str) -> list:
    """布尔因子 0/1 两组对比"""
    s = d[[fac, lab]].dropna()
    if len(s) < 500:
        return []
    rows = []
    for val in (0, 1):
        sub = s[s[fac] == val][lab]
        n, wr, aw, al, ratio, exp = pl_stats(list(sub))
        rows.append((val, n,
                     round(wr * 100, 2) if wr is not None else None,
                     round(aw, 3) if aw is not None else None,
                     round(al, 3) if al is not None else None,
                     round(ratio, 3) if ratio is not None else None,
                     round(exp, 3) if exp is not None else None))
    return rows


def monotonic(rows: list) -> tuple:
    """单调性判定 → (方向, 是否单调)。用盈亏比与期望的档1→档末变化"""
    if len(rows) < 3:
        return "样本不足", False
    exps = [r[6] for r in rows if r[6] is not None]
    if len(exps) < 3:
        return "样本不足", False
    ups = sum(1 for a, b in zip(exps, exps[1:]) if b > a)
    downs = sum(1 for a, b in zip(exps, exps[1:]) if b < a)
    if ups == len(exps) - 1:
        return "严格单调↑", True
    if downs == len(exps) - 1:
        return "严格单调↓", True
    if ups >= downs:
        return "偏↑(非严格)", ups - downs >= 2
    return "偏↓(非严格)", downs - ups >= 2


def regime_consistency(d: pd.DataFrame, fac: str, lab: str, rg: dict) -> tuple:
    """三段市况方向一致性 → (一致段数/3, 各段方向)"""
    d = d.copy()
    d["reg"] = d["trade_date"].map(lambda x: reg_of(x, rg))
    dirs = {}
    for r in ("牛", "熊", "震荡"):
        sub = d[d["reg"] == r]
        rows = (quintile_table(sub, fac, lab) if fac in CONT
                else bool_table(sub, fac, lab))
        if not rows:
            dirs[r] = "无"
            continue
        if fac in BOOL:
            # 布尔: 比较 1 组 vs 0 组的期望差
            e0 = next((x[6] for x in rows if x[0] == 0), None)
            e1 = next((x[6] for x in rows if x[0] == 1), None)
            dirs[r] = ("↑" if e1 is not None and e0 is not None and e1 > e0
                       else "↓" if e1 is not None and e0 is not None else "无")
        else:
            # monotonic 返回如「严格单调↑」「偏↑(非严格)」—— 要取箭头字符,
            # 不能取首字(否则「严格单调↑」会被截成「严」, 一致性永远算0/3)
            m = monotonic(rows)[0]
            dirs[r] = "↑" if "↑" in m else "↓" if "↓" in m else "无"
    pos = sum(1 for v in dirs.values() if v == "↑")
    neg = sum(1 for v in dirs.values() if v == "↓")
    return max(pos, neg), dirs


# ---------------------------------------------------------------- 副口径
def secondary_caliber(d: pd.DataFrame, bars: dict) -> pd.DataFrame:
    """定稿卖出规则口径(仅 intraday_px 存在的近期交易日)。

    simulate_exit 只模拟三条(止损5% / P2 10:10 / 14:57强平), P0/P1/P3
    未模拟 —— 调用方必须知道这是**部分模拟**, 不当完整策略回测。
    封板续持的票延伸到 T+2/T+3(用日线收盘近似, 无分时)。
    """
    ipx_files = sorted(LIVE.glob("intraday_px_*.json"))
    dates = [f.stem.replace("intraday_px_", "") for f in ipx_files]
    if not dates:
        return pd.DataFrame()
    # 缓存每日分时。simulate_exit 要的是**次日**轨迹(见其 docstring),
    # 不是信号日当日 —— 传错会拿当日最低价去比当日最高价算出的止损线,
    # 几乎全部立即触发止损(实测误传时 70.3% 止损、期望 -6.3%, 是假的)。
    cache = {}
    for dt in dates:
        try:
            cache[dt] = json.loads((LIVE / f"intraday_px_{dt}.json")
                                   .read_text(encoding="utf-8"))
        except Exception:
            cache[dt] = {}
    rows = []
    for dt in dates:
        sub = d[d["trade_date"] == dt]
        for r in sub.itertuples():
            b = bars.get(r.ts_code)
            if not b:
                continue
            idx = int(np.searchsorted(b["date"], dt))
            if idx >= len(b["date"]) or b["date"][idx] != dt:
                continue
            if idx + 1 >= len(b["date"]):
                continue
            nxt = b["date"][idx + 1]          # 次日交易日
            pts = (cache.get(nxt) or {}).get(r.ts_code) or []
            if not pts:
                continue
            # 买入价: 用信号日收盘近似(无挂单成交数据)
            pb = float(b["close"][idx])
            day_hi = float(b["high"][idx])
            # 次日涨停价
            pre = b["close"][idx]
            rate = (0.20 if str(r.ts_code)[:3] in ("300", "301", "302",
                                                   "688", "689")
                    else 0.30 if str(r.ts_code).endswith(".BJ") else 0.10)
            lp = round(pre * (1 + rate), 2)
            nx_close = float(b["close"][idx + 1])
            ret, why = simulate_exit(pts, pb, day_hi, lp, nx_close)
            if ret is None:
                continue
            sealed = "封板续持" in why or "14:57强平(封板续持)" == why
            row = {"trade_date": dt, "ts_code": r.ts_code,
                   "exit_ret": ret, "exit_why": why, "sealed": sealed}
            # 封板续持才有 T+2/T+3 敞口
            if sealed:
                for k, key in ((2, "s2_close"), (3, "s3_close")):
                    row[key] = ((b["close"][idx + k] / pb - 1) * 100
                                if idx + k < len(b["date"])
                                and not np.isnan(b["close"][idx + k])
                                else None)
            else:
                row["s2_close"] = None
                row["s3_close"] = None
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="只用最近2年数据(调试用)")
    a = ap.parse_args()

    say("加载触板股宇宙(limitup.events_enriched)…")
    ev = load("limitup.events_enriched",
              columns=["trade_date", "ts_code", "limit_times"])
    ev = ev.drop_duplicates(subset=["trade_date", "ts_code"])
    say(f"  触板股样本 {len(ev)} 条, {ev['trade_date'].nunique()} 个交易日, "
        f"{ev['trade_date'].min()}~{ev['trade_date'].max()}")

    say("加载日线面板(market.daily_panel)…")
    cols = ["trade_date", "ts_code", "open", "high", "low", "close",
            "vol", "pct_chg"]
    p = path_of("market.daily_panel")
    panel = pd.read_parquet(p, columns=cols)
    if a.quick:
        cut = str(int(panel["trade_date"].max()) - 20000)
        panel = panel[panel["trade_date"] >= cut]
        say(f"  --quick: 截断到 {cut} 起")
    panel = panel.dropna(subset=["close"])
    say(f"  面板 {len(panel)} 行, {panel['ts_code'].nunique()} 只票")

    say("构建 per-code bar 数组…")
    bars = build_bars(panel)

    say("计算因子与标签(前视防护: 因子只用T-1及更早, 标签只用T+1及之后)…")
    recs = []
    n_skip = 0
    for r in ev.itertuples():
        b = bars.get(r.ts_code)
        if not b:
            n_skip += 1
            continue
        idx = int(np.searchsorted(b["date"], r.trade_date))
        if idx >= len(b["date"]) or b["date"][idx] != r.trade_date:
            n_skip += 1
            continue
        f = compute_factors(b, idx)
        lb = compute_labels(b, idx)
        if not f or not lb:
            n_skip += 1
            continue
        rec = {"trade_date": r.trade_date, "ts_code": r.ts_code,
               "limit_times": int(r.limit_times or 1),
               "entry_pct": float(b["pct"][idx]) if not np.isnan(b["pct"][idx])
               else None}
        rec.update(f)
        rec.update(lb)
        recs.append(rec)
    d = pd.DataFrame(recs)
    say(f"  有效样本 {len(d)} 条(跳过 {n_skip} 条: 无面板/无T-1/无T+1)")
    if not len(d):
        say("无有效样本, 退出")
        return

    say("市况分段(按月均全A中位涨幅三分位, 同研究31)…")
    rg = build_regimes(panel)

    # ---- 副口径 ----
    say("副口径: 定稿卖出规则(仅 intraday_px 存在的近期交易日)…")
    sec = secondary_caliber(d, bars)
    if len(sec):
        say(f"  副口径样本 {len(sec)} 条, 覆盖 "
            f"{sec['trade_date'].nunique()} 个交易日")

    # ---- 评估 ----
    say("评估: 每因子 × 每持有期 × 5档分位 + 盈亏比 + 三段一致性…")
    results = {}
    for fac in CONT + BOOL:
        results[fac] = {}
        for lab in LABELS:
            rows = (quintile_table(d, fac, lab) if fac in CONT
                    else bool_table(d, fac, lab))
            if not rows:
                results[fac][lab] = None
                continue
            mono, is_mono = monotonic(rows) if fac in CONT else ("布尔", None)
            cons, dirs = regime_consistency(d, fac, lab, rg)
            results[fac][lab] = {"rows": rows, "mono": mono,
                                 "is_mono": is_mono, "cons": cons,
                                 "dirs": dirs}

    # ---- 报告 ----
    write_report(d, sec, results, rg)
    say(f"\n报告已写入 {OUT/'37_shape_factors.md'}")


def write_report(d, sec, results, rg):
    L = []
    A = L.append
    A("# 研究37: K线形态/阻力点/突破点因子的单因子筛选\n")
    A(f"- 宇宙: 触板股 **{len(d)}** 条样本, "
      f"{d['trade_date'].nunique()} 个交易日, "
      f"{d['trade_date'].min()}~{d['trade_date'].max()}")
    A(f"- 因子: {len(CONT)} 个连续 + {len(BOOL)} 个布尔 = "
      f"{len(CONT)+len(BOOL)} 个(全部白盒可复现)")
    A("- 标签: T+1开盘 / T+1收盘 / T+2收盘 / T+3收盘")
    A("- 指标: 胜率 / 平均盈利 / 平均亏损 / **盈亏比** / 期望"
      "(复用 `core/exit_rules.pl_stats`)")
    A("- 市况: 牛/熊/震荡按月均全A中位涨幅三分位(同研究31, 数据驱动)\n")
    A("## 方法论警告\n")
    A("**主判据是胜率与盈亏比, 不是封板率。** 项目研究30/31/08 三次证明"
      "封板率与 EV 系统性反向 —— 竞价量比闸封板率维度极强(过闸12.41% vs "
      "未过闸1.00%, 10倍)但次日胜率反向(档5 37.73% < 档1 41.27%)。"
      "用封板率筛因子会选出「容易涨停但次日亏钱」的票。\n")
    A("研究08 §4 机制矩阵: 全档 EV 均为负, 6%以上入场需81%+封板率才打平, "
      "**数学上不可行**。故标注为「价位风险」的因子必须与入场价位闸门"
      "联合评估。\n")

    # ---- 候选清单(先给结论) ----
    A("## 候选清单(通过判据的因子)\n")
    A("判据: ① 分档单调(或方向一致) ② 三段市况一致性 ≥2/3\n")
    A("| 因子 | 组 | 主判据持有期 | 单调性 | 三段一致 | 盈亏比增量 | 价位风险 |")
    A("|---|---|---|---|---|---|---|")
    cands = []
    for fac in CONT + BOOL:
        best = None
        for lab in LABELS:
            r = results[fac].get(lab)
            if not r or not r["rows"]:
                continue
            ok_mono = (r["is_mono"] if fac in CONT
                       else abs(r["rows"][1][6] or 0) > abs(r["rows"][0][6] or 0))
            if ok_mono and r["cons"] >= 2:
                rows = r["rows"]
                if fac in CONT:
                    delta = ((rows[-1][5] or 0) - (rows[0][5] or 0)
                             if rows[-1][5] is not None
                             and rows[0][5] is not None else 0)
                else:
                    delta = ((rows[1][5] or 0) - (rows[0][5] or 0)
                             if rows[1][5] is not None
                             and rows[0][5] is not None else 0)
                if best is None or abs(delta) > abs(best[3]):
                    best = (lab, r["mono"], r["cons"], delta)
        if best:
            cands.append((fac, best))
            A(f"| `{fac}` | {GROUP[fac]} | {NLABEL[best[0]]} | {best[1]} "
              f"| {best[2]}/3 | {best[3]:+.3f} "
              f"| {'**是**' if fac in PRICE_RISK else '否'} |")
    if not cands:
        A("| (无因子通过判据) | - | - | - | - | - | - |")
    A("")
    A(f"**通过 {len(cands)} / {len(CONT)+len(BOOL)} 个因子。**")
    pr = [f for f, _ in cands if f in PRICE_RISK]
    if pr:
        A(f"\n其中 {len(pr)} 个标注价位风险({', '.join('`'+x+'`' for x in pr)})"
          " —— 这些因子天然推高入场价位, 研究08 已证 6%以上入场数学上"
          "不可行, **必须与入场价位闸门联合评估**, 不能单独进生产。\n")

    # ---- 因子总表 ----
    A("## 因子总表(全窗口 · 无条件持有口径)\n")
    for grp in ("A 形态", "B 阻力", "C 突破", "D 量能"):
        facs = [f for f in CONT + BOOL if GROUP[f] == grp]
        if not facs:
            continue
        A(f"### {grp}组\n")
        for fac in facs:
            A(f"#### `{fac}`\n")
            for lab in LABELS:
                r = results[fac].get(lab)
                if not r or not r["rows"]:
                    A(f"- {NLABEL[lab]}: 样本不足\n")
                    continue
                hdr = ("| 档 | 样本 | 胜率% | 平均盈利 | 平均亏损 | 盈亏比 | 期望 |"
                       if fac in CONT else
                       "| 值 | 样本 | 胜率% | 平均盈利 | 平均亏损 | 盈亏比 | 期望 |")
                A(f"**{NLABEL[lab]}** — 单调性 {r['mono']} · "
                  f"三段一致 {r['cons']}/3 ({r['dirs']})\n")
                A(hdr)
                A("|---|---|---|---|---|---|---|")
                for row in r["rows"]:
                    A("| " + " | ".join(
                        str(x) if x is not None else "-" for x in row) + " |")
                A("")
            A("")

    # ---- 副口径 ----
    A("## 副口径交叉校验(定稿卖出规则)\n")
    if not len(sec):
        A("无 intraday_px 数据, 副口径无法计算。\n")
    else:
        A(f"- 样本 **{len(sec)}** 条, 覆盖 {sec['trade_date'].nunique()} 个交易日"
          f"({sec['trade_date'].min()}~{sec['trade_date'].max()})")
        A("- **只作交叉校验, 不作筛选判据** —— `intraday_px` 只有近7天"
          "(雷达盘中产出, 无法回填), 样本量远小于主口径")
        A("- **部分模拟**: `simulate_exit` 只模拟三条(止损5% / P2 10:10 / "
          "14:57强平), P0/P1/P3 未模拟\n")
        A("| 指标 | 样本 | 胜率% | 平均盈利 | 平均亏损 | 盈亏比 | 期望 |")
        A("|---|---|---|---|---|---|---|")
        for lab, nm in (("exit_ret", "T+1 离场收益"),
                        ("s2_close", "T+2 收盘(仅封板续持)"),
                        ("s3_close", "T+3 收盘(仅封板续持)")):
            n, wr, aw, al, ratio, exp = pl_stats(
                [x for x in sec[lab] if x is not None and not pd.isna(x)])
            A(f"| {nm} | {n} | "
              f"{round(wr*100,2) if wr is not None else '-'} | "
              f"{round(aw,3) if aw is not None else '-'} | "
              f"{round(al,3) if al is not None else '-'} | "
              f"{round(ratio,3) if ratio is not None else '-'} | "
              f"{round(exp,3) if exp is not None else '-'} |")
        A("")
        A("### 离场原因分布\n")
        A("| 原因 | 样本 | 占比 | 期望 |")
        A("|---|---|---|---|")
        for why, g in sec.groupby("exit_why"):
            n, wr, aw, al, ratio, exp = pl_stats(list(g["exit_ret"]))
            A(f"| {why} | {n} | {n/len(sec)*100:.1f}% "
              f"| {round(exp,3) if exp is not None else '-'} |")
        A("")

    # ---- 约束 ----
    A("## 约束与后续\n")
    A("1. **只筛选不进生产** —— 通过判据的因子只进影子字段与看板展示, "
      "不改 S1/S2/S3 触发路径(与 T/S 双轨体系一致)")
    A("2. **下一阶段跑方案矩阵** M0~M6(含入场价位闸门组合), 等本研究"
      "结果出来再定")
    A("3. `tover_cv` 需流通市值, `daily_panel` 无该列, 本阶段未做")
    A("4. 阈值(1.2/0.8倍量, 3%/2%涨幅, 20/60/120窗口)取业内常用值作初筛, "
      "未做二次拟合 —— 筛选阶段目的是看有无区分度, 不是定阈值")
    A("5. 缺失不补缺: 一字板(high==low)时 `body_ratio`/`upper_shadow` 记 "
      "None; bar 不足时 `trapped_ratio`/`dist_high_120` 记 None")

    (OUT / "37_shape_factors.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
