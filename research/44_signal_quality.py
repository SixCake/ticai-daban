# -*- coding: utf-8 -*-
"""研究44: 单日信号质量与买点胜率分析

对 presig_state_{date}.json 的当日信号做**买点胜率**(封板率)与**信号质量**
(离场盈亏比/期望)分析, 并按 T级/S级/方案/入场价位/结构闸 分组。

口径(沿用项目定稿):
  · 封板判定 = zt_shape 非空(贴死涨停价 SEAL_EPS, 同 core/theme_signal)
  · 离场用 core/exit_rules.simulate_exit 现场模拟(当日信号无 ev_exit)。
    **当日盘中入场必须传 hi_pts**(动态最高价, 避免前视 —— 研究39 教训:
    用全天最高价作止损参考价会让止损线高于入场价, 第一根bar 误触发)
  · 入场价位档来自研究08 §4(打平所需封板率): 2~4%可行/4~6%边际/≥6%不可行
  · 方案口径同 research/36 + dashboard psSchemesOf(core=非量爆的S2/S3)
  · 主判据是**离场期望**, 不是封板率(研究08/30/31: 封板率与EV系统性反向)

⚠ 局限: px_hist 来自 radar_log, 只录 pct≥1%或prob≥0.2 的票, 部分票轨迹
  在14:00前就断 → 离场模拟含"轨迹不足回退收盘", 非完整策略回测。

用法: python research/44_signal_quality.py [date YYYYMMDD]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA                                     # noqa: E402
from core.exit_rules import pl_stats, simulate_exit          # noqa: E402

LIVE = DATA / "live"


def entry_tier(pct):
    """入场价位档(研究08 §4 机制矩阵)"""
    if pct is None:
        return None
    if pct < 2:
        return "1_<2%"
    if pct < 4:
        return "2_2-4%可行"
    if pct < 6:
        return "3_4-6%边际"
    return "4_≥6%不可行"


def schemes_of(s):
    """命中方案(同 dashboard psSchemesOf): core=非量爆的S2/S3"""
    if s.get("stage") == "S1":
        return []
    if "量爆" in (s.get("why") or ""):
        return []
    t = (s.get("t_sig") or {}).get("level")
    if t not in ("T3", "T2"):
        return []
    g = (s.get("struct") or {}).get("gate")
    y = (s.get("t_sig") or {}).get("y_ht") or 0
    hit = []
    if t == "T3":
        hit.append("B1")
        if g:
            hit.append("B2")
        if y == 1:
            hit.append("C1")
        if y >= 1 and g:
            hit.append("C5")
    else:                                   # T2
        hit.append("B1")
    return hit


def is_sealed(s):
    return bool(s.get("zt_shape"))


def exit_ret(s):
    """现场模拟当日离场收益%(传 hi_pts 动态最高价, 无前视)"""
    ph = s.get("px_hist") or []
    pts = [[p[0], p[1]] for p in ph if len(p) >= 2 and p[1]]
    if not pts or not s.get("pb"):
        return None
    pb = s["pb"]
    prices = [p[1] for p in pts]
    day_hi = max(prices)
    close_px = prices[-1]
    limit_px = s.get("limit_px") or 0
    hi_pts = [[p[0], p[1]] for p in pts]    # 分时close近似bar high
    ret, _ = simulate_exit(pts, pb, day_hi, limit_px, close_px, hi_pts=hi_pts)
    return ret


def traj_complete(s):
    """轨迹是否覆盖到14:57(离场模拟可靠性)"""
    ph = s.get("px_hist") or []
    if not ph:
        return False
    last = str(ph[-1][0]).replace(":", "")
    return last >= "145700"


def group_stats(sig, keyfn, order=None):
    """按 keyfn 分组 → [(key, 信号数, 封板数, 封板率%, 离场样本, 胜率%, 盈亏比, 期望%)]"""
    groups = {}
    for s in sig:
        k = keyfn(s)
        if k is None:
            continue
        groups.setdefault(k, []).append(s)
    rows = []
    for k, g in groups.items():
        n = len(g)
        sealed = sum(1 for s in g if is_sealed(s))
        rets = [r for r in (exit_ret(s) for s in g) if r is not None]
        nn, wr, aw, al, ratio, exp = pl_stats(rets)
        rows.append((k, n, sealed, sealed / n * 100,
                     nn,
                     round(wr * 100, 1) if wr is not None else None,
                     round(ratio, 2) if ratio is not None else None,
                     round(exp, 2) if exp is not None else None))
    if order:
        rows.sort(key=lambda r: order.index(r[0]) if r[0] in order else 99)
    else:
        rows.sort(key=lambda r: -r[1])
    return rows


def print_table(title, rows):
    print(f"\n=== {title} ===")
    print(f"{'组':<14}{'信号':>5}{'封板':>5}{'封板率%':>8}{'离场样本':>8}"
          f"{'胜率%':>7}{'盈亏比':>7}{'期望%':>7}")
    for k, n, sealed, sr, nn, wr, ratio, exp in rows:
        wr_s = f"{wr}" if wr is not None else "-"
        ra_s = f"{ratio}" if ratio is not None else "-"
        ex_s = f"{exp}" if exp is not None else "-"
        print(f"{k:<14}{n:>5}{sealed:>5}{sr:>8.1f}{nn:>8}"
              f"{wr_s:>7}{ra_s:>7}{ex_s:>7}")


def analyze(date):
    fn = LIVE / f"presig_state_{date}.json"
    if not fn.exists():
        print(f"无 {fn}")
        return
    d = json.load(open(fn, encoding="utf-8"))
    sig = d["signals"]
    n = len(sig)
    sealed = sum(1 for s in sig if is_sealed(s))
    rets = [r for r in (exit_ret(s) for s in sig) if r is not None]
    nn, wr, aw, al, ratio, exp = pl_stats(rets)
    complete = sum(1 for s in sig if traj_complete(s))

    print(f"# 今日({date})买点胜率与信号质量")
    print(f"\n## 总体")
    print(f"  信号 {n} 条 | 封板 {sealed} ({sealed/n*100:.1f}%) | "
          f"轨迹完整(≥14:57) {complete} ({complete/n*100:.0f}%)")
    print(f"  离场模拟样本 {nn} | 胜率 {wr*100:.1f}% | 平均盈利 {aw:.2f}% "
          f"平均亏损 {al:.2f}% | 盈亏比 {ratio:.2f} | 期望 {exp:.2f}%")

    print_table("按 T级(题材级, 数字越大越好)",
                group_stats(sig, lambda s: (s.get("t_sig") or {}).get("level"),
                            order=["T1", "T2", "T3"]))
    print_table("按 S级(个股级)",
                group_stats(sig, lambda s: s.get("stage"),
                            order=["S1", "S2", "S3"]))
    print_table("按 入场价位档(研究08)",
                group_stats(sig, lambda s: entry_tier(s.get("pct"))))
    print_table("按 结构闸(struct.gate)",
                group_stats(sig, lambda s: ("过闸" if (s.get("struct") or {}).get("gate")
                                            else "未过闸")))

    # 方案(一条信号可命中多个, 分别统计)
    scheme_rows = []
    for sc in ["B1", "B2", "C1", "C5"]:
        g = [s for s in sig if sc in schemes_of(s)]
        if not g:
            continue
        sealed_g = sum(1 for s in g if is_sealed(s))
        rets_g = [r for r in (exit_ret(s) for s in g) if r is not None]
        nn_g, wr_g, aw_g, al_g, ratio_g, exp_g = pl_stats(rets_g)
        scheme_rows.append((sc, len(g), sealed_g, sealed_g / len(g) * 100,
                            nn_g,
                            round(wr_g * 100, 1) if wr_g is not None else None,
                            round(ratio_g, 2) if ratio_g is not None else None,
                            round(exp_g, 2) if exp_g is not None else None))
    print_table("按 方案(可命中多个)", scheme_rows)


if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else None
    if not date:
        import re
        fs = sorted(LIVE.glob("presig_state_*.json"))
        ds = [re.search(r"(\d{8})", f.stem).group(1) for f in fs]
        date = max(ds) if ds else None
    analyze(date)
