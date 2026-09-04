# -*- coding: utf-8 -*-
"""净值序列累积与绩效指标 — Sharpe / 信息比率 / 回撤

设计要点:
  Sharpe 与信息比率都需要多日收益序列, 盘中当日只有单日数据算不出。
  故每日 after_trading 结算后把当日净值追加到该 run 的 equity.parquet
  (data/sim/runs/{run_id}/equity.parquet, 路径由 mod config 的 run_dir
  传入), 累计指标从这条跨日序列算; 盘中看板只显示当日盈亏与持仓。

  基准由每个策略自己在 config.yaml 指定(聚宽 set_benchmark 同款),
  框架提供虚拟基准(见 data_source.DBBNCH_ID)。基准缺失时 IR/alpha/beta
  返回 None 而非伪造 0 —— 与项目"数据不足禁止补缺"的一贯口径一致。

年化口径: A股一年约 242 个交易日。
"""
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 242

_EMPTY_COLS = ["trade_date", "equity", "cash", "position_value", "benchmark"]


def load_equity(p) -> pd.DataFrame:
    """跨日净值序列(p = run_dir/equity.parquet); 缺失返回空表(不抛异常)"""
    import pathlib
    p = pathlib.Path(p)
    if not p.exists():
        return pd.DataFrame(columns=_EMPTY_COLS)
    try:
        return pd.read_parquet(p)
    except Exception:
        return pd.DataFrame(columns=_EMPTY_COLS)


def append_equity(p, row: dict) -> pd.DataFrame:
    """追加当日净值(同日覆盖 — 盘中多次结算只留最后一次)"""
    import pathlib
    p = pathlib.Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = load_equity(p)
    row = dict(row)
    d = str(row["trade_date"])
    if len(df):
        df = df[df["trade_date"].astype(str) != d]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values("trade_date").drop_duplicates(
        subset=["trade_date"], keep="last")
    df.to_parquet(p, index=False)
    return df


def daily_returns(df: pd.DataFrame, col: str = "equity") -> pd.Series:
    """日收益率序列(首日无前值 → 跳过)"""
    if len(df) < 2:
        return pd.Series(dtype=float)
    s = df[col].astype(float).reset_index(drop=True)
    return (s / s.shift(1) - 1.0).dropna()


def _sharpe(rets: pd.Series) -> float | None:
    if len(rets) < 2:
        return None
    sd = float(rets.std(ddof=1))
    if sd <= 0:
        return None
    return round(float(rets.mean()) / sd * np.sqrt(TRADING_DAYS_PER_YEAR), 4)


def _max_drawdown(equity: pd.Series) -> float | None:
    if len(equity) < 2:
        return None
    e = equity.astype(float).values
    peak = np.maximum.accumulate(e)
    dd = (e / peak - 1.0)
    return round(float(dd.min()), 4)


def _cagr(equity: pd.Series) -> float | None:
    if len(equity) < 2 or float(equity.iloc[0]) <= 0:
        return None
    n = len(equity)
    total = float(equity.iloc[-1]) / float(equity.iloc[0])
    if total <= 0:
        return None
    return round(total ** (TRADING_DAYS_PER_YEAR / n) - 1.0, 4)


def compute_metrics(df: pd.DataFrame) -> dict:
    """累计绩效指标。df 需含 equity 列; benchmark 列存在且非空才算 IR。

    返回:
      days            样本交易日数
      total_return    累计收益率
      cagr            年化收益率
      sharpe          年化夏普(无风险利率取 0)
      max_drawdown    最大回撤(负值)
      ir              年化信息比率(超额收益/跟踪误差); 无基准返回 None
      alpha/beta      相对基准的年化 alpha 与 beta; 无基准返回 None
      vol             年化波动率
    """
    out = {"days": int(len(df)), "total_return": None, "cagr": None,
           "sharpe": None, "max_drawdown": None, "vol": None,
           "ir": None, "alpha": None, "beta": None}
    if len(df) < 2 or "equity" not in df.columns:
        return out
    eq = df["equity"].astype(float).reset_index(drop=True)
    rets = daily_returns(df)
    out["total_return"] = round(float(eq.iloc[-1] / eq.iloc[0] - 1.0), 4)
    out["cagr"] = _cagr(eq)
    out["sharpe"] = _sharpe(rets)
    out["max_drawdown"] = _max_drawdown(eq)
    out["vol"] = (round(float(rets.std(ddof=1)) * np.sqrt(TRADING_DAYS_PER_YEAR), 4)
                  if len(rets) >= 2 else None)
    # ---- 基准相关(缺失则一律 None, 不伪造) ----
    if "benchmark" in df.columns:
        bm = df["benchmark"].astype(float)
        if bm.notna().sum() >= 2:
            bm = bm.dropna().reset_index(drop=True)
            bm_rets = (bm / bm.shift(1) - 1.0).dropna()
            if len(bm_rets) >= 2:
                excess = rets - bm_rets
                te = float(excess.std(ddof=1))
                out["ir"] = (round(float(excess.mean()) / te
                                   * np.sqrt(TRADING_DAYS_PER_YEAR), 4)
                             if te > 0 else None)
                cov = float(np.cov(rets.values, bm_rets.values)[0, 1])
                var = float(bm_rets.var(ddof=1))
                out["beta"] = round(cov / var, 4) if var > 0 else None
                if out["beta"] is not None:
                    ann_s = float(rets.mean()) * TRADING_DAYS_PER_YEAR
                    ann_b = float(bm_rets.mean()) * TRADING_DAYS_PER_YEAR
                    out["alpha"] = round(ann_s - out["beta"] * ann_b, 4)
    return out


def trade_stats(trades: list) -> dict:
    """平仓交易统计(胜率/盈亏比/均盈亏)。trades 每项需含 pnl_pct。"""
    if not trades:
        return {"n": 0, "win_rate": None, "profit_ratio": None,
                "avg_pnl": None}
    pnls = [float(t.get("pnl_pct") or 0.0) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    avg_w = float(np.mean(wins)) if wins else None
    avg_l = float(np.mean(losses)) if losses else None
    return {
        "n": len(pnls),
        "win_rate": round(len(wins) / len(pnls), 4),
        # 盈亏比 = 平均盈利 / |平均亏损|; 无亏损时为 None(不伪造 inf)
        "profit_ratio": (round(avg_w / abs(avg_l), 4)
                         if avg_w is not None and avg_l not in (None, 0.0)
                         else None),
        "avg_pnl": round(float(np.mean(pnls)), 4),
    }
