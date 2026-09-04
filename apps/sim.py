# -*- coding: utf-8 -*-
"""策略模拟进程管理器 — 一策略一进程一账户

设计依据: rqalpha 原生是单策略单进程(base.strategy_file), 且多策略合并进
一个进程会共享 context 与账户 → 无法独立算 Sharpe/IR, 一个报错全体挂。
故本脚本为 strategies.yaml 里每条 enabled=true 的记录拉起一个独立子进程。

用法:
  python apps/sim.py                             # 拉起所有启用策略(盘中 live, 主模拟)
  python apps/sim.py --strategy v5_daban         # 只跑指定策略
  python apps/sim.py --replay --date 20260903    # 回放单日
  python apps/sim.py --replay --start 20260827 --end 20260903
  python apps/sim.py --list                      # 列出策略清单
  python apps/sim.py --run-one v5_daban --run-id ID --seed-run RID ...  # (内部)子进程入口

落盘(回测与模拟同构, 一次运行一个 run 目录):
  data/sim/runs/{run_id}/meta.json        运行元信息(状态/参数/指标/seed)
  data/sim/runs/{run_id}/equity.parquet   跨日净值序列
  data/sim/runs/{run_id}/trades.parquet   成交明细(累积去重)
  data/sim/runs/{run_id}/positions.parquet 每日持仓(累积去重)
  data/sim/runs/{run_id}/state/{date}.json 盘中状态(live)
  data/sim/runs/{run_id}/run.log          运行日志
run_id 约定: {strategy}__main(每日主模拟) / {strategy}__bt_{ts}(回测) /
{strategy}__sim_{ts}(额外模拟, 可以某次回测为起点)。
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA  # noqa: E402

STRAT_DIR = ROOT / "strategies"
REGISTRY = STRAT_DIR / "strategies.yaml"
SIM_ROOT = DATA / "sim"
LOG_DIR = SIM_ROOT / "logs"
RUNS_DIR = SIM_ROOT / "runs"

# rqalpha 的 sys_analyser 会 import matplotlib; 默认 MPLCONFIGDIR(~/.matplotlib)
# 在无写权限环境下每次启动都刷一屏告警并重建字体缓存, 指到项目 logs 下消噪。
os.environ.setdefault("MPLCONFIGDIR", str(LOG_DIR / "mpl"))
Path(LOG_DIR / "mpl").mkdir(parents=True, exist_ok=True)


# ---------- run 目录与 meta ----------

def run_dir_of(run_id: str) -> Path:
    return RUNS_DIR / run_id


def read_meta(run_dir: Path) -> dict:
    f = Path(run_dir) / "meta.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_meta(run_dir: Path, **fields) -> dict:
    """合并写 meta.json(服务端先建 meta + pid, 子进程收尾补状态, 互不覆盖)"""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    m = read_meta(run_dir)
    m.update(fields)
    (run_dir / "meta.json").write_text(
        json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    return m


def seed_from_run(seed_run_id: str):
    """以某次回测的结束状态作为模拟起点: 结束持仓(数量) + 结束现金。

    rqalpha 的 init_positions 只收 {code: 数量}, 成本取继承日前收盘
    (portfolio/account.py 实测), 故继承持仓的均价会被重置为前收盘 ——
    引擎约束, 看板新建模拟表单里已注明。
    返回 (init_positions, cash, 持仓截止日)。"""
    import pandas as pd
    rd = run_dir_of(seed_run_id)
    pf, ef = rd / "positions.parquet", rd / "equity.parquet"
    if not pf.exists() or not ef.exists():
        raise ValueError(f"{seed_run_id} 缺持仓/净值记录, 不能作模拟起点")
    pos, eq = pd.read_parquet(pf), pd.read_parquet(ef).sort_values("trade_date")
    if not len(pos) or not len(eq):
        raise ValueError(f"{seed_run_id} 持仓/净值记录为空, 不能作模拟起点")
    last_date = str(pos["date"].astype(str).max())[:10]
    last = pos[pos["date"].astype(str).str[:10] == last_date]
    positions = {str(r.order_book_id): int(float(r.quantity))
                 for r in last.itertuples()
                 if float(getattr(r, "quantity", 0) or 0) > 0}
    cash = round(float(eq["cash"].iloc[-1]), 2)
    return positions, cash, last_date


# ---------- 配置读取 ----------

def load_registry() -> list:
    """strategies.yaml 的策略清单"""
    if not REGISTRY.exists():
        return []
    d = yaml.safe_load(REGISTRY.read_text(encoding="utf-8")) or {}
    return d.get("strategies") or []


def load_strategy_config(name: str) -> dict:
    """strategies/{name}/config.yaml; 缺失返回 {}"""
    f = STRAT_DIR / name / "config.yaml"
    if not f.exists():
        return {}
    return yaml.safe_load(f.read_text(encoding="utf-8")) or {}


def strategy_file(name: str) -> Path:
    return STRAT_DIR / name / "strategy.py"


# ---------- rqalpha 配置构造 ----------

def build_rqalpha_config(name: str, cfg: dict, mode: str,
                         start: str, end: str, frequency: str,
                         run_dir: Path, capital=None,
                         init_positions: dict | None = None) -> dict:
    """把策略 config.yaml 转成 rqalpha 的 config dict。

    run_type 一律用 BACKTEST: "模拟"性质来自 sys_simulation 的模拟撮合器,
    而非 run_type; 盘中实时由 TicaiLiveEventSource 的轮询驱动。用
    PAPER_TRADING 会额外触发 rqalpha 的 persist/restore 链路, 本框架的
    状态落盘由 StateRecorder 独立承担, 不需要它。

    init_positions: 以某次回测为起点时继承的结束持仓({rq代码: 数量});
    成本由引擎取继承日前收盘(不可指定, 见 seed_from_run 注释)。
    """
    mod_cfg = dict(cfg.get("mod") or {})
    mod_cfg["mode"] = mode
    mod_cfg["strategy"] = name
    mod_cfg["feeds"] = list(cfg.get("feeds") or [])
    mod_cfg["run_dir"] = str(run_dir)
    capital = int(capital or cfg.get("capital") or 1000000)
    # 基准: 策略 config.yaml 指定(默认自建打板基准 DBBNCH)。
    # 必须写进 base.benchmark 才能让 sys_analyser 算出 alpha/beta/夏普/
    # 信息比率/超额收益(否则全为 nan)。数据源会对面板未覆盖的尾部交易日
    # 做前向填充, 故盘中实时模式(面板未补尾)也不会因基准缺最后一天而崩。
    benchmark = cfg.get("benchmark") or "DBBNCH.XSHG"
    return {
        "base": {
            "start_date": _fmt_date(start),
            "end_date": _fmt_date(end),
            "frequency": frequency,
            "accounts": {"stock": capital},
            "benchmark": benchmark,
            "run_type": "b",
            "strategy_file": str(strategy_file(name)),
            "matching_type": "current_bar",
            "init_positions": dict(init_positions or {}),
            # 显式声明税率(保持 0 = 现状), 消掉 rqalpha 每次启动的
            # capital_gain_tax_rate 未配置 WARN
            "capital_gain_tax_rate": 0.0,
        },
        "extra": {
            "log_level": "info",
        },
        "mod": {
            "ticai": {
                "enabled": True,
                "lib": "rqalpha_mod_ticai",
                **mod_cfg,
            },
            # 保留 sys_simulation(模拟撮合器 + 涨跌停/无量/成交量约束)
            "sys_simulation": {
                "enabled": True,
                "price_limit": True,        # 涨跌停拒单(一字板由此拦住)
                "inactive_limit": True,     # bar 无量撤单
                "volume_limit": True,       # 单笔 ≤ bar 量 25%
                "volume_percent": 0.25,
            },
            "sys_analyser": {
                "enabled": True,
                "benchmark": benchmark,
                "output_file": str(Path(run_dir) / "analyser.pkl"),
                "report_save_path": str(Path(run_dir) / "analyser"),
            },
        },
    }


def _fmt_date(d: str) -> str:
    """YYYYMMDD → YYYY-MM-DD"""
    d = str(d)
    return f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else d


# ---------- 运行 ----------

# ---------- 跨日累积 ----------

def _accumulate_run(run_dir: Path) -> None:
    """把本次 run 的 trades/positions 追加进 run 目录的跨日持久化文件。

    为何需要: rqalpha 的 pkl 是【单次 run】的结果。live 模式每天跑单日,
    会覆盖 pkl → 详情页若只读 pkl 就只剩当天, 丢失"从模拟开始到
    当日"的整体记录(用户明确要求整体收益曲线)。故 trades/positions
    按 (稳定业务键) 去重后追加到 run_dir 下:
      trades.parquet    键 = datetime+order_book_id+side+last_quantity+last_price
        (不能用 order_id: 它跨 run 不稳定, 重启回放会重复追加)
      positions.parquet 键 = date+order_book_id
    净值/基准不在此累积 —— state.py 结算时已逐日 append 到
    run_dir/equity.parquet。

    重启回载: 不读 state 文件恢复账户, 而是靠 live 启动时的 catchup
    从当日开盘确定性回放重建(FILL_SIM 种子固定 → 采样序列确定 →
    同一次 run 重复跑结果逐笔一致, 已实测验证)。"""
    import pandas as pd
    pkl = Path(run_dir) / "analyser.pkl"
    if not pkl.exists():
        return
    try:
        import pickle
        d = pickle.load(open(pkl, "rb"))
    except Exception as e:
        print(f"[sim] 累积跳过(读pkl失败): {e}")
        return
    for key, idcols in (("trades", ["datetime", "order_book_id", "side",
                                  "last_quantity", "last_price"]),
                        ("stock_positions", ["date", "order_book_id"])):
        df = d.get(key)
        if df is None or not len(df):
            continue
        # trades 的 datetime 既是 index 又是列 → reset(drop) 即可;
        # stock_positions 的 date 只在 index 上 → 必须 reset() 保留成列,
        # 否则跨日持仓会丢掉日期(实测踩坑)。
        if key == "trades":
            df = df.reset_index(drop=True).copy()
        else:
            df = df.reset_index().copy()
        out = Path(run_dir) / ("trades.parquet" if key == "trades"
                               else "positions.parquet")
        old = pd.read_parquet(out) if out.exists() else None
        have = set()
        if old is not None and len(old):
            have = {tuple(str(r[c]) for c in idcols)
                    for r in old.itertuples() if all(hasattr(r, c) for c in idcols)}
        add = [r for r in df.itertuples()
               if tuple(str(getattr(r, c, "")) for c in idcols) not in have]
        if not add:
            continue
        add_df = pd.DataFrame([r._asdict() for r in add])
        merged = pd.concat([old, add_df], ignore_index=True) \
            if old is not None else add_df
        merged.to_parquet(out, index=False)
        print(f"[sim] 累积 {key}: +{len(add)} 行 → {out.parent.name}/"
              f"{out.name} (共 {len(merged)})")


def _write_backtest_records(run_dir: Path) -> None:
    """回测(单次 run 即完整记录): 从 pkl 一次性写出 equity/trades/positions。

    equity 取 pkl['portfolio'](逐日 total_value/cash/market_value);
    基准列存 benchmark_unit_net_value(归一净值) —— 看板展示时本来就要
    再归一, 尺度不影响曲线与 IR/alpha/beta。"""
    import numpy as np
    import pandas as pd
    import pickle
    d = pickle.load(open(Path(run_dir) / "analyser.pkl", "rb"))
    pf = d.get("portfolio")
    if pf is not None and len(pf):
        bm = (pf["benchmark_unit_net_value"].astype(float).values
              if "benchmark_unit_net_value" in pf.columns
              else np.full(len(pf), np.nan))
        eq = pd.DataFrame({
            "trade_date": [x.strftime("%Y%m%d") for x in pf.index],
            "equity": pf["total_value"].astype(float).round(2).values,
            "cash": pf["cash"].astype(float).round(2).values,
            "position_value": pf["market_value"].astype(float).round(2).values,
            "benchmark": np.round(bm, 6),
        })
        eq.to_parquet(Path(run_dir) / "equity.parquet", index=False)
    tr = d.get("trades")
    if tr is not None and len(tr):
        tr.reset_index(drop=True).to_parquet(
            Path(run_dir) / "trades.parquet", index=False)
    sp = d.get("stock_positions")
    if sp is not None and len(sp):
        sp.reset_index().to_parquet(
            Path(run_dir) / "positions.parquet", index=False)


def run_one(name: str, mode: str, start: str, end: str, frequency: str,
            run_id: str | None = None, seed_run: str | None = None,
            capital=None) -> int:
    """在当前进程内跑一个策略(子进程入口)"""
    cfg = load_strategy_config(name)
    sf = strategy_file(name)
    if not sf.exists():
        print(f"[sim] 策略文件缺失: {sf}")
        return 2
    if not run_id:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{name}__{'main' if mode == 'live' else 'bt_' + ts}"
    run_dir = run_dir_of(run_id)
    kind = "live" if mode == "live" else "backtest"
    init_positions, seed_cash, seed_date = None, None, None
    if seed_run:
        init_positions, seed_cash, seed_date = seed_from_run(seed_run)
        capital = capital or seed_cash
        print(f"[sim] 以回测 {seed_run} 为起点: 继承持仓 {len(init_positions)} "
              f"只(截至 {seed_date}) + 现金 {seed_cash}")
    write_meta(run_dir, id=run_id, kind=kind, strategy=name, mode=mode,
               start=str(start), end=str(end), freq=frequency,
               capital=int(capital or cfg.get("capital") or 1000000),
               benchmark=cfg.get("benchmark") or "DBBNCH.XSHG",
               seed_run=seed_run, seed_date=seed_date,
               # 保留首次创建时间(服务端可能已先建 meta)
               created_at=read_meta(run_dir).get("created_at")
               or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               status="running", pid=os.getpid())
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    config = build_rqalpha_config(name, cfg, mode, start, end, frequency,
                                  run_dir, capital=capital,
                                  init_positions=init_positions)
    print(f"[sim] 启动策略 {name} mode={mode} {start}~{end} "
          f"freq={frequency} 资金={config['base']['accounts']['stock']} "
          f"run={run_id}")

    import rqalpha
    t0 = time.time()
    try:
        rqalpha.run_file(str(sf), config)
    except Exception as e:
        print(f"[sim] 策略 {name} 运行失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        write_meta(run_dir, status="failed",
                   error=f"{type(e).__name__}: {e}",
                   finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   duration_sec=round(time.time() - t0, 1))
        return 1
    # 写 run 记录: 回测=单次 pkl 即完整记录; live=trades/positions 累积
    # (equity 已由 StateRecorder 结算时追加)。1d 频率的 live 不存在,
    # 回测一律不混入其它 run 的记录(每个 run 独立目录, 天然隔离)。
    try:
        if kind == "backtest":
            _write_backtest_records(run_dir)
        else:
            _accumulate_run(run_dir)
        from rqalpha_mod_ticai import metrics, state
        eq = metrics.load_equity(run_dir / "equity.parquet")
        m = metrics.compute_metrics(eq)
        st = state.load_state(run_dir, end)
        write_meta(run_dir, status="done", error=None, metrics=m,
                   finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   duration_sec=round(time.time() - t0, 1))
        print(f"[sim] {name} 完成: 净值 {st.get('equity')} "
              f"持仓 {len(st.get('positions') or [])} "
              f"样本 {m['days']} 天 sharpe={m['sharpe']} ir={m['ir']}")
    except Exception as e:
        print(f"[sim] 记录写出失败: {e}")
        write_meta(run_dir, status="done",
                   warn=f"记录写出失败: {e}",
                   finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   duration_sec=round(time.time() - t0, 1))
    return 0


def launch(name: str, args, run_id: str) -> subprocess.Popen:
    """拉起一个策略子进程(独立账户, 一个报错不拖垮其他)"""
    run_dir = run_dir_of(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    logf = run_dir / "run.log"
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()),
           "--run-one", name,
           "--mode", args.mode,
           "--start", args.start, "--end", args.end,
           "--freq", args.freq,
           "--run-id", run_id]
    if getattr(args, "seed_run", None):
        cmd += ["--seed-run", args.seed_run]
    if getattr(args, "capital", None):
        cmd += ["--capital", str(args.capital)]
    fh = open(logf, "a", encoding="utf-8")
    p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                         cwd=str(ROOT))
    write_meta(run_dir, pid=p.pid, status="running")
    print(f"[sim] {name} PID {p.pid} 日志 {logf}")
    return p


def migrate_legacy() -> int:
    """一次性把 run 目录模型之前的每策略累积数据迁到 runs/{name}__main。

    幂等: 目标目录已有 meta.json 就跳过; legacy 文件只复制不删除。
    返回迁移的 run 数。"""
    import shutil
    import pandas as pd
    eq_dir = SIM_ROOT / "equity"
    if not eq_dir.exists():
        return 0
    n = 0
    for eqf in sorted(eq_dir.glob("*.parquet")):
        name = eqf.stem
        rd = run_dir_of(f"{name}__main")
        if (rd / "meta.json").exists():
            continue
        rd.mkdir(parents=True, exist_ok=True)
        shutil.copy2(eqf, rd / "equity.parquet")
        for key in ("trades", "positions"):
            src = SIM_ROOT / key / f"{name}.parquet"
            if src.exists():
                shutil.copy2(src, rd / f"{key}.parquet")
        st_src = SIM_ROOT / "state" / name
        if st_src.exists():
            shutil.copytree(st_src, rd / "state", dirs_exist_ok=True)
        logs = sorted(LOG_DIR.glob(f"{name}_*.log"))
        if logs:
            shutil.copy2(logs[-1], rd / "run.log")
        eq = pd.read_parquet(rd / "equity.parquet").sort_values("trade_date")
        write_meta(rd, id=f"{name}__main", kind="live", strategy=name,
                   mode="live", freq="1m",
                   capital=int(round(float(eq["equity"].iloc[0]))),
                   benchmark="DBBNCH.XSHG", seed_run=None, seed_date=None,
                   created_at=str(eq["trade_date"].iloc[0]),
                   start=str(eq["trade_date"].iloc[0]),
                   end=str(eq["trade_date"].iloc[-1]),
                   status="running", pid=None)
        n += 1
        print(f"[sim] 迁移 {name} → runs/{name}__main ({len(eq)} 天)")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="策略模拟进程管理器")
    ap.add_argument("--strategy", help="只跑指定策略(默认按 strategies.yaml)")
    ap.add_argument("--mode", default="live", choices=["live", "replay"],
                    help="live=盘中实时 | replay=历史回放")
    ap.add_argument("--replay", action="store_true",
                    help="等价于 --mode replay")
    ap.add_argument("--date", help="单日(YYYYMMDD); 同时作为 start 与 end")
    ap.add_argument("--start", help="起始日 YYYYMMDD")
    ap.add_argument("--end", help="结束日 YYYYMMDD")
    ap.add_argument("--freq", default="1m", choices=["1m", "1d"],
                    help="1m=盘中快照粒度 | 1d=纯日线回测")
    ap.add_argument("--list", action="store_true", help="列出策略清单")
    ap.add_argument("--migrate", action="store_true",
                    help="迁移 legacy 每策略累积数据到 runs/{name}__main")
    ap.add_argument("--run-one", metavar="NAME",
                    help="(内部)子进程实际执行入口")
    ap.add_argument("--run-id", help="run 目录 ID(缺省: live=__main / 回测=__bt_ts)")
    ap.add_argument("--seed-run", help="以某次回测 run_id 为起点(继承结束持仓+现金)")
    ap.add_argument("--capital", type=float, help="初始资金(缺省: 策略 config/seed 现金)")
    args = ap.parse_args()

    if args.replay:
        args.mode = "replay"

    if args.list:
        for r in load_registry():
            cfg = load_strategy_config(r["name"])
            print(f"{r['name']:<20} enabled={r.get('enabled')} "
                  f"资金={cfg.get('capital')} 基准={cfg.get('benchmark')} "
                  f"feeds={cfg.get('feeds')}")
            print(f"{'':<20} {r.get('description', '')}")
        return 0

    if args.migrate:
        print(f"[sim] 迁移完成: {migrate_legacy()} 个 run")
        return 0

    if args.run_one:
        return run_one(args.run_one, args.mode, args.start, args.end,
                       args.freq, run_id=args.run_id,
                       seed_run=args.seed_run, capital=args.capital)

    # ---- 日期默认值 ----
    today = datetime.now().strftime("%Y%m%d")
    if args.date:
        args.start = args.end = args.date
    else:
        args.start = args.start or today
        args.end = args.end or today

    # ---- 拉起 ----
    targets = []
    if args.strategy:
        targets = [args.strategy]
    else:
        targets = [r["name"] for r in load_registry() if r.get("enabled")]
    if not targets:
        print("[sim] 无启用策略(strategies.yaml 里 enabled=true 为空)")
        return 0

    procs = {}
    for name in targets:
        if not strategy_file(name).exists():
            print(f"[sim] 跳过 {name}: 策略文件缺失")
            continue
        run_id = f"{name}__main"
        # 已被用户关闭的主模拟不再被每日拉起复活(关闭是显式意图)
        if read_meta(run_dir_of(run_id)).get("status") == "closed":
            print(f"[sim] 跳过 {name}: 主模拟已关闭(meta.status=closed)")
            continue
        procs[name] = launch(name, args, run_id)

    if not procs:
        return 0
    print(f"[sim] 已拉起 {len(procs)} 个策略进程, 等待结束...")
    codes = {}
    for name, p in procs.items():
        codes[name] = p.wait()
        print(f"[sim] {name} 退出码 {codes[name]}")
    return 0 if all(c == 0 for c in codes.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
