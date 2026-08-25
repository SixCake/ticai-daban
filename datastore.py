# -*- coding: utf-8 -*-
"""数据目录与统一查询层（唯一出处）

目录规范: data/{域}/{频率}/{数据集}
  域:   limitup 涨停 | theme 题材 | market 行情 | meta 基础
  频率: 1d 日频 | 1m 分钟 | static 无时间维度
  live/ 与 review/ 为盘中瞬态/缓存, 不入目录

注册表 DATASETS 是唯一的路径出处, 其他脚本禁止硬编码 data/ 路径:
  from datastore import load, save, path_of
  df = load("limitup.events_enriched", columns=["trade_date", "ts_code"])
  df = load("limitup.zt_minute", date="20260825")          # 分区数据集
  save("limitup.zt_minute", df, date="20260825")
  p  = path_of("theme.day")

CLI:
  python datastore.py list                 # 数据集清单
  python datastore.py info limitup.events  # 元信息
  python datastore.py head theme.day -n 5 [--date 20260825]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# name → (相对data路径, 频率, 是否按date分区, 说明)
DATASETS = {
    "limitup.events": (
        "limitup/1d/events.parquet", "1d", False,
        "涨停事件原始库 tushare limit_list_d(U) 增量, 2019-11起"),
    "limitup.events_enriched": (
        "limitup/1d/events_enriched.parquet", "1d", False,
        "事件富化: 一字板/T+1开高低收收益/ST标记"),
    "limitup.zt_minute": (
        "limitup/1m/zt_minute_{date}.parquet", "1m", True,
        "当日封板组+炸板组(触板未封)标的1分钟K线, 东财源"),
    "theme.concepts": (
        "theme/static/concepts.parquet", "static", False,
        "同花顺概念列表(过滤噪音后约350个真题材)"),
    "theme.members": (
        "theme/static/members.parquet", "static", False,
        "概念成分快照(当前, 无日期维度)"),
    "theme.attribution": (
        "theme/1d/attribution.parquet", "1d", False,
        "涨停事件×概念独占归属(迭代投票)"),
    "theme.day": (
        "theme/1d/theme_day.parquet", "1d", False,
        "题材日度快照: 涨停数/连板/龙头/归属家数/年龄"),
    "market.daily_panel": (
        "market/1d/daily_panel.parquet", "1d", False,
        "全A日度行情面板(涨跌幅/成交额/换手/涨停价), 约200MB"),
    "meta.trade_cal": (
        "meta/trade_cal.parquet", "static", False,
        "SSE交易日历缓存"),
}

# 分区数据集文件名中的日期格式
DATE_FMT = "{date}"


def _spec(name: str):
    if name not in DATASETS:
        known = ", ".join(sorted(DATASETS))
        raise KeyError(f"未知数据集 '{name}', 可选: {known}")
    return DATASETS[name]


def path_of(name: str, date: str | None = None) -> Path:
    """数据集物理路径; 分区数据集须给 date(YYYYMMDD)"""
    rel, freq, partitioned, _ = _spec(name)
    if partitioned:
        if not date:
            raise ValueError(f"{name} 按日分区, 须提供 date=YYYYMMDD")
        rel = rel.replace(DATE_FMT, str(date))
    elif "{date}" in rel:
        rel = rel.replace("_{date}", "")
    return DATA / rel


def load(name: str, date: str | None = None, columns=None,
         **filters) -> pd.DataFrame:
    """统一读取; filters 为列等值过滤(如 trade_date='20260825')"""
    p = path_of(name, date)
    if not p.exists():
        raise FileNotFoundError(f"{name} 不存在: {p} (先跑对应 collect/build)")
    df = pd.read_parquet(p, columns=columns)
    for col, val in filters.items():
        if col not in df.columns:
            raise KeyError(f"{name} 无列 '{col}'")
        df = df[df[col] == val]
    return df


def save(name: str, df: pd.DataFrame, date: str | None = None) -> Path:
    """统一写出(自动建目录)"""
    p = path_of(name, date)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


def partition_dates(name: str) -> list[str]:
    """分区数据集已有的日期列表(升序)"""
    rel, _, partitioned, _ = _spec(name)
    if not partitioned:
        raise ValueError(f"{name} 非分区数据集")
    prefix, suffix = rel.split(DATE_FMT)   # "limitup/1m/zt_minute_" / ".parquet"
    base = (DATA / prefix).parent          # data/limitup/1m
    stem_prefix = Path(prefix).name        # "zt_minute_"
    out = []
    if base.exists():
        for f in base.glob(f"{stem_prefix}*{suffix}"):
            d = f.stem[len(stem_prefix):]
            if d.isdigit():
                out.append(d)
    return sorted(out)


def _human(n: int) -> str:
    for u in ("B", "K", "M", "G"):
        if n < 1024:
            return f"{n:.0f}{u}" if u == "B" else f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}T"


def cli():
    ap = argparse.ArgumentParser(description="数据目录查询")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list")
    p_info = sub.add_parser("info")
    p_info.add_argument("name")
    p_head = sub.add_parser("head")
    p_head.add_argument("name")
    p_head.add_argument("--date")
    p_head.add_argument("-n", type=int, default=5)
    args = ap.parse_args()

    if args.cmd == "list":
        print(f"{'数据集':<24}{'频率':<8}{'状态':<10}{'体积':<9}路径")
        for name in sorted(DATASETS):
            rel, freq, partitioned, _ = DATASETS[name]
            try:
                if partitioned:
                    ds = partition_dates(name)
                    status = f"{len(ds)}分区" if ds else "缺失"
                    size = sum(f.stat().st_size
                               for d in ds for f in
                               [path_of(name, d)] if f.exists())
                else:
                    p = path_of(name)
                    status = "存在" if p.exists() else "缺失"
                    size = p.stat().st_size if p.exists() else 0
            except Exception:
                status, size = "缺失", 0
            print(f"{name:<24}{freq:<8}{status:<10}"
                  f"{_human(size) if size else '-':<9}{rel}")
    elif args.cmd == "info":
        rel, freq, partitioned, desc = _spec(args.name)
        p = path_of(args.name, date="*") if partitioned else path_of(args.name)
        print(f"名称: {args.name}\n说明: {desc}\n频率: {freq}\n"
              f"分区: {'date' if partitioned else '-'}\n路径: {p}")
        if partitioned:
            ds = partition_dates(args.name)
            print(f"已有分区: {len(ds)} 个 "
                  f"({ds[0]}~{ds[-1]})" if ds else "已有分区: 无")
    elif args.cmd == "head":
        df = load(args.name, date=args.date)
        print(df.head(args.n).to_string())
    else:
        ap.print_help()


if __name__ == "__main__":
    sys.exit(cli())
