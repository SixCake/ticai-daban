# -*- coding: utf-8 -*-
"""pandas 版本变更基线校验 — 降级前后逐项比对

背景: 引入 rqalpha(要求 pandas<3.0)需把项目 pandas 从 3.0.x 降到 2.x。
降级风险不在 API(已扫描: 375处用法零命中 pandas 3.0 独有 API), 而在
① pandas 3.0 写出的 parquet 能否被 2.x 无损读回(string 列 dtype
   会从 str 变 object) ② akshare/tushare 在 pandas 2.x 上行为是否漂移。

用法:
  python research/verify_pandas_env.py --tag before   # 降级前
  pip install "pandas>=2.0,<3.0"
  python research/verify_pandas_env.py --tag after    # 降级后
  python research/verify_pandas_env.py --diff          # 逐项比对

产物 logs/env_baseline_{tag}.json; 比对只校验确定性字段(shape/dtypes/
校验值/行数), 跳过时间戳等易变项。
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA  # noqa: E402
from datastore import DATASETS, partition_dates, path_of  # noqa: E402

# 重点校验的数据集(体积大/被全链路依赖)
KEY_DATASETS = ["market.daily_panel", "factor.longtou",
                "limitup.events_enriched", "limitup.ths_limit",
                "theme.attribution", "theme.day", "meta.trade_cal"]
# 每个数据集取前 N 行做行级指纹(避免全量哈希过慢)
FINGERPRINT_ROWS = 50
# 关键数值列的校验值(降级不应改变任何数值)
NUMERIC_CHECKS = {
    "market.daily_panel": ["close", "open", "high", "low", "vol",
                           "pre_close", "pct_chg", "open_ret"],
    "factor.longtou": None,          # 自动取全部数值列
    "limitup.events_enriched": None,
}


def _dtype_str(df: pd.DataFrame) -> dict:
    return {c: str(t) for c, t in df.dtypes.items()}


def _numeric_stats(df: pd.DataFrame, cols) -> dict:
    """数值列 sum/mean/non-null 计数, 用于逐值比对。
    不存原始浮点值而存相对精度后的值: sum 可达 1e12, round(x,4) 在
    不同 pandas/numpy 上会因浮点累加顺序丢末位, 造成假阳性。"""
    if cols is None:
        cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])]
    out = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        tot = float(s.sum())
        out[c] = {
            # 相对精度: 保留 12 位有效数字(足以捕捉真差异, 又不受末位抖动)
            "sum_sig": f"{tot:.12g}" if tot else "0",
            "mean_sig": f"{float(s.mean()):.12g}" if s.notna().any() else None,
            "nn": int(s.notna().sum()),
            "n_nan": int(s.isna().sum()),
        }
    return out


def _row_fingerprint(df: pd.DataFrame, n: int) -> str:
    """前 n 行的确定性指纹(排序后哈希, 与 dtype/NA 表述无关)。
    数值列统一 round 到 6 位; 非数值列先按 core.times.hhmmss6 同款掩码
    把缺失还原为 '<NA>' 再转文本 — 否则 pandas 3.0 的 str dtype 保留 NA
    而 2.x 的 object dtype 转成 'nan'/'None', 指纹会假性不一致。"""
    head = df.head(n).copy()
    for c in head.columns:
        if pd.api.types.is_numeric_dtype(head[c]):
            head[c] = pd.to_numeric(head[c], errors="coerce").round(6)
            head[c] = head[c].map(lambda v: "<NA>" if pd.isna(v)
                                  else f"{v:.6f}")
        else:
            m = head[c].notna()
            head[c] = head[c].astype(str).where(m, "<NA>")
    blob = json.dumps(head.to_dict(orient="list"), sort_keys=True,
                      default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _dataset_snapshot(name: str) -> dict:
    try:
        p = path_of(name)
        if not p.exists():
            return {"status": "missing"}
        df = pd.read_parquet(p)
    except Exception as e:
        return {"status": f"error: {str(e)[:120]}"}
    return {
        "status": "ok",
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": _dtype_str(df),
        "numeric": _numeric_stats(df, NUMERIC_CHECKS.get(name)),
        "fingerprint": _row_fingerprint(df, FINGERPRINT_ROWS),
    }


def _partition_snapshot(name: str) -> dict:
    """分区数据集: 取最新分区做快照"""
    try:
        ds = partition_dates(name)
        if not ds:
            return {"status": "no-partition"}
        d = ds[-1]
        df = pd.read_parquet(path_of(name, date=d))
        return {"status": "ok", "date": d, "n_partitions": len(ds),
                "shape": list(df.shape), "columns": list(df.columns),
                "dtypes": _dtype_str(df),
                "numeric": _numeric_stats(df, None),
                "fingerprint": _row_fingerprint(df, FINGERPRINT_ROWS)}
    except Exception as e:
        return {"status": f"error: {str(e)[:120]}"}


def _json_snapshot() -> dict:
    """meta/*.json 的键数与抽样值(降级不影响 json, 但一并记录)"""
    out = {}
    for f in ["qmt_names.json", "sw_map.json", "industry_map.json",
              "struct_grids.json", "qmt_avg5vol.json"]:
        p = DATA / "meta" / f
        if not p.exists():
            out[f] = "missing"
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            inner = d.get("data", d) if isinstance(d, dict) else d
            out[f] = {"n": len(inner) if hasattr(inner, "__len__") else -1,
                      "bytes": p.stat().st_size}
        except Exception as e:
            out[f] = f"error: {str(e)[:80]}"
    return out


def _api_probe() -> dict:
    """akshare / tushare 接口探活(降级后须返回同结构)"""
    out = {}
    try:
        import akshare as ak
        df = ak.stock_zt_pool_em(date="20260902")
        out["akshare.stock_zt_pool_em"] = {
            "shape": list(df.shape), "columns": list(df.columns),
            "dtypes": _dtype_str(df)}
    except Exception as e:
        out["akshare.stock_zt_pool_em"] = f"error: {str(e)[:120]}"
    try:
        from config import get_pro
        pro = get_pro()
        df = pro.daily(trade_date="20260902")
        out["tushare.daily"] = {"shape": list(df.shape),
                                "columns": list(df.columns),
                                "dtypes": _dtype_str(df)}
    except Exception as e:
        out["tushare.daily"] = f"error: {str(e)[:120]}"
    return out


def _str_ops_probe() -> dict:
    """string 列操作探活: pandas 3.0 str dtype → 2.x object dtype 后
    .str.zfill/.str.contains 等行为必须一致(降级最大风险点)。
    实测发现: astype(str) 对缺失值的处理两版不一致(3.0 保留 NA,
    2.x 转成 'nan'/'None'), 故除裸操作外额外验证 core.times.hhmmss6
    的版本无关性。"""
    out = {}
    try:
        df = pd.read_parquet(path_of("limitup.events_enriched"),
                             columns=["ts_code"])
        s = df["ts_code"].head(200)
        out["dtype"] = str(s.dtype)
        out["zfill6"] = s.astype(str).str.zfill(6).head(3).tolist()
        out["split"] = s.astype(str).str.split(".").str[0].head(3).tolist()
        out["contains_sz"] = int(s.astype(str).str.contains("SZ").sum())
    except Exception as e:
        out["error"] = str(e)[:120]
    # 缺失值行为对比: 裸 astype(str)(预期两版不一致) vs hhmmss6(预期一致)
    try:
        import numpy as np
        from core.times import hhmmss6, is_before
        raw = pd.Series(["093000", None, np.nan, "094500", 93006],
                        dtype=object)
        out["raw_astype"] = raw.astype(str).str.zfill(6).tolist()
        out["hhmmss6"] = [None if pd.isna(v) else v
                          for v in hhmmss6(raw).tolist()]
        out["is_before"] = is_before(raw, "094500").tolist()
    except Exception as e:
        out["times_error"] = str(e)[:120]
    return out


def collect() -> dict:
    snap = {"env": {"python": sys.version.split()[0],
                    "pandas": pd.__version__,
                    "numpy": np.__version__,
                    "pyarrow": __import__("pyarrow").__version__},
            "datasets": {}, "meta_json": _json_snapshot(),
            "api": _api_probe(), "str_ops": _str_ops_probe()}
    for name in KEY_DATASETS:
        rel, freq, partitioned, _ = DATASETS[name]
        snap["datasets"][name] = (_partition_snapshot(name) if partitioned
                                  else _dataset_snapshot(name))
    # 分区数据集另记
    for name in ["limitup.zt_minute"]:
        snap["datasets"][name] = _partition_snapshot(name)
    return snap


VOLATILE = {"api", "env"}     # 时间戳/版本字段: 不参与 diff


def diff(before: dict, after: dict) -> int:
    """逐项比对; 返回不一致项数"""
    bad = 0
    print(f"pandas: {before['env']['pandas']} → {after['env']['pandas']}")
    print(f"numpy : {before['env']['numpy']} → {after['env']['numpy']}")
    print(f"pyarrow: {before['env']['pyarrow']} → "
          f"{after['env']['pyarrow']}\n")
    # ---- 数据集 ----
    for name in sorted(set(before["datasets"]) | set(after["datasets"])):
        b, a = before["datasets"].get(name), after["datasets"].get(name)
        if b is None or a is None:
            print(f"[缺失] {name}: before={b is not None} after={a is not None}")
            bad += 1
            continue
        if b["status"] != a["status"]:
            print(f"[状态] {name}: {b['status']} → {a['status']}")
            bad += 1
            continue
        if b["status"] != "ok":
            print(f"[跳过] {name}: {b['status']}")
            continue
        diffs = []
        if b["shape"] != a["shape"]:
            diffs.append(f"shape {b['shape']}→{a['shape']}")
        if b["columns"] != a["columns"]:
            diffs.append("columns 变化")
        if b["fingerprint"] != a["fingerprint"]:
            diffs.append(f"行指纹 {b['fingerprint']}→{a['fingerprint']}")
        if b["numeric"] != a["numeric"]:
            for k in set(b["numeric"]) | set(a["numeric"]):
                if b["numeric"].get(k) != a["numeric"].get(k):
                    diffs.append(f"数值列 {k}: "
                                 f"{b['numeric'].get(k)}→"
                                 f"{a['numeric'].get(k)}")
        # dtypes 允许 str→object 这类表述差异, 只报类别变化
        dt_diff = []
        for c in set(b["dtypes"]) | set(a["dtypes"]):
            tb, ta = b["dtypes"].get(c), a["dtypes"].get(c)
            if tb == ta:
                continue
            cat_b = "str" if tb in ("str", "object", "string") else tb
            cat_a = "str" if ta in ("str", "object", "string") else ta
            if cat_b != cat_a:
                dt_diff.append(f"{c}: {tb}→{ta}")
        if dt_diff:
            diffs.append("dtype类别变化 " + "; ".join(dt_diff))
        if diffs:
            print(f"[不一致] {name}: " + " | ".join(diffs))
            bad += 1
        else:
            note = ""
            if b["dtypes"] != a["dtypes"]:
                note = " (dtype表述差异已容忍)"
            print(f"[一致] {name}{note}")
    # ---- meta json ----
    for k in sorted(set(before["meta_json"]) | set(after["meta_json"])):
        b, a = before["meta_json"].get(k), after["meta_json"].get(k)
        if b != a:
            print(f"[不一致] meta/{k}: {b} → {a}")
            bad += 1
        else:
            print(f"[一致] meta/{k}")
    # ---- string 操作 ----
    b, a = before["str_ops"], after["str_ops"]
    # 必须一致(版本无关的封装)
    for k in ["zfill6", "split", "contains_sz", "hhmmss6", "is_before"]:
        if b.get(k) != a.get(k):
            print(f"[不一致] str_ops.{k}: {b.get(k)} → {a.get(k)}")
            bad += 1
        else:
            print(f"[一致] str_ops.{k}")
    # 已知会不一致(裸 astype(str) 对 NA 的处理): 只作提示不计入失败
    if b.get("raw_astype") != a.get("raw_astype"):
        print(f"[已知差异] str_ops.raw_astype: {b.get('raw_astype')} → "
              f"{a.get('raw_astype')}")
        print("           ↑ 裸 astype(str) 对缺失值的处理两版本不同, "
              "已由 core/times.py 统一封装规避")
    if b.get("dtype") != a.get("dtype"):
        print(f"[提示] str_ops.dtype: {b.get('dtype')} → {a.get('dtype')} "
              f"(预期内, pandas 3.0 str → 2.x object)")
    # ---- API(结构比对, 数值随行情变动不比) ----
    for k in sorted(set(before["api"]) | set(after["api"])):
        b, a = before["api"].get(k), after["api"].get(k)
        if isinstance(b, dict) and isinstance(a, dict):
            if b["columns"] != a["columns"]:
                print(f"[不一致] api.{k} columns: {b['columns']}→{a['columns']}")
                bad += 1
            else:
                print(f"[一致] api.{k} columns ({len(a['columns'])}列)")
        elif isinstance(b, str) and isinstance(a, str):
            print(f"[提示] api.{k} 两次均失败: {a[:60]}")
        else:
            print(f"[不一致] api.{k}: {type(b).__name__}→{type(a).__name__}")
            bad += 1
    return bad


def cli():
    ap = argparse.ArgumentParser(description="pandas 版本变更基线校验")
    ap.add_argument("--tag", help="采集基线: before|after|任意标签")
    ap.add_argument("--diff", action="store_true",
                    help="比对 logs/env_baseline_before.json 与 _after.json")
    args = ap.parse_args()
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    if args.diff:
        fb = logs / "env_baseline_before.json"
        fa = logs / "env_baseline_after.json"
        for f in (fb, fa):
            if not f.exists():
                print(f"缺基线文件: {f} (先跑 --tag before / --tag after)")
                return 2
        n = diff(json.loads(fb.read_text(encoding="utf-8")),
                 json.loads(fa.read_text(encoding="utf-8")))
        print(f"\n{'=' * 50}\n比对结果: {n} 项不一致")
        print("可接受: pandas 降级后 0 项不一致(dtype表述差异已容忍)")
        return 1 if n else 0
    if not args.tag:
        ap.print_help()
        return 2
    print(f"采集基线 tag={args.tag} (pandas {pd.__version__}) ...")
    snap = collect()
    out = logs / f"env_baseline_{args.tag}.json"
    out.write_text(json.dumps(snap, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"写出 {out} ({out.stat().st_size // 1024}KB)")
    print(f"数据集 {len(snap['datasets'])} 个, 状态: "
          f"{ {k: v['status'] for k, v in snap['datasets'].items()} }")
    return 0


if __name__ == "__main__":
    sys.exit(cli())
