# -*- coding: utf-8 -*-
"""题材分层树构建器（THS 与 kpl 分口径各建树 + 跨口径映射, 成员包含率驱动）

为什么分口径(修复跨口径混淆):
  同一题材在 THS 与 kpl 下范围差异极大(如"军工" THS 606只 vs kpl 176只),
  若合并成一棵树, kpl军工⊂ths军工=0.93 会被误判为"ths军工的L2子级"——把
  "同题材不同口径"错当"父子分支"。故: **每个口径各自按成员包含率建森林**
  (同口径内包含=真父子), 再用跨口径映射表标注同题材的等价/宽窄关系。

方法(数据驱动, 无需手工 taxonomy):
  1. 读 THS(theme.concepts/members)+kpl(theme.kpl_concepts/kpl_members)的
     is_theme 真题材及成分集合, 各打 source 标签。
  2. 分口径建树: 对每个 source 内部, 节点A的父=满足 cont(A⊂B)=|A∩B|/|A|>=
     PARENT_THR 且 |B|>|A| 的【最小】B(最紧父级); 父级严格更大→无环成森林。
     level: 根=L1主线, BFS 深度递增。实测同口径内: 存储芯片⊂芯片概念=0.93→父子。
  3. 跨口径映射(xcal, 不建层级边): 对每个节点在【另一口径】里找最佳对应:
     同名→equiv(同题材不同范围); 否则按包含率判 narrower/broader/overlap。
  4. scope_tier 按 member_count 分档(窄<30<=中<=100<宽)。

产出 theme.hierarchy(static parquet): ths 森林 + kpl 森林, 每节点带 level/
parent/scope 及其跨口径对应 xcal_*。供 KB 档案挂层级 + 点火探测器"相对父级剥
beta / 兄弟共振"(父级取同口径 parent, 跨口径经 xcal 归一)。

用法:
  python collect/build_theme_taxonomy.py                    # 构建并落库
  python collect/build_theme_taxonomy.py --stats            # 统计(分口径)
  python collect/build_theme_taxonomy.py --branch 芯片概念   # 打印某分支子树
  python collect/build_theme_taxonomy.py --source kpl --stats  # 只看某口径
  python collect/build_theme_taxonomy.py --dump data/theme/hierarchy_tree.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from datastore import load, save  # noqa: E402

PARENT_THR = 0.75        # 同口径 A⊂B 包含率阈值: 达到则 B 是 A 的父级候选
XCAL_THR = 0.50          # 跨口径对应的最小包含率(低于则视为该口径独有)
MIN_MEMBERS = 5          # 成分<5 的概念视为噪音, 不入树
SCOPE_NARROW, SCOPE_WIDE = 30, 100   # 窄<30<=中<=100<宽

HIER_COLS = ["concept_code", "name", "source", "level", "parent_code",
             "parent_name", "cont_to_parent", "member_count", "scope_tier",
             "is_root", "children_count", "n_descendants",
             "xcal_code", "xcal_name", "xcal_source", "xcal_rel", "xcal_cont"]


def _scope_tier(n: int) -> str:
    return "窄" if n < SCOPE_NARROW else "中" if n <= SCOPE_WIDE else "宽"


def _cont(a: set, b: set) -> float:
    return len(a & b) / len(a) if a else 0.0


def load_nodes() -> list:
    """读 THS + kpl 真题材为节点: {code,name,source,members(set),n}"""
    nodes = []
    try:
        ths = load("theme.concepts")
        mem = load("theme.members")
        mem_by = {c: set(g["con_code"]) for c, g in mem.groupby("concept_code")}
        for _, r in ths[ths["is_theme"] == True].iterrows():  # noqa: E712
            ms = mem_by.get(r["ts_code"], set())
            if len(ms) >= MIN_MEMBERS:
                nodes.append({"code": r["ts_code"], "name": r["name"],
                              "source": "ths", "members": ms, "n": len(ms)})
    except Exception as e:
        print(f"[taxonomy] THS 载入失败: {e}")
    try:
        kpl = load("theme.kpl_concepts")
        kmem = load("theme.kpl_members")
        kmem_by = {c: set(g["con_code"]) for c, g in kmem.groupby("concept_code")}
        for _, r in kpl[kpl["is_theme"] == True].iterrows():  # noqa: E712
            ms = kmem_by.get(r["ts_code"], set())
            if len(ms) >= MIN_MEMBERS:
                nodes.append({"code": r["ts_code"], "name": r["name"],
                              "source": "kpl", "members": ms, "n": len(ms)})
    except Exception as e:
        print(f"[taxonomy] kpl 载入失败: {e}")
    return nodes


def build_tree_per_source(nodes: list) -> tuple:
    """分口径建树: 父级只在【同 source】内找(消除跨口径混淆)。
    返回 (parent{code:pcode}, cont{code:包含率})。"""
    parent, cont_p = {}, {}
    for src in ("ths", "kpl"):
        sub = sorted([n for n in nodes if n["source"] == src],
                     key=lambda x: x["n"])       # 升序, 父级只在更大的里找
        for i, a in enumerate(sub):
            best, best_n, best_c = None, None, 0.0
            for b in sub[i + 1:]:
                if b["n"] <= a["n"]:
                    continue
                c = _cont(a["members"], b["members"])
                if c >= PARENT_THR and (best_n is None or b["n"] < best_n):
                    best, best_n, best_c = b, b["n"], c
            if best is not None:
                parent[a["code"]] = best["code"]
                cont_p[a["code"]] = round(best_c, 3)
    return parent, cont_p


def cross_caliber_link(nodes: list) -> dict:
    """跨口径映射(不建层级边): 每个节点在另一口径里的对应。
    只保留两类高置信关联(丢弃宽松误配):
      equiv   = 同名, 同题材不同范围(如军工ths606 ≡ 军工kpl176);
      broader = 本节点在另一口径里的【最紧父容器】(cont(A⊂B)>=PARENT_THR
                且 |B| 最小), 供点火探测剔beta时取跨口径父级。
    无则 None(该口径独有)。返回 {code: (node, rel, cont)}。"""
    ths = [n for n in nodes if n["source"] == "ths"]
    kpl = [n for n in nodes if n["source"] == "kpl"]
    out = {}
    for a in nodes:
        others = kpl if a["source"] == "ths" else ths
        same = [b for b in others if b["name"] == a["name"]]
        if same:
            b = max(same, key=lambda x: _cont(a["members"], x["members"]))
            out[a["code"]] = (b, "equiv", round(_cont(a["members"], b["members"]), 3))
            continue
        # 最紧父容器: 另一口径中包含 A 达阈且规模最小的 B
        cands = [(b, _cont(a["members"], b["members"])) for b in others
                 if b["n"] > a["n"]]
        cands = [(b, c) for b, c in cands if c >= PARENT_THR]
        if cands:
            b, c = min(cands, key=lambda bc: bc[0]["n"])
            out[a["code"]] = (b, "broader", round(c, 3))
        else:
            out[a["code"]] = None
    return out


def assign_levels(nodes: list, parent: dict) -> dict:
    """根(无父)=L1, BFS 深度递增(parent 已限定同口径)"""
    children = defaultdict(list)
    for c, p in parent.items():
        children[p].append(c)
    roots = [n["code"] for n in nodes if n["code"] not in parent]
    level = {r: 1 for r in roots}
    dq = deque(roots)
    while dq:
        cur = dq.popleft()
        for ch in children.get(cur, []):
            if ch not in level:
                level[ch] = level[cur] + 1
                dq.append(ch)
    for n in nodes:
        level.setdefault(n["code"], 1)
    return level


def build_hierarchy() -> pd.DataFrame:
    nodes = load_nodes()
    if not nodes:
        return pd.DataFrame(columns=HIER_COLS)
    parent, cont_p = build_tree_per_source(nodes)
    xcal = cross_caliber_link(nodes)
    level = assign_levels(nodes, parent)
    by_code = {n["code"]: n for n in nodes}
    children = defaultdict(list)
    for c, p in parent.items():
        children[p].append(c)

    def n_desc(code, seen=None):
        seen = seen if seen is not None else set()
        cnt = 0
        for ch in children.get(code, []):
            if ch in seen:
                continue
            seen.add(ch)
            cnt += 1 + n_desc(ch, seen)
        return cnt

    rows = []
    for n in nodes:
        c = n["code"]
        p = parent.get(c)
        x = xcal.get(c)
        rows.append({
            "concept_code": c, "name": n["name"], "source": n["source"],
            "level": int(level.get(c, 1)),
            "parent_code": p, "parent_name": by_code[p]["name"] if p else None,
            "cont_to_parent": cont_p.get(c),
            "member_count": int(n["n"]), "scope_tier": _scope_tier(n["n"]),
            "is_root": p is None,
            "children_count": len(children.get(c, [])),
            "n_descendants": n_desc(c),
            "xcal_code": x[0]["code"] if x else None,
            "xcal_name": x[0]["name"] if x else None,
            "xcal_source": x[0]["source"] if x else None,
            "xcal_rel": x[1] if x else None,
            "xcal_cont": x[2] if x else None,
        })
    df = pd.DataFrame(rows, columns=HIER_COLS)
    return df.sort_values(["source", "level", "member_count"],
                          ascending=[True, True, False]).reset_index(drop=True)


def _stats(df: pd.DataFrame):
    for src in ("ths", "kpl"):
        d = df[df["source"] == src]
        if d.empty:
            continue
        print(f"\n== {src} 口径: {len(d)} 节点 | 根/L1 {int(d['is_root'].sum())} "
              f"| level {d['level'].value_counts().sort_index().to_dict()} "
              f"| scope {d['scope_tier'].value_counts().to_dict()}")
        roots = d[d["is_root"]].nlargest(12, "n_descendants")
        print("   主线(L1, 按后代数):")
        for _, r in roots.iterrows():
            print(f"     {r['name']:12s} |{r['member_count']:>4d}| "
                  f"后代{r['n_descendants']:>3d}")
    both = df[df["xcal_code"].notna()]
    equiv = df[df["xcal_rel"] == "equiv"]
    print(f"\n跨口径映射: {len(both)}/{len(df)} 节点有对应; 其中同名等价 {len(equiv)} 个")
    if len(equiv):
        print("   同名等价样例(同题材不同范围):")
        for _, r in equiv.head(10).iterrows():
            print(f"     {r['name']}[{r['source']} |{r['member_count']}] ≡ "
                  f"{r['xcal_name']}[{r['xcal_source']}] cont={r['xcal_cont']}")


def _branch(df: pd.DataFrame, name: str):
    """打印以 name 为根的子树(同口径内, 缩进=层级)"""
    by_code = {r["concept_code"]: r for _, r in df.iterrows()}
    node = df[df["name"] == name]
    if node.empty:
        node = df[df["concept_code"] == name]
    if node.empty:
        print(f"未找到题材 '{name}'")
        return
    for _, root_row in node.iterrows():
        root = root_row["concept_code"]
        children = defaultdict(list)
        for _, r in df.iterrows():
            if r["parent_code"]:
                children[r["parent_code"]].append(r["concept_code"])

        def walk(code, depth):
            r = by_code[code]
            x = (f" ≡{r['xcal_name']}[{r['xcal_source']}]"
                 if r["xcal_code"] else "")
            print(f"{'  ' * depth}{r['name']} |{r['member_count']}| "
                  f"{r['scope_tier']} L{r['level']} cont={r['cont_to_parent']} "
                  f"[{r['source']}]{x}")
            for ch in sorted(children.get(code, []),
                             key=lambda x: -by_code[x]["member_count"]):
                walk(ch, depth + 1)
        walk(root, 0)


def _dump_tree(df: pd.DataFrame, path: Path):
    """把两口径森林(所有层级)写成缩进 markdown, 供人工浏览。"""
    by_code = {r["concept_code"]: r for _, r in df.iterrows()}
    children = defaultdict(list)
    for _, r in df.iterrows():
        if r["parent_code"] and r["parent_code"] in by_code:
            children[r["parent_code"]].append(r["concept_code"])
    lines = ["# 题材分层树(THS 与 kpl 分口径各建, 成员包含率数据驱动)", "",
             f"共 {len(df)} 节点; 分口径避免'同题材不同范围'被误当父子。",
             "格式: 名称 |成分数| scope L层级 cont=包含父级率 [源] ≡跨口径对应", ""]

    def walk(code, depth):
        r = by_code[code]
        x = f" ≡{r['xcal_name']}[{r['xcal_source']}]" if r["xcal_code"] else ""
        lines.append(f"{'  ' * depth}- {r['name']} |{r['member_count']}| "
                     f"{r['scope_tier']} L{r['level']} "
                     f"cont={r['cont_to_parent']} [{r['source']}]{x}")
        for ch in sorted(children.get(code, []),
                         key=lambda x: -by_code[x]["member_count"]):
            walk(ch, depth + 1)
    for src in ("ths", "kpl"):
        d = df[df["source"] == src]
        lines += [f"## {src} 口径森林 ({len(d)} 节点, "
                  f"{int(d['is_root'].sum())} 条主线)", ""]
        roots = d[d["is_root"]].sort_values(
            ["n_descendants", "member_count"], ascending=False)
        for _, r in roots.iterrows():
            walk(r["concept_code"], 0)
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[taxonomy] 全量树写出 {len(df)} 节点 → {path}")


def cli() -> int:
    ap = argparse.ArgumentParser(description="题材分层树构建器(分口径)")
    ap.add_argument("--branch", help="打印某题材的子树")
    ap.add_argument("--stats", action="store_true", help="打印分口径统计")
    ap.add_argument("--source", choices=["ths", "kpl"], help="只看某口径")
    ap.add_argument("--dump", help="把全量分层树写为 markdown(路径)")
    ap.add_argument("--no-save", action="store_true", help="只算不落库")
    args = ap.parse_args()
    if args.dump:
        df = load("theme.hierarchy")
        if args.source:
            df = df[df["source"] == args.source]
        _dump_tree(df, Path(args.dump))
        return 0
    if args.stats or args.branch:
        df = load("theme.hierarchy")
        if args.source:
            df = df[df["source"] == args.source]
        if args.branch:
            _branch(df, args.branch)
        else:
            _stats(df)
        return 0
    df = build_hierarchy()
    if df.empty:
        print("[taxonomy] 空结果(数据缺失?)")
        return 1
    if not args.no_save:
        p = save("theme.hierarchy", df)
        print(f"[taxonomy] 落库 {len(df)} 节点 → {p}")
    _stats(df)
    return 0


if __name__ == "__main__":
    sys.exit(cli())
