# -*- coding: utf-8 -*-
"""研究25b: 跷跷板对手板块筛选 离线复现（新旧口径对比）

背景: core/seesaw.py 对手口径升级为"资金流入耦合+题材独立"后, 需验证
新筛选是否消除了旧口径的三类错误(自指同主题/跨龙头复用同一批全局强势
板块/与龙头无因果)。

数据源(全部已落盘, 无需盘中):
  data/live/concept_px_YYYYMMDD.json  板块均涨分时 {concept: [[HHMMSS,均涨]]}
  data/live/seesaw_YYYYMMDD.jsonl     旧口径触发事件(kind=trigger, 含旧opp)

复现口径(忠实复现2/3闸, 放量闸需盘中成交额时序, 离线不复现):
  ②题材独立: 与龙头概念成分重叠率≤OVERLAP_MAX 且名称不同主题根
  ③均涨耦合: 对手板块均涨3分钟增量≥OPP_DAVG_MIN(取自concept_px真实均涨)
  排序: 按均涨增量降序(放量/广度项离线缺数据, 不参与)

用法: python research/25b_seesaw_opp_replay.py [YYYYMMDD]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.attribute import load_con2stock, load_maps  # noqa: E402
from core.seesaw import OPP_DAVG_MIN, OVERLAP_MAX  # noqa: E402

LIVE = Path(__file__).resolve().parent.parent / "data" / "live"


def sec(hms: str) -> int:
    """'093414'/'09:34:14' → 当日秒数"""
    h = hms.replace(":", "")
    return int(h[0:2]) * 3600 + int(h[2:4]) * 60 + int(h[4:6])


def at(series: list, target: int):
    """series=[(sec,均涨)] 首个 sec≥target 的均涨; 无则回退最早"""
    for ts, v in series:
        if ts >= target:
            return v
    return series[0][1] if series else None


def replay_opponents(k: str, code: str, t_sec: int, series: dict,
                     con2stock: dict, cname: dict) -> list:
    """新口径离线复现: 同龙头闸 + 题材独立闸 + 均涨耦合闸, 按均涨增量排序"""
    kset = set(con2stock.get(k, []))
    kn = cname.get(k, k)
    cands = []
    for k2, s in series.items():
        if k2 == k or len(s) < 2 or s[-1][0] < t_sec - 180:
            continue
        # ①同龙头闸: 候选含本龙头票=同一批钱, 非对手
        if code in set(con2stock.get(k2, [])):
            continue
        # ②题材独立: 成分重叠率 + 名称同主题根
        if kset and len(kset & set(con2stock.get(k2, []))) / len(kset) \
                > OVERLAP_MAX:
            continue
        n2 = cname.get(k2, k2)
        if kn and n2 and len(kn) >= 2 and (kn[:2] in n2 or n2[:2] in kn):
            continue
        # ③均涨耦合: 板块均涨3分钟增量
        d_avg = round(at(s, t_sec) - at(s, t_sec - 180), 2)
        if d_avg < OPP_DAVG_MIN:
            continue
        cands.append((k2, n2, d_avg, at(s, t_sec)))
    cands.sort(key=lambda x: -x[2])
    return cands[:5]


def main():
    day = sys.argv[1] if len(sys.argv) > 1 else None
    if day is None:
        fs = sorted(LIVE.glob("seesaw_*.jsonl"))
        day = fs[-1].stem.split("_")[1] if fs else None
    if not day:
        print("无 seesaw_*.jsonl 数据")
        return
    cpx_f = LIVE / f"concept_px_{day}.json"
    ss_f = LIVE / f"seesaw_{day}.jsonl"
    if not cpx_f.exists() or not ss_f.exists():
        print(f"缺 {day} 的 concept_px / seesaw 数据")
        return
    con2stock = load_con2stock()
    _, _, cname = load_maps()
    cpx = json.loads(cpx_f.read_text(encoding="utf-8"))
    series = {k: [(sec(hm), v) for hm, v in rows]
              for k, rows in cpx.items()}
    trig = []
    for line in ss_f.read_text(encoding="utf-8").strip().splitlines():
        d = json.loads(line)
        if d.get("kind") == "trigger":
            trig.append(d)
    print(f"=== {day} 对手板块新旧口径对比 (触发{len(trig)}条) ===\n")

    # 概念口径守卫: 若当日seesaw概念码与当前con2stock不匹配(如THS数字码
    # vs 当前kpl名), 无法离线复现, 提醒换kpl口径日
    ss_codes = {e["concept_code"] for e in trig}
    matched = sum(1 for c in ss_codes if c in con2stock)
    if ss_codes and matched < 0.5 * len(ss_codes):
        print(f"⚠ {day} 概念口径与当前con2stock不一致(匹配 "
              f"{matched}/{len(ss_codes)}): 该日可能用THS数字码而当前为"
              f"kpl名, 无法离线复现; 请换kpl口径交易日重跑")
        return

    # ---- 逐事件前后对比(前6条) ----
    for e in trig[:6]:
        t_sec = sec(e["t"])
        new = replay_opponents(e["concept_code"], e["leader_code"], t_sec,
                               series, con2stock, cname)
        old = [(o["name"], o.get("avg_pct")) for o in e.get("opp", [])]
        print(f"[{e['t']}] {e['concept_name']} · 龙头{e['leader_name']}")
        print(f"  旧对手: {old}")
        print(f"  新对手: {[(n, da, f'均涨{ap}') for _, n, da, ap in new]}"
              if new else "  新对手: 无(无独立放量上涨板块→偏消息面杀跌)")
        print()

    # ---- 聚合: 旧对手被闸淘汰比例 / 新旧对手多样性 ----
    n_old = n_selfref = 0
    old_names, new_names, empty = [], [], 0
    n_old_fall = 0
    for e in trig:
        t_sec = sec(e["t"])
        kset = set(con2stock.get(e["concept_code"], []))
        kn = cname.get(e["concept_code"], e["concept_code"])
        for o in e.get("opp", []):
            n_old += 1
            k2 = o["concept_code"]
            old_names.append(o["name"])
            if (o.get("avg_pct") or 0) < 0:   # 仍在下跌却被当对手
                n_old_fall += 1
            ov = (len(kset & set(con2stock.get(k2, []))) / len(kset)
                  if kset else 0)
            n2 = cname.get(k2, k2)
            if (k2 != e["concept_code"]
                    and e["leader_code"] in set(con2stock.get(k2, []))) \
                    or ov > OVERLAP_MAX \
                    or (kn and n2 and len(kn) >= 2
                        and (kn[:2] in n2 or n2[:2] in kn)):
                n_selfref += 1
        new = replay_opponents(e["concept_code"], e["leader_code"], t_sec,
                               series, con2stock, cname)
        if not new:
            empty += 1
        new_names += [n for _, n, _, _ in new]
    print("---- 聚合 ----")
    print(f"旧对手总数 {n_old}, 其中应被三闸淘汰(自指/同龙头/同主题) "
          f"{n_selfref} ({100*n_selfref/max(n_old,1):.0f}%)")
    print(f"旧对手中仍在下跌(avg_pct<0)却被当对手: {n_old_fall} "
          f"({100*n_old_fall/max(n_old,1):.0f}%) — 跷跷板对手应为上涨板块")
    print(f"旧对手多样性: 去重 {len(set(old_names))} 种 / 累计 {len(old_names)} "
          f"个 (越少=同一批全局强势板块被跨龙头复用)")
    print(f"新对手多样性: 去重 {len(set(new_names))} 种 / 累计 "
          f"{len(new_names)} 个; {empty}/{len(trig)} 事件无有效对手(资金未明显切换)")
    print("注: 放量闸(成交额增速)与上涨家数广度需盘中时序, 离线不复现; "
          "盘中实盘由 core/seesaw.py 完整四闸判定。")


if __name__ == "__main__":
    main()
