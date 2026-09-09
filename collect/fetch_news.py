# -*- coding: utf-8 -*-
"""数据源采集器 — 轮询订阅的新闻/RSS流 → 落盘原始条目(供 LLM 线索分析)

产物(见 core/newsfeed.py):
  data/live/news/{source_id}/{date}.json

支持源类型(type, 见 sources.json):
  cls  财联社电报   akshare stock_info_global_cls
  em   东财全球快讯 akshare stock_info_global_em
  ths  同花顺快讯   akshare stock_info_global_ths
  announce 巨潮公司公告  cninfo 官方query接口(个股催化最早出处, 非RSS)
  rss  通用 RSS/Atom  requests + xml.etree 轻量解析(不引 feedparser)

时间戳: 优先用新闻自带发布时间(回放安全); 解析不出则落采集墙钟。
akshare 接口名/列名随版本变动, 全部 try/except 多回退, 缺失或失败打印跳过不崩。

用法:
  python collect/fetch_news.py                    # 采集一次(全部启用源)
  python collect/fetch_news.py --source cls_telegraph
  python collect/fetch_news.py --loop             # 盘中持续(按各源 poll_sec)
"""
import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import newsfeed  # noqa: E402

_DT_FMTS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S",
            "%Y%m%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ")
_TIME_FMTS = ("%H:%M:%S", "%H:%M")


def _dt_to_epoch(s, day: str) -> float | None:
    """把新闻发布时间串转 epoch; 解析失败返回 None(由 append_items 补墙钟)。
    纯时间串(无日期)用 day 补日期。"""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    for f in _DT_FMTS:
        try:
            return datetime.strptime(s, f).timestamp()
        except ValueError:
            continue
    for f in _TIME_FMTS:
        try:
            t = datetime.strptime(s, f).time()
            d = datetime.strptime(day, "%Y%m%d").date()
            return datetime.combine(d, t).timestamp()
        except ValueError:
            continue
    return None


def _strip_html(s) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"<[^>]+>", "", s).strip()


def _row_ts(row: dict, day: str) -> float | None:
    """从一行新闻里挑发布时间(兼容各源列名)"""
    for k in ("发布时间", "发布时间和日期", "时间", "publish_time", "datetime",
              "pubDate", "update_time"):
        if row.get(k):
            ts = _dt_to_epoch(str(row[k]), day)
            if ts:
                return ts
    # 日期+时间两列拼接
    d, t = row.get("发布日期"), row.get("发布时间")
    if d and t:
        ts = _dt_to_epoch(f"{d} {t}", day)
        if ts:
            return ts
    return None


# ---------- 各源抓取: 返回 [{title,text,url,ts}] ----------

def _fetch_cls(day: str) -> list:
    import akshare as ak
    df = None
    for fn in ("stock_info_global_cls", "stock_telegraph_cls"):
        try:
            df = getattr(ak, fn)()
            if df is not None and len(df):
                break
        except Exception:
            df = None
    if df is None or not len(df):
        return []
    out = []
    for _, r in df.iterrows():
        row = r.to_dict()
        title = str(row.get("标题") or "").strip()
        if not title:
            continue
        out.append({"title": title,
                    "text": _strip_html(row.get("内容")) or title,
                    "url": str(row.get("链接") or row.get("url") or ""),
                    "ts": _row_ts(row, day)})
    return out


def _fetch_em(day: str) -> list:
    import akshare as ak
    df = None
    for fn in ("stock_info_global_em", "stock_info_global_cls_em"):
        try:
            df = getattr(ak, fn)()
            if df is not None and len(df):
                break
        except Exception:
            df = None
    if df is None or not len(df):
        return []
    out = []
    for _, r in df.iterrows():
        row = r.to_dict()
        title = str(row.get("标题") or row.get("摘要") or "").strip()
        if not title:
            continue
        out.append({"title": title,
                    "text": _strip_html(row.get("摘要") or row.get("内容"))
                            or title,
                    "url": str(row.get("链接") or row.get("url") or ""),
                    "ts": _row_ts(row, day)})
    return out


def _fetch_ths(day: str) -> list:
    import akshare as ak
    df = None
    for fn in ("stock_info_global_ths", "stock_info_global_sina"):
        try:
            df = getattr(ak, fn)()
            if df is not None and len(df):
                break
        except Exception:
            df = None
    if df is None or not len(df):
        return []
    out = []
    for _, r in df.iterrows():
        row = r.to_dict()
        title = str(row.get("标题") or "").strip()
        if not title:
            continue
        out.append({"title": title,
                    "text": _strip_html(row.get("内容") or row.get("摘要"))
                            or title,
                    "url": str(row.get("链接") or row.get("url") or ""),
                    "ts": _row_ts(row, day)})
    return out


_CNINFO_URL = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
_ANN_BULL = ("中标", "合同", "签署", "合作", "增资", "回购", "增持", "预增",
             "扭亏", "获批", "并购", "重组", "收购", "分红", "举牌")
_ANN_BEAR = ("减持", "处罚", "违规", "立案", "亏损", "预亏", "下调", "停牌",
             "退市", "警示", "问询", "澄清")


def _fetch_announce(day: str) -> list:
    """巨潮资讯当日公司公告(个股催化最早出处, 非RSS结构化源)。
    只保留标题命中利好/利空关键词的公告, 避免全量公告淹没LLM输入。"""
    import requests
    se = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    out = []
    for page in (1, 2):
        payload = {"pageNum": page, "pageSize": 30, "column": "szse",
                   "tabName": "fulltext", "plate": "", "stock": "",
                   "searchkey": "", "secid": "", "category": "", "trade": "",
                   "seDate": se, "sortName": "", "sortType": "",
                   "isHLtitle": "true"}
        r = requests.post(_CNINFO_URL, data=payload, timeout=20,
                          headers={"User-Agent": "Mozilla/5.0",
                                   "Origin": "http://www.cninfo.com.cn",
                                   "Referer": "http://www.cninfo.com.cn/"})
        r.raise_for_status()
        anns = (r.json() or {}).get("announcements") or []
        for a in anns:
            title = _strip_html(a.get("announcementTitle") or "").strip()
            name = (a.get("secName") or "").strip()
            if not title:
                continue
            bull = any(w in title for w in _ANN_BULL)
            bear = any(w in title for w in _ANN_BEAR)
            if not (bull or bear):
                continue
            raw_ts = a.get("announcementTime") or 0
            ts = (raw_ts / 1000.0) if raw_ts > 10 ** 12 \
                else (float(raw_ts) or None)
            adj = a.get("adjunctUrl") or ""
            out.append({
                "title": f"{name}: {title}" if name else title,
                "text": title,
                "url": ("http://static.cninfo.com.cn/" + adj) if adj else "",
                "ts": ts,
                "extra": {"code": a.get("secCode"), "name": name,
                          "hint": "bull" if bull and not bear
                          else "bear" if bear and not bull else "mix"},
            })
        if not anns:
            break
    return out


def _fetch_rss(day: str, url: str) -> list:
    import xml.etree.ElementTree as ET
    import requests
    if not url:
        return []
    r = requests.get(url, timeout=20,
                     headers={"User-Agent": "Mozilla/5.0 (ticai-daban)"})
    r.raise_for_status()
    root = ET.fromstring(r.content)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    nodes = root.findall(".//item") or root.findall(".//a:entry", ns)
    out = []
    for n in nodes[:80]:
        def txt(tag, atom=None):
            e = n.find(tag)
            if e is None and atom:
                e = n.find(atom, ns)
            return (e.text or "").strip() if e is not None and e.text else ""
        title = txt("title") or txt("a:title", "a:title")
        if not title:
            continue
        link_e = n.find("link")
        link = (link_e.text or "").strip() if link_e is not None else ""
        if not link and link_e is not None:
            link = link_e.get("href", "")
        body = _strip_html(txt("description") or txt("summary")
                           or txt("a:summary", "a:summary")) or title
        pub = txt("pubDate") or txt("a:updated", "a:updated") \
            or txt("a:published", "a:published")
        out.append({"title": title, "text": body, "url": link,
                    "ts": _dt_to_epoch(pub, day)})
    return out


def collect_one(src: dict, day: str) -> int:
    """采集一个源并落盘; 返回新增条目数"""
    sid, stype = src.get("id"), src.get("type", "rss")
    try:
        if stype == "cls":
            items = _fetch_cls(day)
        elif stype == "em":
            items = _fetch_em(day)
        elif stype == "ths":
            items = _fetch_ths(day)
        elif stype == "announce":
            items = _fetch_announce(day)
        elif stype == "rss":
            items = _fetch_rss(day, src.get("url", ""))
        else:
            print(f"[fetch_news] 未知源类型 {stype} ({sid}), 跳过")
            return 0
    except Exception as e:
        print(f"[fetch_news] {sid} 采集失败: {type(e).__name__}: {e}")
        return 0
    if not items:
        return 0
    p = newsfeed.append_items(sid, day, items)
    print(f"[fetch_news] {sid}({stype}) {day}: {len(items)} 条 → {p}")
    return len(items)


def cli() -> int:
    ap = argparse.ArgumentParser(description="新闻数据源采集器")
    ap.add_argument("--source", help="只采集指定源 id")
    ap.add_argument("--date", help="目标日期 YYYYMMDD(默认今天)")
    ap.add_argument("--loop", action="store_true", help="盘中持续采集")
    args = ap.parse_args()

    day = args.date or datetime.now().strftime("%Y%m%d")
    sources = newsfeed.enabled_sources()
    if args.source:
        sources = [s for s in newsfeed.load_sources()
                   if s.get("id") == args.source]
        if not sources:
            print(f"[fetch_news] 未知源 {args.source}")
            return 1

    if not args.loop:
        n = sum(collect_one(s, day) for s in sources)
        print(f"[fetch_news] 完成: 共 {n} 条")
        return 0

    print(f"[fetch_news] 持续采集模式, 源 {[s['id'] for s in sources]}, "
          f"Ctrl-C 退出")
    last: dict = {}
    while True:
        now = time.time()
        day = datetime.now().strftime("%Y%m%d")
        # 每轮重读注册表: 看板「源订阅配置」的启用/新增开关即时生效,
        # 无需重启采集器
        for s in newsfeed.enabled_sources():
            if now - last.get(s["id"], 0.0) >= int(s.get("poll_sec", 300)):
                collect_one(s, day)
                last[s["id"]] = now
        time.sleep(5)


if __name__ == "__main__":
    sys.exit(cli())
