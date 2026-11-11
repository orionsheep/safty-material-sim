"""曲线看板用的数据：全局曲线（每日活跃度、低于安全库存的物料数、服务水平扫描、在库变化分布）
和单物料曲线（在库、累计需求、服务水平 / 交货期敏感性）。只依赖 numpy。"""
from __future__ import annotations

import datetime as dt
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .batch import DATASET, RESULTS, TARGET_MTYPES, _mtype, load_rows, run_batch
from .model import LEVELS, Params, analyze, history_onhand, level_params, material_from_dataset, split_demand

ROOT = Path(__file__).resolve().parent.parent
PORTFOLIO_CACHE = ROOT / "data" / "gwms" / "curves-portfolio.json"
LEVELS_CACHE = ROOT / "data" / "gwms" / "curves-levels.json"
INDEX_CACHE = ROOT / "data" / "gwms" / "curves-index.json"
INDEX_CLIP = 5.0               # 单个物料最多按自己平均实际在库的 5 倍计，免得个别物料把指数拉偏
CACHE_VERSION = "v4"           # 引擎改了随机种子 / 上调规则后，旧缓存作废

SL_LEVELS = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98]
LEAD_LEVELS = [3, 5, 7, 10, 14, 20, 30, 45, 60]
PORTFOLIO_LEVELS = [0.80, 0.85, 0.90, 0.95, 0.98]
LEAD_GROUPS = {"f-1采购周期": "f-1 采购周期", "默认值": "缺交货期（默认值）"}


def _r(x, nd=2):
    return None if x is None else round(float(x), nd)


# ---------------------------------------------------------------- 全局

def overview(dataset: dict, results: dict) -> dict:
    """GWMS5 窗口内每天：有出库的物料数（覆盖 / 按单），实际自由在库低于现行 / 建议安全库存的物料数。"""
    n = int(dataset["window"]["days"]) - 1           # 去掉不完整的导出当天
    rows_by_code = {}
    for r in results["rows"]:
        rows_by_code.setdefault(r["code"], r)
    active_cover = np.zeros(n)
    active_bom = np.zeros(n)
    active_in = np.zeros(n)
    below_cur = np.zeros(n)
    below_sug = np.zeros(n)
    tracked = 0
    params = Params()
    for code in dataset["materials"]:
        r = rows_by_code.get(code)
        md = material_from_dataset(dataset, code, (r or {}).get("unit") or "")
        if md is None:
            continue
        cover, bom = split_demand(md, params)
        active_cover += cover[:n] > 0
        active_bom += bom[:n] > 0
        active_in += (md.in_free + md.in_bom)[:n] > 0
        if r and r.get("status") == "ok" and r.get("ss") and r.get("cur"):
            hist = history_onhand(md)
            if hist["negative"]:
                continue
            free = hist["free"][:n]
            below_cur += free < r["cur"]
            below_sug += free < r["ss"]
            tracked += 1
    start = dataset["window"]["start"]
    d0 = dt.date.fromisoformat(start)
    dates = [(d0 + dt.timedelta(days=i)).isoformat() for i in range(n)]
    return {
        "window": dataset["window"], "start": start, "dates": dates,
        "activeCover": active_cover.astype(int).tolist(),
        "activeBom": active_bom.astype(int).tolist(),
        "activeIn": active_in.astype(int).tolist(),
        "belowCur": below_cur.astype(int).tolist(),
        "belowSug": below_sug.astype(int).tolist(),
        "tracked": tracked,
        "savingCdf": saving_cdf(results["rows"]),
        "resultsBuiltAt": results.get("builtAt"),
        "serviceLevel": results.get("params", {}).get("serviceLevel"),
    }


def saving_cdf(rows: list[dict]) -> dict:
    """对照实际自由在库，建议规则在库变化（+ 为增加）的累计分布，按交货期来源分组。"""
    xs = list(range(-100, 301, 10))
    out = {"x": xs, "series": {}}
    groups = {"全部": [r for r in rows if r.get("savingActual") is not None]}
    for src, label in LEAD_GROUPS.items():
        groups[label] = [r for r in groups["全部"] if r.get("leadSrc") == src]
    for label, g in groups.items():
        if not g:
            continue
        ch = np.array([-r["savingActual"] * 100 for r in g])
        out["series"][label] = {"n": len(g), "median": _r(np.median(ch), 1),
                                "y": [_r((ch <= x).mean() * 100, 1) for x in xs]}
    return out


def portfolio_sweep(bom_basis: str = "attr", force: bool = False) -> dict:
    """不同服务水平下整批重算：建议规则在库低于实际的物料占比、在库变化中位数、偏差大的物料数。有缓存。"""
    built = json.loads(DATASET.read_text(encoding="utf-8"))["meta"]["builtAt"]
    key = f"{CACHE_VERSION}|{built}|{bom_basis}"
    if PORTFOLIO_CACHE.exists() and not force:
        cache = json.loads(PORTFOLIO_CACHE.read_text(encoding="utf-8"))
        if cache.get("key") == key:
            return cache
    rows = load_rows()
    levels = []
    for sl in PORTFOLIO_LEVELS:
        res = run_batch(Params(service_level=sl, bom_basis=bom_basis), rows=rows, workers=4)
        ok = [r for r in res["rows"] if r.get("savingActual") is not None]
        item = {"sl": sl, "high": res["summary"]["alert"].get("high", 0), "groups": {}}
        groups = {"全部": ok}
        for src, label in LEAD_GROUPS.items():
            groups[label] = [r for r in ok if r.get("leadSrc") == src]
        for label, g in groups.items():
            if g:
                ch = np.array([-r["savingActual"] for r in g])
                item["groups"][label] = {"n": len(g), "lowerShare": _r((ch < 0).mean() * 100, 1),
                                         "median": _r(np.median(ch) * 100, 1)}
        levels.append(item)
    out = {"key": key, "bomBasis": bom_basis, "levels": levels}
    PORTFOLIO_CACHE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def level_compare(bom_basis: str = "attr", force: bool = False) -> dict:
    """四档激进程度各跑一遍整批：建议方向（降 / 持平 / 升）、对照实际的在库降幅、按规则执行时的模拟缺货。有缓存。"""
    built = json.loads(DATASET.read_text(encoding="utf-8"))["meta"]["builtAt"]
    key = f"{CACHE_VERSION}|{built}|{bom_basis}"
    if LEVELS_CACHE.exists() and not force:
        cache = json.loads(LEVELS_CACHE.read_text(encoding="utf-8"))
        if cache.get("key") == key:
            return cache
    rows = load_rows()
    xs = list(range(-100, 301, 10))
    out = {"key": key, "bomBasis": bom_basis, "x": xs, "levels": []}
    for name, cfg in LEVELS.items():
        res = run_batch(level_params(name, bom_basis=bom_basis), rows=rows, workers=4)
        sm = res["summary"]
        ok = [r for r in res["rows"] if r.get("savingActual") is not None]
        ch = np.array([-r["savingActual"] * 100 for r in ok])
        by_src = {}
        for src, label in {**LEAD_GROUPS, "SAP实际交货天数中位数": "SAP 交货期"}.items():
            g = [r for r in res["rows"] if r.get("leadSrc") == src and r["status"] == "ok" and r.get("simCur")]
            if not g:
                continue
            by_src[label] = {"n": len(g),
                             "down": sum(1 for r in g if (r.get("dev") or 0) < -0.005),
                             "up": sum(1 for r in g if (r.get("dev") or 0) > 0.005),
                             "medianChange": _r(np.median([-r["savingActual"] * 100 for r in g
                                                           if r.get("savingActual") is not None] or [0]), 1)}
        out["levels"].append({
            "name": name, "desc": cfg["desc"], "serviceLevel": cfg["service_level"], "upRule": cfg["up_rule"],
            "defaultLeadDays": cfg["default_lead_days"],
            "n": sm["comparable"], "down": sm["suggestDown"], "same": sm["suggestSame"], "up": sm["suggestUp"],
            "hold": sm["hold"], "alertHigh": sm["alert"].get("high", 0),
            "withActual": len(ok), "lowerThanActual": int((ch < 0).sum()),
            "medianChange": _r(np.median(ch), 1) if len(ch) else None,
            "simShortMedian": sm["simShortMedian"], "simShortP90": sm["simShortP90"],
            "simCurShortMedian": sm["simCurShortMedian"], "fillMean": sm["fillMean"],
            "cdf": [_r((ch <= x).mean() * 100, 1) for x in xs] if len(ch) else [],
            "byLeadSrc": by_src,
        })
    LEVELS_CACHE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


_IX_DS: dict | None = None
_IX_PARAMS: Params | None = None


def _ix_init(path: str, params: Params) -> None:
    global _IX_DS, _IX_PARAMS
    _IX_DS = json.loads(Path(path).read_text(encoding="utf-8"))
    _IX_PARAMS = params


def _ix_one(row: dict):
    """一个物料的在库轨迹 ÷ 它自己近两个月平均实际自由在库：过去（实际）+ 未来 60 天（两种规则 P50）。"""
    md = material_from_dataset(_IX_DS, str(row.get("no") or ""), row.get("unit") or "")
    if md is None:
        return None
    r = analyze(md, row, _IX_PARAMS, with_series=True)
    act, s = r.get("actual"), r.get("series")
    if r.get("status") != "ok" or not act or act["avgFree"] <= 0 or not s or "projCur" not in s:
        return None
    base = act["avgFree"]
    f = lambda arr: np.clip(np.array(arr, dtype=float) / base, 0, INDEX_CLIP)
    return {"mtype": _mtype(row), "past": f(s["onhandFree"][:-1]), "sug": f(s["projSug"]["p50"]),
            "cur": f(s["projCur"]["p50"]), "dates": s["dates"][:-1], "proj": s["projDates"]}


def stock_index(params: Params, results_key: str, force: bool = False) -> dict:
    """全部物料的在库指数（各自近两个月平均实际自由在库 = 100）：过去实际 + 未来 60 天按建议 / 现行规则。有缓存。"""
    key = f"{CACHE_VERSION}|{results_key}"
    if INDEX_CACHE.exists() and not force:
        cache = json.loads(INDEX_CACHE.read_text(encoding="utf-8"))
        if cache.get("key") == key:
            return cache
    rows = [r for r in load_rows() if str(r.get("status", 0)) != "1" and _mtype(r) in TARGET_MTYPES]
    with ProcessPoolExecutor(max_workers=4, initializer=_ix_init, initargs=(str(DATASET), params)) as ex:
        items = [x for x in ex.map(_ix_one, rows, chunksize=64) if x is not None]
    names = {1: "生产类", 2: "非生产类", 7: "发车"}
    groups = {"全部": items}
    for t, nm in names.items():
        groups[nm] = [x for x in items if x["mtype"] == t]
    out = {"key": key, "level": params.level, "clip": INDEX_CLIP,
           "dates": items[0]["dates"] if items else [], "projDates": items[0]["proj"] if items else [], "groups": {}}
    for nm, g in groups.items():
        if not g:
            continue
        tail = slice(-30, None)                       # 后 30 天基本稳定，作为「稳定后」水平
        item = {"n": len(g)}
        for stat, fn in (("median", np.median), ("mean", np.mean)):
            past = fn([x["past"] for x in g], axis=0) * 100
            sug = fn([x["sug"] for x in g], axis=0) * 100
            cur = fn([x["cur"] for x in g], axis=0) * 100
            sug_end = np.array([x["sug"][tail].mean() for x in g])
            item[stat] = {"past": [_r(v, 1) for v in past], "sug": [_r(v, 1) for v in sug], "cur": [_r(v, 1) for v in cur],
                          "pastAvg": _r(past.mean(), 1), "sugSteady": _r(sug[tail].mean(), 1),
                          "curSteady": _r(cur[tail].mean(), 1)}
        item["lowerShare"] = _r((sug_end < 1).mean() * 100, 1)     # 稳定后低于自己实际平均在库的物料占比
        out["groups"][nm] = item
    INDEX_CACHE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


# ---------------------------------------------------------------- 单物料

def material_curves(dataset: dict, row: dict, params: Params) -> dict:
    code = str(row.get("no") or "")
    md = material_from_dataset(dataset, code, row.get("unit") or "")
    base = analyze(md, row, params, with_series=True)
    out = {"base": base}
    if md is None or base.get("ss") is None:
        return out

    # 累计需求
    cover, bom = split_demand(md, params)
    n = md.days - 1
    cum_cover = np.cumsum(cover[:n])
    out["cumulative"] = {
        "dates": md.dates[:n],
        "cover": [_r(v) for v in cum_cover],
        "bom": [_r(v) for v in np.cumsum(bom[:n])],
        "even": [_r(cum_cover[-1] * (i + 1) / n) for i in range(n)] if n else [],
    }

    # 服务水平扫描
    sl_pts = []
    for sl in SL_LEVELS:
        r = analyze(md, row, params.with_(service_level=sl))
        if r.get("ss") is None:
            continue
        sg = r.get("simSug") or {}
        sl_pts.append({"sl": sl, "ss": r["ss"], "max": r["max"], "avg": sg.get("avg"),
                       "stockoutDays": sg.get("stockoutDays"), "fill": sg.get("fill")})
    out["serviceSweep"] = sl_pts

    # 交货期扫描
    lead_pts = []
    for L in sorted(set(LEAD_LEVELS + [int(round(base["lead"]))])):
        r = analyze(md, row, params.with_(lead_override=float(L)))
        if r.get("ss") is None:
            continue
        sg = r.get("simSug") or {}
        lead_pts.append({"lead": L, "ss": r["ss"], "avg": sg.get("avg"), "stockoutDays": sg.get("stockoutDays")})
    out["leadSweep"] = lead_pts

    # 四档激进程度下这个物料的建议值（需求口径、交货期手动值沿用当前参数）
    lv = []
    for name in LEVELS:
        r = analyze(md, row, level_params(name, bom_basis=params.bom_basis, lead_override=params.lead_override,
                                          use_sap_history=params.use_sap_history))
        if r.get("ss") is None:
            continue
        sg = r.get("simSug") or {}
        lv.append({"name": name, "ss": r["ss"], "rawSs": r.get("rawSs"), "max": r["max"], "hold": r.get("hold"),
                   "lead": r["lead"], "avg": sg.get("avg"), "stockoutDays": sg.get("stockoutDays"),
                   "savingActual": r.get("savingActual")})
    out["levels"] = lv
    return out
