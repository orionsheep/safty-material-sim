#!/usr/bin/env python3
"""批量试算：f-1 物料清单 × GWMS5 数据集 → 每个物料的建议安全库存与报警。

输入：data/materials-all.json（f-1 物料清单）、data/gwms/dataset.json（etl_gwms.py 的输出）。
输出：data/gwms/results.json —— 每行一个 f-1 物料（不含图表序列），外加汇总。

只算三类安全库存：生产类（1）、非生产类（2）、发车（7）；已删除（status=1）的行跳过。
同一参数、同一份数据重复跑，结果不变（随机数按物料编码固定种子）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from engine.model import LEVELS, Params, analyze, level_params, material_from_dataset  # type: ignore
else:
    from .model import LEVELS, Params, analyze, level_params, material_from_dataset

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "gwms" / "dataset.json"
MATERIALS = ROOT / "data" / "materials-all.json"
RESULTS = ROOT / "data" / "gwms" / "results.json"

TARGET_MTYPES = {1, 2, 7}        # 生产类 / 非生产类 / 发车安全库存
MTYPE_NAMES = {0: "零库存管理", 1: "生产类安全库存", 2: "非生产类安全库存", 3: "板材管理",
               4: "非标物资", 5: "型材管理", 7: "发车安全库存", 8: "油漆类物资"}

# 汇总表里保留的字段（详情页需要图表时再单独算 with_series=True）
KEEP = ("id", "code", "name", "unit", "unitWms", "status", "alert", "cur", "curMax", "ss", "rawSs", "hold", "max", "batch",
        "dev", "saving", "savingActual", "actual", "simSug", "simCur", "lead", "leadSrc", "leadN", "mu", "nzDays", "historyDays",
        "historySrc", "bomShare", "bomDemand", "onhand", "onhandFree", "nextOrder", "conf", "oldRuleNow",
        "pcycle", "notes", "why")


def load_rows(path: Path = MATERIALS) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return rows if isinstance(rows, list) else rows.get("rows", [])


def load_dataset(path: Path = DATASET) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_one(dataset: dict, row: dict, params: Params) -> dict:
    code = str(row.get("no") or "")
    md = material_from_dataset(dataset, code, row.get("unit") or "")
    try:
        res = analyze(md, row, params)
    except Exception as exc:  # 单个物料出错不影响整批
        res = {"id": str(row.get("id", "")), "code": code, "name": row.get("name", ""), "status": "error",
               "alert": "none", "ss": None, "notes": [], "why": f"计算出错：{exc!r}"}
    out = {k: res.get(k) for k in KEEP if k in res}
    out["mtype"] = row.get("mtype")
    out["mtypeName"] = MTYPE_NAMES.get(row.get("mtype"), str(row.get("mtype")))
    return out


# 子进程各自持有一份数据集，避免每个任务都序列化 1.5MB
_DS: dict | None = None
_PARAMS: Params | None = None


def _init(dataset_path: str, params: Params) -> None:
    global _DS, _PARAMS
    _DS = load_dataset(Path(dataset_path))
    _PARAMS = params


def _work(row: dict) -> dict:
    return run_one(_DS, row, _PARAMS)


def _mtype(row: dict):
    try:
        return int(row.get("mtype"))
    except (TypeError, ValueError):
        return None


def _pct(sorted_vals: list[float], q: float):
    if not sorted_vals:
        return None
    return round(sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))], 1)


def params_meta(params: Params) -> dict:
    return {"level": params.level, "levelDesc": LEVELS.get(params.level, {}).get("desc") if params.level else None,
            "serviceLevel": params.service_level, "bomBasis": params.bom_basis, "upRule": params.up_rule,
            "defaultLeadDays": params.default_lead_days,
            "leadOverride": params.lead_override, "useSapHistory": params.use_sap_history,
            "alertHigh": params.alert_high, "alertMedium": params.alert_medium}


def summarize(results: list[dict]) -> dict:
    status = Counter(r["status"] for r in results)
    alert = Counter(r["alert"] for r in results if r["status"] == "ok")
    comparable = [r for r in results if r["status"] == "ok" and r.get("simCur") and r.get("simSug")]
    up = sum(1 for r in comparable if (r.get("dev") or 0) > 0.005)
    down = sum(1 for r in comparable if (r.get("dev") or 0) < -0.005)
    sim_short = sorted(r["simSug"]["stockoutDays"] for r in comparable)
    cur_short = sorted(r["simCur"]["stockoutDays"] for r in comparable)
    lower_stock = sum(1 for r in comparable if r["simSug"]["avg"] < r["simCur"]["avg"])
    fewer_short = sum(1 for r in comparable if r["simSug"]["stockoutDays"] < r["simCur"]["stockoutDays"])
    with_actual = [r for r in results if r["status"] == "ok" and r.get("savingActual") is not None]
    by_type: dict[str, dict] = {}
    for r in results:
        t = by_type.setdefault(r["mtypeName"], {"total": 0, "ok": 0, "high": 0, "medium": 0})
        t["total"] += 1
        if r["status"] == "ok":
            t["ok"] += 1
            if r["alert"] in ("high", "medium"):
                t[r["alert"]] += 1
    return {
        "total": len(results),
        "status": dict(status),
        "alert": dict(alert),
        "comparable": len(comparable),
        "suggestUp": up,
        "suggestDown": down,
        "suggestSame": len(comparable) - up - down,
        "hold": sum(1 for r in comparable if r.get("hold")),
        "simShortMedian": _pct(sim_short, 0.5),
        "simShortP90": _pct(sim_short, 0.9),
        "simCurShortMedian": _pct(cur_short, 0.5),
        "fillMean": round(sum(r["simSug"]["fill"] for r in comparable) / len(comparable), 4) if comparable else None,
        "simLowerStock": lower_stock,
        "simFewerStockout": fewer_short,
        "simShortDaysCur": round(sum(r["simCur"]["stockoutDays"] for r in comparable), 1),
        "simShortDaysSug": round(sum(r["simSug"]["stockoutDays"] for r in comparable), 1),
        "withActual": len(with_actual),
        "lowerThanActual": sum(1 for r in with_actual if r["savingActual"] > 0),
        "medianSavingActual": (round(sorted(r["savingActual"] for r in with_actual)[len(with_actual) // 2], 4)
                               if with_actual else None),
        "byLeadSrc": {src: {"n": len(g), "lowerThanActual": sum(1 for r in g if r["savingActual"] > 0),
                            "medianSavingActual": round(sorted(r["savingActual"] for r in g)[len(g) // 2], 4)}
                      for src in sorted({r["leadSrc"] for r in with_actual})
                      for g in [[r for r in with_actual if r["leadSrc"] == src]]},
        "byType": by_type,
    }


def run_batch(params: Params = Params(), *, rows: list[dict] | None = None, dataset_path: Path = DATASET,
              workers: int = 4, progress=None) -> dict:
    rows = [r for r in (rows if rows is not None else load_rows())
            if str(r.get("status", 0)) != "1" and _mtype(r) in TARGET_MTYPES]
    t0 = time.time()
    results: list[dict] = []
    if workers > 1 and len(rows) > 200:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                                 initargs=(str(dataset_path), params)) as ex:
            for i, r in enumerate(ex.map(_work, rows, chunksize=64), 1):
                results.append(r)
                if progress and i % 200 == 0:
                    progress(i, len(rows))
    else:
        ds = load_dataset(dataset_path)
        for i, row in enumerate(rows, 1):
            results.append(run_one(ds, row, params))
            if progress and i % 200 == 0:
                progress(i, len(rows))
    ds_meta = json.loads(dataset_path.read_text(encoding="utf-8"))["window"]
    return {
        "builtAt": dt.datetime.now().isoformat(timespec="seconds"),
        "seconds": round(time.time() - t0, 1),
        "window": ds_meta,
        "params": params_meta(params),
        "summary": summarize(results),
        "rows": results,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--level", choices=list(LEVELS), help="激进程度预设；给了就忽略 --service-level")
    ap.add_argument("--service-level", type=float, default=0.90)
    ap.add_argument("--bom-basis", choices=["attr", "attr+planned", "none"], default="attr")
    ap.add_argument("--no-sap-history", action="store_true", help="只用 GWMS5 窗口内的需求")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args(argv)
    if args.level:
        params = level_params(args.level, bom_basis=args.bom_basis, use_sap_history=not args.no_sap_history)
    else:
        params = Params(service_level=args.service_level, bom_basis=args.bom_basis,
                        use_sap_history=not args.no_sap_history)
    res = run_batch(params, workers=args.workers,
                    progress=lambda i, n: print(f"  {i}/{n}", file=sys.stderr))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    s = res["summary"]
    print(f"试算 {s['total']} 个物料，用时 {res['seconds']}s；状态 {s['status']}；报警 {s['alert']}")
    print(f"可与现行值比较 {s['comparable']} 个：建议上调 {s['suggestUp']}、下调 {s['suggestDown']}、"
          f"持平 {s['suggestSame']}（其中模型算高但保持现行 {s['hold']}）；"
          f"模拟平均在库下降 {s['simLowerStock']} 个，缺货天数减少 {s['simFewerStockout']} 个")
    print(f"对照实际自由在库（{s['withActual']} 个）：建议规则在库更低 {s['lowerThanActual']} 个，"
          f"降幅中位数 {s['medianSavingActual']:.0%}" if s["medianSavingActual"] is not None else "无实际在库基线")
    print(f"写入 {args.out}（{args.out.stat().st_size / 1e6:.1f} MB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
