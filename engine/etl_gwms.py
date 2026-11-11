#!/usr/bin/env python3
"""GWMS5（巨沃仓云）镜像 → 按物料、按天的出入库和当前库存。

只读：输入是 zcode 导出的 GWMS5 镜像（zip 或解压后的目录），输出一个紧凑的
data/gwms/dataset.json，供 engine/batch.py 使用。

用到的模块（菜单名）：
  stockOutOrderReport            出库单明细报表  → 每天出库，按批次属性区分「按单(BOM)」与「自由库存」
  receiptreport...               收货单报表      → 每天入库，同样按批次属性区分
  xunihuozhukucunbaobiao-wh      虚拟货主库存报表-库存不为零 → 当前在库 / 可用 / 按单部分
  InventorySummaryReport         库存汇总报表    → 仅用于核对在库总量
另外从 data/workbooks.json（SAP 导出的两本 Excel）读取「实际交货天数」样本。

批次属性（atno / atno1）里带销售订单行号（如 2001346607-40）的库存，是按订单 BOM
申报到货的那一部分；会上说这部分「不优化」，引擎计算建议安全库存时会把它剔除。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SO_ATTR = re.compile(r"^\d{10}-\d+")          # 销售订单-行号
SUFFIX = "_WH"                                 # 卫华（工厂 1000）
SCOPE_WAREHOUSES = ("河南卫华原材料普通仓", "河南卫华电器普通仓")
# 不算作「需求」的出库类型（仍计入在库量倒推）
NON_DEMAND_TYPES = {"采购退货", "报废出库", "借出出库", "供应商暂存出库", "客户暂存"}

MODULES = {
    "out": "stockOutOrderReport",
    "out_new": "stockOutOrderReport1",
    "receipt": "receiptreport20251117143740983",
    "stock": "xunihuozhukucunbaobiao-wh",
    "summary": "InventorySummaryReport",
}


class ModuleSource:
    """从 zip 或目录里按模块名读取 GWMS5 镜像 JSON。"""

    def __init__(self, path: Path):
        self.path = path
        if path.is_file() and zipfile.is_zipfile(path):
            self.zip = zipfile.ZipFile(path)
            self.members = {
                Path(n).name: n
                for n in self.zip.namelist()
                if "/modules/" in n and n.endswith(".json")
            }
        elif path.is_dir():
            self.zip = None
            self.members = {p.name: p for p in path.rglob("*.json") if p.parent.name == "modules"}
        else:
            raise SystemExit(f"找不到 GWMS5 镜像：{path}")
        if not self.members:
            raise SystemExit(f"{path} 里没有 data/modules/*.json")

    def has(self, module: str) -> bool:
        return f"{module}.json" in self.members

    def load(self, module: str) -> dict:
        member = self.members[f"{module}.json"]
        if self.zip is not None:
            with self.zip.open(member) as fh:
                return json.loads(fh.read().decode("utf-8-sig"))
        return json.loads(Path(member).read_text(encoding="utf-8-sig"))


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _code(inco: str | None) -> str | None:
    if inco and inco.endswith(SUFFIX):
        return inco[: -len(SUFFIX)]
    return None


def _day(ts: str | None) -> str:
    return (ts or "")[:10]


def extract_outbound(rows: list[dict]):
    """出库单明细 → {code: {kind: {day: qty}}}，kind ∈ free/bom/otherFree/otherBom。"""
    series: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    types: dict = defaultdict(Counter)
    names: dict = {}
    per_day = Counter()
    for r in rows:
        day = _day(r.get("crdt"))
        if day:
            per_day[day] += 1
        code = _code(r.get("inco"))
        if not code or r.get("flag") != "11" or r.get("whna") not in SCOPE_WAREHOUSES:
            continue
        qty = _num(r.get("qty"))
        if qty <= 0 or not day:
            continue
        bom = bool(SO_ATTR.match(r.get("atno") or ""))
        dena = r.get("dena") or "未标注"
        if dena in NON_DEMAND_TYPES:
            kind = "otherBom" if bom else "otherFree"
        else:
            kind = "bom" if bom else "free"
            if not bom and "计划内" in dena:
                series[code]["planned"][day] += qty      # 自由库存里按生产订单计划领用的部分
        series[code][kind][day] += qty
        types[code][dena] += qty
        names.setdefault(code, r.get("inna") or "")
    return series, types, names, per_day


def extract_receipts(rows: list[dict]):
    series: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    types: dict = defaultdict(Counter)
    suppliers: dict = defaultdict(Counter)
    units: dict = {}
    per_day = Counter()
    for r in rows:
        day = _day(r.get("crdt"))
        if day:
            per_day[day] += 1
        code = _code(r.get("inco"))
        if not code or r.get("flag") != "11" or r.get("whna") not in SCOPE_WAREHOUSES:
            continue
        qty = _num(r.get("qty"))
        if qty <= 0 or not day:
            continue
        bom = bool(SO_ATTR.match(r.get("atno1") or ""))
        series[code]["bom" if bom else "free"][day] += qty
        types[code][r.get("dena") or "未标注"] += qty
        if r.get("suna"):
            suppliers[code][r["suna"]] += 1
        if r.get("igwf5"):
            units.setdefault(code, r["igwf5"])
    return series, types, suppliers, units, per_day


def extract_stock(rows: list[dict]):
    stock: dict = defaultdict(lambda: {"total": 0.0, "bom": 0.0, "avail": 0.0})
    units: dict = {}
    names: dict = {}
    for r in rows:
        code = _code(r.get("inco"))
        if not code or r.get("whna") not in SCOPE_WAREHOUSES:
            continue
        qty = _num(r.get("zqty"))
        s = stock[code]
        s["total"] += qty
        s["avail"] += _num(r.get("lqty"))
        if SO_ATTR.match(r.get("atno1") or ""):
            s["bom"] += qty
        if r.get("gwf5"):
            units.setdefault(code, r["gwf5"])
        names.setdefault(code, r.get("inna") or "")
    return stock, units, names


def extract_summary_totals(rows: list[dict]):
    totals: dict = defaultdict(float)
    units: dict = {}
    for r in rows:
        code = _code(r.get("inco"))
        if not code or r.get("whna") not in SCOPE_WAREHOUSES:
            continue
        totals[code] += _num(r.get("qty"))
        if r.get("gwf5"):
            units.setdefault(code, r["gwf5"])
    return totals, units


def extract_sap_lead(workbooks_path: Path) -> dict:
    """SAP 入库明细里的「实际交货天数」= 过账日期 − 申报时间。"""
    if not workbooks_path.exists():
        return {}
    data = json.loads(workbooks_path.read_text(encoding="utf-8"))
    out: dict = defaultdict(list)
    for src in data.get("sources", []):
        for sheet in src.get("sheets", []):
            if sheet.get("name") != "入库明细":
                continue
            for r in sheet.get("rows", []):
                text = str(r.get("移动类型文本") or "")
                if "初始条目" in text or "余额" in text:
                    continue
                days = _num(r.get("实际交货天数"))
                code = str(r.get("物料") or "").strip()
                if code and days > 0:
                    out[code].append({"date": str(r.get("过账日期") or "")[:10], "days": days,
                                      "supplier": str(r.get("供应商") or "")})
    return dict(out)


def extract_sap_history(workbooks_path: Path, before: str) -> tuple[dict, dict]:
    """SAP 出库明细（两本 Excel）→ GWMS5 窗口之前的日出库，外加 KG/M 这类平行单位的换算系数。

    第一本：数量为负是出库（261/Z09），为正是冲销（262）；工字钢那本：「类型/入库/出库」=出库，数量为正。
    """
    if not workbooks_path.exists():
        return {}, {}
    data = json.loads(workbooks_path.read_text(encoding="utf-8"))
    daily: dict = defaultdict(lambda: defaultdict(float))
    units: dict = {}
    parallel: dict = defaultdict(lambda: {"base": 0.0, "par": 0.0, "baseUnit": "", "parUnit": ""})
    for src in data.get("sources", []):
        for sheet in src.get("sheets", []):
            if sheet.get("name") != "出库明细":
                continue
            has_dir = "类型/入库/出库" in (sheet.get("headers") or [])
            for r in sheet.get("rows", []):
                code = str(r.get("物料") or "").strip()
                day = str(r.get("过账日期") or "")[:10]
                text = str(r.get("移动类型文本") or "")
                if not code or not day or "初始条目" in text or "余额" in text:
                    continue
                qty = _num(r.get("数量"))
                if has_dir:
                    demand = qty if r.get("类型/入库/出库") == "出库" else 0.0
                else:
                    demand = -qty
                units.setdefault(code, r.get("基本计量单位") or "")
                pq, pu = _num(r.get("以并行UoM计的数量")), r.get("平行计量单位")
                if pu and pq:
                    p = parallel[code]
                    p["base"] += abs(qty)
                    p["par"] += abs(pq)
                    p["baseUnit"], p["parUnit"] = r.get("基本计量单位") or "", pu
                if day < before:
                    daily[code][day] += demand
    history = {
        code: {"unit": units.get(code, ""), "start": min(days), "out": {d: round(v, 4) for d, v in sorted(days.items())}}
        for code, days in daily.items() if days
    }
    factors = {
        code: {"base": p["baseUnit"], "parallel": p["parUnit"], "factor": round(p["base"] / p["par"], 6)}
        for code, p in parallel.items() if p["par"] > 0
    }
    return history, factors


def _window(per_day: Counter) -> tuple[str, str]:
    """被 200k 行截断的导出，最早那一天必然不完整，丢掉；最后一天是导出当天（不完整，单独标记）。"""
    days = sorted(d for d in per_day if d)
    if len(days) < 3:
        raise SystemExit("出库明细天数太少，无法确定时间窗")
    return days[1], days[-1]


def _sparse(day_map: dict, start: dt.date, n: int) -> dict:
    out = {}
    for day, qty in day_map.items():
        idx = (dt.date.fromisoformat(day) - start).days
        if 0 <= idx < n and qty:
            out[str(idx)] = round(qty, 4)
    return out


def _onhand(virtual: dict | None, summary_total: float | None) -> dict:
    """在库总量以「库存汇总报表」为准；按单(BOM)占比和可用量取自「虚拟货主库存报表」。

    两张报表不是同一时刻抓的，个别物料差得多，差异写进 virtualTotal 供核对。
    """
    v_total = virtual["total"] if virtual else 0.0
    total = summary_total if summary_total is not None else v_total
    bom_share = (virtual["bom"] / v_total) if virtual and v_total > 0 else 0.0
    avail = min(virtual["avail"], total) if virtual else total
    return {
        "total": round(total, 4),
        "bom": round(total * bom_share, 4),
        "avail": round(max(avail, 0.0), 4),
        "virtualTotal": round(v_total, 4),
    }


def load_f1_codes(path: Path) -> set[str] | None:
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows = rows if isinstance(rows, list) else rows.get("rows", [])
    return {str(r.get("no")) for r in rows if r.get("no")}


def build(src: ModuleSource, workbooks: Path, only_codes: set[str] | None = None) -> dict:
    log = {}
    out_rows = src.load(MODULES["out"])["rows"]
    out_series, out_types, out_names, out_days = extract_outbound(out_rows)
    start_s, end_s = _window(out_days)
    log["outbound_lines"] = len(out_rows)
    if src.has(MODULES["out_new"]):
        keys = {(r.get("biid"), r.get("inco"), r.get("btno"), r.get("qty")) for r in out_rows}
        extra = [r for r in src.load(MODULES["out_new"])["rows"]
                 if (r.get("biid"), r.get("inco"), r.get("btno"), r.get("qty")) not in keys
                 and start_s <= _day(r.get("crdt")) <= end_s]
        log["outbound_new_report_extra_lines"] = len(extra)
    del out_rows

    rec_rows = src.load(MODULES["receipt"])["rows"]
    rec_series, rec_types, suppliers, rec_units, rec_days = extract_receipts(rec_rows)
    log["receipt_lines"] = len(rec_rows)
    log["receipt_first_day"] = min(d for d in rec_days if d)
    del rec_rows
    if log["receipt_first_day"] >= start_s:
        print(f"[警告] 收货单报表最早 {log['receipt_first_day']}，没有覆盖出库窗口起点 {start_s}", file=sys.stderr)

    stock, stock_units, stock_names = extract_stock(src.load(MODULES["stock"])["rows"])
    summary_totals, summary_units = extract_summary_totals(src.load(MODULES["summary"])["rows"])
    diffs = [abs(stock.get(c, {}).get("total", 0.0) - t) for c, t in summary_totals.items()]
    log["stock_vs_summary_codes_differing"] = sum(1 for d in diffs if d > 1e-6)
    log["stock_codes"] = len(stock)

    start = dt.date.fromisoformat(start_s)
    n = (dt.date.fromisoformat(end_s) - start).days + 1
    codes = set(out_series) | set(rec_series) | set(stock) | set(summary_totals)
    if only_codes is not None:
        codes &= only_codes
    materials = {}
    for code in sorted(codes):
        o, r = out_series.get(code, {}), rec_series.get(code, {})
        materials[code] = {
            "name": out_names.get(code) or stock_names.get(code) or "",
            "unit": stock_units.get(code) or summary_units.get(code) or rec_units.get(code) or "",
            "outFree": _sparse(o.get("free", {}), start, n),
            "outPlanned": _sparse(o.get("planned", {}), start, n),
            "outBom": _sparse(o.get("bom", {}), start, n),
            "outOtherFree": _sparse(o.get("otherFree", {}), start, n),
            "outOtherBom": _sparse(o.get("otherBom", {}), start, n),
            "inFree": _sparse(r.get("free", {}), start, n),
            "inBom": _sparse(r.get("bom", {}), start, n),
            "outTypes": {k: round(v, 3) for k, v in out_types.get(code, Counter()).most_common(6)},
            "inTypes": {k: round(v, 3) for k, v in rec_types.get(code, Counter()).most_common(4)},
            "suppliers": [k for k, _ in suppliers.get(code, Counter()).most_common(3)],
            "onhand": _onhand(stock.get(code), summary_totals.get(code)),
            "hasStockRow": code in stock or code in summary_totals,
        }
    sap_history, unit_factors = extract_sap_history(workbooks, start_s)
    return {
        "meta": {
            "source": str(src.path.name),
            "builtAt": dt.datetime.now().isoformat(timespec="seconds"),
            "scopeWarehouses": list(SCOPE_WAREHOUSES),
            "nonDemandTypes": sorted(NON_DEMAND_TYPES),
            "modules": MODULES,
            "log": log,
        },
        "window": {"start": start_s, "end": end_s, "days": n, "partialLastDay": True},
        "materials": materials,
        "leadSamples": extract_sap_lead(workbooks),
        "sapHistory": sap_history,
        "unitFactors": unit_factors,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gwms", required=True, type=Path, help="zcode 导出的 zip，或解压后的目录")
    ap.add_argument("--workbooks", type=Path, default=ROOT / "data" / "workbooks.json")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "gwms" / "dataset.json")
    ap.add_argument("--materials", type=Path, default=ROOT / "data" / "materials-all.json",
                    help="只保留 f-1 物料清单里的编码")
    ap.add_argument("--all-codes", action="store_true", help="保留 GWMS5 里出现的全部编码")
    args = ap.parse_args(argv)
    only = None if args.all_codes else load_f1_codes(args.materials)
    dataset = build(ModuleSource(args.gwms.expanduser()), args.workbooks, only)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dataset, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    w, m = dataset["window"], dataset["materials"]
    print(f"窗口 {w['start']} ~ {w['end']}（{w['days']} 天，最后一天不完整）；物料 {len(m)} 个；"
          f"SAP 交货期样本 {sum(len(v) for v in dataset['leadSamples'].values())} 条")
    print("日志:", json.dumps(dataset["meta"]["log"], ensure_ascii=False))
    print(f"写入 {args.out}（{args.out.stat().st_size / 1e6:.1f} MB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
