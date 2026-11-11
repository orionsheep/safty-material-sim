"""SAP 每日出入库小样本 → 看计划补货、按真实 6 个月过拟合安全库存 → 单文件 HTML 报告。

用法：
  python3 scripts/validate_report.py 表1.xlsx 表2.xlsx --out /Users/mychanging/Desktop/HTML报告/安全库存曲线验证_2026-11-10.html
现行安全库存从 data/materials-all.json（f-1 导出，status=0 的那行）里取。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.validate import read_workbook, validate_plan  # noqa: E402


def current_rules() -> dict:
    rows = json.loads((ROOT / "data/materials-all.json").read_text())
    out = {}
    for r in rows:
        code = str(r.get("no") or "")
        if not code:
            continue
        if code in out and r.get("status") == 1:
            continue                          # 优先启用中的那行
        s = float(r.get("lquantity") or 0)
        h = float(r.get("hquantity") or 0)
        out[code] = {"s": s, "S": h if h > s else 2 * s, "pack": float(r.get("pquantity") or 0),
                     "mtype": r.get("mtype"), "pcycleF1": r.get("pcycle")}
    return out


def pct(a, b):
    return None if not b else round((a - b) / b * 100)


def g(x) -> str:
    x = float(x)
    return f"{x:,.0f}" if abs(x) >= 100 or x.is_integer() else f"{x:,.1f}"


def brief(r: dict) -> dict:
    f, t = r["fit"], r["test"]
    return {"lookBy": r["lookBy"], "look": r["look"], "B": f["B"], "cover": f["cover"], "fill": f["fill"],
            "shortDays": f["shortDays"], "orders": f["orders"], "avg": f["avg"], "testFill": t["fill"],
            "planQty": f["planQty"]}


def pick(a: dict, b: dict) -> tuple[dict, dict]:
    """两种下单口径里选一个：后 3 个月检验满足率 >= 95% 的里面平均在库最低；都不到 95% 就选检验更稳的。"""
    ok = [r for r in (a, b) if r["test"]["fill"] >= 0.95]
    if ok:
        best = min(ok, key=lambda r: (r["fit"]["avg"], r["fit"]["B"]))
    else:
        best = max((a, b), key=lambda r: (r["test"]["fill"], -r["fit"]["avg"]))
    return best, (b if best is a else a)


def verdict(r: dict) -> dict:
    cur, fit, act, te, alt = r["cur"], r["fit"], r["actual"], r["test"], r["alt"]
    u, pc, med = r["unit"], int(r["pcycle"]), r["leadMedian"]
    drop = pct(fit["B"], cur["s"])
    vs = pct(fit["avg"], act["avg"])
    gap = pc - med
    mu = r["stats"]["mean"]
    if r["lookBy"] == "采购周期":
        how = (f"按系统采购周期 {pc} 天往前看计划用量下单。采购周期比实际交货期（中位 {med} 天）长 {gap} 天，"
               f"这 {gap} 天的用量（约 {g(mu * gap)}{u}）本身就在起安全库存的作用，所以额外的安全库存只要 <b>{g(fit['B'])}</b>{u}。"
               f"如果把采购周期改成实际的 {alt['look']} 天，安全库存要 {g(alt['B'])}、平均在库 {g(alt['avg'])}，"
               f"但后 3 个月满足率只有 {alt['testFill']*100:.1f}%，不如按采购周期稳。")
    else:
        side = "短" if gap < 0 else "长"
        how = (f"系统采购周期 {pc} 天比实际交货期（中位 {med} 天）{side} {abs(gap)} 天。"
               + (f"按采购周期下单会下晚，安全库存得放到 {g(alt['B'])} 才顶得住；" if gap < 0 else
                  f"按采购周期下单会下早，安全库存可以是 {g(alt['B'])}，但平均在库 {g(alt['avg'])} 偏高；")
               + f"按实际交货期 {med} 天往前看下单，安全库存 <b>{g(fit['B'])}</b>{u}、平均在库 {g(fit['avg'])}。"
                 f"建议把系统采购周期改成 {med} 天。")
    head = (f"安全库存 {g(cur['s'])} → <b>{g(fit['B'])}</b>{u}（{drop}%）。" + how +
            f"按这个规则把 6 个月真实出库原样走一遍：满足率 {fit['fill']*100:.1f}%（缺货 {fit['shortDays']} 天），"
            f"下单 {fit['orders']} 次（现场实际到货 {act['receipts']} 次），平均在库 {g(fit['avg'])}，现场实际 {g(act['avg'])}"
            f"（{'+' if vs > 0 else ''}{vs}%）。")
    risk = (f"防过拟合检查：只用前 3 个月拟合，放到后 3 个月满足率 {te['fill']*100:.1f}%"
            + ("，比较稳。" if te["fill"] >= 0.95 else "，偏低——这个物料出库成批，上线时建议用「均衡」档并每月重算。"))
    if vs <= 0:
        return {"tag": "降库存", "tone": "good", "text": head + "安全库存和在库同时降下来。" + risk}
    if act["zeroEndDays"] >= 0.15 * len(r["dates"]):
        why = f"在库没降：现场本来就是按单到货、日终常常为 0（{act['zeroEndDays']} 天），实际在库已经很低。"
    else:
        why = ("在库没降：平均在库主要由每次到货的批量决定"
               + (f"（包装 {g(cur['pack'])}{u} 一次）" if cur["pack"] else "") + "，要降在库得把批量拆小、分批到货。")
    return {"tag": "安全库存可降", "tone": "warn", "text": head + why + risk}


def build(paths: list[str], out: Path, level: str):
    rules = current_rules()
    mats = []
    for p in paths:
        for ss in read_workbook(p):
            c = rules.get(ss.code)
            if not c:
                print("跳过（f-1 里没有现行值）:", ss.code)
                continue
            by_pc = validate_plan(ss, c["s"], c["S"], c["pack"], level=level, look=int(round(ss.pcycle)) or None)
            by_lt = validate_plan(ss, c["s"], c["S"], c["pack"], level=level)
            r, other = pick(by_pc, by_lt)
            r["alt"] = brief(other)
            r["schemes"] = [brief(by_pc), brief(by_lt)]
            r["mtype"] = c["mtype"]
            r["pcycleF1"] = c["pcycleF1"]
            r["verdict"] = verdict(r)
            r["source"] = Path(p).name
            mats.append(r)
            print(ss.code, ss.name.split("/")[0], r["lookBy"], r["look"], "cur", c["s"], "-> B", r["fit"]["B"],
                  "avg", r["fit"]["avg"], "test", r["test"]["fill"], "| alt", other["lookBy"], other["fit"]["B"])
    data = {"builtAt": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "level": level, "materials": mats}
    html = TEMPLATE.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, default=float))
    lib = (ROOT / "scripts/vendor/echarts-5.5.1.min.js").read_text(encoding="utf-8")
    html = html.replace('<script src="https://cdnjs.cloudflare.com/ajax/libs/echarts/5.5.1/echarts.min.js"></script>',
                        "<script>" + lib.replace("</script", "<\\/script") + "</script>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print("写出", out)


TEMPLATE = (ROOT / "scripts/validate_report.html").read_text(encoding="utf-8") if (ROOT / "scripts/validate_report.html").exists() else ""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", default="激进")
    a = ap.parse_args()
    build(a.xlsx, Path(a.out), a.level)
