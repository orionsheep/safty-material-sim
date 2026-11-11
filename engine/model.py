"""单物料安全库存引擎。

思路（对应会上的要求）：
1. 在库量 = 按安全库存申报到货的部分 + 按订单 BOM 申报到货的部分。BOM 部分是订单上已确定的需求，
   不优化：批次属性带销售订单号的出库，从需求里剔除（可切换为「计划内领料也剔除」或「不剔除」）。
2. 剩下的需求按交货期累计，取服务水平分位数（默认 P90，与 phase3 一致），得到建议安全库存。
   f-1 的安全库存实际是申报触发点：可用 + 在途低于它就申报，补到最高库存。
3. 用同一套「低于安全库存就申报、补到最高库存」的规则，分别按现行值和建议值模拟一年在库量，
   比较平均在库量和缺货天数。会上的成功标准：预测在库量比现行固定安全库存有所下降。
   注意：现场并不严格按 f-1 现行值补货，实际在库普遍远高于「按现行值模拟」的结果，所以另外
   用出入库倒推的实际自由在库（GWMS5 窗口内）作为基线，比较建议规则能把在库量降多少。
4. 建议值和现行值差得大就报警，留给人工确认；人工结果写入反馈日志，作为以后学习的样本。

交货期取 SAP「实际交货天数」的中位数（样本 ≥5 个），否则用 f-1 采购周期。实际交货天数有长尾
（提前很久的 BOM 申报会被算成几十上百天），所以不直接用整条分布。
只依赖 numpy；随机数按物料编码固定种子，同一份数据重复计算结果不变。
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import math
from dataclasses import dataclass, field

import numpy as np

COUNT_UNITS = {"PC", "TAO", "PS", "EA", "ST", "SET", "个", "件", "套", "台", "根", "只", "把", "片"}
NOT_APPLICABLE_MTYPES = {0}       # 零库存管理：按单采购，不设安全库存
BOM_BASIS = {
    "attr": "只剔除带销售订单号的批次",
    "attr+planned": "带销售订单号的批次 + 计划内领料都剔除",
    "none": "不剔除",
}


@dataclass(frozen=True)
class Params:
    service_level: float = 0.90
    default_lead_days: float = 30.0
    min_lead_samples: int = 5
    min_demand_days: int = 3
    alert_high: float = 0.50
    alert_medium: float = 0.30
    bootstrap: int = 4000
    block_days: int = 7
    sim_days: int = 365
    sim_warmup: int = 90
    sim_paths: int = 48
    horizon: int = 60
    proj_paths: int = 200
    bom_basis: str = "attr"
    lead_override: float | None = None
    use_sap_history: bool = True
    # 上调规则：free = 按模型可上调；evidence = 近两个月实际断货过才上调，否则保持现行；never = 只降不升
    up_rule: str = "free"
    level: str | None = None          # 激进程度预设名（只用于展示）

    def with_(self, **kw) -> "Params":
        kw = {k: v for k, v in kw.items() if v is not None}
        return dataclasses.replace(self, **kw)


@dataclass
class MaterialData:
    code: str
    name: str
    unit: str                      # 计算所用单位（能换算时与 f-1 一致）
    unit_wms: str
    start: dt.date
    days: int
    out_free: np.ndarray
    out_planned: np.ndarray
    out_bom: np.ndarray
    out_other_free: np.ndarray
    out_other_bom: np.ndarray
    in_free: np.ndarray
    in_bom: np.ndarray
    onhand_total: float = 0.0
    onhand_bom: float = 0.0
    avail: float = 0.0
    virtual_total: float = 0.0
    has_stock_row: bool = False
    lead_samples: list = field(default_factory=list)
    out_types: dict = field(default_factory=dict)
    in_types: dict = field(default_factory=dict)
    suppliers: list = field(default_factory=list)
    sap_out: np.ndarray | None = None          # GWMS5 窗口之前的 SAP 日出库（已换算单位）
    unit_note: str | None = None
    unit_mismatch: bool = False

    @property
    def dates(self) -> list[str]:
        return [(self.start + dt.timedelta(days=i)).isoformat() for i in range(self.days)]


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _dense(sparse: dict, n: int, factor: float = 1.0) -> np.ndarray:
    arr = np.zeros(n)
    for k, v in (sparse or {}).items():
        i = int(k)
        if 0 <= i < n:
            arr[i] = float(v) * factor
    return arr


def _same(a: str, b: str) -> bool:
    return (a or "").upper() == (b or "").upper()


def material_from_dataset(dataset: dict, code: str, f1_unit: str = "") -> MaterialData | None:
    m = dataset["materials"].get(code)
    if m is None:
        return None
    n = int(dataset["window"]["days"])
    start = dt.date.fromisoformat(dataset["window"]["start"])
    unit_wms = m.get("unit", "") or ""
    uf = dataset.get("unitFactors", {}).get(code)
    factor, unit, note, mismatch = 1.0, (f1_unit or unit_wms), None, False
    if f1_unit and unit_wms and not _same(f1_unit, unit_wms):
        if uf and _same(uf["base"], f1_unit) and _same(uf["parallel"], unit_wms) and uf["factor"] > 0:
            factor = uf["factor"]
            note = f"GWMS5 按 {unit_wms} 记账，已按 SAP 换算系数 {factor:g} {f1_unit}/{unit_wms} 换成 {f1_unit}"
        else:
            unit, mismatch = unit_wms, True
            note = f"单位不一致：f-1 是 {f1_unit}，GWMS5 是 {unit_wms}，建议值按 {unit_wms} 计算，不和现行值比较"

    sap_out = None
    sh = dataset.get("sapHistory", {}).get(code)
    if sh and sh.get("out"):
        s_factor = None
        if _same(sh.get("unit"), unit):
            s_factor = 1.0
        elif uf and _same(sh.get("unit"), uf["base"]) and _same(unit, uf["parallel"]) and uf["factor"] > 0:
            s_factor = 1.0 / uf["factor"]
        if s_factor is not None:
            s0 = dt.date.fromisoformat(sh["start"])
            k = (start - s0).days
            if k > 0:
                sap_out = np.zeros(k)
                for day, qty in sh["out"].items():
                    i = (dt.date.fromisoformat(day) - s0).days
                    if 0 <= i < k:
                        sap_out[i] += float(qty) * s_factor
                sap_out = np.maximum(sap_out, 0.0)

    oh = m.get("onhand") or {}
    return MaterialData(
        code=code,
        name=m.get("name", ""),
        unit=unit,
        unit_wms=unit_wms,
        start=start,
        days=n,
        out_free=_dense(m.get("outFree"), n, factor),
        out_planned=_dense(m.get("outPlanned"), n, factor),
        out_bom=_dense(m.get("outBom"), n, factor),
        out_other_free=_dense(m.get("outOtherFree"), n, factor),
        out_other_bom=_dense(m.get("outOtherBom"), n, factor),
        in_free=_dense(m.get("inFree"), n, factor),
        in_bom=_dense(m.get("inBom"), n, factor),
        onhand_total=float(oh.get("total", 0.0)) * factor,
        onhand_bom=float(oh.get("bom", 0.0)) * factor,
        avail=float(oh.get("avail", 0.0)) * factor,
        virtual_total=float(oh.get("virtualTotal", 0.0)) * factor,
        has_stock_row=bool(m.get("hasStockRow")),
        lead_samples=[s["days"] for s in dataset.get("leadSamples", {}).get(code, [])],
        out_types=m.get("outTypes", {}),
        in_types=m.get("inTypes", {}),
        suppliers=m.get("suppliers", []),
        sap_out=sap_out,
        unit_note=note,
        unit_mismatch=mismatch,
    )


def seed_for(code: str, salt: str = "") -> int:
    return int(hashlib.md5(f"{code}|{salt}".encode("utf-8")).hexdigest()[:8], 16)


def round_qty(x: float, unit: str) -> float:
    """计数单位向上取整；KG/M 等连续单位保留一位小数（量大时取整）。"""
    if x is None or x <= 0:
        return 0.0
    if (unit or "").upper() in COUNT_UNITS or abs(x) >= 100:
        return float(math.ceil(x - 1e-9))
    return math.ceil(x * 10 - 1e-9) / 10


# ---------------------------------------------------------------- 基础计算

# 激进程度预设：从稳健到极限，逐步放松服务水平、交货期兜底值和上调条件（需求口径另选，不随预设变）。
# 「缺交货期」只影响 f-1 没填采购周期、也没有 SAP 交货样本的物料；10 天是 f-1 已填采购周期的中位数。
LEVELS = {
    "稳健": {"service_level": 0.90, "default_lead_days": 30.0, "up_rule": "free",
             "desc": "P90，缺交货期按 30 天，模型算高就建议上调"},
    "均衡": {"service_level": 0.85, "default_lead_days": 10.0, "up_rule": "evidence",
             "desc": "P85，缺交货期按 10 天，近两个月实际断货过才上调"},
    "激进": {"service_level": 0.75, "default_lead_days": 10.0, "up_rule": "evidence",
             "desc": "P75，缺交货期按 10 天，近两个月实际断货过才上调"},
    "极限": {"service_level": 0.60, "default_lead_days": 10.0, "up_rule": "never",
             "desc": "P60，缺交货期按 10 天，只降不升"},
}
UP_RULES = {"free": "可上调", "evidence": "实际断货才上调", "never": "只降不升"}


def level_params(name: str, **override) -> "Params":
    if name not in LEVELS:
        raise ValueError(f"激进程度只能是 {' / '.join(LEVELS)}")
    kw = {k: v for k, v in LEVELS[name].items() if k != "desc"}
    kw.update({k: v for k, v in override.items() if v is not None})
    return Params(level=name, **kw)


def split_demand(md: MaterialData, params: Params) -> tuple[np.ndarray, np.ndarray]:
    """GWMS5 窗口内（含导出当天）：(要由安全库存覆盖的需求, 视为按单/BOM 覆盖的需求)。"""
    if params.bom_basis == "none":
        return md.out_free + md.out_bom, np.zeros(md.days)
    if params.bom_basis == "attr+planned":
        return md.out_free - md.out_planned, md.out_bom + md.out_planned
    return md.out_free.copy(), md.out_bom.copy()


def demand_series(md: MaterialData, params: Params) -> np.ndarray:
    """统计用的日需求：去掉导出当天（不完整）；能接上 SAP 历史就接上（SAP 分不出计划内，那种口径不接）。"""
    cover, _ = split_demand(md, params)
    gw = cover[:-1] if md.days > 1 else cover
    if params.use_sap_history and md.sap_out is not None and params.bom_basis != "attr+planned":
        return np.concatenate([md.sap_out, gw])
    return gw


def history_onhand(md: MaterialData) -> dict:
    """用当前在库倒推窗口内每天日终在库量（总量 / 自由库存）。"""
    n = md.days
    moves_total = (md.in_free + md.in_bom) - (md.out_free + md.out_bom + md.out_other_free + md.out_other_bom)
    moves_free = md.in_free - (md.out_free + md.out_other_free)
    total = np.empty(n)
    free = np.empty(n)
    total[-1] = md.onhand_total
    free[-1] = md.onhand_total - md.onhand_bom
    for t in range(n - 1, 0, -1):
        total[t - 1] = total[t] - moves_total[t]
        free[t - 1] = free[t] - moves_free[t]
    return {"total": total, "free": free, "negative": bool(total.min() < -1e-6 or free.min() < -1e-6)}


def lead_info(md: MaterialData | None, row: dict, params: Params) -> dict:
    if params.lead_override and params.lead_override > 0:
        return {"days": float(params.lead_override), "source": "手动设定", "n": 0}
    samples = np.array([s for s in (md.lead_samples if md else []) if s and s > 0], dtype=float)
    if len(samples) >= params.min_lead_samples:
        p25, p50, p75 = np.percentile(samples, [25, 50, 75])
        return {"days": float(p50), "source": "SAP实际交货天数中位数", "n": int(len(samples)),
                "p25": float(p25), "p75": float(p75)}
    pcycle = _num(row.get("pcycle"))
    if pcycle > 0:
        return {"days": pcycle, "source": "f-1采购周期", "n": 0}
    return {"days": float(params.default_lead_days), "source": "默认值", "n": 0}


def bootstrap_paths(d: np.ndarray, length: int, paths: int, block: int, rng) -> np.ndarray:
    """循环分块自助法：按 7 天一块重抽历史日需求，保留周内节奏和间歇性。"""
    n = len(d)
    block = max(1, min(block, n))
    nb = math.ceil(max(length, 1) / block)
    starts = rng.integers(0, n, size=(paths, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return d[idx].reshape(paths, nb * block)[:, :length]


def lead_time_demand(d: np.ndarray, lead_days: float, params: Params, rng) -> np.ndarray:
    L = max(int(round(lead_days)), 1)
    return bootstrap_paths(d, L, params.bootstrap, params.block_days, rng).sum(axis=1)


def simulate(D: np.ndarray, s: float, S: float, lead_days: float, *, init: float, warmup: int,
             pack: float = 0.0, record: bool = False) -> dict:
    """(s, S) 申报规则：可用+在途 ≤ s 就申报，补到 S（按包装数取整）；L 天后到货；缺货延后满足。"""
    P, T = D.shape
    L = max(int(round(lead_days)), 1)
    arrivals = np.zeros((P, T + L + 2))
    net = np.full(P, float(init))
    onorder = np.zeros(P)
    onhand_sum = np.zeros(P)
    short_days = np.zeros(P)
    short_qty = np.zeros(P)
    dem_qty = np.zeros(P)
    orders = np.zeros(P)
    first_order = np.full(P, -1)
    trace = np.empty((P, T)) if record else None
    S = max(S, s)
    for t in range(T):
        arr = arrivals[:, t]
        net += arr
        onorder -= arr
        dem = D[:, t]
        short = np.maximum(dem - np.maximum(net, 0.0), 0.0)
        net -= dem
        pos = net + onorder
        need = pos <= s
        if need.any():
            q = np.where(need, S - pos, 0.0)
            if pack > 0:
                q = np.ceil(q / pack - 1e-9) * pack
            q = np.maximum(q, 0.0)
            arrivals[:, t + 1 + L] += q
            onorder += q
            first_order[need & (first_order < 0)] = t
            if t >= warmup:
                orders += need & (q > 0)
        if record:
            trace[:, t] = np.maximum(net, 0.0)
        if t >= warmup:
            onhand_sum += np.maximum(net, 0.0)
            short_days += short > 0
            short_qty += short
            dem_qty += dem
    span = max(T - warmup, 1)
    total_dem = dem_qty.sum()
    out = {
        "avg": float(onhand_sum.mean() / span),
        "stockoutDays": float(short_days.mean() * 365.0 / span),
        "fill": float(1.0 - short_qty.sum() / total_dem) if total_dem > 0 else 1.0,
        "orders": float(orders.mean() * 365.0 / span),
        "firstOrder": first_order,
    }
    if record:
        out["trace"] = trace
    return out


# ---------------------------------------------------------------- 结论

def alert_level(status: str, current: float, suggested: float | None, params: Params) -> str:
    if status != "ok" or suggested is None:
        return "none"
    if current <= 0:
        return "high" if suggested > 0 else "ok"
    dev = abs((suggested - current) / current)
    if dev >= params.alert_high:
        return "high"
    if dev >= params.alert_medium:
        return "medium"
    return "ok"


def current_policy(row: dict) -> tuple[float, float]:
    s = _num(row.get("lquantity"))
    h = _num(row.get("hquantity"))
    return s, (h if h > s else 2 * s)


def _fmt(x) -> str:
    if x is None:
        return "—"
    x = float(x)
    if abs(x) >= 100 or x.is_integer():
        return f"{x:,.0f}"
    return f"{x:,.1f}"


def analyze(md: MaterialData | None, row: dict, params: Params = Params(), *, with_series: bool = False) -> dict:
    code = str(row.get("no") or (md.code if md else ""))
    cur_s, cur_S = current_policy(row)
    base = {"id": str(row.get("id", "")), "code": code, "name": row.get("name") or (md.name if md else ""),
            "cur": cur_s, "curMax": cur_S, "unit": row.get("unit") or (md.unit if md else ""),
            "notes": [], "bomBasis": params.bom_basis, "serviceLevel": params.service_level}
    try:
        mtype = int(row.get("mtype"))
    except (TypeError, ValueError):
        mtype = None
    if mtype in NOT_APPLICABLE_MTYPES:
        return {**base, "status": "not_applicable", "alert": "none", "ss": None,
                "why": "零库存管理物料按单采购，不设安全库存，不做建议。"}
    if md is None or (not md.has_stock_row and md.out_free.sum() + md.out_bom.sum() + md.in_free.sum() + md.in_bom.sum() == 0):
        return {**base, "status": "no_data", "alert": "none", "ss": None,
                "why": "GWMS5 近两个月没有这个物料的出入库，也没有库存记录，保留现行值。"}

    notes = base["notes"]
    unit = md.unit
    base["unit"] = unit
    if md.unit_note:
        notes.append(md.unit_note)
    if md.virtual_total and abs(md.virtual_total - md.onhand_total) > max(1.0, 0.2 * md.onhand_total):
        notes.append(f"两张库存报表不一致：库存汇总 {_fmt(md.onhand_total)}，虚拟货主 {_fmt(md.virtual_total)}")

    rng = np.random.default_rng(seed_for(code, f"{params.service_level}|{params.bom_basis}|{params.lead_override}"
                                                f"|{params.default_lead_days}"))
    d = demand_series(md, params)
    cover, bom_part = split_demand(md, params)
    gw_days = md.days - 1
    nz = int((d > 0).sum())
    cover_total = float(cover[:-1].sum())
    bom_total = float(bom_part[:-1].sum())
    bom_share = bom_total / (cover_total + bom_total) if cover_total + bom_total > 0 else 0.0
    mu = float(d.mean()) if len(d) else 0.0
    lead = lead_info(md, row, params)
    L = lead["days"]
    hist = history_onhand(md)
    if hist["negative"]:
        notes.append("按出入库倒推的历史在库出现负数，出入库或库存数据有缺口")
    pack = _num(row.get("pquantity"))
    all_daily = (md.out_free + md.out_bom)[:-1]
    if md.sap_out is not None:
        all_daily = np.concatenate([md.sap_out, all_daily])
    pcycle = _num(row.get("pcycle"))

    result = {
        **base,
        "unitWms": md.unit_wms,
        "historyDays": int(len(d)),
        "historySrc": ("SAP+GWMS5" if len(d) > gw_days else "GWMS5"),
        "windowDays": gw_days,
        "nzDays": nz,
        "mu": round(mu, 4),
        "sigma": round(float(d.std(ddof=1)), 4) if len(d) > 1 else 0.0,
        "coverDemand": round(cover_total, 3),
        "bomDemand": round(bom_total, 3),
        "bomShare": round(bom_share, 4),
        "lead": round(L, 1),
        "leadSrc": lead["source"],
        "leadN": lead["n"],
        "leadIqr": [round(lead["p25"], 1), round(lead["p75"], 1)] if "p25" in lead else None,
        "onhand": round(md.onhand_total, 3),
        "onhandBom": round(md.onhand_bom, 3),
        "onhandFree": round(md.onhand_total - md.onhand_bom, 3),
        "avail": round(md.avail, 3),
        "outTypes": md.out_types,
        "suppliers": md.suppliers,
        "oldRuleNow": round_qty(float(all_daily.mean()) * pcycle, unit) if pcycle > 0 and len(all_daily) else None,
        "pcycle": pcycle or None,
    }

    if nz < params.min_demand_days:
        result.update(status="insufficient", alert="none", ss=None, max=None,
                      why=(f"近 {len(d)} 天需由安全库存覆盖的出库只有 {nz} 天，样本太少，先保留现行安全库存 {_fmt(cur_s)}{unit}。"
                           + (f"期间按单/BOM 出库 {_fmt(bom_total)}{unit}，已剔除。" if bom_total else "")))
        if with_series:
            result["series"] = _series(md, hist, cover, bom_part, None, None, params)
        return result

    ss = round_qty(float(np.quantile(lead_time_demand(d, L, params, rng), params.service_level)), unit)
    raw_ss = ss
    hold = None                      # 模型算高、但按上调规则保持现行的原因
    actual = actual_baseline(md, hist, cover)
    if cur_s > 0 and ss > cur_s and not md.unit_mismatch and params.up_rule != "free":
        short = actual["stockoutDays"] if actual else None
        if params.up_rule == "never":
            hold = "只降不升"
        elif short is None:
            hold = "实际在库倒推不出来，没有断货证据"
        elif short <= 0:
            hold = f"近 {actual['days']} 天实际没有断货"
        if hold:
            ss = cur_s
    batch = max(mu * L, pack if pack > 0 else 0.0)
    if pack > 0:
        batch = math.ceil(batch / pack - 1e-9) * pack
    batch = round_qty(batch, unit)
    smax = ss + batch
    status = "unit_mismatch" if md.unit_mismatch else "ok"

    # 同一组需求路径分别模拟现行 / 建议规则（公平比较）
    T = params.sim_days + params.sim_warmup
    D = bootstrap_paths(d, T, params.sim_paths, params.block_days, rng)
    sim_sug = simulate(D, ss, smax, L, init=smax, warmup=params.sim_warmup, pack=pack)
    sim_cur = None
    if status == "ok" and cur_s > 0:
        sim_cur = simulate(D, cur_s, cur_S, L, init=cur_S, warmup=params.sim_warmup, pack=pack)
    saving = None
    if sim_cur and sim_cur["avg"] > 0:
        saving = (sim_cur["avg"] - sim_sug["avg"]) / sim_cur["avg"]
    saving_actual = None
    if actual and actual["avgFree"] > 0 and not md.unit_mismatch:
        saving_actual = (actual["avgFree"] - sim_sug["avg"]) / actual["avgFree"]

    # 未来 60 天：从当前自由库存出发（镜像里没有可靠的在途数据，未计在途）
    Dp = bootstrap_paths(d, params.horizon, params.proj_paths, params.block_days, rng)
    start_free = max(md.onhand_total - md.onhand_bom, 0.0)
    proj_sug = simulate(Dp, ss, smax, L, init=start_free, warmup=0, pack=pack, record=True)
    proj_cur = (simulate(Dp, cur_s, cur_S, L, init=start_free, warmup=0, pack=pack, record=True)
                if sim_cur is not None else None)
    fo = proj_sug["firstOrder"]
    next_order = None
    if (fo >= 0).mean() >= 0.5:
        next_order = (md.start + dt.timedelta(days=md.days - 1 + int(np.median(fo[fo >= 0])))).isoformat()

    dev = None
    if status == "ok" and cur_s > 0:
        dev = round((ss - cur_s) / cur_s, 4)
    result.update(
        status=status,
        ss=ss,
        rawSs=raw_ss,
        hold=hold,
        level=params.level,
        upRule=params.up_rule,
        max=smax,
        batch=batch,
        dev=dev,
        alert=alert_level(status, cur_s, ss, params),
        simSug=_sim_summary(sim_sug),
        simCur=_sim_summary(sim_cur),
        saving=None if saving is None else round(saving, 4),
        actual=actual,
        savingActual=None if saving_actual is None else round(saving_actual, 4),
        nextOrder=next_order,
        conf=_confidence(nz, len(d), lead),
    )
    result["why"] = _explain(result)
    if with_series:
        result["series"] = _series(md, hist, cover, bom_part, proj_sug, proj_cur, params)
    return result


def actual_baseline(md: MaterialData, hist: dict, cover: np.ndarray) -> dict | None:
    """GWMS5 窗口内的实际情况（不含导出当天）：平均在库、平均自由在库、断货天数（折年）。倒推出负数时不给。"""
    if hist["negative"] or md.days < 8:
        return None
    free = np.maximum(hist["free"][:-1], 0.0)
    total = np.maximum(hist["total"][:-1], 0.0)
    d = cover[:-1]
    n = len(d)
    return {"avgTotal": round(float(total.mean()), 2), "avgFree": round(float(free.mean()), 2),
            "stockoutDays": round(float(((free <= 1e-9) & (d > 0)).sum() * 365.0 / n), 1), "days": n}


def _sim_summary(sim: dict | None) -> dict | None:
    if not sim:
        return None
    return {"avg": round(sim["avg"], 2), "stockoutDays": round(sim["stockoutDays"], 1),
            "fill": round(sim["fill"], 4), "orders": round(sim["orders"], 1)}


def _confidence(nz: int, n: int, lead: dict) -> str:
    if lead["source"] == "默认值":       # 没有交货期数据时，30 天默认值会把建议值明显抬高
        return "低"
    if nz >= 20 and n >= 90 and lead["source"].startswith("SAP"):
        return "高"
    if nz >= 10:
        return "中"
    return "低"


def _explain(r: dict) -> str:
    u = r["unit"] or ""
    hist = f"近 {r['historyDays']} 天（{r['historySrc']}）"
    parts = ([f"激进程度：{r['level']}"] if r.get("level") else []) + [f"{hist}需由安全库存覆盖的出库，{r['nzDays']} 天有出库，日均 {_fmt(r['mu'])}{u}"]
    if r["bomDemand"] > 0:
        parts.append(f"GWMS5 窗口内按单/BOM 出库 {_fmt(r['bomDemand'])}{u}（占 {r['bomShare']:.0%}）已剔除，不参与优化")
    lead = f"交货期 {_fmt(r['lead'])} 天，取自{r['leadSrc']}"
    if r.get("leadIqr"):
        lead += f"（{r['leadN']} 个样本，中间一半在 {_fmt(r['leadIqr'][0])}–{_fmt(r['leadIqr'][1])} 天）"
    parts.append(lead)
    if r.get("hold"):
        parts.append(f"交货期内需求的 P{round(r['serviceLevel'] * 100)} 是 {_fmt(r['rawSs'])}{u}，比现行值高，"
                     f"但{r['hold']}，按「{UP_RULES.get(r.get('upRule'), '')}」保持现行 {_fmt(r['ss'])}{u}；"
                     f"建议最高库存 {_fmt(r['max'])}{u}，每次申报约 {_fmt(r['batch'])}{u}")
    else:
        parts.append(f"交货期内需求的 P{round(r['serviceLevel'] * 100)} 是 {_fmt(r['ss'])}{u}，即建议安全库存；"
                     f"建议最高库存 {_fmt(r['max'])}{u}，每次申报约 {_fmt(r['batch'])}{u}")
    if r["status"] == "ok" and r["cur"] > 0 and r["dev"] is not None:
        if abs(r["dev"]) < 0.005:
            parts.append(f"现行安全库存 {_fmt(r['cur'])}{u}，与建议值基本一致")
        else:
            direction = "下调" if r["dev"] < 0 else "上调"
            parts.append(f"现行安全库存 {_fmt(r['cur'])}{u}，建议{direction} {abs(r['dev']):.0%}")
    if r.get("oldRuleNow") is not None:
        parts.append(f"按 f-1 现行公式（日均 × 采购周期 {_fmt(r['pcycle'])} 天）用同一段历史算是 {_fmt(r['oldRuleNow'])}{u}")
    sc, sg = r.get("simCur"), r.get("simSug")
    if sc and sg:
        parts.append(f"用同一组需求模拟一年：平均在库 {_fmt(sc['avg'])} → {_fmt(sg['avg'])}{u}"
                     + (f"（{-r['saving']:+.0%}）" if r["saving"] is not None else "")
                     + f"，缺货天数 {sc['stockoutDays']:.0f} → {sg['stockoutDays']:.0f} 天/年")
    elif sg:
        parts.append(f"按建议规则模拟一年：平均在库 {_fmt(sg['avg'])}{u}，缺货 {sg['stockoutDays']:.0f} 天/年")
    act = r.get("actual")
    if act and sg and r.get("savingActual") is not None:
        parts.append(f"对照实际：近 {act['days']} 天平均自由在库 {_fmt(act['avgFree'])}{u}，按建议规则模拟 {_fmt(sg['avg'])}{u}"
                     f"（{-r['savingActual']:+.0%}）；实际断货约 {act['stockoutDays']:.0f} 天/年")
    if r.get("nextOrder"):
        parts.append(f"按建议规则，预计 {r['nextOrder']} 前后触发申报（未计在途）")
    return "；".join(parts) + "。"


def _round_list(arr, nd=2):
    return [None if a is None else round(float(a), nd) for a in arr]


def _series(md: MaterialData, hist: dict, cover: np.ndarray, bom: np.ndarray,
            proj_sug: dict | None, proj_cur: dict | None, params: Params) -> dict:
    kernel = np.ones(7) / 7.0
    full = cover[:-1]
    trend = np.convolve(full, kernel, mode="same") if len(full) >= 7 else full
    out = {
        "dates": md.dates,
        "demandCover": _round_list(cover),
        "demandBom": _round_list(bom),
        "receipts": _round_list(md.in_free + md.in_bom),
        "trend": _round_list(list(trend)) + [None],
        "onhandTotal": _round_list(np.maximum(hist["total"], 0)),
        "onhandFree": _round_list(np.maximum(hist["free"], 0)),
    }
    if proj_sug is not None:
        last = md.start + dt.timedelta(days=md.days - 1)
        out["projDates"] = [(last + dt.timedelta(days=i + 1)).isoformat() for i in range(params.horizon)]
        for key, proj in (("projSug", proj_sug), ("projCur", proj_cur)):
            if proj is None:
                continue
            q = np.quantile(proj["trace"], [0.1, 0.5, 0.9], axis=0)
            out[key] = {"p10": _round_list(q[0]), "p50": _round_list(q[1]), "p90": _round_list(q[2])}
    return out
