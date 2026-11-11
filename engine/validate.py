"""用 SAP 每日出入库 + 每日库存（副本出入库数据.xlsx 这类表）做小样本回测验证。

做法（前训练、后检验）：
  1. 训练期（默认前 92 天）：用日出库做 7 天分块自助抽样，交货期从实际到货记录里随机抽，
     得到「交货期内需求」的分布，取 P 分位作为安全库存（= 申报点 s），补到 S = s + 批量。
  2. 检验期（剩下的天数）：把真实的每日出库原样回放，按 (s, S) 规则申报，每张单的交货期
     从实际交货期里随机抽（重复 300 次），得到模拟在库曲线；和现场实际在库、按现行值补货对比。
  3. 安全库存曲线：每天用此前的数据重算一次安全库存，看它随时间怎么变。
  4. 取舍曲线：不同服务水平下，检验期的平均在库 vs 缺货天数。

模型只知道训练期的数据；回放时的「现实」用全部实际交货期。
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import numpy as np

from .model import bootstrap_paths, round_qty, seed_for

P_LEVELS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]
LEVEL_P = {"稳健": 0.90, "均衡": 0.85, "激进": 0.75, "极限": 0.60}


@dataclass
class SapSeries:
    code: str
    name: str
    unit: str
    start: dt.date
    out: np.ndarray            # 日出库（正数）
    ret: np.ndarray            # 日退库（出库列里的正数）
    rec: np.ndarray            # 日入库
    stock: np.ndarray          # 日终库存（表里给的，没有发生的日子沿用前一天）
    lead: list = field(default_factory=list)   # [(到货日序号, 交货天数)]
    pcycle: float = 0.0
    daily_avg: float = 0.0
    check_gap: float = 0.0     # 库存与出入库递推的最大差

    @property
    def days(self) -> int:
        return len(self.out)

    def date(self, i: int) -> str:
        return (self.start + dt.timedelta(days=i)).isoformat()


def _f(v) -> float:
    try:
        return float(v) if v not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0


def read_workbook(path: str) -> list[SapSeries]:
    """每个 sheet 一个物料：过账日期 / 物料 / 物料描述 / [单位] / 出库 / 入库 / 每日库存 / 采购周期 / 日均用量 / 交货期。"""
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = []
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue
        head = [str(h or "").replace("\n", "") for h in rows[0]]

        def col(*keys):
            for i, h in enumerate(head):
                if any(k in h for k in keys):
                    return i
            return None
        c_date, c_code, c_name, c_unit = col("过账日期"), col("物料") , col("物料描述"), col("单位")
        c_out, c_in, c_stock = col("出库数量"), col("入库数量"), col("每日库存")
        c_pc, c_avg, c_lead = col("采购周期"), col("日均用量"), col("交货")
        unit = "KG" if "KG" in "".join(head) else ""
        recs = []
        code = name = ""
        for r in rows[1:]:
            d = r[c_date]
            if not hasattr(d, "date"):
                continue
            recs.append(r)
            if r[c_code] and not code:
                code, name = str(r[c_code]).strip(), str(r[c_name] or "").strip()
            if c_unit is not None and r[c_unit] and not unit:
                unit = str(r[c_unit]).strip()
        if not recs:
            continue
        start = recs[0][c_date].date()
        end = recs[-1][c_date].date()
        n = (end - start).days + 1
        o, rt, rc = np.zeros(n), np.zeros(n), np.zeros(n)
        st = np.full(n, np.nan)
        lead, pc, avg = [], 0.0, 0.0
        for r in recs:
            i = (r[c_date].date() - start).days
            q = _f(r[c_out])
            if q < 0:
                o[i] += -q
            elif q > 0:
                rt[i] += q
            rc[i] += _f(r[c_in])
            if r[c_stock] is not None:
                st[i] = _f(r[c_stock])
            if c_pc is not None and _f(r[c_pc]) > 0:
                pc = _f(r[c_pc])
            if c_avg is not None and _f(r[c_avg]) > 0:
                avg = _f(r[c_avg])
            if c_lead is not None and _f(r[c_lead]) > 0 and _f(r[c_in]) > 0:
                lead.append((i, _f(r[c_lead])))
        for i in range(n):                      # 没发生的日子库存不变
            if np.isnan(st[i]):
                st[i] = st[i - 1] if i else 0.0
        st = np.maximum(st, 0.0)
        rec = st[0] + np.cumsum(rc - o + rt) - (rc[0] - o[0] + rt[0])
        gap = float(np.max(np.abs(rec - st))) if n else 0.0
        out.append(SapSeries(code=code or ws.title, name=name, unit=unit or "PC", start=start, out=o, ret=rt,
                             rec=rc, stock=st, lead=lead, pcycle=pc, daily_avg=avg, check_gap=gap))
    return out


# ------------------------------------------------------------------ 拟合

def ltd_samples(d: np.ndarray, leads: np.ndarray, paths: int, block: int, rng) -> np.ndarray:
    """交货期内需求：交货期从实际样本里抽，需求按 7 天分块抽。"""
    L = np.maximum(np.rint(rng.choice(leads, size=paths)).astype(int), 1)
    D = bootstrap_paths(d, int(L.max()), paths, block, rng)
    cs = np.cumsum(D, axis=1)
    return cs[np.arange(paths), L - 1]


def fit_policy(d: np.ndarray, leads: np.ndarray, p: float, unit: str, pack: float, rng,
               paths: int = 4000, block: int = 7) -> dict:
    ltd = ltd_samples(d, leads, paths, block, rng)
    s = round_qty(float(np.quantile(ltd, p)), unit)
    mu = float(d.mean())
    batch = max(mu * float(np.median(leads)), pack if pack > 0 else 0.0)
    if pack > 0:
        batch = math.ceil(batch / pack - 1e-9) * pack
    batch = round_qty(batch, unit)
    return {"s": s, "S": s + batch, "batch": batch, "ltdMean": round(float(ltd.mean()), 2)}


# ------------------------------------------------------------------ 回放

FILL_TARGET = {"稳健": 0.995, "均衡": 0.99, "激进": 0.98, "极限": 0.95}


def lead_matrix(leads: np.ndarray, reps: int, T: int, rng) -> np.ndarray:
    """共用随机数：第 t 天下的单，交货期 = Lmat[rep, t]，所有规则用同一张表，比较才公平。"""
    return np.maximum(np.rint(rng.choice(leads, size=(reps, T))).astype(int), 1)


def replay(dem: np.ndarray, ret: np.ndarray | None, s: float, S: float, Lmat: np.ndarray, *, init: float,
           pipeline: np.ndarray, pack: float, warmup: int = 0) -> dict:
    """按 (s, S) 规则逐日走一遍。dem 是一维（真实需求原样回放）或 (reps, T)（抽样路径）。
    缺货不丢单，等到货再发；前 warmup 天只走不计。"""
    reps = Lmat.shape[0]
    D = np.broadcast_to(dem, (reps, dem.shape[-1]))
    T = D.shape[1]
    if ret is None:
        ret = np.zeros(T)
    arrivals = np.zeros((reps, T + int(Lmat.max()) + len(pipeline) + 2))
    arrivals[:, :len(pipeline)] += pipeline[None, :]
    net = np.full(reps, float(init))
    onorder = np.full(reps, float(pipeline.sum()))
    trace = np.empty((reps, T))
    short_days = np.zeros(reps)
    short_qty = np.zeros(reps)
    orders = np.zeros(reps)
    S = max(S, s)
    rows = np.arange(reps)
    for t in range(T):
        a = arrivals[:, t]
        net += a + ret[t]
        onorder -= a
        short = np.maximum(D[:, t] - np.maximum(net, 0.0), 0.0)
        net -= D[:, t]
        if t >= warmup:
            short_days += short > 0
            short_qty += short
        pos = net + onorder
        need = pos <= s
        if need.any():
            q = np.where(need, S - pos, 0.0)
            if pack > 0:
                q = np.ceil(q / pack - 1e-9) * pack
            q = np.maximum(q, 0.0)
            arrivals[rows, t + Lmat[:, t]] += q
            onorder += q
            orders += need & (q > 0)
        trace[:, t] = np.maximum(net, 0.0)
    trace = trace[:, warmup:]
    tot = D[:, warmup:].sum(axis=1).mean()
    p10, p50, p90 = np.percentile(trace, [10, 50, 90], axis=0)
    return {
        "avg": float(trace.mean()),
        "stockoutDays": float(short_days.mean()),
        "stockoutDaysP90": float(np.percentile(short_days, 90)),
        "fill": float(1 - short_qty.mean() / tot) if tot > 0 else 1.0,
        "orders": float(orders.mean()),
        "p10": p10, "p50": p50, "p90": p90,
    }


def _batch(mu: float, lead_med: float, pack: float, unit: str) -> float:
    b = max(mu * lead_med, pack if pack > 0 else 0.0)
    if pack > 0:
        b = math.ceil(b / pack - 1e-9) * pack
    return round_qty(b, unit)


def _grid(top: float, unit: str, k: int = 60) -> list[float]:
    vals = sorted({round_qty(x, unit) for x in np.linspace(0, max(top, 1.0), k)})
    return vals


def _state(ss: "SapSeries", t0: int, lead_med: float) -> tuple[float, np.ndarray]:
    """回放起点：t0 前一天的实际库存 + 已在路上的单（起点后一个中位交货期内的实际到货）。"""
    init = float(ss.stock[t0 - 1]) if t0 > 0 else float(ss.stock[0] + ss.out[0] - ss.rec[0] - ss.ret[0])
    k = max(int(round(lead_med)), 1)
    pipe = np.zeros(k + 1)
    seg = ss.rec[t0:t0 + k]
    pipe[:len(seg)] = seg
    return max(init, 0.0), pipe


def fit_by_backtest(ss: "SapSeries", t0: int, t1: int, leads: np.ndarray, pack: float, target: float,
                    reps: int, rng, *, warmup: int = 60, horizon: int = 180) -> dict:
    """用 [t0, t1) 的真实日需求按 7 天一块抽 reps 条路径，交货期从实际样本抽，
    稳态模拟（补满起步、丢掉预热期），找满足率 >= target 的最小申报点 s。"""
    dem = ss.out[t0:t1]
    mu = float(dem.mean())
    lead_med = float(np.median(leads))
    batch = _batch(mu, lead_med, pack, ss.unit)
    T = warmup + horizon
    D = bootstrap_paths(dem, T, reps, 7, rng)
    Lmat = lead_matrix(leads, reps, T, rng)
    top = float(np.quantile(ltd_samples(dem, leads, 3000, 7, rng), 0.995)) * 1.3
    best, curve = None, []
    for s in _grid(top, ss.unit):
        r = replay(D, None, s, s + batch, Lmat, init=s + batch, pipeline=np.zeros(1), pack=pack, warmup=warmup)
        curve.append((s, r["fill"], r["avg"]))
        if best is None and r["fill"] >= target:
            best = s
            break
    return {"s": best if best is not None else curve[-1][0], "batch": batch, "met": best is not None,
            "curve": curve}


def _r(a, nd=2):
    return [round(float(x), nd) for x in a]


def validate(ss: SapSeries, cur_s: float, cur_S: float, pack: float, *, train_days: int = 92,
             level: str = "激进", reps: int = 300) -> dict:
    n = ss.days
    target = FILL_TARGET[level]
    p_level = LEVEL_P[level]
    d_tr = ss.out[:train_days]
    lead_all = np.array([x for _, x in ss.lead], dtype=float)
    lead_tr = np.array([x for i, x in ss.lead if i < train_days], dtype=float)
    lead_src = "训练期实际交货期"
    if len(lead_tr) < 3:
        lead_tr = lead_all if len(lead_all) >= 3 else np.array([ss.pcycle or 10.0])
        lead_src = "全部实际交货期（训练期到货记录不足 3 次）" if len(lead_all) >= 3 else "采购周期"
    world = lead_all if len(lead_all) >= 3 else lead_tr
    lead_med = float(np.median(world))

    # 1) 拟合：训练期回放找最小 s
    fit = fit_by_backtest(ss, 0, train_days, lead_tr, pack, target, reps,
                          np.random.default_rng(seed_for(ss.code, "fit")))
    # 2) 原引擎做法：交货期内需求 P 分位
    eng = fit_policy(d_tr, lead_tr, p_level, ss.unit, pack, np.random.default_rng(seed_for(ss.code, "eng")))

    # 3) 检验期：同一张交货期表回放
    dem, ret = ss.out[train_days:], ss.ret[train_days:]
    init, pipe = _state(ss, train_days, lead_med)
    Lmat = lead_matrix(world, reps, len(dem), np.random.default_rng(seed_for(ss.code, "test")))

    def run(s, S):
        return replay(dem, ret, s, S, Lmat, init=init, pipeline=pipe, pack=pack)

    def pack_res(r, extra=None):
        o = {"avg": round(r["avg"], 2), "stockoutDays": round(r["stockoutDays"], 1),
             "stockoutDaysP90": r["stockoutDaysP90"], "fill": round(r["fill"], 4), "orders": round(r["orders"], 1),
             "p50": _r(r["p50"])}
        if extra:
            o.update(extra)
        return o

    r_fit = run(fit["s"], fit["s"] + fit["batch"])
    r_eng = run(eng["s"], eng["S"])
    r_cur = run(cur_s, cur_S) if cur_s > 0 else None

    # 取舍曲线：检验期里 s 从 0 往上扫
    top = max(fit["s"], eng["s"], cur_s) * 1.6 + 1
    sweep = []
    for s in _grid(top, ss.unit, 40):
        r = run(s, s + fit["batch"])
        sweep.append({"s": s, "avg": round(r["avg"], 2), "stockoutDays": round(r["stockoutDays"], 2),
                      "fill": round(r["fill"], 4)})
    test_best = next((x["s"] for x in sweep if x["fill"] >= target), None)

    act = ss.stock[train_days:]
    actual = {"avg": round(float(act.mean()), 2), "zeroDemandDays": int(((act <= 0) & (dem > 0)).sum()),
              "zeroEndDays": int((act <= 0).sum()), "receipts": int((ss.rec[train_days:] > 0).sum())}

    # 4) 安全库存曲线：每 7 天用此前（最多 92 天）的数据重新拟合一次
    roll = []
    for t in list(range(60, n, 7)) + [n]:
        t0 = max(0, t - 92)
        ld = np.array([x for i, x in ss.lead if i < t], dtype=float)
        if len(ld) < 3:
            ld = lead_tr
        f = fit_by_backtest(ss, t0, t, ld, pack, target, 200, np.random.default_rng(seed_for(ss.code, f"roll|{t}")))
        e = fit_policy(ss.out[t0:t], ld, p_level, ss.unit, pack, np.random.default_rng(seed_for(ss.code, f"rolle|{t}")))
        roll.append({"date": ss.date(t - 1), "fit": f["s"], "eng": e["s"], "ltdMean": e["ltdMean"],
                     "met": f["met"]})

    return {
        "code": ss.code, "name": ss.name, "unit": ss.unit, "level": level, "fillTarget": target, "levelP": p_level,
        "dates": [ss.date(i) for i in range(n)],
        "trainEnd": ss.date(train_days - 1), "testStart": ss.date(train_days), "testDays": n - train_days,
        "out": _r(ss.out), "rec": _r(ss.rec), "stock": _r(ss.stock), "checkGap": round(ss.check_gap, 3),
        "stats": {"days": n, "nzDays": int((ss.out > 0).sum()), "mean": round(float(ss.out.mean()), 3),
                  "meanTrain": round(float(d_tr.mean()), 3), "meanTest": round(float(dem.mean()), 3),
                  "pcycle": ss.pcycle, "dailyAvgSheet": ss.daily_avg},
        "lead": {"all": [x for _, x in ss.lead], "train": lead_tr.tolist(), "src": lead_src,
                 "median": round(lead_med, 1), "p25": round(float(np.percentile(world, 25)), 1),
                 "p75": round(float(np.percentile(world, 75)), 1)},
        "cur": {"s": cur_s, "S": cur_S, "pack": pack},
        "fit": pack_res(r_fit, {"s": fit["s"], "S": fit["s"] + fit["batch"], "batch": fit["batch"], "met": fit["met"],
                                "p10": _r(r_fit["p10"]), "p90": _r(r_fit["p90"])}),
        "eng": pack_res(r_eng, {"s": eng["s"], "S": eng["S"]}),
        "curSim": None if r_cur is None else pack_res(r_cur, {"s": cur_s, "S": cur_S}),
        "actual": actual,
        "sweep": sweep, "testBest": test_best,
        "rolling": roll,
    }


# ================================================================== 看计划补货 + 过拟合（当前报告用这一套）
#
# 安全库存的含义改成：交货期内的计划用量之外，额外多放的量。
# 补货：预计库存 = 在库 + 在途 − 未来 Lp 天的计划用量，低于安全库存就下单，补 (安全库存 − 预计) + 若干天用量。
# 过拟合：计划用量直接用历史真实出库；每张单的交货期用时间上最近那次实际到货的交货期。

PLAN_TARGET = {"稳健": 1.0, "均衡": 0.99, "激进": 0.98, "极限": 0.95}
COVER_DAYS = [1, 2, 3, 5, 7, 10, 14, 21, 30]


def real_lead_row(ss: SapSeries) -> np.ndarray:
    """第 t 天下的单 → 隐含下单日（到货日 − 交货期）离 t 最近的那次实际到货的交货期。"""
    if not ss.lead:
        return np.full(ss.days, max(int(round(ss.pcycle or 10)), 1))
    rec = np.array([(i - L, L) for i, L in ss.lead], float)
    idx = np.abs(rec[:, 0][None, :] - np.arange(ss.days)[:, None]).argmin(axis=1)
    return np.maximum(np.rint(rec[idx, 1]).astype(int), 1)


def plan_replay(ss: SapSeries, B: float, cover: float, *, t0: int = 0, t1: int | None = None, pack: float = 0.0,
                Lrow: np.ndarray | None = None, Lp: int = 1, vis: int | None = None, record: bool = False) -> dict:
    t1 = ss.days if t1 is None else t1
    Lrow = real_lead_row(ss) if Lrow is None else Lrow
    vis = Lp if vis is None else vis
    mu = float(ss.out[t0:t1].mean()) if t1 > t0 else 0.0
    warm = max(int(np.median(Lrow)), 1)
    init = float(ss.stock[t0 - 1]) if t0 > 0 else float(ss.stock[0] + ss.out[0] - ss.rec[0] - ss.ret[0])
    arr = np.zeros(ss.days + int(Lrow.max()) + 5)
    arr[t0:t0 + warm] += ss.rec[t0:t0 + warm]          # 起点时已经在路上的单
    net, onorder = max(init, 0.0), float(ss.rec[t0:t0 + warm].sum())
    trace, sdays, sq, dq, orders = [], 0, 0.0, 0.0, 0
    for t in range(t0, t1):
        a = arr[t]
        net += a + ss.ret[t]
        onorder -= a
        sh = max(ss.out[t] - max(net, 0.0), 0.0)
        net -= ss.out[t]
        counted = t >= t0 + warm
        if counted:
            sdays += sh > 0
            sq += sh
            dq += ss.out[t]
        fut = ss.out[t + 1:min(t + 1 + min(vis, Lp), ss.days)].sum() + mu * max(Lp - vis, 0)
        proj = net + onorder - fut
        if proj < B:
            q = B - proj + mu * cover
            if pack > 0:
                q = math.ceil(q / pack - 1e-9) * pack
            q = round_qty(q, ss.unit)
            if q > 0:
                arr[t + Lrow[t]] += q
                onorder += q
                orders += counted
        if counted or record:
            trace.append(max(net, 0.0))
    body = trace[warm:] if record else trace
    out = {"shortDays": int(sdays), "fill": 1 - sq / dq if dq > 0 else 1.0, "avg": float(np.mean(body)) if body else 0.0,
           "orders": int(orders), "warm": warm}
    if record:
        out["trace"] = trace
    return out


def plan_fit(ss: SapSeries, pack: float, target: float, *, t0: int = 0, t1: int | None = None, cap: float | None = None,
             max_orders: int | None = None, tie: float = 0.02, look: int | None = None) -> dict:
    """在 [t0, t1) 上过拟合：满足率 >= target、下单次数不超过实际到货次数的方案里，平均在库最低的；
    在库和最低的差不到 2% 的方案里，再挑安全库存最低的。"""
    t1 = ss.days if t1 is None else t1
    Lrow = real_lead_row(ss)
    Lp = int(look) if look else int(np.ceil(np.median(Lrow)))
    seg = ss.out[t0:t1]
    top = max(cap or 0.0, float(np.quantile(np.convolve(seg, np.ones(Lp), "valid"), 0.95)) if len(seg) > Lp else float(seg.sum()), 1.0)
    grid = sorted({round_qty(x, ss.unit) for x in np.linspace(0, top * 1.2, 90)})
    feas, loose = [], []
    for B in grid:
        for cv in COVER_DAYS:
            r = plan_replay(ss, B, cv, t0=t0, t1=t1, pack=pack, Lrow=Lrow, Lp=Lp)
            row = (B, cv, r)
            loose.append(row)
            if r["fill"] + 1e-9 >= target and (max_orders is None or r["orders"] <= max_orders):
                feas.append(row)
    if feas:
        lo = min(x[2]["avg"] for x in feas)
        B, cv, r = min((x for x in feas if x[2]["avg"] <= lo * (1 + tie) + 1e-9), key=lambda x: (x[0], x[2]["avg"]))
        return {"B": B, "cover": cv, "Lp": Lp, **r, "met": True}
    B, cv, r = max(loose, key=lambda x: (x[2]["fill"], -x[2]["avg"]))
    return {"B": B, "cover": cv, "Lp": Lp, **r, "met": False}


def validate_plan(ss: SapSeries, cur_s: float, cur_S: float, pack: float, *, level: str = "激进",
                  train_days: int = 92, look: int | None = None) -> dict:
    """look = 下单时往前看几天的计划用量；传采购周期 = 按系统里的采购周期下单，None = 按实际交货期中位数。"""
    n = ss.days
    target = PLAN_TARGET[level]
    receipts = int((ss.rec > 0).sum())
    Lrow = real_lead_row(ss)
    fit = plan_fit(ss, pack, target, cap=cur_s, max_orders=receipts, look=look)
    full = plan_replay(ss, fit["B"], fit["cover"], pack=pack, Lrow=Lrow, Lp=fit["Lp"], record=True)
    warm = full["warm"]

    levels = []
    for name, tg in PLAN_TARGET.items():
        f = fit if name == level else plan_fit(ss, pack, tg, cap=cur_s, max_orders=receipts, look=look)
        levels.append({"name": name, "target": tg, "B": f["B"], "cover": f["cover"], "avg": round(f["avg"], 2),
                       "fill": round(f["fill"], 4), "shortDays": f["shortDays"], "orders": f["orders"], "met": f["met"]})

    # 取舍曲线：固定补货天数，B 从 0 往上扫
    top = max(fit["B"], cur_s) * 1.4 + 1
    sweep = []
    for B in sorted({round_qty(x, ss.unit) for x in np.linspace(0, top, 50)}):
        r = plan_replay(ss, B, fit["cover"], pack=pack, Lrow=Lrow, Lp=fit["Lp"])
        sweep.append({"s": B, "avg": round(r["avg"], 2), "fill": round(r["fill"], 4), "shortDays": r["shortDays"]})

    # 防过拟合检查：只用前 92 天拟合，后 92 天照样回放
    tr = plan_fit(ss, pack, target, t1=train_days, cap=cur_s,
                  max_orders=int((ss.rec[:train_days] > 0).sum()), look=look)
    te = plan_replay(ss, tr["B"], tr["cover"], t0=train_days, pack=pack, Lrow=Lrow, Lp=tr["Lp"])
    test = {"B": tr["B"], "cover": tr["cover"], "fill": round(te["fill"], 4), "shortDays": te["shortDays"],
            "avg": round(te["avg"], 2), "actualAvg": round(float(ss.stock[train_days + te["warm"]:].mean()), 2)}

    # 安全库存曲线：每 7 天用截至当天的全部数据重新过拟合（数据越多越稳）
    roll = []
    for t in list(range(60, n, 7)) + [n]:
        t0 = 0
        f = plan_fit(ss, pack, target, t0=t0, t1=t, cap=cur_s, max_orders=max(int((ss.rec[t0:t] > 0).sum()), 1), look=look)
        roll.append({"date": ss.date(t - 1), "B": f["B"], "met": f["met"]})

    act = ss.stock[warm:]
    lt = np.array([x for _, x in ss.lead], float) if ss.lead else np.array([ss.pcycle or 10.0])
    return {
        "code": ss.code, "name": ss.name, "unit": ss.unit, "level": level, "target": target,
        "pcycle": ss.pcycle, "look": fit["Lp"], "lookBy": "采购周期" if look else "实际交货期",
        "leadMedian": int(np.ceil(np.median(Lrow))),
        "dates": [ss.date(i) for i in range(n)], "warm": warm,
        "out": [round(float(x), 2) for x in ss.out], "rec": [round(float(x), 2) for x in ss.rec],
        "stock": [round(float(x), 2) for x in ss.stock],
        "trace": [round(float(x), 2) for x in full["trace"]],
        "stats": {"days": n, "nzDays": int((ss.out > 0).sum()), "mean": round(float(ss.out.mean()), 3),
                  "maxDay": round(float(ss.out.max()), 2), "pcycle": ss.pcycle, "receipts": receipts},
        "lead": {"all": lt.tolist(), "median": round(float(np.median(lt)), 1),
                 "p25": round(float(np.percentile(lt, 25)), 1), "p75": round(float(np.percentile(lt, 75)), 1)},
        "cur": {"s": cur_s, "S": cur_S, "pack": pack},
        "fit": {"B": fit["B"], "cover": fit["cover"], "Lp": fit["Lp"], "avg": round(full["avg"], 2),
                "fill": round(fit["fill"], 4), "shortDays": fit["shortDays"], "orders": fit["orders"], "met": fit["met"],
                "planQty": round_qty(float(ss.out.mean()) * fit["Lp"], ss.unit)},
        "actual": {"avg": round(float(act.mean()), 2), "zeroEndDays": int((act <= 0).sum()), "receipts": receipts},
        "levels": levels, "sweep": sweep, "test": test, "rolling": roll,
    }
