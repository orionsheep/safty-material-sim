#!/usr/bin/env python3
"""本地服务：静态页面 + 安全库存试算 API。只用标准库和 numpy。

  python3 server.py [--port 8770]

API
  GET  /api/status                         数据窗口、结果时间、AI 是否可用
  GET  /api/results                        最近一次批量试算（data/gwms/results.json）
  POST /api/batch     {serviceLevel, bomBasis, useSapHistory}   重跑批量试算并保存
  POST /api/analyze   {row, params}        单物料试算（含图表序列）；row 用页面上的 f-1 行，改过的现行值也生效
  GET  /api/feedback[?code=]               人工确认记录
  POST /api/feedback  {code, decision, value, ...}   追加一条人工确认（data/feedback.jsonl）
  POST /api/ask       {row, params, question, history}  AI 助手
  GET  /api/overview                       曲线看板：全局每日曲线 + 在库变化分布
  GET  /api/portfolio-curve[?bomBasis=&force=1]  曲线看板：不同服务水平整批重算（有缓存，首次约 15 秒）
  POST /api/curves    {row, params}        曲线看板：单物料曲线（含四档激进程度下的建议值）
  GET  /api/stock-index[?force=1]            全部物料在库指数（各自实际平均自由在库 = 100）：过去实际 + 未来按当前档位 / 现行规则
  GET  /api/levels[?bomBasis=&force=1]      激进程度对比：四档各整批重算（有缓存，首次约 15 秒）
  params 里可以带 level（稳健 / 均衡 / 激进 / 极限），带了就用预设的服务水平、缺交货期兜底值和上调规则
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import threading
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine import assist, curves  # noqa: E402
from engine.batch import DATASET, RESULTS, MTYPE_NAMES, run_batch  # noqa: E402
from engine.model import LEVELS, Params, analyze, level_params, material_from_dataset  # noqa: E402

FEEDBACK = ROOT / "data" / "feedback.jsonl"
DECISIONS = {"accept": "采纳建议", "modify": "改为人工值", "reject": "保留现行值"}

_lock = threading.Lock()
_dataset: dict | None = None


def dataset() -> dict:
    global _dataset
    if _dataset is None:
        _dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    return _dataset


def params_from(p: dict | None) -> Params:
    p = p or {}
    kw = {}
    level = p.get("level") or None
    if level is not None and level not in LEVELS:
        raise ValueError(f"激进程度只能是 {' / '.join(LEVELS)}")
    if p.get("serviceLevel") is not None and level is None:
        sl = float(p["serviceLevel"])
        if not 0.5 <= sl < 1:
            raise ValueError("服务水平要在 0.5 到 0.99 之间")
        kw["service_level"] = sl
    if p.get("bomBasis") is not None:
        if p["bomBasis"] not in ("attr", "attr+planned", "none"):
            raise ValueError("bomBasis 只能是 attr / attr+planned / none")
        kw["bom_basis"] = p["bomBasis"]
    if p.get("leadOverride") not in (None, "", 0):
        kw["lead_override"] = float(p["leadOverride"])
    if p.get("useSapHistory") is not None:
        kw["use_sap_history"] = bool(p["useSapHistory"])
    return level_params(level, **kw) if level else Params().with_(**kw)


def analyze_row(row: dict, p: dict | None) -> dict:
    code = str(row.get("no") or "").strip()
    if not code:
        raise ValueError("缺少物料编码")
    md = material_from_dataset(dataset(), code, row.get("unit") or "")
    res = analyze(md, row, params_from(p), with_series=True)
    res["mtypeName"] = MTYPE_NAMES.get(_int(row.get("mtype")), str(row.get("mtype")))
    return res


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def read_feedback(code: str | None = None) -> list[dict]:
    if not FEEDBACK.exists():
        return []
    out = []
    for line in FEEDBACK.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if code is None or rec.get("code") == code:
            out.append(rec)
    return out


def add_feedback(body: dict) -> dict:
    code = str(body.get("code") or "").strip()
    decision = body.get("decision")
    if not code or decision not in DECISIONS:
        raise ValueError("需要物料编码和 decision（accept / modify / reject）")
    rec = {
        "time": dt.datetime.now().isoformat(timespec="seconds"),
        "code": code,
        "name": body.get("name"),
        "decision": decision,
        "decisionText": DECISIONS[decision],
        "cur": body.get("cur"),
        "curMax": body.get("curMax"),
        "suggested": body.get("suggested"),
        "suggestedMax": body.get("suggestedMax"),
        "final": body.get("final"),
        "finalMax": body.get("finalMax"),
        "reason": (body.get("reason") or "").strip()[:500],
        "user": (body.get("user") or "").strip()[:40],
        "params": body.get("params"),
        "alert": body.get("alert"),
        "conf": body.get("conf"),
        "leadSrc": body.get("leadSrc"),
        "window": dataset()["window"],
    }
    with _lock:
        FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
        with FEEDBACK.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, fmt, *args):
        if args and "/api/" in str(args[0]):
            sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % args))

    def end_headers(self):
        if self.path.startswith("/data/") or self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 2_000_000:
            raise ValueError("请求太大")
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self):
        u = urlparse(self.path)
        if not u.path.startswith("/api/"):
            # 结果、反馈、原始镜像之外的数据文件照常静态读取
            if u.path.startswith("/engine/") or u.path.endswith(".py"):
                return self.send_error(404)
            return super().do_GET()
        try:
            if u.path == "/api/status":
                res_time = None
                if RESULTS.exists():
                    res_time = json.loads(RESULTS.read_text(encoding="utf-8")).get("builtAt")
                return self._json({"window": dataset()["window"], "meta": dataset()["meta"],
                                   "resultsBuiltAt": res_time, "ai": assist.configured()})
            if u.path == "/api/results":
                if not RESULTS.exists():
                    return self._json({"error": "还没有批量试算结果，先运行一次"}, 404)
                data = RESULTS.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return self.wfile.write(data)
            if u.path == "/api/overview":
                if not RESULTS.exists():
                    return self._json({"error": "还没有批量试算结果，先运行一次"}, 404)
                return self._json(curves.overview(dataset(), json.loads(RESULTS.read_text(encoding="utf-8"))))
            if u.path == "/api/portfolio-curve":
                qs = parse_qs(u.query)
                basis = qs.get("bomBasis", ["attr"])[0]
                if basis not in ("attr", "attr+planned", "none"):
                    raise ValueError("bomBasis 只能是 attr / attr+planned / none")
                with _lock:
                    return self._json(curves.portfolio_sweep(basis, force=qs.get("force", ["0"])[0] == "1"))
            if u.path == "/api/stock-index":
                if not RESULTS.exists():
                    return self._json({"error": "还没有批量试算结果，先运行一次"}, 404)
                res = json.loads(RESULTS.read_text(encoding="utf-8"))
                rp = res.get("params") or {}
                params = params_from({"level": rp.get("level"), "serviceLevel": rp.get("serviceLevel"),
                                      "bomBasis": rp.get("bomBasis"), "useSapHistory": rp.get("useSapHistory")})
                force = parse_qs(u.query).get("force", ["0"])[0] == "1"
                with _lock:
                    return self._json(curves.stock_index(params, res.get("builtAt", ""), force=force))
            if u.path == "/api/levels":
                qs = parse_qs(u.query)
                basis = qs.get("bomBasis", ["attr"])[0]
                if basis not in ("attr", "attr+planned", "none"):
                    raise ValueError("bomBasis 只能是 attr / attr+planned / none")
                with _lock:
                    return self._json(curves.level_compare(basis, force=qs.get("force", ["0"])[0] == "1"))
            if u.path == "/api/feedback":
                code = parse_qs(u.query).get("code", [None])[0]
                return self._json({"rows": read_feedback(code)})
            return self._json({"error": "没有这个接口"}, 404)
        except ValueError as exc:
            return self._json({"error": str(exc)}, 400)
        except Exception as exc:
            traceback.print_exc()
            return self._json({"error": str(exc)}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            body = self._body()
            if u.path == "/api/analyze":
                return self._json(analyze_row(body.get("row") or {}, body.get("params")))
            if u.path == "/api/curves":
                row = body.get("row") or {}
                if not str(row.get("no") or "").strip():
                    raise ValueError("缺少物料编码")
                out = curves.material_curves(dataset(), row, params_from(body.get("params")))
                out["base"]["mtypeName"] = MTYPE_NAMES.get(_int(row.get("mtype")), str(row.get("mtype")))
                return self._json(out)
            if u.path == "/api/batch":
                with _lock:
                    res = run_batch(params_from(body), dataset_path=DATASET, workers=4)
                    RESULTS.write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
                return self._json({"builtAt": res["builtAt"], "seconds": res["seconds"], "summary": res["summary"],
                                   "params": res["params"]})
            if u.path == "/api/feedback":
                return self._json(add_feedback(body))
            if u.path == "/api/ask":
                q = (body.get("question") or "").strip()
                if not q:
                    raise ValueError("问题不能为空")
                if not assist.configured():
                    return self._json({"error": "AI 助手未配置：启动服务前设置 ANTHROPIC_API_KEY（或 ANTHROPIC_AUTH_TOKEN）"}, 503)
                res = analyze_row(body.get("row") or {}, body.get("params"))
                res.pop("series", None)
                try:
                    answer = assist.ask(res, q[:1000], history=body.get("history") or None,
                                        feedback=read_feedback(res["code"]))
                except RuntimeError as exc:      # 上游 AI 接口的问题（密钥、网络），不是本服务出错
                    return self._json({"error": str(exc)}, 502)
                return self._json({"answer": answer})
            return self._json({"error": "没有这个接口"}, 404)
        except ValueError as exc:
            return self._json({"error": str(exc)}, 400)
        except Exception as exc:
            traceback.print_exc()
            return self._json({"error": str(exc)}, 500)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="卫华安全库存插件本地服务")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)
    dataset()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"打开 http://{args.host}:{args.port}/   （AI 助手：{'可用' if assist.configured() else '未配置'}）")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
