"""AI 助手：拿单物料的试算结果回答业务人员的问题（为什么这么建议、能不能再降、风险在哪）。

走 Anthropic Messages API（兼容中转）。读环境变量：
  ANTHROPIC_BASE_URL（默认 https://api.anthropic.com）
  ANTHROPIC_API_KEY 或 ANTHROPIC_AUTH_TOKEN
  SAFETY_AI_MODEL（默认取 ANTHROPIC_DEFAULT_SONNET_MODEL，再默认 claude-sonnet-5-5）
没配置就返回 None，由调用方提示「AI 助手未配置」。只用标准库。
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

SYSTEM = """你是卫华集团仓储部的安全库存分析助手，回答业务人员关于某个物料安全库存建议的问题。
规则：
- 只依据下面给出的试算结果和方法说明回答，数据里没有的不要编；不确定就直说需要补什么数据。
- 用中文，口语化、简短，先给结论再给理由，数字带单位。不要用 markdown 标题。
- 方法说明：需求 = 出库里扣掉按销售订单/BOM 到货的部分（这部分不优化）；建议安全库存 = 交货期内需求的服务水平分位数（默认 P90，用 7 天分块自助法重抽历史）；
  f-1 的安全库存是申报触发点：可用+在途低于它就申报，补到最高库存；建议最高库存 = 建议安全库存 + 一次申报量（日均 × 交货期，按包装数取整）。
  simCur/simSug 是用同一组模拟需求按现行值/建议值跑一年的平均在库和缺货天数；actual 是 GWMS5 近两个月出入库倒推的实际自由在库和实际断货天数。
  现场并不严格按 f-1 现行值补货，所以 actual 往往比 simCur 高很多，判断「能不能降库存」以 actual 对比 simSug 为准。
  交货期来源是「默认值」（30 天）时，建议值偏高、可信度低，要提醒补交货期数据。
- 人工确认是最终决定，你只给参考意见。"""

FIELDS = ("code", "name", "unit", "mtypeName", "status", "alert", "cur", "curMax", "ss", "max", "batch", "dev",
          "lead", "leadSrc", "leadN", "leadIqr", "mu", "sigma", "nzDays", "historyDays", "historySrc",
          "coverDemand", "bomDemand", "bomShare", "onhand", "onhandFree", "onhandBom", "simCur", "simSug",
          "saving", "actual", "savingActual", "nextOrder", "conf", "oldRuleNow", "pcycle", "notes", "why",
          "serviceLevel", "bomBasis", "outTypes", "suppliers")


def configured() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def ask(result: dict, question: str, history: list[dict] | None = None, feedback: list[dict] | None = None,
        timeout: float = 60.0) -> str | None:
    if not configured():
        return None
    base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    model = (os.environ.get("SAFETY_AI_MODEL") or os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL")
             or "claude-sonnet-5-5")
    ctx = {k: result.get(k) for k in FIELDS if result.get(k) is not None}
    content = "试算结果（JSON）：\n" + json.dumps(ctx, ensure_ascii=False)
    if feedback:
        content += "\n\n这个物料以往的人工确认记录：\n" + json.dumps(feedback[-5:], ensure_ascii=False)
    turns = [t for t in (history or []) if t.get("q") and t.get("a")][-6:]
    questions = [t["q"] for t in turns] + [question]
    messages = [{"role": "user", "content": content + "\n\n问题：" + questions[0]}]
    for turn, nxt in zip(turns, questions[1:]):
        messages.append({"role": "assistant", "content": turn["a"]})
        messages.append({"role": "user", "content": nxt})

    body = json.dumps({"model": model, "max_tokens": 900, "system": SYSTEM, "messages": messages}).encode("utf-8")
    headers = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
    if os.environ.get("ANTHROPIC_API_KEY"):
        headers["x-api-key"] = os.environ["ANTHROPIC_API_KEY"]
    else:
        headers["authorization"] = "Bearer " + os.environ["ANTHROPIC_AUTH_TOKEN"]
    req = urllib.request.Request(base + "/v1/messages", data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"AI 接口返回 {exc.code}：{exc.read().decode('utf-8', 'replace')[:300]}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"连不上 AI 接口：{exc.reason}") from None
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()
