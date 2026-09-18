#!/usr/bin/env python3
"""
model-router — TypeSafe (jev) classification → deterministic model dispatch → execution.

Pipeline
--------
  1. CLASSIFY   one jev call: task_type (choice), complexity (score),
                high_stakes (noul), speed_priority (noul)
  2. DECIDE     deterministic policy in code picks a tier + ordered fallback chain
  3. EXECUTE    first model in the chain answers; failures fall through the chain
  4. TRACE      full decision record returned alongside the answer

Pools (verified live 2026-09-18)
-------------------------------
  opencode-go : OpenAI-compatible, 30+ models, requires browser UA + x-opencode-session
  bailian     : Anthropic Messages API (Aliyun Token Plan)

Run
---
  python3 router.py "your prompt"                        # route + execute
  python3 router.py --route-only "your prompt"           # decision only
  python3 router.py --json "your prompt"                 # raw trace
  python3 router.py --model bailian/qwen3.7-max "..."    # force a model
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field

import httpx

# ----------------------------------------------------------------------------
# env / credentials (read from ~/.hermes/.env at runtime — never stored here)
# ----------------------------------------------------------------------------

HOME_ENV = os.path.expanduser("~/.hermes/.env")
ENV_KEYS = ("TYPESAFE_API_KEY", "TOKEN_PLAN_API_KEY", "OPENCODE_GO_API_KEY")


def load_env(path: str = HOME_ENV) -> dict:
    env = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip("'\"")
    except FileNotFoundError:
        pass
    for k in ENV_KEYS:  # real environment wins when set
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


ENV = load_env()

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"

OPENCODE_URL = "https://opencode.ai/zen/go/v1"
OPENCODE_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0 Safari/537.36"
)
OPENCODE_SESSION = str(uuid.uuid4())  # per-process session id (required by zen/go)

BAILIAN_URL = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/apps/anthropic/v1/messages"
# OpenAI-compatible sibling of the Token Plan endpoint (verified live 2026-09-18:
# accepts Bearer auth, supports stream + stream_options + tool calls).
BAILIAN_COMPAT_URL = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1/chat/completions"

DEFAULT_MAX_TOKENS = 2048
JEV_TIMEOUT = 30.0
MODEL_TIMEOUT = 180.0

# ----------------------------------------------------------------------------
# model pool (verified 2026-09-18; unavailable models intentionally omitted so
# the fallback chain never wastes a hop)
#   excluded: grok-4.5/4.6 (not on plan), gpt-5.6-luna (500),
#             deepseek-direct (402 balance), kimi-coding (quota),
#             openai (billing), minimax direct (plan limit)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Route:
    model: str
    provider: str  # "opencode-go" | "bailian"
    note: str = ""

    @property
    def id(self) -> str:
        return f"{self.provider}/{self.model}"


def oc(model: str, note: str = "") -> Route:
    return Route(model, "opencode-go", note)


def bl(model: str, note: str = "") -> Route:
    return Route(model, "bailian", note)


# Ordered tiers. First entry is the primary; the rest are same-tier fallbacks.
TIERS: dict[str, list[Route]] = {
    "fast": [
        bl("deepseek-v4.1-flash", "proven daily driver, ~1.4s"),
        oc("glm-5.3-flash", "fast GLM"),
        oc("qwen3.8-flash", "fast Qwen"),
        bl("qwen3.6-flash", "backup flash"),
    ],
    "balanced": [
        oc("glm-5.2", "strong generalist, ~0.9s"),
        oc("deepseek-v4-pro", "deep reasoning, fast"),
        bl("qwen3.7-plus", "qwen balanced"),
        bl("deepseek-v4-pro", "deepseek via token plan"),
    ],
    "code": [
        oc("kimi-k2.7-code", "code specialist, ~1.4s"),
        oc("kimi-k3", "strong coder"),
        bl("qwen3.7-max", "strong generalist backup"),
    ],
    "strong": [
        oc("kimi-k3", "top reasoning"),
        bl("qwen3.8-max-preview", "flagship qwen"),
        bl("qwen3.7-max", "strong qwen"),
        oc("qwen3.8-max", "strong qwen (zen)"),
    ],
    "creative": [
        bl("qwen3.7-max", "strong expressive"),
        oc("glm-5.3", "creative GLM"),
        oc("hy4-preview", "alternative voice"),
    ],
}

# ----------------------------------------------------------------------------
# 1. CLASSIFY — one TypeSafe call, four independent questions
# ----------------------------------------------------------------------------

QUESTIONS = {
    "task_type": {
        "type": "choice",
        "instructions": "What kind of work does this request primarily require from an AI assistant?",
        "criteria": {
            "code": "Writing, debugging, refactoring, or explaining software code",
            "analysis": "Reasoning, math, logic, planning, or multi-step problem solving",
            "creative": "Creative writing, brainstorming, naming, storytelling, or open-ended generation",
            "factual": "Looking up or recalling a specific fact, definition, or short piece of information",
            "conversation": "Chitchat, greetings, or simple conversational exchange",
        },
    },
    "complexity": {
        "type": "score",
        "instructions": "How much capability does this request demand to answer well?",
        "criteria": [
            "Simple: a short answer, greeting, or one-step lookup",
            "Moderate: some reasoning, domain knowledge, or a few steps",
            "Demanding: multi-step reasoning, large context, or precision-critical work",
        ],
    },
    "high_stakes": {
        "type": "noul",
        "instructions": (
            "Would a wrong or sloppy answer to this request have costly or serious "
            "consequences (money, health, legal, safety, irreversible actions)?"
        ),
        "criteria": {
            "true": "Consequences are serious or hard to reverse",
            "false": "Low stakes or easily corrected",
        },
    },
    "speed_priority": {
        "type": "noul",
        "instructions": (
            "Does this request feel time-critical, where a fast rough answer "
            "is much better than a slow precise one?"
        ),
        "criteria": {
            "true": "Urgent, latency matters most",
            "false": "Quality matters more than speed",
        },
    },
}


class RouterError(RuntimeError):
    pass


def classify(prompt: str, system: str | None = None) -> tuple[dict, dict, float]:
    """Ask jev the four routing questions. Returns (answers, usage, latency_s)."""
    state = prompt if not system else {"system": system, "request": prompt}
    payload = {"state": state, "model": JEV_MODEL, "questions": QUESTIONS}
    headers = {
        "Authorization": f"Bearer {ENV.get('TYPESAFE_API_KEY', '')}",
        "Content-Type": "application/json",
    }
    t0 = time.time()
    try:
        r = httpx.post(JEV_URL, json=payload, headers=headers, timeout=JEV_TIMEOUT)
    except httpx.HTTPError as e:
        raise RouterError(f"jev classification failed: {e}") from e
    dt = time.time() - t0
    if r.status_code != 200:
        raise RouterError(f"jev classification HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    return data["answers"], data.get("usage", {}), dt


# ----------------------------------------------------------------------------
# 2. DECIDE — deterministic policy. Code owns the workflow.
# ----------------------------------------------------------------------------

CONF_GATE = 0.40      # task_type confidence below this → safe default
STAKES_GATE = 0.65    # high_stakes noul at/above this → strong tier
SPEED_GATE = 0.60     # speed_priority noul at/above this → fast tier
COMPLEX_HARD = 1.20   # complexity score at/above this counts as hard


def chain_for_tier(tier: str) -> list[Route]:
    """Tier chain + balanced fallbacks appended (dedup, order preserved)."""
    chain = list(TIERS[tier])
    if tier != "balanced":
        seen = {r.id for r in chain}
        chain += [r for r in TIERS["balanced"] if r.id not in seen]
    return chain


def decide(answers: dict) -> dict:
    task = answers["task_type"]
    comp = answers["complexity"]
    stakes = answers["high_stakes"]
    speed = answers["speed_priority"]

    task_choice = task.get("choice", "conversation")
    task_conf = float(task.get("confidence", 0.0))
    comp_score = float(comp.get("score", 1.0))
    comp_conf = float(comp.get("confidence", 0.0))
    stakes_p = float(stakes.get("noul", 0.0))
    speed_p = float(speed.get("noul", 0.0))

    reasons: list[str] = []
    gates: dict[str, str] = {}

    if comp_conf < 0.30:
        gates["complexity"] = f"low confidence ({comp_conf:.2f}) — treated as moderate"
        comp_score = max(1.0, min(comp_score, 1.49))

    if task_conf < CONF_GATE:
        tier = "balanced"
        gates["task_type"] = f"low confidence ({task_conf:.2f} < {CONF_GATE}) — safe default"
        reasons.append(f"task unclear → balanced default")
    elif stakes_p >= STAKES_GATE:
        tier = "strong"
        reasons.append(f"high stakes p={stakes_p:.2f} → strongest model")
    elif speed_p >= SPEED_GATE and comp_score < COMPLEX_HARD:
        tier = "fast"
        reasons.append(f"speed priority p={speed_p:.2f} → fastest model")
    elif task_choice == "code":
        tier = "code"
        reasons.append(f"code task (complexity {comp_score:.2f}) → code specialist")
    elif task_choice == "analysis":
        tier = "strong" if comp_score >= COMPLEX_HARD else "balanced"
        reasons.append(
            f"analysis, complexity {comp_score:.2f} → "
            + ("strong tier" if tier == "strong" else "balanced tier")
        )
    elif task_choice == "creative":
        tier = "creative"
        reasons.append("creative task → creative tier")
    elif task_choice in ("factual", "conversation"):
        tier = "fast"
        reasons.append(f"{task_choice} → fast tier")
    else:
        tier = "balanced"
        reasons.append("unmapped task → balanced")

    chain = chain_for_tier(tier)

    chosen = chain[0]
    return {
        "tier": tier,
        "chosen": chosen.id,
        "chain": [r.id for r in chain],
        "reasons": reasons,
        "gates": gates,
        "_routes": chain,  # internal, stripped before serialization
    }


# ----------------------------------------------------------------------------
# 3. EXECUTE — call the chosen model; fall through the chain on failure
# ----------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.S)


def _clean(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _call_opencode(route: Route, prompt: str, system: str | None,
                   max_tokens: int) -> tuple[str, dict, str]:
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    headers = {
        "Authorization": f"Bearer {ENV.get('OPENCODE_GO_API_KEY', '')}",
        "User-Agent": OPENCODE_UA,
        "x-opencode-session": OPENCODE_SESSION,
        "Content-Type": "application/json",
    }
    r = httpx.post(
        f"{OPENCODE_URL}/chat/completions",
        json={"model": route.model, "messages": messages, "max_tokens": max_tokens},
        headers=headers,
        timeout=MODEL_TIMEOUT,
    )
    if r.status_code != 200:
        raise RouterError(f"{route.id} HTTP {r.status_code}: {r.text[:180]}")
    d = r.json()
    ch = d.get("choices") or []
    if not ch:
        raise RouterError(f"{route.id}: no choices in response")
    msg = ch[0].get("message") or {}
    text = _clean(msg.get("content") or "")
    if not text and msg.get("reasoning_content"):
        text = _clean(msg["reasoning_content"])
    if not text:
        raise RouterError(f"{route.id}: empty content")
    usage = d.get("usage") or {}
    return text, usage, d.get("model", route.model)


def _call_bailian(route: Route, prompt: str, system: str | None,
                  max_tokens: int) -> tuple[str, dict, str]:
    headers = {
        "x-api-key": ENV.get("TOKEN_PLAN_API_KEY", ""),
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    body = {
        "model": route.model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system
    r = httpx.post(BAILIAN_URL, json=body, headers=headers, timeout=MODEL_TIMEOUT)
    if r.status_code != 200:
        raise RouterError(f"{route.id} HTTP {r.status_code}: {r.text[:180]}")
    d = r.json()
    parts = [
        b.get("text", "") for b in d.get("content", []) if b.get("type") == "text"
    ]
    text = _clean("".join(parts))
    if not text:
        raise RouterError(f"{route.id}: empty content")
    usage = {
        "input_tokens": (d.get("usage") or {}).get("input_tokens"),
        "output_tokens": (d.get("usage") or {}).get("output_tokens"),
    }
    return text, usage, d.get("model", route.model)


def call_model(route: Route, prompt: str, system: str | None = None,
               max_tokens: int = DEFAULT_MAX_TOKENS) -> tuple[str, dict, float, str]:
    """Call one route. Returns (text, usage, latency_s, model_returned)."""
    t0 = time.time()
    if route.provider == "bailian":
        text, usage, model_ret = _call_bailian(route, prompt, system, max_tokens)
    elif route.provider == "opencode-go":
        text, usage, model_ret = _call_opencode(route, prompt, system, max_tokens)
    else:
        raise RouterError(f"unknown provider {route.provider}")
    return text, usage, time.time() - t0, model_ret


def _upstream_headers(provider: str) -> dict:
    if provider == "bailian":
        return {
            "Authorization": f"Bearer {ENV.get('TOKEN_PLAN_API_KEY', '')}",
            "Content-Type": "application/json",
        }
    if provider == "opencode-go":
        return {
            "Authorization": f"Bearer {ENV.get('OPENCODE_GO_API_KEY', '')}",
            "User-Agent": OPENCODE_UA,
            "x-opencode-session": OPENCODE_SESSION,
            "Content-Type": "application/json",
        }
    raise RouterError(f"unknown provider {provider}")


def _upstream_url(provider: str) -> str:
    if provider == "bailian":
        return BAILIAN_COMPAT_URL
    if provider == "opencode-go":
        return f"{OPENCODE_URL}/chat/completions"
    raise RouterError(f"unknown provider {provider}")


def call_model_passthrough(route: Route, body: dict) -> tuple[httpx.Response, httpx.Client]:
    """Forward an OpenAI chat-completions body untouched to the route's
    provider (conversation history, tools, streaming all preserved).

    Returns (open streaming Response, client) — caller must close BOTH."""
    payload = dict(body)
    payload["model"] = route.model
    client = httpx.Client(timeout=MODEL_TIMEOUT)
    try:
        req = client.build_request(
            "POST",
            _upstream_url(route.provider),
            json=payload,
            headers=_upstream_headers(route.provider),
        )
        r = client.send(req, stream=True)
    except httpx.HTTPError as e:
        client.close()
        raise RouterError(f"{route.id}: {e}") from e
    if r.status_code != 200:
        raw = r.read()[:400]
        r.close()
        client.close()
        raise RouterError(f"{route.id} HTTP {r.status_code}: {raw.decode('utf-8', 'replace')}")
    return r, client


def execute(chain: list[Route], prompt: str, system: str | None = None,
            max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    tried: list[dict] = []
    for route in chain[:4]:  # cap hops to avoid hammering the pools
        try:
            text, usage, dt, model_ret = call_model(route, prompt, system, max_tokens)
            return {
                "model": route.id,
                "model_returned": model_ret,
                "latency_s": round(dt, 2),
                "usage": usage,
                "fallbacks_tried": tried,
                "answer": text,
            }
        except RouterError as e:
            tried.append({"model": route.id, "error": str(e)[:160]})
            continue
    raise RouterError(f"all models in chain failed: {json.dumps(tried, indent=1)}")


# ----------------------------------------------------------------------------
# 4. TRACE — full pipeline
# ----------------------------------------------------------------------------


def parse_forced(model: str) -> Route:
    """'provider/model' → Route; bare name searches the pool."""
    if "/" in model:
        prov, _, name = model.partition("/")
        return Route(name, prov)
    for routes in TIERS.values():
        for r in routes:
            if r.model == model:
                return r
    raise RouterError(f"unknown model '{model}' — use provider/model form")


def run(prompt: str, route_only: bool = False, force: str | None = None,
        system: str | None = None, max_tokens: int = DEFAULT_MAX_TOKENS) -> dict:
    trace: dict = {"prompt_preview": prompt[:140]}

    if force:
        route = parse_forced(force)
        trace["decision"] = {
            "tier": "forced",
            "chosen": route.id,
            "chain": [route.id],
            "reasons": ["model forced on the command line"],
            "gates": {},
        }
        chain = [route]
        jev = None
    else:
        answers, jev_usage, jev_dt = classify(prompt, system)
        trace["jev"] = {
            "answers": answers,
            "latency_s": round(jev_dt, 2),
            "usage": jev_usage,
        }
        decision = decide(answers)
        chain = decision.pop("_routes")
        trace["decision"] = decision

    if not route_only:
        trace["execution"] = execute(chain, prompt, system, max_tokens)
    return trace


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _fmt(trace: dict) -> str:
    out: list[str] = []
    jev = trace.get("jev")
    if jev:
        a = jev["answers"]
        t, c, s, f = a["task_type"], a["complexity"], a["high_stakes"], a["speed_priority"]
        out.append(
            f"🧠 jev: {t['choice']} (conf {t['confidence']:.2f}) · "
            f"complexity {c['score']:.2f} (conf {c['confidence']:.2f}) · "
            f"stakes {s['noul']:.2f} · speed {f['noul']:.2f}  [{jev['latency_s']}s]"
        )
    d = trace["decision"]
    out.append(f"🧭 → {d['chosen']}  [{d['tier']} tier]")
    for r in d["reasons"]:
        out.append(f"   · {r}")
    for g in d.get("gates", {}).values():
        out.append(f"   ⚠ {g}")
    ex = trace.get("execution")
    if ex:
        fb = ex.get("fallbacks_tried") or []
        line = f"⚡ {ex['model']} answered in {ex['latency_s']}s"
        if fb:
            line += f" (after {len(fb)} fallback(s): " + ", ".join(
                f["model"] for f in fb) + ")"
        out.append(line)
        out.append("")
        out.append(ex["answer"])
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="TypeSafe-powered model router")
    ap.add_argument("prompt", nargs="?", help="prompt (or pipe via stdin)")
    ap.add_argument("--route-only", action="store_true", help="decision only, no model call")
    ap.add_argument("--json", action="store_true", help="print raw JSON trace")
    ap.add_argument("--model", help="force a model (provider/model)")
    ap.add_argument("--system", help="system prompt")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = ap.parse_args()

    prompt = args.prompt
    if not prompt:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        if not prompt:
            ap.error("no prompt given (argument or stdin)")

    try:
        trace = run(prompt, route_only=args.route_only, force=args.model,
                    system=args.system, max_tokens=args.max_tokens)
    except RouterError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(trace, indent=2, ensure_ascii=False))
    else:
        print(_fmt(trace))
    return 0


if __name__ == "__main__":
    sys.exit(main())