#!/usr/bin/env python3
"""
model-router server v0.2 — OpenAI-compatible passthrough proxy.

In front of every request: ONE TypeSafe jev classification (~0.4–0.6 s) routes
the conversation to a tier pool; the chosen model then receives the request
VERBATIM — full message history, tools, streaming — and its response is relayed
back untouched. Usable as a real agent backend (Hermes custom provider,
OpenAI-compatible clients, curl).

Endpoints
---------
  GET  /health                 pool + key status
  GET  /v1/models              callable ids: "auto" + every pooled model
  POST /route                  {"prompt": "..."} → decision trace (no model call)
  POST /v1/chat/completions    model="auto"/"router" → jev routing;
                               "provider/model" (or pooled bare name) forces that
                               model; unknown bare names → 400. Stream + tools
                               pass through verbatim. Non-stream responses carry
                               an extra "x_router" trace field.
  GET  /decisions?limit=N      recent routing decisions (jsonl tail)

Routing traces are also emitted as response headers:
  X-Router-Model, X-Router-Tier, X-Router-Jev, X-Router-Fallbacks

Decisions are appended to logs/decisions.jsonl (one JSON object per request).

Run
---
  uvicorn server:app --host 127.0.0.1 --port 8790     (or ./serve.sh)
  Production: systemd --user unit model-router.service
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

import router as R

app = FastAPI(title="model-router", version="0.2.0")

LOG_PATH = Path(__file__).resolve().parent / "logs" / "decisions.jsonl"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- route cache
# Tool loops re-send the same conversation many times (one model call per tool
# hop). Classifying the same (system, user) pair again would burn a jev call
# (~625 in / 105 out tokens) per hop — cache the decision instead.
_CACHE_TTL = 900.0     # 15 min
_CACHE_MAX = 512
_ROUTE_CACHE: "OrderedDict[str, tuple[float, dict, dict]]" = OrderedDict()


def _cache_key(system: str | None, prompt: str) -> str:
    h = hashlib.sha1()
    h.update((system or "").encode())
    h.update(b"\x00")
    h.update(prompt.encode())
    return h.hexdigest()


def _cache_get(key: str):
    hit = _ROUTE_CACHE.get(key)
    if not hit:
        return None
    ts, decision, jev = hit
    if time.time() - ts > _CACHE_TTL:
        _ROUTE_CACHE.pop(key, None)
        return None
    _ROUTE_CACHE.move_to_end(key)
    return decision, jev


def _cache_put(key: str, decision: dict, jev: dict | None) -> None:
    _ROUTE_CACHE[key] = (time.time(), decision, jev or {})
    _ROUTE_CACHE.move_to_end(key)
    while len(_ROUTE_CACHE) > _CACHE_MAX:
        _ROUTE_CACHE.popitem(last=False)


def _log(entry: dict) -> None:
    try:
        with LOG_PATH.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------- misc


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": app.version,
        "pools": {tier: [r.id for r in routes] for tier, routes in R.TIERS.items()},
        "jev_model": R.JEV_MODEL,
        "keys_present": {
            k: bool(R.ENV.get(k))
            for k in ("TYPESAFE_API_KEY", "TOKEN_PLAN_API_KEY", "OPENCODE_GO_API_KEY")
        },
        "cache_size": len(_ROUTE_CACHE),
    }


@app.get("/v1/models")
def list_models():
    ids = {"auto"}
    for routes in R.TIERS.values():
        ids.update(r.model for r in routes)
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "owned_by": "model-router"}
            for m in sorted(ids)
        ],
    }


@app.get("/decisions")
def decisions(limit: int = 20):
    limit = max(1, min(int(limit or 20), 200))
    try:
        lines = LOG_PATH.read_text().splitlines()[-limit:]
    except FileNotFoundError:
        lines = []
    out = []
    total_prompt = total_completion = 0
    per_model: dict[str, dict] = {}
    for ln in lines:
        try:
            d = json.loads(ln)
        except Exception:
            continue
        out.append(d)
        u = d.get("usage") or {}
        pt = u.get("prompt_tokens") or 0
        ct = u.get("completion_tokens") or 0
        total_prompt += pt
        total_completion += ct
        m = d.get("chosen") or d.get("model") or "?"
        slot = per_model.setdefault(m, {"calls": 0, "prompt": 0, "completion": 0})
        slot["calls"] += 1
        slot["prompt"] += pt
        slot["completion"] += ct
    return {
        "count": len(out),
        "tokens": {"prompt": total_prompt, "completion": total_completion,
                   "total": total_prompt + total_completion},
        "per_model": per_model,
        "decisions": out,
    }


@app.post("/route")
async def route_only(body: dict):
    prompt = body.get("prompt") or ""
    if not prompt:
        return JSONResponse({"error": "prompt required"}, status_code=400)
    try:
        trace = R.run(prompt, route_only=True, system=body.get("system"))
    except R.RouterError as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    return trace


# ------------------------------------------------------- chat completions


JEV_SYSTEM_CAP = 1500  # chars of system prompt sent to jev (classification
                       # signal lives in the user request; full Hermes system
                       # prompts are 20-40k chars and would burn jev tokens)


def _extract_messages(body: dict) -> tuple[str, str | None]:
    """OpenAI messages → (last user prompt, concatenated system).

    Scans backwards for the last user message so tool-loop turns (last message
    = role "tool") still classify on the original ask. The system string is
    capped at JEV_SYSTEM_CAP chars before classification/caching.
    """
    system_parts: list[str] = []
    last_user: str | None = None
    for m in body.get("messages") or []:
        role, content = m.get("role"), m.get("content")
        if isinstance(content, list):  # multimodal-style parts
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        content = content or ""
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            last_user = content
    system = "\n\n".join(system_parts) or None
    if system and len(system) > JEV_SYSTEM_CAP:
        system = system[:JEV_SYSTEM_CAP]
    return last_user or "", system


def _resolve_chain(body: dict) -> tuple[list[R.Route], dict, dict | None]:
    """Decide the model chain for a request.

    Returns (chain, decision, jev_info). Raises R.RouterError on a bad
    explicit model name.
    """
    model = (body.get("model") or "auto").strip()
    if model in ("auto", "router", "default", "router/auto", ""):
        prompt, system = _extract_messages(body)
        if not prompt:
            raise R.RouterError("no user message found")
        key = _cache_key(system, prompt)
        hit = _cache_get(key)
        if hit:
            decision, jev = hit
            decision = {**decision, "cached": True}
            chain = [R.parse_forced(rid) for rid in decision["chain"]]
            return chain, decision, jev or None
        try:
            answers, jev_usage, jev_dt = R.classify(prompt, system)
        except R.RouterError as e:
            # Classifier down → deterministic balanced chain, no hard failure.
            decision = {
                "tier": "balanced",
                "chosen": R.TIERS["balanced"][0].id,
                "chain": [r.id for r in R.chain_for_tier("balanced")],
                "reasons": [f"jev unavailable ({str(e)[:80]}) → balanced default"],
                "gates": {},
                "cached": False,
            }
            chain = R.chain_for_tier("balanced")
            _log({"ts": time.time(), "kind": "classify_failed", "error": str(e)[:200]})
            return chain, decision, None
        jev = {"answers": answers, "latency_s": round(jev_dt, 2), "usage": jev_usage}
        decision = R.decide(answers)
        chain = decision.pop("_routes")
        decision["cached"] = False
        _cache_put(key, decision, jev)
        return chain, decision, jev

    # Explicit model — force it.
    route = R.parse_forced(model)
    decision = {
        "tier": "forced",
        "chosen": route.id,
        "chain": [route.id],
        "reasons": ["model forced by the caller"],
        "gates": {},
        "cached": False,
    }
    return [route], decision, None


def _trace_headers(decision: dict, jev: dict | None) -> dict:
    h = {
        "X-Router-Model": decision.get("chosen", ""),
        "X-Router-Tier": decision.get("tier", ""),
        "X-Router-Cached": "1" if decision.get("cached") else "0",
    }
    if jev:
        a = jev.get("answers") or {}
        tt = (a.get("task_type") or {}).get("choice", "")
        h["X-Router-Jev"] = (
            f"{tt};stakes={(a.get('high_stakes') or {}).get('noul', '')};"
            f"speed={(a.get('speed_priority') or {}).get('noul', '')}"
        )
    return h


def _relay(upstream: httpx.Response, client: httpx.Client, req_id: str,
           decision: dict, t0: float, tried: list[dict]):
    """Sync byte-level SSE relay (Starlette runs it in a threadpool).

    Passes bytes through verbatim; opportunistically captures the upstream
    usage payload (include_usage chunk) for the decisions log."""
    usage = None
    buf = b""
    try:
        for chunk in upstream.iter_bytes():
            yield chunk
            if b'"usage"' in chunk:
                try:
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if line.startswith(b"data: ") and b'"usage"' in line:
                            evt = json.loads(line[6:])
                            if isinstance(evt.get("usage"), dict):
                                usage = evt["usage"]
                except Exception:
                    buf = b""
                if len(buf) > 65536:
                    buf = b""
    except Exception as e:  # noqa: BLE001 — upstream died mid-stream
        _log({
            "ts": time.time(), "req_id": req_id, "kind": "stream_error",
            "model": decision.get("chosen"), "error": str(e)[:200],
        })
    finally:
        upstream.close()
        client.close()
        _log({
            "ts": time.time(), "req_id": req_id, "kind": "stream_done",
            "chosen": decision.get("chosen"), "tier": decision.get("tier"),
            "cached": decision.get("cached"), "fallbacks": tried,
            "usage": usage, "elapsed_s": round(time.time() - t0, 2),
        })


def _open_upstream(chain: list[R.Route], body: dict):
    """Try each route until one accepts the request. Returns (open stream,
    client, route, failed hops)."""
    tried: list[dict] = []
    for route in chain[:4]:
        try:
            upstream, client = R.call_model_passthrough(route, body)
            return upstream, client, route, tried
        except R.RouterError as e:
            tried.append({"model": route.id, "error": str(e)[:200]})
            continue
    raise R.RouterError(f"all models failed: {json.dumps(tried)[:400]}")


@app.post("/v1/chat/completions")
def chat_completions(body: dict):
    req_id = uuid.uuid4().hex[:12]
    t0 = time.time()

    try:
        chain, decision, jev = _resolve_chain(body)
    except R.RouterError as e:
        return JSONResponse(
            {"error": {"message": str(e), "type": "router_error"}}, status_code=400
        )

    try:
        upstream, client, route, tried = _open_upstream(chain, body)
    except R.RouterError as e:
        _log({
            "ts": time.time(), "req_id": req_id, "kind": "failed",
            "decision": decision, "error": str(e)[:400],
        })
        return JSONResponse(
            {"error": {"message": str(e), "type": "router_error"}}, status_code=502
        )

    headers = _trace_headers(decision, jev)
    headers["X-Router-Upstream"] = route.id
    stream = bool(body.get("stream"))

    if stream:
        headers["X-Accel-Buffering"] = "no"
        return StreamingResponse(
            _relay(upstream, client, req_id, {**decision, "chosen": route.id}, t0, tried),
            media_type="text/event-stream",
            headers=headers,
        )

    # Non-stream: buffer the upstream answer, add the router trace, relay.
    try:
        raw = upstream.read()
        upstream.close()
        client.close()
        parsed = json.loads(raw)
    except Exception as e:  # noqa: BLE001
        _log({
            "ts": time.time(), "req_id": req_id, "kind": "buffer_error",
            "chosen": route.id, "error": str(e)[:200],
        })
        return JSONResponse(
            {"error": {"message": f"upstream parse error: {e}", "type": "router_error"}},
            status_code=502,
        )

    usage = parsed.get("usage") or {}
    _log({
        "ts": time.time(), "req_id": req_id, "kind": "done",
        "chosen": route.id, "tier": decision.get("tier"),
        "cached": decision.get("cached"), "fallbacks": tried,
        "upstream_model": parsed.get("model"),
        "usage": usage, "elapsed_s": round(time.time() - t0, 2),
    })

    parsed["x_router"] = {
        "decision": {k: v for k, v in decision.items() if k != "cached" or v},
        "jev": jev,
        "upstream": route.id,
        "fallbacks": tried,
    }
    return JSONResponse(parsed, headers=headers)