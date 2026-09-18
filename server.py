#!/usr/bin/env python3
"""
model-router server — OpenAI-compatible endpoint in front of the router.

Endpoints
---------
  GET  /health                  pool status + jev reachability
  POST /route                   {"prompt": "..."} → decision trace (no model call)
  POST /v1/chat/completions     OpenAI-compatible; model="auto" (or "router")
                                triggers jev routing, "provider/model" forces,
                                anything else routes and notes the override.
                                Response carries an extra "x_router" trace field.

Run
---
  uvicorn server:app --host 127.0.0.1 --port 8790
  (or: ./serve.sh)
"""
from __future__ import annotations

import time
import uuid

from fastapi import FastAPI
from fastapi.responses import JSONResponse

import router as R

app = FastAPI(title="model-router", version="0.1.0")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pools": {tier: [r.id for r in routes] for tier, routes in R.TIERS.items()},
        "jev_model": R.JEV_MODEL,
        "keys_present": {
            k: bool(R.ENV.get(k)) for k in ("TYPESAFE_API_KEY", "TOKEN_PLAN_API_KEY", "OPENCODE_GO_API_KEY")
        },
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


def _extract_messages(body: dict) -> tuple[str, str | None]:
    """OpenAI messages → (last user prompt, concatenated system)."""
    system_parts, user_parts = [], []
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
            user_parts.append(content)
    return (user_parts[-1] if user_parts else ""), ("\n\n".join(system_parts) or None)


@app.post("/v1/chat/completions")
async def chat_completions(body: dict):
    prompt, system = _extract_messages(body)
    if not prompt:
        return JSONResponse(
            {"error": {"message": "no user message found", "type": "invalid_request_error"}},
            status_code=400,
        )

    model = body.get("model") or "auto"
    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or R.DEFAULT_MAX_TOKENS
    forced = None
    if model not in ("auto", "router", "default"):
        if "/" in model:
            forced = model  # explicit provider/model passthrough
        else:
            forced = model  # bare name → parse_forced resolves against the pool

    try:
        trace = R.run(prompt, route_only=False, force=forced,
                      system=system, max_tokens=int(max_tokens))
    except R.RouterError as e:
        return JSONResponse(
            {"error": {"message": str(e), "type": "router_error"}}, status_code=502
        )

    ex = trace["execution"]
    created = int(time.time())
    resp = {
        "id": f"chatcmpl-router-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": created,
        "model": ex["model"],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ex["answer"]},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": (ex.get("usage") or {}).get("input_tokens")
            or (ex.get("usage") or {}).get("prompt_tokens"),
            "completion_tokens": (ex.get("usage") or {}).get("output_tokens")
            or (ex.get("usage") or {}).get("completion_tokens"),
            "total_tokens": None,
        },
        "x_router": {
            "decision": trace["decision"],
            "jev": trace.get("jev"),
            "fallbacks": ex.get("fallbacks_tried", []),
            "model_latency_s": ex["latency_s"],
        },
    }
    return resp