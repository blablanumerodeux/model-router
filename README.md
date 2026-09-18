# model-router

A TypeSafe-powered model router: **Jev** (System One) classifies each incoming
request, deterministic code picks the best available model, and the router
proxies the request to it — streaming, tools, and full conversation history
passed through verbatim.

```
request ──► jev classify ──► policy (code) ──► tier chain ──► upstream model
            ~0.5s (1st)       deterministic      first healthy    verbatim relay
            cached (loop)                                       (SSE + tools)
```

Dual interface:

- **OpenAI-compatible proxy** (`/v1/chat/completions`) — drop-in backend for
  Hermes (registered as provider `router`), any OpenAI client, or curl.
- **CLI** (`./route`) — route + execute a single prompt with a decision trace.

## Why

Instead of sending every request to one fixed model (expensive + slow) or
hand-coded if/else routing (brittle), the semantic call is made by TypeSafe's
Jev — which returns *typed*, *calibrated* answers (choice + confidence, score +
confidence, noul probability) that ordinary code can act on safely.

## Pipeline

1. **CLASSIFY** — one jev call, four independent questions asked together:
   - `task_type` (choice): code / analysis / creative / factual / conversation
   - `complexity` (score, 3 levels): simple → demanding
   - `high_stakes` (noul): would a wrong answer be costly?
   - `speed_priority` (noul): fast-and-rough beats slow-and-precise?
2. **DECIDE** — explicit policy in `router.py` (thresholds visible, tunable):
   - task confidence < 0.40 → safe `balanced` default
   - high_stakes ≥ 0.65 → `strong` tier (stakes outrank everything)
   - speed ≥ 0.60 & not demanding → `fast` tier
   - code → `code` tier · analysis → `balanced`/`strong` · creative → `creative`
   - factual/conversation → `fast`
3. **EXECUTE** — first model in the tier accepts the request; on failure the
   chain falls through (max 4 hops). Every hop recorded in the trace.
4. **TRACE** — jev answers + confidences + tier + reasons + fallbacks +
   token usage. Response headers carry `X-Router-Model/-Tier/-Jev/-Cached`;
   non-stream bodies carry `x_router`; `logs/decisions.jsonl` keeps every hop.

### Classification cache

The decision is cached per (system, user) pair for 15 min — tool loops re-send
the same ask several times per turn, and each uncached hop would cost one jev
call. The system prompt sent to jev is capped at 1,500 chars (Hermes system
prompts are 20–40k chars; the signal lives in the user request).

## Pools (verified live 2026-09-18)

| Tier | Primary | Chain |
|---|---|---|
| fast | `bailian/deepseek-v4.1-flash` | glm-5.3-flash → qwen3.8-flash → qwen3.6-flash |
| balanced | `opencode-go/glm-5.2` | deepseek-v4-pro → qwen3.7-plus → deepseek-v4-pro (bl) |
| code | `opencode-go/kimi-k2.7-code` | kimi-k3 → qwen3.7-max |
| strong | `opencode-go/kimi-k3` | qwen3.8-max-preview → qwen3.7-max → qwen3.8-max |
| creative | `bailian/qwen3.7-max` | glm-5.3 → hy4-preview |

Provider notes:
- **opencode-go** (`opencode.ai/zen/go/v1`) needs a browser User-Agent and an
  `x-opencode-session` header — handled automatically.
- **bailian** = Aliyun Token Plan. The proxy uses the OpenAI-compatible
  sibling `…/compatible-mode/v1/chat/completions` (Bearer auth; live-verified
  stream + tool calls). `router.py`'s direct single-shot calls still use the
  Anthropic Messages endpoint (`…/apps/anthropic/v1/messages`).

Deliberately excluded (failed probes): grok-4.5/4.6, gpt-5.6-luna (500),
deepseek-direct (no balance), kimi-coding (quota), openai (billing),
minimax-direct (plan limit). Re-check before adding.

## Usage

### As a Hermes provider (model picker)

Registered in `~/.hermes/config.yaml` as provider `router`
(`http://127.0.0.1:8790/v1`, default model `auto`). In Telegram: `/model` →
**Router (TypeSafe jev)** → `auto`. Session-scoped by default — perfect for
testing one conversation; `--global` to make it the default everywhere.

`auto` = jev routing. Any pooled model name = forced passthrough.

### CLI — route + execute

```bash
./route "Refactor this function to be async: def fetch(url): ..."
./route --route-only "I need to decide whether to sign this contract"
./route --json "Write a haiku about Montreal"
./route --model bailian/qwen3.7-max "Summarize this"
echo "explain quicksort" | ./route
```

### Server (OpenAI-compatible)

```bash
./serve.sh                      # uvicorn on 127.0.0.1:8790
# durable: systemctl --user enable --now model-router   (unit already installed)

curl localhost:8790/health
curl localhost:8790/v1/models
curl -X POST localhost:8790/route -d '{"prompt": "debug this stack trace"}'
curl -X POST localhost:8790/v1/chat/completions \
     -d '{"model": "auto", "messages": [{"role": "user", "content": "hi!"}]}'
curl "localhost:8790/decisions?limit=20"   # token totals + per-model split
```

The chat completions response is OpenAI-shaped (drop-in for most clients) and
carries an extra `x_router` field with the full decision trace. Streamed
responses pass through byte-for-byte as SSE.

## Files

- `router.py` — pipeline, policy, pools, providers, CLI
- `server.py` — FastAPI: /health, /v1/models, /route, /v1/chat/completions, /decisions
- `serve.sh`, `route` — wrappers · `logs/decisions.jsonl` — routing log

## Config / secrets

Keys are read at runtime from `~/.hermes/.env` (`TYPESAFE_API_KEY`,
`TOKEN_PLAN_API_KEY`, `OPENCODE_GO_API_KEY`) — never stored in this repo.
Server binds `127.0.0.1` only.

## Tuning

Thresholds live at the top of the policy section in `router.py`
(`CONF_GATE`, `STAKES_GATE`, `SPEED_GATE`, `COMPLEX_HARD`). Tier membership is
the `TIERS` dict. Validate changes against the routing battery — thresholds
are policy, not truth. Re-probe the pool before trusting it:
`python3 ~/.hermes/skills/platform/llm-model-routing/scripts/probe_pool.py`.