# model-router

A TypeSafe-powered model router: **Jev** (System One) classifies each incoming
request, deterministic code picks the best available model, and the router
executes it — returning the answer plus a full decision trace.

```
prompt ──► jev classify ──► policy (code) ──► chain of models ──► answer + trace
           ~0.4–1.5s        deterministic      first healthy wins
```

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
3. **EXECUTE** — first model in the tier answers; on failure the chain falls
   through (max 4 hops). Every hop is recorded in the trace.
4. **TRACE** — jev answers + confidences + chosen tier + reasons + fallbacks.

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
- **bailian** = Aliyun Token Plan, Anthropic Messages API format.

Deliberately excluded (failed probes): grok-4.5/4.6, gpt-5.6-luna (500),
deepseek-direct (no balance), kimi-coding (quota), openai (billing),
minimax-direct (plan limit). Re-check before adding.

## Usage

```bash
# CLI — route + execute
./route "Refactor this function to be async: def fetch(url): ..."

# Decision only (no model call — ~0.5s)
./route --route-only "I need to decide whether to sign this contract"

# Raw JSON trace
./route --json "Write a haiku about Montreal"

# Force a model
./route --model bailian/qwen3.7-max "Summarize this"

# stdin works too
echo "explain quicksort" | ./route
```

### Server (OpenAI-compatible)

```bash
./serve.sh                      # uvicorn on 127.0.0.1:8790

curl localhost:8790/health
curl -X POST localhost:8790/route -H 'Content-Type: application/json' \
     -d '{"prompt": "debug this stack trace"}'
curl -X POST localhost:8790/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model": "auto", "messages": [{"role": "user", "content": "hi!"}]}'
```

The chat completions response is OpenAI-shaped (drop-in for most clients) and
carries an extra `x_router` field with the full decision trace.

## Files

- `router.py` — pipeline, policy, pools, CLI
- `server.py` — FastAPI: /health, /route, /v1/chat/completions
- `serve.sh`, `route` — wrappers

## Config / secrets

Keys are read at runtime from `~/.hermes/.env` (`TYPESAFE_API_KEY`,
`TOKEN_PLAN_API_KEY`, `OPENCODE_GO_API_KEY`) — never stored in this repo.
Server binds `127.0.0.1` only.

## Tuning

Thresholds live at the top of the policy section in `router.py`
(`CONF_GATE`, `STAKES_GATE`, `SPEED_GATE`, `COMPLEX_HARD`). Tier membership is
the `TIERS` dict. Validate changes against representative cases — thresholds
are policy, not truth.