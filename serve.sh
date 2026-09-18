#!/usr/bin/env bash
# Start the model-router server (localhost only).
cd "$(dirname "$0")"
exec python3 -m uvicorn server:app --host 127.0.0.1 --port 8790 "$@"