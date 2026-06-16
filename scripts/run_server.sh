#!/usr/bin/env bash
set -euo pipefail

exec uvicorn server.app:app --host "${ARTALK_WEB_HOST:-0.0.0.0}" --port "${ARTALK_WEB_PORT:-8961}"
