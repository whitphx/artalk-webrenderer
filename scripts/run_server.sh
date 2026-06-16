#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import importlib.util

missing = [
    name
    for name in ("app.web_api", "core.web_api")
    if importlib.util.find_spec(name) is None
]
if missing:
    raise SystemExit(
        "Missing renderer backend dependencies: {}. "
        "Install local checkouts with `pip install -e /path/to/ARTalk --no-deps` "
        "and `pip install -e /path/to/GAGAvatar --no-deps`.".format(", ".join(missing))
    )
PY

exec uvicorn server.app:app --host "${ARTALK_WEB_HOST:-0.0.0.0}" --port "${ARTALK_WEB_PORT:-8961}"
