# ARTalk Web Renderer

Standalone browser renderer for ARTalk-generated animation artifacts.

This repository owns the React/Vite frontend and a thin FastAPI adapter server.
The server expects ARTalk and GAGAvator to be available as external Python
dependencies through `PYTHONPATH` or an installed package.

## Frontend

```bash
pnpm install
pnpm dev
```

The Vite dev server proxies `/api` to `http://127.0.0.1:8961` by default. Set
`ARTALK_API_TARGET` to point at another backend.

## Backend

Example local run against sibling worktrees:

```bash
export PYTHONPATH=/path/to/ARTalk:/path/to/GAGAvatar
export ARTALK_ASSET_DIR=/path/to/ARTalk/assets
export GAGAVATAR_MODEL_PATH=/path/to/GAGAvatar/assets/GAGAvatar.pt
export GAGAVATAR_TRACKED_PATH=/path/to/GAGAvatar/assets/tracked.pt
uvicorn server.app:app --host 0.0.0.0 --port 8961
```

`mesh` mode only requires ARTalk. `gagavatar` and `browser-gaussian` modes also
require GAGAvator, CUDA, and the 32-channel Gaussian rasterizer.

The ONNX upsampler model is not checked in. Generate it from GAGAvator and place
it at:

```text
public/models/gagavatar_upsampler.onnx
```
