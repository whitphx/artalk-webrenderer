# ARTalk Web Renderer

Standalone browser renderer for ARTalk-generated animation artifacts.

This repository owns the React/Vite frontend and a thin FastAPI adapter server.
The server expects ARTalk and GAGAvator to be available as external Python
dependencies.

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
pip install -e /path/to/ARTalk --no-deps
pip install -e /path/to/GAGAvatar --no-deps

export ARTALK_ASSET_DIR=/path/to/ARTalk/assets
export GAGAVATAR_MODEL_PATH=/path/to/GAGAvatar/assets/GAGAvatar.pt
export GAGAVATAR_TRACKED_PATH=/path/to/GAGAvatar/assets/tracked.pt
scripts/run_server.sh
```

The editable installs provide the ARTalk import package `app` and the
GAGAvator import package `core`. `--no-deps` is intentional for now; install
the heavy PyTorch/CUDA dependencies through the upstream environment files.

`mesh` mode only requires ARTalk. `gagavatar` and `browser-gaussian` modes also
require GAGAvator, CUDA, and the 32-channel Gaussian rasterizer.

The ONNX upsampler model is not checked in. Generate it from GAGAvator and place
it at:

```text
public/models/gagavatar_upsampler.onnx
```
