#!/usr/bin/env python

from __future__ import annotations

import json
import os
import subprocess
import threading
import traceback
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from gtts import gTTS

from app.runtime import ARTalkRuntime, ARTalkRuntimeConfig, available_styles
from server.artifacts import export_browser_gaussian_artifacts, write_animation_artifacts


MESH_RENDER_MODE = "mesh"
BROWSER_GAUSSIAN_RENDER_MODE = "browser-gaussian"
GAGAVATAR_RENDER_MODE = "gagavatar"
GTTS_LANG = {
    "English": "en",
    "中文": "zh",
    "日本語": "ja",
    "Deutsch": "de",
    "Français": "fr",
    "Español": "es",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
JOB_ROOT = Path(os.environ.get("ARTALK_WEB_JOB_ROOT", REPO_ROOT / "render_results" / "web_jobs")).resolve()
FRONTEND_DIST = REPO_ROOT / "dist"

app = FastAPI(title="ARTalk Standalone Web Renderer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"https?://(localhost|127\\.0\\.0\\.1|192\\.168\\.\\d+\\.\\d+):5173",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_artalk_engines = {}
_artalk_engines_lock = threading.Lock()
_gagavatar_runtime = None
_gagavatar_runtime_lock = threading.Lock()


@app.middleware("http")
async def add_cross_origin_isolation_headers(_request: Request, call_next):
    response = await call_next(_request)
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    return response


def get_artalk_engine(device: str):
    asset_dir = Path(os.environ.get("ARTALK_ASSET_DIR", "assets"))
    key = (str(asset_dir.resolve()), device)
    with _artalk_engines_lock:
        if key not in _artalk_engines:
            _artalk_engines[key] = ARTalkRuntime(
                ARTalkRuntimeConfig(asset_dir=asset_dir, device=device)
            )
        return _artalk_engines[key]


def get_gagavatar_runtime():
    global _gagavatar_runtime
    model_path = os.environ.get("GAGAVATAR_MODEL_PATH")
    tracked_path = os.environ.get("GAGAVATAR_TRACKED_PATH")
    if not model_path:
        raise RuntimeError("GAGAVATAR_MODEL_PATH is required for GAGAvator render modes.")
    with _gagavatar_runtime_lock:
        if _gagavatar_runtime is None:
            from core.runtime import GAGAvatarRuntime, GAGAvatarRuntimeConfig

            _gagavatar_runtime = GAGAvatarRuntime(
                GAGAvatarRuntimeConfig(
                    model_path=model_path,
                    tracked_path=tracked_path,
                    flame_model_path=os.environ.get(
                        "GAGAVATAR_FLAME_MODEL_PATH",
                        str(Path(os.environ.get("ARTALK_ASSET_DIR", "assets")) / "FLAME_with_eye.pt"),
                    ),
                    device=os.environ.get("GAGAVATAR_DEVICE", "auto"),
                )
            )
        return _gagavatar_runtime


def available_avatars():
    avatars = [{"id": "mesh", "label": "Neutral mesh", "source": "mesh", "previewUrl": None}]
    try:
        tracked_path = os.environ.get("GAGAVATAR_TRACKED_PATH")
        if not tracked_path:
            return avatars
        import torch

        tracked = torch.load(tracked_path, map_location="cpu", weights_only=False)
    except Exception:
        return avatars
    avatar_ids = ["avatar"] if "avatar" in tracked and len(tracked) == 1 else sorted(tracked)
    for avatar_id in avatar_ids:
        avatars.append(
            {
                "id": f"gagavatar:{avatar_id}",
                "label": f"GAGAvator {Path(avatar_id).stem}",
                "source": "gagavatar",
                "previewUrl": None,
            }
        )
    return avatars


def job_dir(job_id: str) -> Path:
    path = JOB_ROOT / job_id
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return path


def write_state(path: Path, state: dict):
    path.mkdir(parents=True, exist_ok=True)
    tmp_path = path / f".state.{uuid.uuid4().hex}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f)
    tmp_path.replace(path / "state.json")


def read_state(path: Path) -> dict:
    with open(path / "state.json") as f:
        return json.load(f)


def start_thread(target, **kwargs):
    thread = threading.Thread(target=target, kwargs=kwargs, daemon=True)
    thread.start()
    return thread


def write_text_audio(text: str, language: str, output_dir: Path) -> Path:
    mp3_path = output_dir / "input.mp3"
    wav_path = output_dir / "input.wav"
    gTTS(text=text, lang=GTTS_LANG[language]).save(str(mp3_path))
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(mp3_path), "-ac", "1", "-ar", "16000", str(wav_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail="Text input requires ffmpeg.") from exc
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=400, detail=f"Failed to convert generated speech: {exc.stderr}") from exc
    return wav_path


def split_avatar_id(avatar_id: str) -> tuple[str, str | None]:
    if avatar_id in ("", "mesh", None):
        return "mesh", None
    if ":" not in avatar_id:
        raise ValueError(f"Unknown avatar id: {avatar_id}")
    source, key = avatar_id.split(":", 1)
    if source != "gagavatar" or not key:
        raise ValueError(f"Unknown avatar id: {avatar_id}")
    return source, key


def run_job(job_id: str, *, input_path: str, style_id: str, clip_length: int, device: str, avatar_id: str, render_mode: str):
    path = JOB_ROOT / job_id
    try:
        write_state(path, {"id": job_id, "status": "running", "stage": "loading models"})
        avatar_source, avatar_key = split_avatar_id(avatar_id)
        gagavatar = None
        shape_code = None
        if avatar_source == "gagavatar":
            gagavatar = get_gagavatar_runtime()
            tracked = gagavatar.load_tracked_avatar(avatar_key)
            shape_code = gagavatar.shape_code(tracked)
            gagavatar.set_tracked_avatar(tracked, avatar_key)
        elif render_mode in {BROWSER_GAUSSIAN_RENDER_MODE, GAGAVATAR_RENDER_MODE}:
            raise ValueError("Choose a GAGAvator avatar for GAGAvator render modes.")

        engine = get_artalk_engine(device)
        write_state(path, {"id": job_id, "status": "running", "stage": "generating motion"})
        result = engine.generate(
            input_path,
            style_id=style_id,
            clip_length=clip_length,
            avatar_id=avatar_id,
            shape_code=shape_code,
        )
        write_state(path, {"id": job_id, "status": "running", "stage": "writing mesh artifacts"})
        metadata = write_animation_artifacts(result, path)
        if render_mode == BROWSER_GAUSSIAN_RENDER_MODE:
            write_state(path, {"id": job_id, "status": "running", "stage": "exporting gaussian artifacts"})
            metadata.update(export_browser_gaussian_artifacts(gagavatar, result.motions, path, fps=result.fps))
            metadata["renderMode"] = BROWSER_GAUSSIAN_RENDER_MODE
        elif render_mode == GAGAVATAR_RENDER_MODE:
            write_state(path, {"id": job_id, "status": "running", "stage": "rendering GAGAvator video"})
            from app.utils_videos import write_video
            import torch

            frames = torch.stack([gagavatar.render_rgb_frame(motion) for motion in result.motions]) * 255.0
            video_path = path / "gagavatar.mp4"
            write_video(frames, str(video_path), result.fps, result.audio, result.sample_rate, "aac")
            metadata["renderMode"] = GAGAVATAR_RENDER_MODE
            metadata["videoUrl"] = video_path.name
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)
        write_state(
            path,
            {
                "id": job_id,
                "status": "complete",
                "stage": "complete",
                "metadata": f"/api/jobs/{job_id}/metadata",
                "frameCount": metadata["frameCount"],
            },
        )
    except Exception as exc:
        write_state(
            path,
            {
                "id": job_id,
                "status": "failed",
                "stage": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )


@app.get("/api/config")
def config():
    styles = ["default"] + available_styles(Path(os.environ.get("ARTALK_ASSET_DIR", "assets")))
    return {
        "styles": styles,
        "avatars": available_avatars(),
        "languages": list(GTTS_LANG.keys()),
        "defaultStyle": "natural_0" if "natural_0" in styles else "default",
        "defaultAvatar": "mesh",
        "renderModes": [
            {"id": MESH_RENDER_MODE, "label": "Browser mesh"},
            {"id": BROWSER_GAUSSIAN_RENDER_MODE, "label": "Browser Gaussian (experimental)"},
            {"id": GAGAVATAR_RENDER_MODE, "label": "Colored video (server)"},
        ],
        "defaultRenderMode": MESH_RENDER_MODE,
    }


@app.get("/api/avatars")
def list_avatars():
    return {"avatars": available_avatars()}


@app.post("/api/avatar-jobs")
async def create_avatar_job(image_file: UploadFile = File(...)):
    raise HTTPException(status_code=501, detail="Uploaded avatar tracking is not wired into the standalone server yet.")


@app.get("/api/avatar-jobs/{avatar_id}")
def get_avatar_job(avatar_id: str):
    raise HTTPException(status_code=501, detail="Uploaded avatar tracking is not wired into the standalone server yet.")


@app.post("/api/jobs")
async def create_job(
    input_type: Literal["audio", "text"] = Form("audio"),
    style_id: str = Form("default"),
    clip_length: int = Form(750),
    device: str = Form("auto"),
    avatar_id: str = Form("mesh"),
    render_mode: Literal["mesh", "browser-gaussian", "gagavatar"] = Form(MESH_RENDER_MODE),
    text: str | None = Form(None),
    text_language: str = Form("English"),
    audio_file: UploadFile | None = File(None),
):
    job_id = uuid.uuid4().hex
    path = JOB_ROOT / job_id
    path.mkdir(parents=True, exist_ok=True)
    if input_type == "text":
        if text is None or not text.strip():
            raise HTTPException(status_code=400, detail="Text input is required")
        if text_language not in GTTS_LANG:
            raise HTTPException(status_code=400, detail="Unsupported text language")
        input_path = write_text_audio(text, text_language, path)
    else:
        if audio_file is None:
            raise HTTPException(status_code=400, detail="Audio file is required")
        suffix = Path(audio_file.filename or "input.wav").suffix or ".wav"
        input_path = path / f"input{suffix}"
        with open(input_path, "wb") as f:
            f.write(await audio_file.read())
    state = {"id": job_id, "status": "queued", "stage": "queued"}
    write_state(path, state)
    start_thread(
        run_job,
        job_id=job_id,
        input_path=str(input_path),
        style_id=style_id,
        clip_length=clip_length,
        device=device,
        avatar_id=avatar_id,
        render_mode=render_mode,
    )
    return state


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return read_state(job_dir(job_id))


@app.get("/api/jobs/{job_id}/metadata")
def get_metadata(job_id: str):
    path = job_dir(job_id)
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Metadata not ready")
    with open(metadata_path) as f:
        metadata = json.load(f)
    return {
        **metadata,
        "verticesUrl": f"/api/jobs/{job_id}/vertices.f32",
        "facesUrl": f"/api/jobs/{job_id}/faces.i32",
        "regionLabelsUrl": f"/api/jobs/{job_id}/regions.u8" if metadata.get("regionLabelsUrl") else None,
        "audioUrl": f"/api/jobs/{job_id}/audio.wav",
        "motionsUrl": f"/api/jobs/{job_id}/motions.pt",
        "videoUrl": f"/api/jobs/{job_id}/{metadata['videoUrl']}" if metadata.get("videoUrl") else None,
        "gaussianUrls": {
            key: (
                [f"/api/jobs/{job_id}/{item}" for item in value]
                if isinstance(value, list)
                else (f"/api/jobs/{job_id}/{value}" if value else None)
            )
            for key, value in metadata.get("gaussianUrls", {}).items()
        },
    }


@app.get("/api/jobs/{job_id}/{name}")
def get_job_file(job_id: str, name: str):
    if not is_allowed_artifact_name(name):
        raise HTTPException(status_code=404, detail="File not found")
    file_path = job_dir(job_id) / name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not ready")
    if name.endswith(".gz"):
        return FileResponse(file_path, media_type="application/octet-stream", headers={"Content-Encoding": "gzip"})
    return FileResponse(file_path)


def is_allowed_artifact_name(name: str) -> bool:
    fixed = {
        "vertices.f32",
        "faces.i32",
        "regions.u8",
        "audio.wav",
        "motions.pt",
        "gagavatar.mp4",
        "gaussians.xyz.f32",
        "gaussians.head_xyz.f32",
        "gaussians.transforms.f32",
        "gaussians.reference.mp4",
        "gaussians.upsampler_input_first.u8.gz",
        "gaussians.colors.f32",
        "gaussians.opacities.f32",
        "gaussians.scales.f32",
        "gaussians.rotations.f32",
    }
    if name in fixed:
        return True
    prefix = "gaussians.upsampler_input_"
    suffix = ".u8.gz"
    return name.startswith(prefix) and name.endswith(suffix) and name[len(prefix) : -len(suffix)].isdigit()


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
