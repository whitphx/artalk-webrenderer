#!/usr/bin/env python

from __future__ import annotations

import gzip
import os
from pathlib import Path

import av
import torch

from app.runtime import ARTalkResult, MESH_REGION_LABELS, save_audio


ARTIFACT_FORMAT_VERSION = "artalk-web-animation-v1"
UPSAMPLER_PREVIEW_FRAME_COUNT = 32
UPSAMPLER_QUANTIZATION_LEVELS = 32
UPSAMPLER_INPUT_DTYPE = "uint8-linear"


def write_animation_artifacts(result: ARTalkResult, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.vertices.tofile(output_dir / "vertices.f32")
    result.faces.tofile(output_dir / "faces.i32")
    result.region_labels.tofile(output_dir / "regions.u8")
    torch.save(result.motions, output_dir / "motions.pt")
    save_audio(output_dir / "audio.wav", result.audio[None], result.sample_rate)
    return {
        "artifactFormatVersion": ARTIFACT_FORMAT_VERSION,
        "renderMode": "mesh",
        "fps": result.fps,
        "sampleRate": result.sample_rate,
        "frameCount": int(result.vertices.shape[0]),
        "vertexCount": int(result.vertices.shape[1]),
        "faceCount": int(result.faces.shape[0]),
        "verticesUrl": "vertices.f32",
        "facesUrl": "faces.i32",
        "regionLabelsUrl": "regions.u8",
        "regionLabelFormat": "uint8-vertex",
        "regionLabels": MESH_REGION_LABELS,
        "regionSource": result.region_source,
        "audioUrl": "audio.wav",
        "motionsUrl": "motions.pt",
        "videoUrl": None,
        "avatarId": result.avatar_id,
    }


@torch.no_grad()
def export_browser_gaussian_artifacts(
    gagavatar,
    motions: torch.Tensor,
    output_dir: str | Path,
    *,
    write_reference_video: bool = True,
    upsampler_preview_frame_count: int | str | None = None,
    upsampler_preview_stride: int | None = None,
    quantization_levels: int = UPSAMPLER_QUANTIZATION_LEVELS,
    fps: int = 25,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    motions = motions.to(gagavatar.device)
    head_frames = []
    transform_frames = []
    reference_frames = []
    first_batch = None
    first_gs_params = None
    upsampler_input_files = []
    upsampler_input_shape = None
    preview_indices = upsampler_preview_frame_indices(
        int(motions.shape[0]),
        upsampler_preview_frame_count,
        upsampler_preview_stride,
    )
    preview_index_set = set(preview_indices)

    for frame_index, motion in enumerate(motions):
        batch = gagavatar.build_forward_batch(motion[None])
        if first_batch is None:
            first_batch = batch
        head_frames.append(batch["t_points"][0].detach().float().cpu())
        transform_frames.append(batch["t_transform"][0].detach().float().cpu())
        if write_reference_video:
            reference_frames.append(gagavatar.render_rgb_batch(batch).cpu()[0])
        if frame_index in preview_index_set:
            gs_params_frame = gagavatar.forward_gaussians_for_batch(batch)
            if first_gs_params is None:
                first_gs_params = {key: value.clone() for key, value in gs_params_frame.items()}
            upsampler_input = gagavatar.rasterize_gaussians(
                gs_params_frame,
                batch["t_transform"],
            )[0].detach().to(torch.float16).cpu()
            upsampler_input_shape = list(upsampler_input.shape)
            file_name = f"gaussians.upsampler_input_{len(upsampler_input_files):03d}.u8.gz"
            write_quantized_upsampler_input(
                upsampler_input,
                output_dir / file_name,
                quantization_levels=quantization_levels,
            )
            if not upsampler_input_files:
                write_quantized_upsampler_input(
                    upsampler_input,
                    output_dir / "gaussians.upsampler_input_first.u8.gz",
                    quantization_levels=quantization_levels,
                )
            upsampler_input_files.append(file_name)

    if first_batch is None:
        raise ValueError("Cannot export Gaussian artifacts for an empty motion sequence.")
    gs_params = first_gs_params if first_gs_params is not None else gagavatar.forward_gaussians_for_batch(first_batch)
    head_positions = torch.stack(head_frames)
    transforms = torch.stack(transform_frames)
    snapshot = {
        "xyz": gs_params["xyz"][0].detach().float().cpu(),
        "colors": gs_params["colors"][0].detach().float().cpu(),
        "opacities": gs_params["opacities"][0].detach().float().cpu(),
        "scales": gs_params["scales"][0].detach().float().cpu(),
        "rotations": gs_params["rotations"][0].detach().float().cpu(),
    }
    for name, tensor in snapshot.items():
        tensor.numpy().astype("float32", copy=False).tofile(output_dir / f"gaussians.{name}.f32")
    head_positions.numpy().astype("float32", copy=False).tofile(output_dir / "gaussians.head_xyz.f32")
    transforms.numpy().astype("float32", copy=False).tofile(output_dir / "gaussians.transforms.f32")
    if write_reference_video and reference_frames:
        frames = (torch.stack(reference_frames) * 255.0).to(torch.uint8).permute(0, 2, 3, 1)
        write_rgb_video(frames, output_dir / "gaussians.reference.mp4", fps=fps)

    camera_params = gagavatar.camera_params
    return {
        "gaussianCount": int(snapshot["xyz"].shape[0]),
        "gaussianFormat": "gagavatar-first-frame-f32-v1",
        "gaussianColorChannels": int(snapshot["colors"].shape[1]),
        "gaussianUpsamplerInput": {
            "dtype": UPSAMPLER_INPUT_DTYPE,
            "shape": upsampler_input_shape,
            "frameCount": len(upsampler_input_files),
            "frameIndices": preview_indices,
            "quantizationLevels": quantization_levels,
        },
        "gaussianHeadCount": int(head_positions.shape[1]),
        "gaussianHeadFrameCount": int(head_positions.shape[0]),
        "gaussianTransformFrameCount": int(transforms.shape[0]),
        "gaussianUrls": {
            "xyz": "gaussians.xyz.f32",
            "headXyz": "gaussians.head_xyz.f32",
            "transforms": "gaussians.transforms.f32",
            "referenceVideo": "gaussians.reference.mp4" if write_reference_video else None,
            "upsamplerInputFirst": "gaussians.upsampler_input_first.u8.gz",
            "upsamplerInputFrames": upsampler_input_files,
            "colors": "gaussians.colors.f32",
            "opacities": "gaussians.opacities.f32",
            "scales": "gaussians.scales.f32",
            "rotations": "gaussians.rotations.f32",
        },
        "gaussianCamera": {
            "focalX": float(camera_params["focal_x"]),
            "focalY": float(camera_params["focal_y"]),
            "size": list(camera_params["size"]),
        },
    }


def sample_frame_indices(frame_count: int, max_frames: int) -> list[int]:
    if frame_count <= 0 or max_frames <= 0:
        return []
    if frame_count <= max_frames:
        return list(range(frame_count))
    if max_frames == 1:
        return [0]
    return sorted({round(index * (frame_count - 1) / (max_frames - 1)) for index in range(max_frames)})


def upsampler_preview_frame_indices(
    frame_count: int,
    max_frames: int | str | None = None,
    stride: int | None = None,
) -> list[int]:
    if frame_count <= 0:
        return []
    if stride is None:
        raw_stride = first_env_value(
            "ARTALK_WEB_UPSAMPLER_PREVIEW_STRIDE",
            "ARTALK_UPSAMPLER_PREVIEW_STRIDE",
            "GAGAVATAR_UPSAMPLER_PREVIEW_STRIDE",
        )
        stride = int(raw_stride) if raw_stride and raw_stride.isdigit() else None
    if stride is not None:
        return list(range(0, frame_count, max(1, stride)))
    if max_frames is None:
        max_frames = first_env_value(
            "ARTALK_WEB_UPSAMPLER_PREVIEW_FRAMES",
            "ARTALK_UPSAMPLER_PREVIEW_FRAMES",
            "GAGAVATAR_UPSAMPLER_PREVIEW_FRAMES",
        ) or UPSAMPLER_PREVIEW_FRAME_COUNT
    if isinstance(max_frames, str):
        if max_frames.lower() == "all":
            return list(range(frame_count))
        try:
            max_frames = int(max_frames)
        except ValueError:
            max_frames = UPSAMPLER_PREVIEW_FRAME_COUNT
    return sample_frame_indices(frame_count, max(1, int(max_frames)))


def first_env_value(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def write_quantized_upsampler_input(
    tensor: torch.Tensor,
    output_path: str | Path,
    *,
    quantization_levels: int = UPSAMPLER_QUANTIZATION_LEVELS,
):
    levels = min(max(int(quantization_levels), 2), 256)
    channels = int(tensor.shape[0])
    values = tensor.float().flatten(1)
    mins = values.min(dim=1).values
    maxs = values.max(dim=1).values
    scales = ((maxs - mins) / float(levels - 1)).clamp_min(1e-8)
    quantized = torch.clamp(
        torch.round((tensor.float() - mins[:, None, None]) / scales[:, None, None]),
        0,
        levels - 1,
    ).to(torch.uint8)
    params = torch.stack([mins, scales], dim=1).cpu().numpy().astype("float32", copy=False)
    with gzip.open(output_path, "wb", compresslevel=1) as f:
        f.write(quantized.cpu().numpy().tobytes(order="C"))
        f.write(params.tobytes(order="C"))
    if params.shape != (channels, 2):
        raise ValueError("Invalid upsampler quantization parameters.")


def write_rgb_video(frames: torch.Tensor, output_path: str | Path, *, fps: int):
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"frames must be (T, H, W, 3), got {tuple(frames.shape)}")
    frames_np = frames.detach().cpu().numpy()
    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("h264", rate=int(fps))
    stream.width = int(frames_np.shape[2])
    stream.height = int(frames_np.shape[1])
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18"}
    try:
        for frame in frames_np:
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
