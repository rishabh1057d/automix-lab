"""Pinned UMX-HQ vocal activity inference for transition-relevant audio."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import httpx
import numpy as np
from scipy import signal

SR = 44100
FFT = 4096
HOP = 1024
MODEL_REVISION = "87f26a5b2ed19488b5e453059a0a1d530423fe4a"
MODEL_SHA256 = "3d05709ff7197bbd4a33aee759e4e82002acb491f41c94770227ab57507b6ccb"
MODEL_SIZE = 17_820_856
MODEL_URL = f"https://huggingface.co/edgetools/umx-hq/resolve/{MODEL_REVISION}/vocals.onnx"


def _valid_model(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size != MODEL_SIZE:
        return False
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() == MODEL_SHA256


def ensure_model(model_dir: Path) -> Path:
    model_dir = Path(model_dir)
    path = model_dir / "vocals.onnx"
    if _valid_model(path):
        return path
    model_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".onnx.part")
    digest, size = hashlib.sha256(), 0
    with httpx.stream("GET", MODEL_URL, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > MODEL_SIZE:
                    raise ValueError("UMX-HQ download exceeds its pinned size")
                digest.update(chunk)
                stream.write(chunk)
    if size != MODEL_SIZE or digest.hexdigest() != MODEL_SHA256:
        temporary.unlink(missing_ok=True)
        raise ValueError("UMX-HQ download did not match its pinned checksum")
    temporary.replace(path)
    return path


@lru_cache(maxsize=1)
def _session(path: str):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
    input_meta, output_meta = session.get_inputs()[0], session.get_outputs()[0]
    if input_meta.type != "tensor(float)" or input_meta.shape[:3] != [1, 2, 2049]:
        raise ValueError(f"Unexpected UMX-HQ input contract: {input_meta.type} {input_meta.shape}")
    if output_meta.type != "tensor(float)" or output_meta.shape[:3] != [1, 2, 2049]:
        raise ValueError(f"Unexpected UMX-HQ output contract: {output_meta.type} {output_meta.shape}")
    return session, input_meta.name, output_meta.name


def _magnitude(audio: np.ndarray) -> np.ndarray:
    import torch
    wave = torch.from_numpy(np.ascontiguousarray(audio.T, dtype=np.float32))
    window = torch.hann_window(FFT, periodic=True)
    return torch.stft(wave, FFT, HOP, window=window, center=True, pad_mode="reflect",
                      return_complex=True).abs().numpy()[None].astype(np.float32)


def _reduce_activity(mix: np.ndarray, vocal: np.ndarray) -> np.ndarray:
    low, high = round(200 * FFT / SR), round(4000 * FFT / SR)
    mixture = mix[0, :, low:high]
    estimate = np.minimum(np.maximum(vocal[0, :, low:high], 0), mixture)
    frame_power = np.mean(mix[0] ** 2, axis=(0, 1))
    gate = max(1e-12, float(np.quantile(frame_power, .85)) * .01)
    ratio = np.sqrt(np.sum(estimate ** 2, axis=(0, 1)) /
                    (np.sum(mixture ** 2, axis=(0, 1)) + 1e-12))
    ratio[frame_power < gate] = 0
    kernel = max(3, round(SR / HOP) | 1)
    return np.clip(signal.medfilt(ratio, kernel_size=kernel), 0, 1).astype(np.float32)


def activity_curve(audio: np.ndarray, model_dir: Path, points: int, progress=None) -> tuple[list, list, str, None]:
    path = ensure_model(model_dir)
    session, input_name, output_name = _session(str(path))
    duration = len(audio) / SR
    # ponytail: only transition-relevant head/tail; add chunked full-track inference if cue search expands.
    windows = [(0.0, duration)] if duration <= 60 else [(0.0, 30.0), (duration - 30.0, duration)]
    times = np.arange(points, dtype=np.float32) / 4
    result = np.full(points, .5, dtype=np.float32)
    for index, (start, end) in enumerate(windows):
        if progress:
            progress(f"UMX-HQ analyzing {'full track' if len(windows) == 1 else ('head' if index == 0 else 'tail')}")
        chunk = audio[round(start * SR):round(end * SR)]
        magnitude = _magnitude(chunk)
        estimate = session.run([output_name], {input_name: magnitude})[0]
        if estimate.shape != magnitude.shape or not np.isfinite(estimate).all():
            raise ValueError("UMX-HQ returned invalid output")
        activity = _reduce_activity(magnitude, estimate)
        frame_times = start + np.arange(len(activity)) * HOP / SR
        selected = (times >= start) & (times <= end)
        result[selected] = np.interp(times[selected], frame_times, activity)
    return result.tolist(), [[round(a, 3), round(b, 3)] for a, b in windows], "UMX-HQ ONNX", None
