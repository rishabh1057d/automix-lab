"""Measured rhythm layers and finite stereo echo, independent of the renderer."""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

SR = 44100


def local_pulse(analysis: dict, start: float, end: float, rate: float = 1.0) -> float | None:
    beats = np.asarray(analysis.get("beats", []), dtype=float)
    beats = np.unique(beats[np.isfinite(beats) & (beats >= start - 1e-6) & (beats <= end + 1e-6)])
    intervals = np.diff(beats)
    if len(intervals) < 4 or not np.isfinite(rate) or rate <= 0:
        return None
    pulse = float(np.median(intervals))
    if not .08 <= pulse <= 2.5 or np.mean(np.abs(intervals - pulse) <= pulse * .10) < .8:
        return None
    return pulse / rate


def ping_pong_echo(send: np.ndarray, delay_seconds: float, history: np.ndarray | None = None,
                   wet: float = .45, feedback: float = .65, tail_seconds: float = 6.0) -> np.ndarray:
    """Return only repeats; prehistory seeds them without extending the source send."""
    if not .08 <= delay_seconds <= 6 or not 0 <= feedback < 1 or wet < 0 or tail_seconds < 0:
        raise ValueError("Echo delay, feedback, wet level or tail is outside its supported range.")
    n, tail = len(send), round(tail_seconds * SR)
    output = np.zeros((n + tail, 2), dtype=np.float32)
    if not n:
        return output
    prior = np.asarray(history[-6 * SR:] if history is not None else np.empty((0, 2)), dtype=np.float32)
    source = np.concatenate((prior, np.asarray(send, dtype=np.float32)))
    filtered = signal.sosfilt(signal.butter(3, [200, 5000], "bandpass", fs=SR, output="sos"),
                              source, axis=0).astype(np.float32)
    cutoff = min(n, round(.02 * SR))
    filtered[-cutoff:] *= np.linspace(1, 0, cutoff, dtype=np.float32)[:, None]
    delay = max(1, round(delay_seconds * SR))
    for tap in range(1, max(1, round(6 * SR) // delay) + 1):
        offset = tap * delay - len(prior)
        lo, hi = max(0, offset), min(len(output), offset + len(filtered))
        if hi <= lo:
            continue
        emphasis = np.array([1.0, .2] if tap % 2 else [.2, 1.0], dtype=np.float32)
        # A mid component keeps alternate repeats audible in both ears even for hard-panned sends.
        chunk = filtered[lo - offset:hi - offset]
        stereo = .5 * chunk + .5 * chunk.mean(axis=1, keepdims=True)
        output[lo:hi] += stereo * emphasis * (wet * feedback ** (tap - 1))
    ramp = min(n, round(.08 * SR))
    output[:ramp] *= np.linspace(0, 1, ramp, dtype=np.float32)[:, None]
    if tail:
        output[n:] *= (np.cos(np.linspace(0, np.pi / 2, tail)) ** 2).astype(np.float32)[:, None]
        output[-1] = 0
    return output


def _clean_bar(analysis: dict, start: float, end: float) -> bool:
    if analysis.get("vocal_source") not in ("UMX-HQ ONNX", "test-model"):
        return False
    covered_to = start
    for lo, hi in sorted(analysis.get("vocal_windows", [])):
        if lo > covered_to + 1e-6:
            break
        if hi >= covered_to:
            covered_to = hi
        if covered_to >= end - 1e-6:
            break
    vocal = np.asarray(analysis.get("vocal", []), dtype=float)
    lo, hi = int(np.floor(start * 4)), int(np.ceil(end * 4))
    if covered_to < end - 1e-6 or lo < 0 or hi > len(vocal) or hi <= lo:
        return False
    active = vocal[lo:hi] >= .35
    return bool(np.isfinite(vocal[lo:hi]).all() and np.mean(active) < .1 and
                not (len(active) >= 2 and np.any(active[:-1] & active[1:])))


def _percussion(audio: np.ndarray) -> np.ndarray:
    _, _, spectrum = signal.stft(audio.T, fs=SR, nperseg=2048, noverlap=1536)
    magnitude = np.abs(spectrum).mean(axis=0)
    harmonic = median_filter(magnitude, size=(1, 17))
    percussive = median_filter(magnitude, size=(17, 1))
    mask = percussive ** 2 / (harmonic ** 2 + percussive ** 2 + 1e-12)
    _, audio = signal.istft(spectrum * mask[None, :, :], fs=SR, nperseg=2048, noverlap=1536)
    return signal.sosfilt(signal.butter(3, [180, 5000], "bandpass", fs=SR, output="sos"),
                          audio.T, axis=0).astype(np.float32)


def render_rhythm_loop(source_audio: np.ndarray, a: dict, b: dict, plan: dict) -> np.ndarray:
    seconds = plan["overlap_seconds"]
    output = np.zeros((round(seconds * SR), 2), dtype=np.float32)
    details = plan["rhythm_loop"] = {"enabled": False, "reason": "No covered instrumental bar with regular measured beats."}
    if plan.get("tier") == "plain-crossfade":
        details["reason"] = "Plain crossfade has no rhythm layer."
        return output
    end = min(plan["outgoing_end"], len(source_audio) / SR)
    downbeats = sorted(set(a.get("downbeats", [])))
    bars = [(lo, hi) for lo, hi in zip(downbeats, downbeats[1:])
            if max(0, end - 30) <= lo < hi <= end and 1 <= hi - lo <= 6
            and local_pulse(a, lo, hi) is not None and _clean_bar(a, lo, hi)]
    if not bars:
        return output
    cue, rate = plan["incoming_cue"], plan["incoming_playback_rate"]
    delay = plan.get("incoming_delay_seconds", 0)
    earliest = max(plan.get("incoming_fade_start", .25) * seconds, delay)
    incoming = sorted(set(b.get("downbeats", [])))
    candidates = [(lo, hi, delay + (lo - cue) / rate)
                  for lo, hi in zip(incoming, incoming[1:])
                  if lo >= cue and earliest - 1e-6 <= delay + (lo - cue) / rate < .75 * seconds
                  and 1 <= hi - lo <= 6 and local_pulse(b, lo, hi, rate) is not None]
    if not candidates:
        details["reason"] = "No regular measured incoming bar before the final quarter of the overlap."
        return output
    target_start, target_end, entry = candidates[0]
    target_bar = (target_end - target_start) / rate
    selected = None
    for lo, hi in reversed(bars):
        mappings = [(abs((hi - lo) / (target_bar * multiple) - 1), multiple)
                    for multiple in (.5, 1, 2)
                    if .88 - 1e-9 <= (hi - lo) / (target_bar * multiple) <= 1.12 + 1e-9]
        if mappings:
            _, multiple = min(mappings)
            selected = lo, hi, target_bar * multiple
            break
    if selected is None:
        details["reason"] = "Instrumental and incoming bars need more than 12% loop stretching."
        return output
    lo, hi, target_seconds = selected
    original = source_audio[round(lo * SR):round(hi * SR)]
    loop = _percussion(original)[:len(original)]
    source_rms = float(np.sqrt(np.mean(original ** 2)))
    loop_rms = float(np.sqrt(np.mean(loop ** 2)))
    if loop_rms < max(1e-5, source_rms * .08):
        details["reason"] = "The eligible instrumental bar has too little percussion."
        return output
    speed = (hi - lo) / target_seconds
    if abs(speed - 1) > 1e-5:
        from audiotsm import wsola
        from audiotsm.io.array import ArrayReader, ArrayWriter
        writer = ArrayWriter(channels=2)
        # WSOLA can flush short; a repeated guard bar supplies the final analysis window.
        wsola(channels=2, speed=speed, frame_length=2048, synthesis_hop=512).run(
            ArrayReader(np.tile(loop, (3, 1)).T), writer)
        loop = writer.data.T.copy()
    target_frames = round(target_seconds * SR)
    loop = np.pad(loop[:target_frames], ((0, max(0, target_frames - len(loop))), (0, 0)))
    seam = min(round(.02 * SR), len(loop) // 2)
    loop[:seam] *= np.linspace(0, 1, seam, dtype=np.float32)[:, None]
    loop[-seam:] *= np.linspace(1, 0, seam, dtype=np.float32)[:, None]
    energy = np.asarray(b.get("energy", []), dtype=float)
    local_energy = energy[max(0, int(target_start * 4)):int(np.ceil(target_end * 4))]
    reference = float(np.median(local_energy)) if len(local_energy) else source_rms
    reference = max(0, reference) if np.isfinite(reference) else source_rms
    gain = min(10 ** (6 / 20), reference * 10 ** (-6 / 20) / (float(np.sqrt(np.mean(loop ** 2))) + 1e-12))
    start = min(len(output), max(0, round(entry * SR)))
    output[start:] = np.tile(loop, (int(np.ceil((len(output) - start) / len(loop))), 1))[:len(output) - start] * gain
    ramp = min(round(.08 * SR), len(output) - start)
    output[start:start + ramp] *= np.linspace(0, 1, ramp, dtype=np.float32)[:, None]
    fade_start = max(start, round(.75 * len(output)))
    output[fade_start:] *= (np.cos(np.linspace(0, np.pi / 2, len(output) - fade_start)) ** 2).astype(np.float32)[:, None]
    if len(output):
        output[-1] = 0
    details.update(enabled=True, reason="Measured instrumental percussion bar aligned to the incoming downbeat.",
                   source_start=lo, source_end=hi, playback_rate=round(speed, 6), entry_seconds=entry,
                   gain_db=round(float(20 * np.log10(max(gain, 1e-12))), 2))
    return output
