"""Measured rhythm layers and diffuse stereo reverb, independent of the renderer."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

SR = 44100


def limit_effect_peak(audio: np.ndarray, ceiling: float) -> tuple[np.ndarray, float]:
    """Limit only this effect window with linked stereo gain and a short release."""
    audio = np.asarray(audio, dtype=np.float32)
    if not np.isfinite(ceiling) or ceiling <= 0 or not np.isfinite(audio).all():
        raise ValueError("Effect audio must be finite and its peak ceiling positive.")
    if not len(audio) or float(np.max(np.abs(audio))) <= ceiling:
        return audio.copy(), 0.0
    peaks = np.max(np.abs(audio), axis=1)
    block = 128
    padded = np.pad(peaks, (0, (-len(peaks)) % block))
    block_peaks = padded.reshape(-1, block).max(axis=1)
    safe_ceiling = float(np.nextafter(np.float32(ceiling), np.float32(0)))
    required = np.minimum(1.0, safe_ceiling / np.maximum(block_peaks, 1e-12))
    controls = np.minimum.reduce((required, np.r_[1., required[:-1]], np.r_[required[1:], 1.]))
    release = np.exp(-block / (.08 * SR))
    gain = 1.0
    for i, required_gain in enumerate(controls):
        gain = min(float(required_gain), 1 - (1 - gain) * release)
        controls[i] = gain
    controls[controls > 1 - 1e-5] = 1
    envelope = np.interp(np.arange(len(audio)), np.arange(len(controls)) * block, controls)
    recovery = min(len(audio), round(.25 * SR))
    blend = .5 - .5 * np.cos(np.linspace(0, np.pi, recovery))
    envelope[-recovery:] += (1 - envelope[-recovery:]) * blend
    envelope[0] = 1
    # Interpolation and end recovery can miss an isolated peak; cap only those samples.
    envelope = np.minimum(envelope, np.where(peaks > ceiling, safe_ceiling / np.maximum(peaks, 1e-12), 1.0))
    return (audio * envelope[:, None]).astype(np.float32), float(-20 * np.log10(max(float(envelope.min()), 1e-12)))


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


@lru_cache(maxsize=8)
def _reverb_ir(decay_seconds: float) -> np.ndarray:
    predelay = round(.03 * SR)
    frames = 6 * SR - predelay
    noise = np.random.default_rng(712).standard_normal((frames, 2))
    diffuse = signal.sosfilt(signal.butter(3, [200, 6500], "bandpass", fs=SR, output="sos"), noise, axis=0)
    diffuse *= np.exp(-np.log(1000) * np.arange(frames) / (SR * decay_seconds))[:, None]
    diffuse /= np.sqrt(np.sum(diffuse ** 2, axis=0, keepdims=True))
    impulse = np.pad(diffuse.astype(np.float32), ((predelay, 0), (0, 0)))
    impulse.setflags(write=False)
    return impulse


def progressive_reverb(send: np.ndarray, wet: float = .65, decay_start_seconds: float = .7,
                       decay_end_seconds: float = 3.5, tail_seconds: float = 6.0) -> np.ndarray:
    """Return a diffuse wet bed whose newly arriving sound gradually rings longer."""
    send = np.asarray(send, dtype=np.float32)
    parameters = [wet, decay_start_seconds, decay_end_seconds, tail_seconds]
    if (send.ndim != 2 or send.shape[1] != 2 or not np.isfinite(send).all()
            or not np.isfinite(parameters).all() or not 0 <= wet <= 1
            or not .1 <= decay_start_seconds <= decay_end_seconds <= 6 or not 0 <= tail_seconds <= 6):
        raise ValueError("Reverb needs finite stereo audio, wet 0–1, increasing decay 0.1–6s and tail 0–6s.")
    n, tail = len(send), round(tail_seconds * SR)
    output = np.zeros((n + tail, 2), dtype=np.float32)
    if not n or not wet or not np.any(send):
        return output
    source = .5 * send + .5 * send.mean(axis=1, keepdims=True)
    ramp, cutoff = min(n, round(.08 * SR)), min(n, round(.02 * SR))
    source[:ramp] *= np.linspace(0, 1, ramp, dtype=np.float32)[:, None]
    source[-cutoff:] *= np.linspace(1, 0, cutoff, dtype=np.float32)[:, None]
    progress = np.linspace(0, 2, n, dtype=np.float32)
    for index, decay in enumerate(np.linspace(decay_start_seconds, decay_end_seconds, 3)):
        # Morph the send, not the returning tail: late sounds retain their longer decay.
        weight = np.maximum(0, 1 - np.abs(progress - index))
        convolved = signal.fftconvolve(source * weight[:, None], _reverb_ir(float(decay)), axes=0)
        count = min(len(output), len(convolved))
        output[:count] += convolved[:count] * wet
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
