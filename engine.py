"""Offline audio analysis and mixing."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from transition_effects import limit_effect_peak, progressive_reverb, render_rhythm_loop

SR = 44100
VERSION = "analysis-8-vocal-entry-45s"
VOCAL_ACTIVE = .35
MAX_DISCARDED_MUSIC_SECONDS = 12.0
MIN_AUTOMIX_OVERLAP_SECONDS = 12.0
INCOMING_VOCAL_TARGET = .50


def decode(path: Path) -> np.ndarray:
    try:
        info = sf.info(path)
        if info.channels not in (1, 2) or not 0 < info.duration <= 600:
            raise ValueError("Audio must be mono/stereo and between 0 and 10 minutes.")
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
    except (RuntimeError, sf.LibsndfileError) as exc:
        raise ValueError("The file is not supported, decodable audio.") from exc
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite samples.")
    if rate != SR:
        common = math.gcd(rate, SR)
        audio = signal.resample_poly(audio, SR // common, rate // common).astype(np.float32)
    return np.repeat(audio, 2, axis=1) if audio.shape[1] == 1 else audio


def peaks(audio: np.ndarray, count: int = 1200) -> list:
    width = max(1, math.ceil(len(audio) / count))
    return [round(float(np.max(np.abs(audio[i:i + width]))), 5)
            for i in range(0, len(audio), width)]


def _audible_seconds(energy: list[float], start: float, end: float) -> float:
    curve = np.asarray(energy, dtype=float)
    if len(curve) == 0:
        return max(0.0, end - start)
    floor = max(1e-5, float(np.quantile(curve, .85)) * .1)
    lo, hi = max(0, round(start * 4)), min(len(curve), round(end * 4))
    return float(np.count_nonzero(curve[lo:hi] > floor) / 4)


def _mix_out_candidates(energy: list[float], content_end: float) -> list[dict]:
    """Find the onset of a sustained final energy drop, not arbitrary quiet gaps."""
    curve = np.asarray(energy, dtype=float)
    end = min(len(curve), round(content_end * 4))
    candidates = []
    floor = max(1e-5, float(np.quantile(curve, .85)) * .25)
    for i in range(max(16, end - 160), end - 24, 4):
        pre = float(np.median(curve[i - 12:i]))
        after = curve[i:end]
        if pre < floor:
            continue
        early = float(np.median(after[:16]))
        if early > pre * .72 or len(after) < 24:
            continue
        # Reject a verse dip that recovers into another full-strength section.
        if np.max(np.convolve(after, np.ones(8) / 8, mode="valid")) > pre * .9:
            continue
        time = i / 4
        discarded = _audible_seconds(energy, time, content_end)
        candidates.append({"time": time, "type": "sustained_outro_drop",
                           "strength": round(1 - early / (pre + 1e-12), 3),
                           "discarded_music_seconds": discarded})
    return candidates


@lru_cache(maxsize=1)
def _model():
    import torch
    from beat_this.inference import Audio2Beats
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    return Audio2Beats(checkpoint_path="small0", device="cpu", dbn=False)


def _beats(audio: np.ndarray, progress=None) -> dict:
    """Only measure head and tail; never extrapolate a fictitious full-song grid."""
    from beat_this.inference import Audio2Frames
    model = _model()
    duration = len(audio) / SR
    starts = [0.0]
    if duration > 45:
        starts.append(30.0)
    if duration > 75:
        starts.append(duration - 30.0)
    elif duration > 60:
        starts.append(max(30.0, duration - 30.0))
    starts = sorted(set(starts))
    beats, downbeats, confidences, intervals, local_tempos = [], [], [], [], []
    for i, start in enumerate(starts):
        if progress:
            progress(f"Beat This! analyzing {'head' if i == 0 else 'tail'}")
        chunk_end = min(duration, start + (duration if duration <= 45 else 15 if start == 30 and duration > 75 else 30))
        chunk = audio[round(start * SR):round(chunk_end * SR)]
        logits, down_logits = Audio2Frames.__call__(model, chunk, SR)
        beat, downbeat = model.frames2beats(logits, down_logits)
        beat = np.asarray(beat)
        delta = np.diff(beat)
        valid = delta[(delta >= 60 / 220) & (delta <= 60 / 40)]
        confidence = 0.0
        if len(valid) >= 6:
            median = float(np.median(valid))
            regularity = float(np.mean(np.abs(valid - median) < median * .10))
            positions = np.clip(np.rint(beat * 50).astype(int), 0, len(logits) - 1)
            evidence = float(logits.sigmoid().cpu().numpy()[positions].mean())
            coverage = min(1.0, len(valid) * median / max(1, len(chunk) / SR) / .65)
            confidence = evidence * regularity * coverage
            intervals.extend(valid.tolist())
            local_tempos.append(60 / median)
        confidences.append(confidence)
        beats.extend((beat + start).tolist())
        downbeats.extend((np.asarray(downbeat) + start).tolist())
    stable = not local_tempos or max(local_tempos) / min(local_tempos) <= 1.04
    confidence = min(confidences) if stable else min(.19, min(confidences))
    return {"beats": sorted(set(beats)), "downbeats": sorted(set(downbeats)),
            "bpm": round(60 / float(np.median(intervals)), 2) if intervals else 0.0,
            "beat_confidence": round(confidence, 3), "beat_source": "Beat This! small0 (head/tail)",
            "beat_warning": None if stable else "Head/tail tempos disagree; track may change tempo, so stretching is disabled."}


def analyze(path: Path, cache_dir: Path, progress=None) -> dict:
    path, cache_dir = Path(path), Path(cache_dir)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{VERSION}-{digest}.json"
    if cached.exists():
        try:
            result = json.loads(cached.read_text(encoding="utf-8"))
            return dict(result, cache_hit=True)
        except (ValueError, OSError):
            pass
    audio = decode(path)
    duration = len(audio) / SR
    hop = SR // 4
    energy, heuristic = [], []
    frequencies = np.fft.rfftfreq(hop, 1 / SR)
    voice_band = (frequencies >= 250) & (frequencies <= 4000)
    previous = 0.0
    for start in range(0, len(audio), hop):
        frame = audio[start:start + hop]
        rms = float(np.sqrt(np.mean(frame * frame)))
        energy.append(rms)
        center = frame.mean(axis=1)
        spectrum = np.abs(np.fft.rfft(center, n=hop)) ** 2
        band = spectrum[voice_band] + 1e-15
        band_fraction = float(band.sum() / (spectrum.sum() + 1e-12))
        flatness = float(np.exp(np.log(band).mean()) / band.mean())
        centered = float(np.mean(center ** 2) / (np.mean(frame ** 2) + 1e-12))
        percussive = max(0.0, (rms - previous) / (rms + 1e-8))
        heuristic.append(float(np.clip(band_fraction * centered * (1 - flatness) * (1 - .7 * percussive), 0, 1)))
        previous = rms
    reference = float(np.quantile(energy, .85))
    active = np.flatnonzero(np.asarray(energy) > max(1e-5, reference * .1))
    audible_start = float(active[0] / 4) if len(active) else 0.0
    content_end = min(duration, float((active[-1] + 1) / 4)) if len(active) else duration
    mix_out_candidates = _mix_out_candidates(energy, content_end) if len(active) else []
    try:
        beat = _beats(audio, progress) if reference > 1e-5 and duration >= 8 else {
            "beats": [], "downbeats": [], "bpm": 0.0, "beat_confidence": 0.0,
            "beat_source": "unavailable", "beat_warning": "Silent or very short audio: beat analysis skipped."}
    except Exception as exc:
        logging.getLogger(__name__).exception("Beat This! inference unavailable for %s", path.name)
        beat = {"beats": [], "downbeats": [], "bpm": 0.0, "beat_confidence": 0.0,
                "beat_source": "unavailable", "beat_warning": f"Beat model unavailable ({type(exc).__name__}); safe crossfade used."}
    vocal_cacheable = True
    if reference > 1e-5:
        try:
            from vocal_model import activity_curve
            vocal, vocal_windows, vocal_source, vocal_warning = activity_curve(
                audio, cache_dir.parent / "models" / "umx-hq", len(energy), progress)
        except Exception as exc:
            logging.getLogger(__name__).exception("UMX-HQ inference unavailable for %s", path.name)
            vocal, vocal_windows = heuristic, []
            vocal_source = "DSP heuristic fallback"
            vocal_warning = f"Vocal model unavailable ({type(exc).__name__}); heuristic affects cue ranking only."
            vocal_cacheable = False
    else:
        vocal, vocal_windows = [0.0] * len(energy), []
        vocal_source, vocal_warning = "unavailable", "Silent audio: vocal analysis skipped."
    result = dict(beat, duration=duration, audible_start=audible_start, content_end=content_end,
                  mix_out_candidates=mix_out_candidates,
                  energy=[round(x, 6) for x in energy], vocal=[round(x, 4) for x in vocal],
                  vocal_source=vocal_source, vocal_windows=vocal_windows, vocal_warning=vocal_warning,
                  waveform=peaks(audio), sha256=digest, cache_hit=False)
    # Failed model downloads are retryable; do not turn one outage into a permanent cached fallback.
    if (beat["beat_source"] != "unavailable" or duration < 8 or reference <= 1e-5) and vocal_cacheable:
        temporary = cached.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
        temporary.replace(cached)
    return result


def _model_vocals(analysis: dict) -> bool:
    return analysis.get("vocal_source") in ("UMX-HQ ONNX", "test-model")


def _covered(analysis: dict, times: np.ndarray) -> np.ndarray:
    if not _model_vocals(analysis):
        return np.ones(len(times), dtype=bool)
    covered = np.zeros(len(times), dtype=bool)
    for start, end in analysis.get("vocal_windows", []):
        covered |= (times >= start) & (times <= end)
    return covered


def _activity(analysis: dict, times: np.ndarray) -> np.ndarray:
    values = np.asarray(analysis.get("vocal", [0.0]))
    activity = values[np.clip((times * 4).astype(int), 0, len(values) - 1)]
    return np.where(_covered(analysis, times), activity, 0.0)


def _clash(a: dict, b: dict, start: float, cue: float, duration: float, rate: float, delay: float = 0) -> float:
    times = np.arange(0, max(.25, duration), .25)
    audible = times >= max(.25 * duration, delay)
    return float(np.mean(audible & (_activity(a, start + times) >= VOCAL_ACTIVE) &
                         (_activity(b, cue + np.maximum(0, times - delay) * rate) >= VOCAL_ACTIVE)))


def _model_pair_covered(a: dict, b: dict, start: float, cue: float, duration: float, rate: float) -> bool:
    times = np.arange(0, max(.25, duration), .25)
    return (_model_vocals(a) and _model_vocals(b) and
            bool(_covered(a, start + times).all()) and bool(_covered(b, cue + times * rate).all()))


def _safe_outro_exit(a: dict, end: float) -> bool:
    if end > a["content_end"] or _audible_seconds(a.get("energy", []), end, a["content_end"]) > MAX_DISCARDED_MUSIC_SECONDS:
        return False
    if _model_vocals(a):
        times = np.arange(end, a["content_end"], .25)
        if not _covered(a, times).all():
            return False
        active = (_activity(a, times) >= VOCAL_ACTIVE).astype(int)
        return not ((len(active) and active[0]) or
                    (len(active) >= 4 and np.any(np.convolve(active, np.ones(4, dtype=int), mode="valid") == 4)))
    return True


def _first_vocal_phrase(analysis: dict) -> float | None:
    if not _model_vocals(analysis):
        return None
    limit = min(45.0, analysis["duration"])
    times = np.arange(0, limit, .25)
    if not _covered(analysis, times).all():
        return None
    strength = _activity(analysis, times)
    active = np.flatnonzero(strength >= VOCAL_ACTIVE)
    if not len(active):
        return None
    groups = np.split(active, np.flatnonzero(np.diff(active) > 3) + 1)
    for group in groups:
        if len(group) < 12:
            continue
        if max(np.count_nonzero((group >= t) & (group < t + 20)) for t in group) < 12:
            continue
        onset = int(group[0])
        while onset and onset - group[0] >= -4 and strength[onset - 1] >= .18:
            onset -= 1
        return onset / 4
    return None


def _vocal_entry_timing(vocal_entry: float, cue: float, overlap: float, rate: float) -> tuple[float, float]:
    delay = max(0.0, INCOMING_VOCAL_TARGET * overlap - (vocal_entry - cue) / rate)
    return delay, delay + (vocal_entry - cue) / rate


def plan_transition(a: dict, b: dict, mode: str = "automix", vocal_entry_override: float | None = None) -> dict:
    plain = mode == "plain"
    ca, cb = a["beat_confidence"], b["beat_confidence"]
    bpm_a, bpm_b = a["bpm"], b["bpm"]
    short = min(a["duration"], b["duration"]) < 45
    ratios = [bpm_a / (bpm_b * factor) for factor in (.5, 1, 2)] if bpm_a and bpm_b else [1.0]
    ratio = min(ratios, key=lambda value: abs(value - 1))
    reasons = []
    vocal_entry = None if plain else (vocal_entry_override if vocal_entry_override is not None else _first_vocal_phrase(b))
    entry_source = "manual" if vocal_entry_override is not None and not plain else "automatic" if vocal_entry is not None else "fallback"
    has_structural_cues = bool(a.get("downbeats") and b.get("downbeats"))
    if plain:
        tier, rate = "plain-crossfade", 1.0
        reasons.append("A/B reference: six-second equal-power fade, full track beginnings.")
    elif short or not (40 <= bpm_a <= 220 and 40 <= bpm_b <= 220) or max(ca, cb) < .15 or not has_structural_cues:
        tier, rate = "safe-crossfade", 1.0
        reasons.append("Short track or insufficient measured beat evidence: conservative crossfade.")
    elif min(ca, cb) >= .55 and abs(ratio - 1) <= .04000001:
        tier, rate = "beatmatched", ratio
        reasons.append("Both measured beat grids are confident; tempo correction is within ±4%.")
    else:
        tier, rate = "dj-assisted", 1.0
        reasons.append("Tempo distance or beat confidence prevents stretching; structural cues and EQ remain available.")
    if not plain:
        reasons.append("Song 1 fades under a 5 kHz-to-250 Hz low-pass; Song 2 rises from 25% through a 650 Hz low-pass opening to full tone and a short room effect.")
    for item in (a, b):
        if item.get("beat_warning"):
            reasons.append(item["beat_warning"])
        if item.get("vocal_warning"):
            reasons.append(item["vocal_warning"])
    outros = [] if plain else a.get("mix_out_candidates", [])
    outro = next(iter(outros), None)
    overlap = (6.0 if plain else float(np.clip(
        (16 if outro else 8) * 60 / bpm_a if tier != "safe-crossfade" else MIN_AUTOMIX_OVERLAP_SECONDS,
        MIN_AUTOMIX_OVERLAP_SECONDS, 16)))
    end = a["duration"] if plain else (outro["time"] if outro else a["content_end"])
    cue = 0.0 if plain else b["audible_start"]
    if outro and tier == "safe-crossfade":
        outro = next((candidate for candidate in outros
                      if _safe_outro_exit(a, candidate["time"] + overlap)), None)
        end = outro["time"] + overlap if outro else a["content_end"]
    if tier in ("beatmatched", "dj-assisted"):
        incoming = [t for t in b.get("downbeats", [])
                    if b["audible_start"] <= t <= min(b["audible_start"] + 12,
                                                       b["duration"] - overlap * rate - 3)] or [cue]
        curve = np.asarray(b.get("energy", [0.0]))
        def entry_score(t):
            at = min(len(curve) - 1, round(t * 4))
            before = float(curve[max(0, at - 8):max(1, at)].mean())
            after = float(curve[at:min(len(curve), at + 8)].mean())
            rise = (after - before) / (max(after, before) + 1e-8)
            voice = float(_activity(b, np.arange(t, t + overlap, .25)).mean())
            return rise * .6 - voice * .35 - (t - b["audible_start"]) * .015
        starts = []
        for candidate in outros:
            anchor = candidate["time"]
            starts = [t for t in a.get("downbeats", []) if anchor - .5 <= t + overlap <= anchor + 4
                      and t >= overlap and _safe_outro_exit(a, t + overlap)]
            if not starts:
                starts = [t for t in a.get("downbeats", []) if anchor - overlap / 2 <= t <= anchor + 4
                          and _safe_outro_exit(a, t + overlap)]
            if starts:
                outro = candidate
                break
        if not starts:
            outro, overlap, end = None, float(np.clip(8 * 60 / bpm_a, MIN_AUTOMIX_OVERLAP_SECONDS, 16)), a["content_end"]
            starts = [t for t in a.get("downbeats", []) if end - 16 <= t + overlap <= end
                      and t >= overlap and _safe_outro_exit(a, t + overlap)]
        starts = starts or [end - overlap]
        pairs = [(start, cue) for start in starts for cue in incoming]
        covered_pairs = [pair for pair in pairs
                         if _model_pair_covered(a, b, pair[0], pair[1], overlap, rate)]
        if vocal_entry is not None:
            start = max(starts)
            reasons.append("Outgoing measured beat selected while the incoming cue follows the sustained vocal phrase.")
        elif covered_pairs:
            start, cue = min(covered_pairs,
                             key=lambda pair: (_clash(a, b, pair[0], pair[1], overlap, rate),
                                               -entry_score(pair[1]), -pair[0]))
            reasons.append("Measured downbeat pair selected to minimize model-detected vocal collision.")
        else:
            start, cue = max(starts), max(incoming, key=entry_score)
            reasons.append("Incoming downbeat selected by energy rise, vocal estimate, and ≤12-second intro budget.")
        end = start + overlap
        if start in a.get("downbeats", []):
            reasons.append("Outgoing overlap begins on a measured downbeat near the audible ending.")
    if outro:
        reasons.append("Sustained late-song energy drop; overlap carries the outro until the remaining tail is safe to skip.")
    if vocal_entry is not None:
        target_cue = max(0.0, vocal_entry - INCOMING_VOCAL_TARGET * overlap * rate)
        measured = [t for t in b.get("downbeats", []) if t <= vocal_entry and abs(t - target_cue) <= .5]
        cue = min(measured, key=lambda t: abs(t - target_cue)) if measured else target_cue
        reasons.append(f"{entry_source.capitalize()} sustained vocal phrase at {vocal_entry:.1f}s is placed near the midpoint of the handoff.")
    elif not plain:
        reasons.append("No sustained vocal phrase detected in first 45 seconds; using a conservative entry.")
    outgoing_span = end if plain else max(1 / SR, end - a["audible_start"])
    incoming_end = b["duration"] if plain else b["content_end"]
    available = min(outgoing_span, max(1 / SR, (incoming_end - cue) / max(rate, 1e-6)))
    overlap = min(overlap, available) if not plain else min(overlap, available / 3)
    if not plain and overlap < MIN_AUTOMIX_OVERLAP_SECONDS:
        reasons.append("Selected audio is shorter than twelve seconds; the longest physically possible handoff is used.")
    start = end - overlap
    if tier in ("beatmatched", "dj-assisted") and (
            not any(abs(t - start) <= .125 for t in a.get("downbeats", [])) or
            not any(abs(t - cue) <= .125 for t in b.get("downbeats", []))):
        tier, rate = "safe-crossfade", 1.0
        reasons.append("Selected cue pair lacks measured downbeats; tempo stretching and beat effects are disabled.")
    if vocal_entry is not None:
        target_cue = max(0.0, vocal_entry - INCOMING_VOCAL_TARGET * overlap * rate)
        measured = [t for t in b.get("downbeats", []) if t <= vocal_entry and abs(t - target_cue) <= .5]
        cue = min(measured, key=lambda t: abs(t - target_cue)) if measured else target_cue
        if tier in ("beatmatched", "dj-assisted") and not measured:
            tier, rate = "safe-crossfade", 1.0
            cue = max(0.0, vocal_entry - INCOMING_VOCAL_TARGET * overlap)
            reasons.append("Vocal entry has no nearby measured downbeat; tempo stretching is disabled.")
    delay, entry_at = _vocal_entry_timing(vocal_entry, cue, overlap, rate) if vocal_entry is not None else (0.0, None)
    clash = _clash(a, b, start, cue, overlap, rate, delay)
    vocal_evidence = ("model" if _model_pair_covered(a, b, start, cue, overlap, rate) else
                      "heuristic" if "heuristic" in (a.get("vocal_source", "") + b.get("vocal_source", "")).lower()
                      else "unavailable")
    if not plain and tier != "safe-crossfade" and vocal_evidence == "model" and clash > .05:
        reduced = max(MIN_AUTOMIX_OVERLAP_SECONDS, overlap / 2)
        if (vocal_entry is None and reduced < overlap and (not outro or start + reduced >= outro["time"] - .5) and
                _safe_outro_exit(a, start + reduced)):
            alternative = _clash(a, b, start, cue, reduced, rate)
            if alternative < clash:
                overlap, end, clash = reduced, start + reduced, alternative
                reasons.append("Shorter overlap reduces simultaneous model-detected vocals.")
        if clash > .05:
            reasons.append("Unavoidable model-detected vocal overlap; complementary vocal-band ducking applied.")
    vocal_duck_db = (min(6.0, 2.0 + 4.0 * clash)
                     if not plain and tier != "safe-crossfade" and vocal_evidence == "model" and clash > .05 else 0.0)
    bass = overlap * .55
    nearby = [(t - start) for t in a.get("downbeats", []) if overlap * .35 <= t - start <= overlap * .75]
    if nearby:
        measured = min(nearby, key=lambda t: abs(t - bass))
        if abs(measured - bass) <= min(.5, .1 * overlap):
            bass = measured
    discarded_music = _audible_seconds(a.get("energy", []), end, a["content_end"]) if not plain else 0.0
    reverb_wet = .1 if outro and (tier in ("beatmatched", "dj-assisted") or
                                 (vocal_entry is not None and ca >= .15 and cb >= .15 and
                                  any(abs(t - start) <= .125 for t in a.get("downbeats", [])))) else 0.0
    if reverb_wet:
        reasons.append("A short stereo room tail carries the outgoing outro beneath the new track.")
    return {"tier": tier, "outgoing_start": start, "outgoing_end": end, "incoming_cue": cue,
            "mix_out_type": outro["type"] if outro else ("full_track_end" if plain else "content_end"),
            "mix_out_anchor": outro["time"] if outro else end,
            "discarded_music_seconds": round(discarded_music, 2), "reverb_wet": reverb_wet,
            "incoming_reverb_wet": .2 if not plain else 0.0,
            "reverb_tail_seconds": .7 if reverb_wet else 0.0,
            "overlap_seconds": overlap, "overlap_beats": overlap * bpm_a / 60 if bpm_a else 0,
            "incoming_playback_rate": rate, "bass_handoff_seconds": bass, "vocal_overlap": clash,
            "vocal_duck_db": round(vocal_duck_db, 2), "vocal_evidence": vocal_evidence,
            "vocal_entry": vocal_entry, "vocal_entry_source": entry_source,
            "incoming_delay_seconds": delay, "incoming_vocal_at": entry_at,
            "incoming_fade_start": .25 if not plain else 0.0,
            "bass_crossover_hz": 180 if not plain else 0,
            "reverb": {"enabled": not plain, "wet": .65, "decay_start_seconds": .7,
                       "decay_end_seconds": 3.5, "tail_seconds": 6.0,
                       "vocal_duck_db": 3 if _model_vocals(b) and not plain else 0},
            "confidence": min(ca, cb), "reasons": reasons}


def stretch_intro(audio: np.ndarray, rate: float, overlap: float) -> tuple[np.ndarray, dict]:
    """WSOLA at transition tempo, 2s rate ramp, then waveform-aligned original audio."""
    if abs(rate - 1) < 1e-5:
        return audio, {"source_resume": 0.0, "output_resume": 0.0, "rate_ramp_seconds": 0.0}
    from audiotsm import wsola
    from audiotsm.io.array import ArrayReader, ArrayWriter
    tsm = wsola(channels=2, speed=rate, frame_length=2048, synthesis_hop=512)
    reader, writer = ArrayReader(audio.T), ArrayWriter(channels=2)
    target = round(min(overlap + 2.0, len(audio) / SR - .2) * SR)
    written, consumed_nominal = 0, 0.0
    while written < target:
        time = written / SR
        current_rate = rate + (1 - rate) * np.clip((time - overlap) / 2.0, 0, 1)
        # AudioTSM uses integer analysis hops; account for the actual quantized ratio.
        effective = int(512 * current_rate) / 512
        tsm.set_speed(current_rate)
        tsm.read_from(reader)
        count, _ = tsm.write_to(writer)
        written += count
        consumed_nominal += count * effective
        if reader.empty and count == 0:
            break
    rendered = writer.data.T.copy()
    if len(rendered) < 2048:
        raise ValueError("Track too short for tempo correction.")
    cross = 1024
    expected = round(consumed_nominal) - cross
    left, right = max(0, expected - 4096), min(len(audio) - cross, expected + 4096)
    template = rendered[-cross:].mean(axis=1)
    search = audio[left:right + cross].mean(axis=1)
    correlation = signal.correlate(search, template, mode="valid", method="fft")
    energy = signal.convolve(search ** 2, np.ones(cross), mode="valid", method="fft")
    match = left + int(np.argmax(correlation / np.sqrt(np.maximum(energy, 1e-12))))
    blend = np.linspace(0, 1, cross, dtype=np.float32)[:, None]
    rendered[-cross:] = rendered[-cross:] * (1 - blend) + audio[match:match + cross] * blend
    resumed = match + cross
    return np.concatenate((rendered, audio[resumed:])), {
        "source_resume": resumed / SR, "output_resume": len(rendered) / SR,
        "rate_ramp_seconds": 2.0}


def _filter_sweep(audio: np.ndarray, cutoffs: list[float], kind: str) -> np.ndarray:
    """Crossfade adjacent fixed filters: continuously moving tone without coefficient jumps."""
    position = np.linspace(0, len(cutoffs) - 1, len(audio), dtype=np.float32)
    result = np.zeros_like(audio)
    for i, cutoff in enumerate(cutoffs):
        filtered = signal.sosfilt(signal.butter(4, cutoff, btype=kind, fs=SR, output="sos"), audio, axis=0)
        weight = np.maximum(0, 1 - np.abs(position - i))[:, None]
        result += (filtered * weight).astype(np.float32)
    return result


def blend_audio(outgoing: np.ndarray, incoming: np.ndarray, plan: dict, *, return_decks=False):
    n = min(len(outgoing), len(incoming))
    outgoing, incoming = outgoing[:n], incoming[:n]
    x = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    if plan["tier"] != "plain-crossfade":
        # ponytail: 150 ms ramps avoid hard timbre jumps; add an adjustable curve only if listening tests need it.
        seconds = plan.get("overlap_seconds", n / SR)
        out_wet = np.clip(x * seconds / .15, 0, 1)
        entry = np.clip((x - .25) / .75, 0, 1)
        outgoing = outgoing * (1 - out_wet) + _filter_sweep(outgoing, [5000, 2000, 700, 250], "lowpass") * out_wet
        entry_sample = min(n - 1, round(n * .25))
        filtered = _filter_sweep(incoming[entry_sample:], [650, 1800, 5500, 18000], "lowpass")
        incoming = incoming.copy()
        incoming[:entry_sample] = 0
        incoming[entry_sample:] = filtered * (1 - entry[entry_sample:]) + incoming[entry_sample:] * entry[entry_sample:]
        if plan.get("incoming_reverb_wet", 0):
            vocal_at = plan.get("incoming_vocal_at")
            clear_at = max(0.0, vocal_at - .25) if vocal_at is not None else seconds * .5
            incoming += _reverb_send(incoming, plan["incoming_reverb_wet"], clear_at)[:n]
    if plan["tier"] in ("beatmatched", "dj-assisted") and not plan.get("bass_crossover_hz"):
        low = signal.butter(2, 200, fs=SR, output="sos")
        a_low = signal.sosfilt(low, outgoing, axis=0).astype(np.float32)
        b_low = signal.sosfilt(low, incoming, axis=0).astype(np.float32)
        handoff = plan["bass_handoff_seconds"] / plan["overlap_seconds"]
        bass_gain = np.clip((x - handoff + .1) / .2, 0, 1)
        # Smooth bass exchange while preserving the exact full-band signal at both endpoints.
        outgoing = outgoing - a_low * bass_gain
        incoming = incoming - b_low * (1 - bass_gain)
    if plan["tier"] in ("beatmatched", "dj-assisted") and plan.get("vocal_duck_db", 0) > 0:
        middle = signal.butter(2, [250, 4000], btype="bandpass", fs=SR, output="sos")
        a_mid = signal.sosfilt(middle, outgoing, axis=0).astype(np.float32)
        b_mid = signal.sosfilt(middle, incoming, axis=0).astype(np.float32)
        depth = 1 - 10 ** (-plan["vocal_duck_db"] / 20)
        outgoing -= a_mid * (depth * np.sin(np.pi * x) * x)
        incoming -= b_mid * (depth * np.sin(np.pi * x) * (1 - x))
    incoming_gain = np.sin(entry * np.pi / 2) ** .8 if plan["tier"] != "plain-crossfade" else np.sin(x * np.pi / 2)
    outgoing_gain = np.cos(x * np.pi / 2)
    if plan["tier"] != "plain-crossfade" and plan.get("bass_crossover_hz"):
        low = signal.butter(2, plan["bass_crossover_hz"], fs=SR, output="sos")
        a_low = signal.sosfiltfilt(low, outgoing, axis=0, padlen=min(12, n - 1)).astype(np.float32)
        b_low = signal.sosfiltfilt(low, incoming, axis=0, padlen=min(12, n - 1)).astype(np.float32)
        available = max(.25 * seconds, plan.get("incoming_audible_at", plan.get("incoming_delay_seconds", 0)))
        center = max(plan["bass_handoff_seconds"], available + .05 * seconds)
        end = min(seconds, center + .1 * seconds)
        start = min(end - 1 / SR, max(available, center - .1 * seconds))
        bass = np.clip((x * seconds - start) / max(1 / SR, end - start), 0, 1)
        plan.update(bass_handoff_start_seconds=start, bass_handoff_end_seconds=end)
        body = np.clip(x * seconds / .08, 0, 1) * np.clip((1 - x) * seconds / .5, 0, 1)
        a_body = a_low + body * (1.6 * np.tanh(1.8 * a_low) / 1.8 - a_low)
        b_body = b_low + body * (1.6 * np.tanh(1.8 * b_low) / 1.8 - b_low)
        outgoing = (outgoing - a_low) * outgoing_gain + a_body * np.cos(bass * np.pi / 2)
        incoming = (incoming - b_low) * incoming_gain + b_body * np.sin(bass * np.pi / 2)
    else:
        outgoing = outgoing * outgoing_gain
        incoming = incoming * incoming_gain
    return (outgoing, incoming) if return_decks else outgoing + incoming


def _reverb_vocal_gain(analysis: dict, plan: dict, frames: int) -> np.ndarray:
    """Duck only the wet return, using a small control-rate envelope rather than audio-rate Python."""
    depth = plan["reverb"]["vocal_duck_db"]
    if not depth:
        return np.ones(frames, np.float32)
    times = np.arange(math.ceil(frames / SR * 100) + 1) / 100
    delay, overlap = plan["incoming_delay_seconds"], plan["overlap_seconds"]
    if plan["source_resume"]:
        positions = np.interp(times, [delay, delay + overlap, plan["output_resume"]],
                              [0, overlap * plan["incoming_playback_rate"], plan["source_resume"]])
        after = times >= plan["output_resume"]
        positions[after] = plan["source_resume"] + times[after] - plan["output_resume"]
    else:
        positions = np.maximum(0, times - delay)
    active = (_activity(analysis, plan["incoming_cue"] + positions) >= VOCAL_ACTIVE) & (times >= delay)
    targets = np.where(active, 10 ** (-depth / 20), 1.0)
    envelope = np.empty_like(targets)
    value = 1.0
    for i, target in enumerate(targets):
        coefficient = math.exp(-.01 / (.08 if target < value else .25))
        value = target + coefficient * (value - target)
        envelope[i] = value
    return np.interp(np.arange(frames) / SR, times, envelope).astype(np.float32)


@lru_cache(maxsize=1)
def _reverb_ir() -> np.ndarray:
    """Deterministic stereo room response: 30 ms pre-delay, 0.7 s decay."""
    rng = np.random.default_rng(3127)
    tail = round(.7 * SR)
    noise = rng.standard_normal((tail, 2)).astype(np.float32)
    band = signal.butter(2, [180, 6500], btype="bandpass", fs=SR, output="sos")
    noise = signal.sosfilt(band, noise, axis=0).astype(np.float32)
    noise *= np.exp(-np.arange(tail, dtype=np.float32)[:, None] / (.18 * SR))
    noise /= np.sqrt(np.sum(noise * noise, axis=0, keepdims=True))
    return np.pad(noise, ((round(.03 * SR), 0), (0, 0)))


def _reverb_send(outgoing: np.ndarray, wet: float, clear_at: float | None = None) -> np.ndarray:
    """Put a quiet room under the handoff, clearing it before incoming vocals."""
    length = len(outgoing)
    n = min(length, round(length * .55))
    source = outgoing[:n].copy()
    source *= np.clip(np.arange(n, dtype=np.float32)[:, None] / max(1, .15 * SR), 0, 1)
    rendered = signal.fftconvolve(source, _reverb_ir(), mode="full", axes=0).astype(np.float32)
    result = np.zeros((length + len(_reverb_ir()) - 1, 2), np.float32)
    result[:len(rendered)] = rendered * wet
    stop = min(length, round((clear_at if clear_at is not None else length / SR) * SR))
    fade = min(round(.35 * SR), stop)
    if fade:
        result[stop - fade:stop] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]
    result[stop:] = 0
    return result


def render_mix(tracks: list[dict], mode: str, output_dir: Path, cache_dir: Path, progress=None) -> dict:
    if mode not in ("automix", "plain") or not 2 <= len(tracks) <= 8:
        raise ValueError("Choose automix/plain and between 2 and 8 tracks.")
    notify = progress or (lambda *args: None)
    analyses = []
    for i, track in enumerate(tracks):
        notify("analyzing", int(i / len(tracks) * 55), f"Analyzing {track['title']} ({i + 1} of {len(tracks)})")
        analyses.append(analyze(Path(track["path"]), cache_dir,
                                lambda msg, i=i: notify("analyzing", int(i / len(tracks) * 55), msg)))
    notify("planning", 55, "Selecting structural cues and transition tiers")
    current = decode(Path(tracks[0]["path"]))
    original_current = current
    source_peak = float(np.max(np.abs(current)))
    segments, transitions = [], []
    effect_beds = []
    timeline = 0.0
    source_offset = 0.0
    track_details = [dict(id=t["id"], title=t["title"], **a) for t, a in zip(tracks, analyses)]
    track_details[0].update(timeline_start=0.0, incoming_cue=0.0, source_resume=0.0, output_resume=0.0)
    for i in range(len(tracks) - 1):
        notify("rendering", 55 + int(i / (len(tracks) - 1) * 35), f"Rendering transition {i + 1} of {len(tracks) - 1}")
        plan = plan_transition(analyses[i], analyses[i + 1], mode, tracks[i + 1].get("vocal_entry_override"))
        # Source positions after the first transition's rate ramp map with slope 1.
        end = round((plan["outgoing_end"] - source_offset) * SR)
        n = round(plan["overlap_seconds"] * SR)
        if end <= 0:
            # A stale or exhausted analysis must never reuse the incoming track as outgoing audio.
            end = 1
            plan["tier"] = "safe-crossfade"
            plan["incoming_playback_rate"] = 1.0
            plan["reverb_wet"] = 0.0
            plan["reverb_tail_seconds"] = 0.0
            plan["mix_out_type"] = "content_end"
            plan["reasons"].append("Audible ending precedes available audio; fading the remaining samples conservatively.")
        end = max(1, min(len(current), end))
        n = min(n, max(1, end // 2))
        start = end - n
        incoming_full = decode(Path(tracks[i + 1]["path"]))
        source_peak = max(source_peak, float(np.max(np.abs(incoming_full))))
        cue_sample = round(plan["incoming_cue"] * SR)
        incoming, mapping = stretch_intro(incoming_full[cue_sample:], plan["incoming_playback_rate"], n / SR)
        if plan["vocal_entry"] is not None:
            delay, vocal_at = _vocal_entry_timing(plan["vocal_entry"], plan["incoming_cue"], n / SR,
                                                   plan["incoming_playback_rate"])
            delay_samples = round(delay * SR)
            if delay_samples:
                incoming = np.pad(incoming, ((delay_samples, 0), (0, 0)))
            mapping["output_resume"] += delay_samples / SR
            plan["incoming_delay_seconds"] = delay_samples / SR
            plan["incoming_vocal_at"] = vocal_at
        n = min(n, len(incoming))
        start = end - n
        plan["outgoing_start"] = source_offset + start / SR
        plan["outgoing_end"] = source_offset + end / SR
        plan["overlap_beats"] = n / SR * analyses[i]["bpm"] / 60 if analyses[i]["bpm"] else 0.0
        plan["bass_handoff_seconds"] = min(n / SR, plan["bass_handoff_seconds"])
        plan["incoming_audible_at"] = plan["incoming_delay_seconds"] + max(
            0.0, analyses[i + 1]["audible_start"] - plan["incoming_cue"]) / plan["incoming_playback_rate"]
        transition_start = timeline + start / SR
        plan.update(index=i, outgoing_title=tracks[i]["title"], incoming_title=tracks[i + 1]["title"],
                    timeline_start=transition_start, timeline_end=transition_start + n / SR,
                    overlap_seconds=n / SR, **mapping)
        outgoing_deck, incoming_deck = blend_audio(current[start:end], incoming[:n], plan, return_decks=True)
        blended = outgoing_deck + incoming_deck
        next_audio = incoming[n:]
        if plan["reverb"]["enabled"]:
            rhythm = render_rhythm_loop(original_current, analyses[i], analyses[i + 1], plan)
            blended += rhythm
            settings = plan["reverb"]
            progress = np.linspace(0, 1, n, dtype=np.float32)[:, None]
            outgoing_send = current[start:end] * (1 - .65 * np.clip((progress - .65) / .35, 0, 1))
            incoming_send = incoming[:n] * np.clip((progress - .25) / .35, 0, 1)
            wet = progressive_reverb(outgoing_send + incoming_send + rhythm, wet=settings["wet"],
                                     decay_start_seconds=settings["decay_start_seconds"],
                                     decay_end_seconds=settings["decay_end_seconds"],
                                     tail_seconds=settings["tail_seconds"])
            wet *= _reverb_vocal_gain(analyses[i + 1], plan, len(wet))[:, None]
            effect_beds.append((round(transition_start * SR), wet))
            plan["effects_end"] = plan["timeline_end"] + settings["tail_seconds"]
            plan["reasons"].append("Diffuse stereo reverb grows from a 0.7s to a 3.5s decay through the handoff, then fades to dry over six seconds; no added echo repeats. Vocal timing and the warm bass exchange are preserved.")
        else:
            plan["rhythm_loop"] = {"enabled": False, "reason": "Plain comparison uses no rhythm loop."}
        if plan["reverb_wet"]:
            clear_at = max(0.0, plan["incoming_vocal_at"] - .25) if plan["incoming_vocal_at"] is not None else n / SR
            room = _reverb_send(current[start:end], plan["reverb_wet"], clear_at)
            effect_beds.append((round(transition_start * SR), room))
        segments.extend((current[:start], blended))
        timeline += end / SR
        current = next_audio
        original_current = incoming_full
        source_offset = plan["incoming_cue"] + mapping["source_resume"] - mapping["output_resume"] + n / SR
        track_details[i + 1].update(timeline_start=transition_start, incoming_cue=plan["incoming_cue"], **mapping)
        transitions.append(plan)
    segments.append(current)
    notify("rendering", 94, "Writing audio and waveform")
    audio = np.concatenate(segments)
    ceiling = 10 ** (-1 / 20)
    reference_peak = source_peak if mode == "automix" else float(np.max(np.abs(audio)))
    attenuation = min(1.0, ceiling / max(reference_peak, 1e-12))
    audio *= attenuation
    for offset, bed in effect_beds:
        count = min(len(bed), len(audio) - offset)
        if count > 0:
            audio[offset:offset + count] += bed[:count] * attenuation
    regions = []
    for transition in transitions:
        if "effects_end" in transition:
            transition["effects_end"] = min(len(audio) / SR, transition["effects_end"])
        if mode == "automix":
            start, end = round(transition["timeline_start"] * SR), round(transition.get("effects_end", transition["timeline_end"]) * SR)
            if regions and start <= regions[-1][1]:
                regions[-1] = (regions[-1][0], max(end, regions[-1][1]))
            else:
                regions.append((start, end))
    effect_reduction = 0.0
    for start, end in regions:
        audio[start:end], reduction = limit_effect_peak(audio[start:end], ceiling)
        effect_reduction = max(effect_reduction, reduction)
    peak = float(np.max(np.abs(audio)))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(output_dir / "mix.wav", audio, SR, subtype="PCM_24")
    result = {"mode": mode, "duration": len(audio) / SR, "sample_rate": SR,
              "tracks": track_details, "transitions": transitions, "waveform": peaks(audio, 1800),
              "peak_dbfs": round(20 * math.log10(max(peak, 1e-12)), 3),
              "analysis_cache_hits": sum(a["cache_hit"] for a in analyses),
              "effect_peak_reduction_db": round(effect_reduction, 3),
              "gain_reduction_db": round(-20 * math.log10(attenuation), 3)}
    (output_dir / "plan.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    notify("complete", 100, "Mix ready")
    return result
