import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
from scipy import signal

import engine


def analysis(duration=60, bpm=120, confidence=.9, cue=0):
    return {"duration": duration, "bpm": bpm, "beat_confidence": confidence,
            "beats": np.arange(0, duration, 60 / bpm).tolist() if bpm else [],
            "downbeats": np.arange(cue, duration, 240 / bpm).tolist() if bpm else [],
            "audible_start": cue, "content_end": duration, "energy": [.1] * int(duration * 4),
            "vocal": [0.] * int(duration * 4), "waveform": [.1], "cache_hit": False,
            "beat_source": "test", "beat_warning": None, "vocal_source": "test-model",
            "vocal_windows": [[0, duration]], "vocal_warning": None}


class EngineTests(unittest.TestCase):
    def test_policy_tempo_cap_octaves_and_short_tracks(self):
        for bpm, expected in [(120, "beatmatched"), (60, "beatmatched"), (115, "dj-assisted")]:
            plan = engine.plan_transition(analysis(), analysis(bpm=bpm))
            self.assertEqual(plan["tier"], expected)
            self.assertLessEqual(abs(plan["incoming_playback_rate"] - 1), .04)
            self.assertGreaterEqual(plan["overlap_seconds"], 4)
            self.assertLessEqual(plan["overlap_seconds"], 12)
        self.assertEqual(engine.plan_transition(analysis(confidence=.1), analysis(confidence=.1))["tier"], "safe-crossfade")
        self.assertEqual(engine.plan_transition(analysis(confidence=.19), analysis(confidence=.19))["tier"], "dj-assisted")
        self.assertEqual(engine.plan_transition(analysis(duration=5), analysis())["tier"], "safe-crossfade")

    def test_cue_budget_and_plain_beginning(self):
        a, b = analysis(), analysis(cue=2)
        b["energy"][80:] = [1.] * (len(b["energy"]) - 80)
        plan = engine.plan_transition(a, b)
        self.assertLessEqual(plan["incoming_cue"] - b["audible_start"], 12)
        self.assertLessEqual(a["content_end"] - plan["outgoing_end"], 12)
        self.assertIn(plan["outgoing_start"], a["downbeats"])
        self.assertIn(plan["incoming_cue"], b["downbeats"])
        self.assertEqual(engine.plan_transition(a, b, "plain")["incoming_cue"], 0)

    def test_vocal_overlap_is_simultaneous_not_mean(self):
        a, b = analysis(), analysis()
        a["vocal"][0:4] = [1.] * 4
        b["vocal"][0:4] = [1.] * 4
        self.assertAlmostEqual(engine._clash(a, b, 0, 0, 4, 1), .25)
        self.assertEqual(engine._clash(a, b, 1, 0, 4, 1), 0)

    def test_model_vocals_choose_clean_cue_and_gate_ducking(self):
        a, b = analysis(), analysis()
        a["vocal"] = [1.] * len(a["vocal"])
        b["vocal"][0:8] = [1.] * 8
        clean = engine.plan_transition(a, b)
        self.assertGreaterEqual(clean["incoming_cue"], 2)
        self.assertEqual(clean["vocal_overlap"], 0)
        self.assertEqual(clean["vocal_duck_db"], 0)

        a, b = analysis(bpm=60), analysis(bpm=60)
        a["vocal"] = b["vocal"] = [1.] * len(a["vocal"])
        unavoidable = engine.plan_transition(a, b)
        self.assertEqual(unavoidable["vocal_evidence"], "model")
        self.assertEqual(unavoidable["vocal_duck_db"], 6)
        a["vocal_source"] = b["vocal_source"] = "DSP heuristic fallback"
        fallback = engine.plan_transition(a, b)
        self.assertEqual(fallback["vocal_evidence"], "heuristic")
        self.assertEqual(fallback["vocal_duck_db"], 0)

        a, b = analysis(120, bpm=100), analysis(120, bpm=100)
        a.update(content_end=104.8, downbeats=[100], vocal=[1.] * 480)
        b.update(downbeats=[0], vocal=[0.] * 16 + [1.] * 4 + [0.] * 460)
        shortened = engine.plan_transition(a, b)
        self.assertEqual(shortened["outgoing_start"], 100)
        self.assertEqual(shortened["overlap_seconds"], 4)

        a, b = analysis(120, bpm=60), analysis(120, bpm=60)
        a.update(content_end=120, downbeats=[100], vocal=[1.] * 480)
        b.update(downbeats=[0], vocal=[0.] * 16 + [1.] * 464)
        bounded = engine.plan_transition(a, b)
        self.assertEqual((bounded["outgoing_start"], bounded["outgoing_end"]), (100, 108))
        self.assertEqual(bounded["overlap_seconds"], 8)
        self.assertGreater(bounded["vocal_duck_db"], 0)

    def test_unmeasured_vocals_are_unknown_and_safe_tier_never_claims_duck(self):
        a, b = analysis(120), analysis(120, cue=40)
        for item in (a, b):
            item["vocal"] = [.5] * 480
            item["vocal_windows"] = [[0, 30], [90, 120]]
        a.update(content_end=44, downbeats=[40])
        b["downbeats"] = [40]
        unknown = engine.plan_transition(a, b)
        self.assertEqual(unknown["vocal_evidence"], "unavailable")
        self.assertEqual(unknown["vocal_overlap"], 0)
        self.assertEqual(unknown["vocal_duck_db"], 0)

        a, b = analysis(20), analysis(20)
        a["vocal"] = b["vocal"] = [1.] * 80
        safe = engine.plan_transition(a, b)
        self.assertEqual(safe["tier"], "safe-crossfade")
        self.assertEqual(safe["vocal_duck_db"], 0)

    def test_equal_power_and_endpoints(self):
        n = 4410
        first = np.tile([1., 0.], (n, 1)).astype(np.float32)
        second = np.tile([0., 1.], (n, 1)).astype(np.float32)
        audio = engine.blend_audio(first, second, {"tier": "safe-crossfade"})
        np.testing.assert_allclose((audio ** 2).sum(axis=1), 1, atol=2e-7)
        np.testing.assert_allclose(audio[0], first[0], atol=1e-7)
        np.testing.assert_allclose(audio[-1], second[-1], atol=1e-7)

    def test_vocal_duck_is_audible_signal_processing(self):
        t = np.arange(4410) / engine.SR
        tone = np.column_stack((np.sin(2 * np.pi * 1000 * t),) * 2).astype(np.float32) * .2
        plan = {"tier": "dj-assisted", "bass_handoff_seconds": .05,
                "overlap_seconds": .1, "vocal_duck_db": 0}
        plain = engine.blend_audio(tone, tone, plan)
        ducked = engine.blend_audio(tone, tone, dict(plan, vocal_duck_db=6))
        middle = slice(1800, 2600)
        self.assertLess(float(np.sqrt(np.mean(ducked[middle] ** 2))),
                        float(np.sqrt(np.mean(plain[middle] ** 2))) * .85)

    def test_decode_silence_boundaries_and_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "test.wav"
            audio = np.concatenate((np.zeros(engine.SR), np.full(engine.SR * 2, .1), np.zeros(engine.SR)))
            sf.write(path, audio, engine.SR)
            fake_vocals = lambda audio, model_dir, points, progress: ([0.] * points, [[0, 4]], "test-model", None)
            with patch.object(engine, "_beats", side_effect=AssertionError("No inference for short tracks")), \
                    patch("vocal_model.activity_curve", side_effect=fake_vocals):
                first = engine.analyze(path, root / "cache")
                second = engine.analyze(path, root / "cache")
            self.assertEqual(first["audible_start"], 1)
            self.assertEqual(first["content_end"], 3)
            self.assertTrue(second["cache_hit"])
            retry_cache = root / "retry-cache"
            with patch("vocal_model.activity_curve", side_effect=RuntimeError("offline")):
                fallback = engine.analyze(path, retry_cache)
            with patch("vocal_model.activity_curve", side_effect=fake_vocals):
                recovered = engine.analyze(path, retry_cache)
            self.assertEqual(fallback["vocal_source"], "DSP heuristic fallback")
            self.assertFalse(recovered["cache_hit"])
            sf.write(path, np.zeros(engine.SR), engine.SR)
            silent = engine.analyze(path, root / "cache")
            self.assertEqual(silent["beat_confidence"], 0)
            self.assertTrue(engine.analyze(path, root / "cache")["cache_hit"])
            sf.write(path, np.zeros((10, 3)), engine.SR)
            with self.assertRaises(ValueError):
                engine.decode(path)
            sf.write(path, np.empty((0, 2)), engine.SR)
            with self.assertRaises(ValueError):
                engine.decode(path)

    def test_wsola_preserves_pitch_and_continuous_source_resume(self):
        t = np.arange(engine.SR * 10) / engine.SR
        tone = (.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        stereo = np.column_stack((tone, tone))
        for rate in (.97, 1.03):
            rendered, mapping = engine.stretch_intro(stereo, rate, 4)
            resume_src = round(mapping["source_resume"] * engine.SR)
            resume_out = round(mapping["output_resume"] * engine.SR)
            self.assertEqual(len(rendered), resume_out + len(stereo) - resume_src)
            np.testing.assert_array_equal(rendered[resume_out:], stereo[resume_src:])
            frequencies, powers = signal.periodogram(rendered[engine.SR:3 * engine.SR, 0], engine.SR)
            self.assertAlmostEqual(frequencies[np.argmax(powers)], 440, delta=1)
            expected_consumed = (4 * rate + 2 * (rate + 1) / 2) * engine.SR
            self.assertLess(abs(resume_src - expected_consumed), .15 * engine.SR)
            self.assertLess(abs(float(rendered[resume_out, 0] - rendered[resume_out - 1, 0])), .02)

    def test_render_three_tracks_timeline_and_second_transition_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            t = np.arange(engine.SR * 48) / engine.SR
            audio = (.85 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            tracks = []
            for i in range(3):
                path = root / f"{i}.wav"
                sf.write(path, audio, engine.SR)
                tracks.append({"id": str(i), "title": str(i), "path": path})
            data = [analysis(48, bpm=120), analysis(48, bpm=118), analysis(48, bpm=120)]
            with patch.object(engine, "analyze", side_effect=data):
                result = engine.render_mix(tracks, "automix", root / "out", root / "cache")
            output, sr = sf.read(root / "out" / "mix.wav")
            self.assertEqual(sr, engine.SR)
            self.assertEqual(len(output), round(result["duration"] * sr))
            self.assertTrue(np.isfinite(output).all())
            self.assertLessEqual(float(np.max(np.abs(output))), 10 ** (-1 / 20) + 2e-7)
            first, second = result["transitions"]
            expected_start = (first["timeline_start"] + first["output_resume"] +
                              second["outgoing_start"] - first["incoming_cue"] - first["source_resume"])
            self.assertAlmostEqual(second["timeline_start"], expected_start, delta=2 / sr)
            self.assertGreater(second["timeline_start"], first["timeline_end"])
            self.assertEqual(json.loads((root / "out" / "plan.json").read_text())["mode"], "automix")

    def test_short_audible_middle_track_keeps_monotonic_timeline(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tracks = []
            data = [analysis(2, bpm=0, confidence=0) for _ in range(3)]
            data[1]["content_end"] = .25
            for i in range(3):
                path = root / f"{i}.wav"
                audio = np.full(engine.SR * 2, .1, np.float32)
                if i == 1:
                    audio[engine.SR // 4:] = 0
                sf.write(path, audio, engine.SR)
                tracks.append({"id": str(i), "title": str(i), "path": path})
            with patch.object(engine, "analyze", side_effect=data):
                result = engine.render_mix(tracks, "automix", root / "out", root / "cache")
            first, second = result["transitions"]
            self.assertGreaterEqual(second["timeline_start"], first["timeline_end"])
            self.assertGreater(second["outgoing_start"], 0)
            self.assertLessEqual(second["outgoing_end"], .25 + 1 / engine.SR)
            self.assertGreater(result["duration"], second["timeline_end"])


if __name__ == "__main__":
    unittest.main()
