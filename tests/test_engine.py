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
            self.assertGreaterEqual(plan["overlap_seconds"], 12)
            self.assertLessEqual(plan["overlap_seconds"], 16)
            self.assertEqual(plan["incoming_reverb_wet"], .2)
            self.assertLessEqual(plan["bass_handoff_seconds"], plan["overlap_seconds"] * .75)
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
        plain = engine.plan_transition(a, b, "plain")
        self.assertEqual(plain["incoming_cue"], 0)
        self.assertEqual(plain["overlap_seconds"], 6)
        self.assertEqual(plain["incoming_reverb_wet"], 0)

    def test_late_outro_exit_is_early_but_does_not_skip_music(self):
        a, b = analysis(120), analysis(120)
        a["energy"] = [.4] * (105 * 4) + [.03] * (15 * 4)
        a["mix_out_candidates"] = engine._mix_out_candidates(a["energy"], a["content_end"])
        self.assertTrue(a["mix_out_candidates"])
        early = engine.plan_transition(a, b)
        self.assertEqual(early["mix_out_type"], "sustained_outro_drop")
        self.assertGreaterEqual(120 - early["outgoing_start"], 18)
        self.assertGreaterEqual(early["outgoing_end"], early["mix_out_anchor"] - .5)
        self.assertLessEqual(early["discarded_music_seconds"], 12)
        self.assertGreater(early["reverb_wet"], 0)
        self.assertEqual(engine.plan_transition(a, b, "plain")["reverb_wet"], 0)

        # An ordinary energy dip followed by a full-strength return is not an outro.
        rebound = [.4] * (90 * 4) + [.06] * (5 * 4) + [.4] * (25 * 4)
        self.assertEqual(engine._mix_out_candidates(rebound, 120), [])

        # Quiet music above the same audible floor as content_end still spends the budget.
        quiet_outro = [.4] * (85 * 4) + [.06] * (35 * 4)
        self.assertEqual(engine._audible_seconds(quiet_outro, 85, 120), 35)
        a, b = analysis(120), analysis(120)
        a["energy"] = quiet_outro
        a["mix_out_candidates"] = engine._mix_out_candidates(quiet_outro, 120)
        self.assertTrue(a["mix_out_candidates"])
        self.assertEqual(engine.plan_transition(a, b)["mix_out_type"], "content_end")

        a, b = analysis(120), analysis(120, bpm=118)
        a.update(mix_out_candidates=[{"time": 84, "type": "sustained_outro_drop"}],
                 downbeats=[t for t in a["downbeats"] if t < 30 or t >= 90])
        unmeasured = engine.plan_transition(a, b)
        self.assertEqual(unmeasured["mix_out_type"], "content_end")
        self.assertEqual(unmeasured["reverb_wet"], 0)

    def test_audible_outro_can_mix_early_only_after_later_vocals(self):
        a, b = analysis(224), analysis(224)
        a["energy"] = [.38] * (207 * 4) + [.2] * (2 * 4) + [.1] * (15 * 4)
        a["downbeats"] = [200, 204, 208, 212, 216, 220]
        a["vocal"][216 * 4:219 * 4] = [.8] * (3 * 4)
        a["mix_out_candidates"] = engine._mix_out_candidates(a["energy"], a["content_end"])
        self.assertTrue(a["mix_out_candidates"])
        plan = engine.plan_transition(a, b)
        self.assertEqual(plan["mix_out_type"], "sustained_outro_drop")
        self.assertGreaterEqual(plan["mix_out_anchor"], a["mix_out_candidates"][0]["time"])
        self.assertLess(plan["outgoing_start"], 216)
        self.assertGreaterEqual(plan["outgoing_end"], 219)
        self.assertLess(plan["outgoing_end"], a["content_end"])
        self.assertLessEqual(plan["discarded_music_seconds"], 12)

        # A long overlap may still fit from the last measured downbeat; it must keep the voice.
        a["downbeats"] = [200, 204, 208]
        guarded = engine.plan_transition(a, b)
        self.assertGreaterEqual(guarded["outgoing_end"], 219)
        self.assertLessEqual(guarded["discarded_music_seconds"], 12)

    def test_later_one_second_vocal_return_is_not_discarded(self):
        a = analysis(120)
        a["vocal"][109 * 4:111 * 4] = [.8] * (2 * 4)
        self.assertFalse(engine._safe_outro_exit(a, 108))
        a["vocal"] = [0.] * len(a["vocal"])
        a["vocal"][108 * 4:108 * 4 + 5] = [.8] * 5
        self.assertFalse(engine._safe_outro_exit(a, 108))
        a["vocal"][108 * 4 + 5] = .8
        self.assertFalse(engine._safe_outro_exit(a, 108))

        # A late measured beat can carry even an active tail until vocals subside.
        a, b = analysis(120, bpm=100), analysis(120, bpm=100)
        a["downbeats"] = [106]
        a["mix_out_candidates"] = [{"time": 110, "type": "sustained_outro_drop"}]
        a["vocal"][110 * 4:115 * 4] = [.8] * (5 * 4)
        carried = engine.plan_transition(a, b)
        self.assertEqual(carried["mix_out_type"], "sustained_outro_drop")
        self.assertGreater(carried["outgoing_end"], 115)
        self.assertGreater(carried["reverb_wet"], 0)

    def test_vocal_overlap_is_simultaneous_not_mean(self):
        a, b = analysis(), analysis()
        a["vocal"][4:8] = [1.] * 4
        b["vocal"][4:8] = [1.] * 4
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
        self.assertGreater(unavoidable["vocal_duck_db"], 2)
        a["vocal_source"] = b["vocal_source"] = "DSP heuristic fallback"
        fallback = engine.plan_transition(a, b)
        self.assertEqual(fallback["vocal_evidence"], "heuristic")
        self.assertEqual(fallback["vocal_duck_db"], 0)

        a, b = analysis(120, bpm=100), analysis(120, bpm=100)
        a.update(content_end=104.8, downbeats=[100], vocal=[1.] * 480)
        b.update(downbeats=[0], vocal=[0.] * 16 + [1.] * 4 + [0.] * 460)
        shortened = engine.plan_transition(a, b)
        self.assertAlmostEqual(shortened["outgoing_start"], 92.8)
        self.assertEqual(shortened["overlap_seconds"], 12)

        a, b = analysis(120, bpm=60), analysis(120, bpm=60)
        a.update(content_end=120, downbeats=[100], vocal=[1.] * 480)
        b.update(downbeats=[0], vocal=[0.] * 16 + [1.] * 464)
        bounded = engine.plan_transition(a, b)
        self.assertEqual((bounded["outgoing_start"], bounded["outgoing_end"]), (108, 120))
        self.assertEqual(bounded["overlap_seconds"], 12)
        self.assertEqual(bounded["tier"], "safe-crossfade")
        self.assertEqual(bounded["vocal_duck_db"], 0)

        too_short = engine.plan_transition(analysis(5), analysis(5))
        self.assertEqual(too_short["overlap_seconds"], 5)
        self.assertIn("shorter than twelve seconds", " ".join(too_short["reasons"]))

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
        audio = engine.blend_audio(first, second, {"tier": "plain-crossfade"})
        np.testing.assert_allclose((audio ** 2).sum(axis=1), 1, atol=2e-7)
        np.testing.assert_allclose(audio[0], first[0], atol=1e-7)
        np.testing.assert_allclose(audio[-1], second[-1], atol=1e-7)

    def test_automix_handoff_audibly_filters_both_tracks(self):
        t = np.arange(engine.SR * 2) / engine.SR
        silence = np.zeros((len(t), 2), np.float32)
        plan = {"tier": "safe-crossfade", "overlap_seconds": 2,
                "bass_handoff_seconds": .8, "vocal_duck_db": 0}
        early = slice(round(.3 * len(t)), round(.36 * len(t)))
        middle = slice(round(.53 * len(t)), round(.57 * len(t)))
        before_entry = slice(round(.2 * len(t)), round(.24 * len(t)))
        late = slice(round(.7 * len(t)), round(.8 * len(t)))

        def level(frequency, outgoing, window):
            tone = np.repeat(np.sin(2 * np.pi * frequency * t)[:, None], 2, axis=1).astype(np.float32)
            filtered = engine.blend_audio(tone if outgoing else silence,
                                          silence if outgoing else tone, plan)
            return float(np.sqrt(np.mean(filtered[window] ** 2)))

        # Song 1 has already begun fading and strongly loses highs at transition start.
        self.assertLess(level(4000, True, early), level(100, True, early) * .45)
        # Song 2 is silent until 25%, then rises through its opening filter.
        self.assertLess(level(100, False, before_entry), 1e-5)
        self.assertGreater(level(100, False, early), .02)
        self.assertGreater(level(100, False, middle), level(100, True, middle) * .9)
        self.assertGreater(level(100, False, late), .15)
        self.assertLess(level(3000, False, early), level(100, False, early) * .3)
        self.assertLess(level(10000, False, early), level(100, False, early) * .25)
        end = slice(round(.93 * len(t)), round(.98 * len(t)))
        self.assertGreater(level(10000, False, end), level(100, False, end) * .7)

    def test_incoming_room_is_audible_before_vocals_and_clears_before_entry(self):
        t = np.arange(engine.SR * 2) / engine.SR
        silence = np.zeros((len(t), 2), np.float32)
        incoming = np.repeat((.2 * np.sin(2 * np.pi * 440 * t))[:, None], 2, axis=1).astype(np.float32)
        plan = {"tier": "safe-crossfade", "overlap_seconds": 2, "incoming_vocal_at": 1.4,
                "incoming_reverb_wet": 0}
        dry = engine.blend_audio(silence, incoming, plan)
        wet = engine.blend_audio(silence, incoming, dict(plan, incoming_reverb_wet=.2))
        audible = slice(round(.7 * engine.SR), round(1.0 * engine.SR))
        after_vocal = slice(round(1.2 * engine.SR), round(1.5 * engine.SR))
        self.assertGreater(float(np.sqrt(np.mean((wet[audible] - dry[audible]) ** 2))), .001)
        np.testing.assert_allclose(wet[after_vocal], dry[after_vocal], atol=1e-7)

    def test_bass_stays_present_until_incoming_audio_can_take_over(self):
        t = np.arange(engine.SR * 4) / engine.SR
        bass = np.repeat((.2 * np.sin(2 * np.pi * 70 * t))[:, None], 2, axis=1).astype(np.float32)
        plan = {"tier": "safe-crossfade", "overlap_seconds": 4, "bass_crossover_hz": 180,
                "bass_handoff_seconds": 2.2, "incoming_delay_seconds": 1.6}
        outgoing, incoming = engine.blend_audio(bass, bass, plan, return_decks=True)
        early = slice(round(.8 * engine.SR), round(1.2 * engine.SR))
        self.assertGreater(float(np.sqrt(np.mean(outgoing[early] ** 2))), .12)
        self.assertLess(float(np.sqrt(np.mean(incoming[early] ** 2))), .005)
        self.assertGreaterEqual(plan["bass_handoff_start_seconds"], 1.6)
        self.assertAlmostEqual(plan["bass_handoff_end_seconds"], 2.6)
        np.testing.assert_allclose(outgoing[-1], [0, 0], atol=1e-7)
        np.testing.assert_allclose(incoming[-1], bass[-1], atol=1e-7)

    def test_echo_lingers_after_handoff_then_returns_to_original_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            t = np.arange(engine.SR * 24) / engine.SR
            tracks = []
            for i, frequency in enumerate((330, 550)):
                audio = np.repeat((.08 * np.sin(2 * np.pi * frequency * t))[:, None], 2, axis=1)
                path = root / f"{i}.wav"
                sf.write(path, audio, engine.SR, subtype="PCM_24")
                tracks.append({"id": str(i), "title": str(i), "path": path})
            data = [analysis(24, bpm=0, confidence=0), analysis(24, bpm=0, confidence=0)]
            with patch.object(engine, "analyze", side_effect=data):
                result = engine.render_mix(tracks, "automix", root / "out", root / "cache")
            output, _ = sf.read(root / "out" / "mix.wav", dtype="float32")
            source = engine.decode(tracks[1]["path"])
            transition = result["transitions"][0]
            self.assertTrue(transition["echo"]["enabled"])
            self.assertEqual(transition["echo"]["incoming_timing"], "fixed")
            self.assertAlmostEqual(transition["effects_end"], transition["timeline_end"] + 6)
            self.assertFalse(transition["rhythm_loop"]["enabled"])
            attenuation = 10 ** (-result["gain_reduction_db"] / 20)
            start = round(transition["timeline_start"] * engine.SR)
            tail = slice(round((transition["timeline_end"] + .5) * engine.SR),
                         round((transition["timeline_end"] + 1.5) * engine.SR))
            dry_tail = source[tail.start - start:tail.stop - start] * attenuation
            self.assertGreater(float(np.sqrt(np.mean((output[tail] - dry_tail) ** 2))), .001)
            restored = round((transition["effects_end"] + .1) * engine.SR)
            np.testing.assert_allclose(output[restored:], source[restored - start:] * attenuation, atol=2e-6)
            self.assertLessEqual(float(np.max(np.abs(output))), 10 ** (-1 / 20) + 2e-7)

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

    def test_reverb_is_stereo_and_clears_before_vocals(self):
        outgoing = np.zeros((engine.SR * 4, 2), np.float32)
        outgoing[:engine.SR * 2] = .2 * np.sin(2 * np.pi * 440 * np.arange(engine.SR * 2) / engine.SR)[:, None]
        room = engine._reverb_send(outgoing, .1, 2.75)
        self.assertGreater(float(np.sqrt(np.mean(room[engine.SR:engine.SR * 2] ** 2))), .001)
        self.assertEqual(float(np.max(np.abs(room[round(2.75 * engine.SR):]))), 0)
        self.assertGreater(float(np.mean(np.abs(room[:, 0] - room[:, 1]))), 1e-4)

    def test_render_carries_outro_reverb_into_next_track(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            t = np.arange(engine.SR * 48) / engine.SR
            audio = (.12 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            tracks = []
            for i in range(2):
                path = root / f"{i}.wav"
                sf.write(path, audio, engine.SR)
                tracks.append({"id": str(i), "title": str(i), "path": path})
            a, b = analysis(48), analysis(48)
            a["energy"] = [.4] * (36 * 4) + [.03] * (12 * 4)
            a["mix_out_candidates"] = engine._mix_out_candidates(a["energy"], 48)
            with patch.object(engine, "analyze", side_effect=[a, b]):
                result = engine.render_mix(tracks, "automix", root / "wet", root / "cache")
            with patch.object(engine, "analyze", side_effect=[a, b]), \
                    patch.object(engine, "_reverb_send", side_effect=lambda samples, wet, clear_at:
                                 np.zeros((len(samples) + len(engine._reverb_ir()) - 1, 2), np.float32)):
                engine.render_mix(tracks, "automix", root / "dry", root / "cache")
            wet, _ = sf.read(root / "wet" / "mix.wav", dtype="float32")
            dry, _ = sf.read(root / "dry" / "mix.wav", dtype="float32")
            start = round(result["transitions"][0]["timeline_start"] * engine.SR)
            self.assertGreater(float(np.sqrt(np.mean((wet[start + engine.SR:start + 2 * engine.SR] -
                                                       dry[start + engine.SR:start + 2 * engine.SR]) ** 2))), .001)
            self.assertEqual(result["transitions"][0]["reverb_wet"], .1)

    def test_main_verse_skips_tag_and_arrives_inside_handoff(self):
        a, b = analysis(120), analysis(120)
        b["vocal"][4:8] = [1.] * 4
        b["vocal"][60:84] = [1.] * 24
        phrase = engine.plan_transition(a, b)
        self.assertAlmostEqual(phrase["vocal_entry"], 15, delta=.25)
        self.assertEqual(phrase["vocal_entry_source"], "automatic")
        self.assertAlmostEqual(phrase["incoming_vocal_at"], phrase["overlap_seconds"] * .5, delta=.5)
        self.assertGreater(phrase["incoming_cue"], 5)
        immediate = analysis(120)
        immediate["vocal"][:20] = [1.] * 20
        padded = engine.plan_transition(a, immediate)
        self.assertGreater(padded["incoming_delay_seconds"], 0)
        self.assertAlmostEqual(padded["incoming_vocal_at"], padded["overlap_seconds"] * .5, delta=.5)
        manual = engine.plan_transition(a, b, vocal_entry_override=50)
        self.assertEqual((manual["vocal_entry"], manual["vocal_entry_source"]), (50, "manual"))
        b["vocal_source"] = "DSP heuristic fallback"
        self.assertEqual(engine.plan_transition(a, b)["vocal_entry_source"], "fallback")

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
            data[1]["vocal"][:20] = [1.] * 20
            with patch.object(engine, "analyze", side_effect=data):
                result = engine.render_mix(tracks, "automix", root / "out", root / "cache")
            output, sr = sf.read(root / "out" / "mix.wav")
            self.assertEqual(sr, engine.SR)
            self.assertEqual(len(output), round(result["duration"] * sr))
            self.assertTrue(np.isfinite(output).all())
            self.assertLessEqual(float(np.max(np.abs(output))), 10 ** (-1 / 20) + 2e-7)
            first, second = result["transitions"]
            self.assertGreater(first["incoming_delay_seconds"], 0)
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
