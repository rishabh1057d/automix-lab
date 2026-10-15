import unittest

import numpy as np

from transition_effects import SR, limit_effect_peak, local_pulse, progressive_reverb, render_rhythm_loop


def evidence(duration=40, pulse=.5):
    return {"beats": np.arange(0, duration + pulse / 2, pulse).tolist(),
            "downbeats": np.arange(0, duration + .01, 4 * pulse).tolist(),
            "vocal_source": "test-model", "vocal_windows": [[0, duration]],
            "vocal": [0.] * (duration * 4), "energy": [.1] * (duration * 4)}


def percussion(duration=40):
    audio = np.zeros((round(duration * SR), 2), dtype=np.float32)
    rng = np.random.default_rng(14)
    burst = rng.normal(0, .2, (round(.04 * SR), 2)).astype(np.float32)
    burst *= np.exp(-np.linspace(0, 7, len(burst)))[:, None]
    for time in np.arange(0, duration, .5):
        start = round(time * SR)
        audio[start:start + len(burst)] += burst[:min(len(burst), len(audio) - start)]
    return audio


class TransitionEffectsTests(unittest.TestCase):
    def test_local_limiter_bounds_bursts_preserves_stereo_and_recovers(self):
        time = np.arange(3 * SR) / SR
        baseline = .18 * np.sin(2 * np.pi * 440 * time)
        burst = np.zeros(len(time))
        for start in (.6, 1., 1.4, 1.8):
            local = time - start
            burst += np.where((local >= 0) & (local < .16), 2.4 * np.cos(2 * np.pi * 900 * local), 0)
        audio = np.stack((baseline + burst, .4 * (baseline + burst)), axis=1).astype(np.float32)
        original = audio.copy()
        limited, reduction = limit_effect_peak(audio, .89)
        self.assertTrue(np.array_equal(audio, original))
        self.assertLessEqual(float(np.max(np.abs(limited))), .89)
        self.assertTrue(np.isfinite(limited).all())
        self.assertGreater(reduction, 6)
        np.testing.assert_allclose(limited[:, 1], .4 * limited[:, 0], atol=1e-7)
        np.testing.assert_array_equal(limited[:round(.5 * SR)], audio[:round(.5 * SR)])
        np.testing.assert_array_equal(limited[-round(.1 * SR):], audio[-round(.1 * SR):])
        self.assertEqual(limited.dtype, np.float32)

    def test_local_limiter_leaves_sub_ceiling_silence_and_tiny_audio_unchanged(self):
        rng = np.random.default_rng(3)
        for audio in (np.zeros((0, 2), dtype=np.float32), np.zeros((1, 2), dtype=np.float32),
                      rng.uniform(-.4, .4, (1024, 2)).astype(np.float32)):
            limited, reduction = limit_effect_peak(audio, .89)
            np.testing.assert_array_equal(limited, audio)
            self.assertEqual(reduction, 0)
        limited, reduction = limit_effect_peak(np.array([[2., -1.]], dtype=np.float32), .89)
        self.assertLessEqual(float(np.max(np.abs(limited))), .89)
        self.assertAlmostEqual(float(limited[0, 1] / limited[0, 0]), -.5)
        self.assertGreater(reduction, 6)

    def test_measured_local_pulse_and_rate(self):
        measured = evidence()
        self.assertEqual(local_pulse(measured, 2, 4), .5)
        self.assertAlmostEqual(local_pulse(measured, 2, 4, 1.1), .5 / 1.1)
        self.assertIsNone(local_pulse(measured, 2, 3.5))
        self.assertIsNone(local_pulse({"beats": [0, .3, .9, 1.1, 2, 3]}, 0, 3))
        self.assertIsNone(local_pulse({"bpm": 120}, 0, 20))

    def test_reverb_impulses_decay_progressively_longer_without_discrete_repeats(self):
        ratios = []
        for time in (.5, 4, 7.5):
            send = np.zeros((8 * SR, 2), dtype=np.float32)
            send[round(time * SR)] = 1
            wet = progressive_reverb(send)
            onset = round((time + .03) * SR)
            first = wet[onset:onset + round(.25 * SR)]
            later = wet[onset + round(.5 * SR):onset + SR]
            ratios.append(float(np.sum(later ** 2) / np.sum(first ** 2)))
            self.assertLess(float(np.max(np.abs(wet[:onset]))), 1e-7)
            self.assertGreater(np.mean(np.abs(first) > 1e-7), .99)
            self.assertLess(abs(float(np.corrcoef(first.T)[0, 1])), .1)
            self.assertEqual(wet.shape, (14 * SR, 2))
            np.testing.assert_array_equal(wet[-1], [0, 0])
            self.assertEqual(wet.dtype, np.float32)
        self.assertGreater(ratios[1], 3 * ratios[0])
        self.assertGreater(ratios[2], 3 * ratios[1])

    def test_reverb_preserves_source_has_finite_tail_and_deterministic_output(self):
        time = np.arange(SR) / SR
        send = np.repeat((.1 * np.cos(2 * np.pi * 1000 * time))[:, None], 2, axis=1).astype(np.float32)
        original = send.copy()
        wet = progressive_reverb(send)
        np.testing.assert_array_equal(send, original)
        np.testing.assert_array_equal(wet, progressive_reverb(send))
        self.assertTrue(np.isfinite(wet).all())
        self.assertGreater(float(np.sum(wet[SR:2 * SR] ** 2)), 0)
        np.testing.assert_array_equal(wet[-1], [0, 0])
        self.assertLess(float(np.max(np.abs(wet[-round(.01 * SR):]))), 1e-6)
        endpoints = np.zeros_like(send)
        endpoints[[0, -1]] = 1
        self.assertFalse(np.any(progressive_reverb(endpoints)))

    def test_reverb_empty_tiny_silent_and_zero_wet_inputs(self):
        for n in (0, 1, 12, SR):
            for value in (0, .1) if n < SR else (0,):
                send = np.full((n, 2), value, dtype=np.float32)
                wet = progressive_reverb(send)
                self.assertEqual(wet.shape, (n + 6 * SR, 2))
                self.assertTrue(np.isfinite(wet).all())
                np.testing.assert_array_equal(wet[-1], [0, 0])
                if not value:
                    self.assertFalse(np.any(wet))
        send = np.ones((100, 2), dtype=np.float32)
        self.assertFalse(np.any(progressive_reverb(send, wet=0)))
        self.assertEqual(progressive_reverb(send, tail_seconds=0).shape, send.shape)

    def test_reverb_rejects_nonfinite_mono_and_invalid_parameters(self):
        for send in (np.zeros(SR), np.zeros((10, 1)), np.zeros((2, 2, 2)),
                     np.array([[np.nan, 0]]), np.array([[0, np.inf]])):
            with self.subTest(shape=send.shape), self.assertRaises(ValueError):
                progressive_reverb(send)
        send = np.zeros((1, 2), dtype=np.float32)
        for parameters in ({"wet": -.1}, {"wet": 1.1}, {"wet": np.nan},
                           {"decay_start_seconds": 0}, {"decay_start_seconds": 4},
                           {"decay_end_seconds": 7}, {"decay_end_seconds": np.inf},
                           {"tail_seconds": -1}, {"tail_seconds": 7}, {"tail_seconds": np.nan}):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                progressive_reverb(send, **parameters)

    def test_instrumental_bar_is_aligned_and_stretched_with_exact_clearance(self):
        a, b = evidence(), evidence(pulse=.525)
        plan = {"tier": "beatmatched", "overlap_seconds": 12, "outgoing_start": 20,
                "outgoing_end": 32, "incoming_cue": 0, "incoming_playback_rate": 1,
                "incoming_delay_seconds": 0, "incoming_fade_start": .25}
        output = render_rhythm_loop(percussion(), a, b, plan)
        details = plan["rhythm_loop"]
        self.assertTrue(details["enabled"], details)
        self.assertEqual((details["source_start"], details["source_end"]), (30, 32))
        self.assertAlmostEqual(details["playback_rate"], 2 / 2.1, places=5)
        self.assertAlmostEqual(details["entry_seconds"], 4.2)
        self.assertFalse(np.any(output[:round(4.2 * SR)]))
        self.assertGreater(float(np.sqrt(np.mean(output[5 * SR:9 * SR] ** 2))), .005)
        self.assertTrue(np.array_equal(output[-1], [0, 0]))
        self.assertEqual(output.shape, (12 * SR, 2))

    def test_vocal_uncovered_heuristic_and_incompatible_bars_are_rejected(self):
        a, b = evidence(), evidence()
        plan = {"tier": "safe-crossfade", "overlap_seconds": 12, "outgoing_end": 32,
                "incoming_cue": 0, "incoming_playback_rate": 1}
        audio = percussion()
        for change in ({"vocal": [1.] * 160}, {"vocal_windows": [[0, 1]]},
                       {"vocal_source": "DSP heuristic fallback"}):
            with self.subTest(change=next(iter(change))):
                self.assertFalse(np.any(render_rhythm_loop(audio, dict(a, **change), b, plan)))
                self.assertFalse(plan["rhythm_loop"]["enabled"])
        self.assertFalse(np.any(render_rhythm_loop(audio, a, evidence(pulse=.7), plan)))
        self.assertIn("12%", plan["rhythm_loop"]["reason"])
        self.assertFalse(np.any(render_rhythm_loop(np.zeros_like(audio), a, b, plan)))
        self.assertIn("percussion", plan["rhythm_loop"]["reason"])
        time = np.arange(len(audio)) / SR
        pitched = np.repeat((.1 * np.sin(2 * np.pi * 440 * time))[:, None], 2, axis=1).astype(np.float32)
        self.assertFalse(np.any(render_rhythm_loop(pitched, a, b, plan)))
        self.assertIn("percussion", plan["rhythm_loop"]["reason"])


if __name__ == "__main__":
    unittest.main()
