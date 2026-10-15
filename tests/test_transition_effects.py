import unittest

import numpy as np

from transition_effects import SR, limit_effect_peak, local_pulse, ping_pong_echo, render_rhythm_loop


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

    def test_echo_spacing_stereo_feedback_and_exact_tail(self):
        send = np.zeros((SR, 2), dtype=np.float32)
        send[round(.1 * SR)] = 1
        wet = ping_pong_echo(send, .25, feedback=.65)
        self.assertEqual(wet.shape, (7 * SR, 2))
        energies = []
        for repeat, time in enumerate((.35, .6, .85)):
            chunk = wet[round(time * SR):round((time + .02) * SR)]
            energies.append(np.sum(chunk ** 2, axis=0))
            dominant, quiet = (0, 1) if repeat % 2 == 0 else (1, 0)
            self.assertGreater(energies[-1][dominant], energies[-1][quiet] * 15)
        self.assertAlmostEqual(float(energies[1].sum() / energies[0].sum()), .65 ** 2, places=3)
        self.assertLess(float(np.max(np.abs(wet[:round(.35 * SR)]))), 1e-7)
        self.assertTrue(np.any(wet[SR:2 * SR]))
        self.assertTrue(np.array_equal(wet[-1], [0, 0]))
        self.assertEqual(wet.dtype, np.float32)

    def test_history_seeds_early_echo_and_silence_is_finite(self):
        history = np.zeros((SR, 2), dtype=np.float32)
        history[-round(.15 * SR)] = 1
        send = np.zeros((SR, 2), dtype=np.float32)
        seeded = ping_pong_echo(send, .25, history)
        self.assertGreater(float(np.max(np.abs(seeded[round(.09 * SR):round(.13 * SR)]))), .01)
        self.assertFalse(np.any(ping_pong_echo(send, .25)))
        self.assertTrue(np.isfinite(ping_pong_echo(np.zeros((1, 2), dtype=np.float32), .25)).all())
        self.assertEqual(ping_pong_echo(np.empty((0, 2), dtype=np.float32), .25).shape, (6 * SR, 2))

    def test_echo_send_cutoff_is_tapered_without_changing_source(self):
        time = np.arange(SR) / SR
        send = np.repeat((.1 * np.cos(2 * np.pi * 1000 * time))[:, None], 2, axis=1).astype(np.float32)
        original = send.copy()
        wet = ping_pong_echo(send, 1.5, wet=.45, feedback=0)
        self.assertTrue(np.array_equal(send, original))
        self.assertLess(float(np.max(np.abs(wet[round(2.499 * SR):round(2.5 * SR)]))), .003)
        self.assertGreater(float(np.max(np.abs(wet[round(2.46 * SR):round(2.47 * SR)]))), .02)

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
