import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import vocal_model


class VocalModelTests(unittest.TestCase):
    def test_energy_ratio_and_unmeasured_middle(self):
        mix = np.ones((1, 2, 2049, 50), np.float32)
        activity = vocal_model._reduce_activity(mix, mix * .8)
        self.assertAlmostEqual(float(activity[25]), .8, places=4)

        class Session:
            def run(self, *_):
                return [mix * .8]

        audio = np.zeros((vocal_model.SR * 61, 2), np.float32)
        with patch.object(vocal_model, "ensure_model", return_value=Path("vocals.onnx")), \
                patch.object(vocal_model, "_session", return_value=(Session(), "mag", "estimate")), \
                patch.object(vocal_model, "_magnitude", return_value=mix):
            curve, windows, source, warning = vocal_model.activity_curve(audio, Path("models"), 244)
        self.assertEqual(windows, [[0.0, 30.0], [31.0, 61.0]])
        self.assertAlmostEqual(curve[2], .8, places=4)
        self.assertEqual(curve[122], .5)
        self.assertEqual((source, warning), ("UMX-HQ ONNX", None))


if __name__ == "__main__":
    unittest.main()
