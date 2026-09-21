# Third-party notices and provenance

## Runtime dependencies

- Python: PSF license.
- FastAPI, Uvicorn, HTTPX and Beat This!: MIT.
- NumPy and SciPy: BSD-3-Clause.
- SoundFile: BSD-3-Clause; bundled libsndfile and codec libraries carry their own notices, including LGPL. They remain dynamically linked in the Docker environment. See [SoundFile licensing](https://github.com/bastibe/python-soundfile/tree/master/licensing).
- AudioTSM: MIT. Copyright Muges and contributors. [Upstream](https://github.com/Muges/audiotsm).
- ONNX Runtime: MIT. The CPU package executes the vocal model locally.
- PyTorch and torchaudio: BSD-style, with their upstream third-party notices retained in their installed distributions.
- Additional transitive dependencies retain their package metadata/license files in the Docker image.

A deployment/distribution license audit must include the installed packages' bundled native components. This list does not replace their original license texts.

## Beat model

[Beat This!](https://github.com/CPJKU/beat_this), Foscarin, Schlüter and Widmer, ISMIR 2024. Code and published weights are MIT licensed by the Institute of Computational Perception, JKU Linz, Austria. This project uses the upstream `small0` checkpoint downloaded by the official package. No model weights are in Git. The upstream authors note that training-data rights can require separate assessment.

## Vocal model

[Open-Unmix UMX-HQ](https://zenodo.org/records/3370489) weights are identified as MIT. The app downloads the `vocals.onnx` export from the `edgetools/umx-hq` Hugging Face repository at pinned revision `87f26a5b2ed19488b5e453059a0a1d530423fe4a`, verifies SHA-256 `3d05709ff7197bbd4a33aee759e4e82002acb491f41c94770227ab57507b6ccb`, and does not commit model weights to Git. UMX-HQ estimates are labelled as model evidence, not calibrated singing probabilities.

User-provided recordings remain the user's responsibility to license for the intended processing and distribution.
