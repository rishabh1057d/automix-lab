# AutoMix Lab: an explainable transition experiment

## The proposition

Improve the experience between songs by choosing entry/exit cues from the audio, managing vocal and bass competition, and applying small tempo corrections only when evidence supports them. Show the chosen cue and its rationale so the quality is inspectable.

Music platforms already offer transition features. The credible pitch is a measured approach to better handoffs, demonstrated with local licensed recordings.

## Demonstration

Play the same fixed queue in AutoMix and plain crossfade modes. Jump to either transition, inspect the incoming start time and overlap, and listen for continuity, conflicting voices and competing low frequencies. Reorder/import additional licensed material for broader evaluation; no recommendation model is claimed.

The demo catalog shares a nominal 100 BPM. That makes it a useful controlled example, but a weak standalone benchmark for diverse musical material. Lasting Hope has choir/synth textures; it does not establish reliable lead-singer onset detection.

## Engineering fit

Python is appropriate for model inference, experimentation and offline audio analysis.

A production deployment would retain Python for analysis/evaluation and implement sample-accurate rendering in its native playback engine. This prototype is a local workbench, not a drop-in streaming component.

The portable output contract contains cue times, overlap, playback-rate correction, bass handoff, confidence and reasons. A production handoff would add recording-version identity, calibrated model scores, decoder clock contracts, streaming buffering behavior and measured resource limits.

## Evidence required before a quality claim

Compare transition versions with a representative licensed evaluation set. Use blinded, loudness-matched listening tests across electronic, acoustic, vocal-led, tempo-changing, sparse and short tracks. Record preference, perceived discontinuity, phrase interruption and artifacts alongside objective beat error, clipping, latency, memory and fallback frequency.

This version has automated correctness checks and an interactive audition path. It does not claim universal beat correctness, accurate singer onset, proven listener preference, production real-time performance or a deployed streaming partnership.

## Integration boundary

This local workbench analyzes only recordings supplied to it. A future streaming integration would require appropriate platform and rightsholder authorization.

## Practical next step

After listening to this demo, prioritize the weak transition examples. Improve cue selection and vocal evidence against those examples, then measure whether the change improves listener preference. Keep the portable analysis/plan contract stable while replacing the prototype renderer with a streaming client implementation only when an integration partner needs it.
