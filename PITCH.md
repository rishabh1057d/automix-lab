# AutoMix Lab: an explainable transition experiment

## The proposition

Improve the experience between songs by choosing entry/exit cues from the audio, managing vocal and bass competition, and applying small tempo corrections only when evidence supports them. Show the chosen cue and its rationale so the quality is inspectable.

Spotify already offers Automix and mixed playlists. The credible pitch is a measured approach to better handoffs, not inventing automated mixing or asserting that Spotify lacks it. This prototype demonstrates a subset with local licensed recordings.

## Demonstration

Play the same fixed queue in AutoMix and plain crossfade modes. Jump to either transition, inspect the incoming start time and overlap, and listen for continuity, conflicting voices and competing low frequencies. Reorder/import additional licensed material for broader evaluation; no recommendation model is claimed.

The demo catalog shares a nominal 100 BPM. That makes it a useful controlled example, but a weak standalone benchmark for diverse musical material. Lasting Hope has choir/synth textures; it does not establish reliable lead-singer onset detection.

## Engineering fit

Python is appropriate for model inference, experimentation and offline audio analysis. Spotify's public [audio research infrastructure article](https://engineering.atspotify.com/2020/11/its-all-just-wiggly-air-building-infrastructure-to-support-audio-research) describes Python alongside native media libraries; its [ML platform article](https://engineering.atspotify.com/2023/2/unleashing-ml-innovation-at-spotify-with-ray) also emphasizes Python research workflows.

Spotify has multiple clients and languages. Its [web-player architecture article](https://engineering.atspotify.com/2019/03/building-spotifys-new-web-player) describes React/Redux and desktop web technology historically, not a guarantee of today's complete stack. Our proposed first-party deployment would retain Python for analysis/evaluation and implement sample-accurate rendering in the existing native playback engine. This prototype does not know Spotify's private APIs and is not a drop-in component.

The portable output contract contains cue times, overlap, playback-rate correction, bass handoff, confidence and reasons. A production handoff would add recording-version identity, calibrated model scores, decoder clock contracts, streaming buffering behavior and measured resource limits.

## Evidence required before a quality claim

Compare with Spotify's existing features using authorized playback conditions and a representative licensed evaluation set. Use blinded, loudness-matched listening tests across electronic, acoustic, vocal-led, tempo-changing, sparse and short tracks. Record preference, perceived discontinuity, phrase interruption and artifacts alongside objective beat error, clipping, latency, memory and fallback frequency.

This version has automated correctness checks and an interactive audition path. It does not claim universal beat correctness, accurate singer onset, proven listener preference, production real-time performance or a deployed Spotify partnership.

## Integration boundary

Public [developer policy](https://developer.spotify.com/policy) restricts mixing and analysis of Spotify content. Spotify's selected [DJ integrations](https://support.spotify.com/us/article/dj-integration/) demonstrate that separately authorized partnerships exist. A future integration requires appropriate Spotify and rightsholder authorization; adding Web API credentials to this app only enables now-playing metadata.

## Practical next step

After listening to this demo, prioritize the weak transition examples. Improve cue selection and vocal evidence against those examples, then measure whether the change improves listener preference. Keep the portable analysis/plan contract stable while replacing the prototype renderer with a streaming client implementation only when an integration partner needs it.
