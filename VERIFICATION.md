# Verification — 17 September 2026

Status: working local prototype, with the integration and quality limits below. The delayed-entry, dual-low-pass handoff has passed final review.

## Automated checks

Executed in the Docker image:

```sh
docker compose run --rm automix python -m unittest discover -s tests -v
```

24 tests passed (17 engine, 6 API, 1 vocal-model), final run 3.303 seconds. `node --check static/app.js` also passed. No line-coverage percentage was measured.

| Acceptance criterion | Implementation | Evidence |
| --- | --- | --- |
| Download and mix licensed demo audio | Manifest checksums, download job, Python renderer | Three official downloads and two completed full-queue exports |
| Choose cues and safe transition tiers | `engine.py` analysis/planner | Tempo/cue-budget unit tests and measured transitions below |
| Preserve pitch during small tempo corrections | WSOLA intro and native-tempo return | 440 Hz fixtures at 0.97 and 1.03 rates, within 1 Hz; source-resume continuity assertions |
| Render a continuous three-track timeline | Source/output position mapping | Three-track mapping and short-audible-middle regression tests |
| Bound output peaks and handle silence | Global output gain, conservative fallback | Unit tests plus full real WAV scan |
| Audition and compare transitions in browser | Audio player, audition buttons, handoff-relative A/B positioning | Both audition buttons, advancing playback, pause and mode switching exercised |
| Validate uploads and recover sessions | API boundary, content hashes, persisted source lists | Invalid upload, byte limit, cross-origin, range and restored-source tests |
| Reproducible local deployment | Dockerfile and Compose | Clean-copy startup and original-session restart passed |
| Model-backed vocal-aware transitions | Pinned UMX-HQ ONNX, pair scoring, guarded ducking | Model contract/checksum check, synthetic collision tests, and real two-track render below |
| Early structural outro and reverb | Whole-song 250 ms energy scan, audible-music budget, stereo convolution tail | Fade/rebound fixtures, rendered-tail comparison, two real queues, and browser audition below |

## Early-outro and reverb verification

For the reported *Swim* → *Timeless* miss, old AutoMix `6801a42b67ac43679a35b3d512eb6acf` waited until 219.09s and had no reverb. Final export `d52c7b2ef6344f6bbad9b54bd8050b53` finds the sustained drop at 209s, starts the overlap at 212.067s, exits at 220.067s, discards 4.0s of measured audible music, and applies the 20%-wet stereo room tail. It cues *Timeless* at 2.0s with an 8.0s overlap and 2.5 dB of model-gated vocal ducking. The 20,553,540 stereo samples are finite with a -1.0 dBFS peak. A synthetic signal-path test verifies the new handoff: Song 1 begins fading and loses 4 kHz from the start; Song 2 remains silent before halfway, enters through a low-pass, and recovers its high-frequency tone near the end. Plain crossfade bypasses these effects.

Browser audition advanced beyond the Swim handoff to 236.2s with ready state 4 and no media error. After the final container restart, the player restored the same export and was left paused at 208.4s, four seconds before the overlap, for listening. The filter and reverb signal paths are measured, but whether this cue sounds better still needs the user's ears.

Review found that a later 1.75s model-active phrase could still be skipped. A new regression test rejects a later run of at least 1s, while allowing at most 1.25s of activity immediately at the exit as model-tail decay. Cached real-plan checks after the correction keep Swim at 212.07–220.07s, Weeknd at 221.17–229.49s, and Masakali at 274.91–284.51s, each with its structural reverb.

A second review found that the end-region fallback bypassed the vocal guard. Fallback cue selection and overlap-shortening now use the same safety check; the regression asserts that a later vocal-active section is retained through the actual outgoing end.

On the user-provided Weeknd pair, the full-song scan detected a sustained final energy decline at 227.25 seconds in *I Was Never There*. Export `5107146f029e4b4aa6106cefb04bbe25` starts the outgoing overlap at 221.1667 seconds (19.90 seconds before the 241.0667-second file end), finishes it at 229.4870 seconds, and skips 7.0 seconds of measured audible music under the same threshold used for audible boundaries. The 8.3203-second overlap has 2.94 dB of vocal-band ducking and a 20%-wet, 1.4-second stereo reverb tail. The earlier measured tail RMS in the first second after the outgoing track stops was 11.61 dB below the incoming audio RMS. The rendered 21,042,168 stereo samples are all finite, and the sample peak is -1.0 dBFS.

The three-track licensed demo (`6975871aec364fc5a9810daa313bb062`) kept Style Funk's first transition at its content end with no added reverb. Its second transition found Leopard Print Elevator's final decline, mixed out after that onset, and added the reverb tail. This demonstrates that the effect is conditional rather than applied to every transition.

The browser restored final user-pair mix `d4103efbc49a4692930ac2dd5837cd90` and showed `Outro exit at 229.5s`, `7.0s audible music skipped`, `Stereo reverb 20%`, and `Vocal duck 2.9 dB`. Audition playback advanced through the handoff with ready state 4 and no media error. It was left paused four seconds before that transition for user testing. Listening quality remains subjective and requires the user's ears; these checks prove the processing and playback paths, not superiority over BitChord.

## Vocal-model verification

The pinned `vocals.onnx` file was downloaded at 17,820,856 bytes and matched SHA-256 `3d05709ff7197bbd4a33aee759e4e82002acb491f41c94770227ab57507b6ccb`. ONNX Runtime reported float32 input/output tensors shaped `[1, 2, 2049, frames]`. Both user-provided tracks completed warm head/tail vocal analysis in 1.812 and 1.819 seconds respectively on CPU, below the 60-second per-track budget.

AutoMix export `0ef54e3042194552a742cd9d8adef103` used UMX-HQ evidence for both tracks. It selected the measured pair at 231.1667 seconds outgoing and 0.02 seconds incoming with a 4.1602-second overlap. Vocal collision was 0%, so the balanced policy correctly reported `vocal_duck_db: 0` rather than applying an unnecessary effect. The cached repeat render produced 21,483,168 finite stereo frames at 44.1 kHz with a -1.0 dBFS sample peak. Browser audition advanced through the handoff with ready state 4 and no media error; the inspector showed `UMX-HQ model` and `Vocal duck not needed`.

Synthetic tests separately force avoidable and unavoidable vocal collisions. They verify clean-cue selection, downbeat-preserving overlap shortening, the 6 dB cap, measurable mid-band attenuation, and that DSP fallback, unmeasured windows, and safe-crossfade tiers cannot claim ducking. This is deterministic signal-path evidence, not a claim that the supplied song pair needed ducking.

## Real audio measurements

AutoMix export `3fc18ceb2c1a467492e779828c568116`:

- 32,914,217 stereo frames at 44,100 Hz: 746.354127 seconds.
- All decoded output samples finite; sample count matches reported duration.
- Measured sample peak approximately -1.000000 dBFS, within PCM quantization tolerance.
- All three track analyses reused the local cache on the repeat build.

| Handoff | Tier | Incoming source cue | Overlap | Mix timeline start |
| --- | --- | --- | --- | --- |
| Style Funk → Leopard Print Elevator | Beatmatched | 2.44 s | 4.80 s | 300.052245 s |
| Leopard Print Elevator → Lasting Hope | DJ-assisted | 9.62 s | 4.80 s | 612.046122 s |

Both real-demo rates were 1.0; non-unity stretching is covered by synthetic tests, not this musical example. Lasting Hope's head/tail tempo disagreement correctly prevented stretching. Its low-confidence estimate is not a claim of ground-truth tempo.

Browser playback advanced from 608.054 to 628.894 seconds during the second audition, with no media error. The first audition also started and advanced, and the plain reference played after mode switching. Session data and both exports recovered after a server restart. The desktop view was visually inspected; responsive-device and accessibility audits were not completed.

## Deployment check

Copied source/configuration to an isolated directory with no `data/`, copied `.env.example` to `.env`, then ran `docker compose -p automix-clean up --build -d --wait`. The service became healthy and the browser showed the empty, build-ready studio. `/api/latest` returned null for both modes, confirming no accidental dependency on the existing demo data. Docker dependency layers were cached; this was not a second uncached dependency-download test.

Stopped that verification container and restarted the original project with `docker compose up -d --wait`. The original demo data was preserved. The stack remains available at http://127.0.0.1:8765.

## Limits and outstanding verification

- Not a streaming playback extension or first-party integration.
- UMX-HQ activity is source-separation evidence, not a calibrated singing probability or lyric-onset transcript. Queue order is user-supplied; recommendations are not implemented.
- Automated correctness and browser playback do not prove musical quality. No blinded listening study, true-peak/loudness analysis, load test or streaming-client benchmark was performed.
- GitHub issue #1 tracks the prototype. No separate defect issues were open at handoff. Earlier review findings were addressed, but the final independent re-review attempt failed because its agent reached a usage limit. No final approval is claimed and no merge was performed.

See the README for setup and the pitch for the authorization and evaluation work needed before presenting this as a production integration.
