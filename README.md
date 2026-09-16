# AutoMix Lab

A Python audio workbench for studying musical transitions. Import recordings you have permission to use, or try the included three-track licensed demo. Compare automatic cue selection, beat alignment, gentle tempo correction, bass handoff and EQ with a conventional equal-power crossfade.

## Start

Install/start Docker Desktop, then run in this directory:

```sh
docker compose up --build
```

Open **http://127.0.0.1:8765**. The first analysis downloads pinned Beat This! and UMX-HQ model weights. No GPU, Python installation, Spotify subscription, or account is needed for the demo.

1. Click **Build demo mix**. The app downloads three tracks from Kevin MacLeod's official catalog, checks their SHA-256 hashes, analyzes them and renders AutoMix plus a plain reference.
2. Click **Hear transition 01** or **02** to start four seconds before a handoff.
3. Switch **AutoMix / Plain crossfade** to compare the corresponding transition.
4. Inspect the incoming cue, overlap, tempo adjustment, waveform and reasoning below the player. Use **WAV** or **Mix plan** to export.
5. Use **Add audio** for 2–6 MP3/WAV/FLAC files in filename-selection order. Each file must be mono/stereo, 2 seconds–10 minutes, at most 100 MB, and sampled between 8 and 192 kHz.

The server binds only to loopback. Audio and model weights stay in `data/`, excluded from Git. Once the demo and models are cached, ordinary mixing works offline. Failed model downloads produce an explicitly labelled fallback and are retried on a later build. Rendered sessions survive restarts. A partially rendered job is marked interrupted on restart; rebuild it using the same files.

Stop with `docker compose stop`. Uploaded sources and exports accumulate in `data/`; this is a local workbench with manual retention.

## What it does

The engine scans the entire song at 250 ms resolution for energy and audible boundaries, including sustained late-song drops that can serve as an outro exit. Beat This! small0 measures beats/downbeats, while UMX-HQ estimates vocal activity from source-separated magnitude in each track's first and last 30 seconds. Unmeasured middle sections stay explicitly neutral; neither model invents evidence there.

The planner evaluates incoming and outgoing downbeat pairs and prefers the pair with the least model-detected vocal collision. A sustained final energy drop can move the exit earlier, but it may discard no more than 12 seconds of measured audible music, and the outgoing fade must finish at or after the detected outro. If vocals remain unavoidable, it can shorten the overlap and apply at most 6 dB of complementary vocal-band ducking. Model failure falls back to the old heuristic for cue ranking only. Both confident beat grids and an octave-normalized tempo difference within 4% permit beatmatching.

The renderer processes two tracks simultaneously during each overlap, using pitch-preserving WSOLA on the incoming region, a return to native tempo, a bass handoff and deliberate DJ-style timing. Song 1 fades from the first sample under a low-pass that closes from 5 kHz to 250 Hz. Song 2 is silent during the first half, then rises through its own low-pass, initially around 1.4 kHz and opening to full tone by the end. Both filters are intentionally audible; they shape the handoff and are separate from reverb. A detected, beat-supported outro also gets a short, 20%-wet stereo convolution reverb tail. Plain crossfade has neither filter sweep nor reverb. Completed WAVs have a -1 dBFS sample-peak ceiling. Outside the overlap and short reverb tail, playback uses the original decoded samples, subject to one global output gain.

The plain comparison preserves track beginnings and uses a six-second equal-power crossfade, shortened for tiny files.

This is **offline rendering with interactive playback**, not an extension that manipulates live streaming audio. The worker processes one mix at a time. The app accepts a maximum of three pending jobs.

## Spotify context

Spotify already provides [Automix](https://support.spotify.com/us/article/tracks-transitions/), [mixed playlists](https://support.spotify.com/us/article/mixed-playlists/), and selected [DJ integrations](https://support.spotify.com/us/article/dj-integration/). This prototype is an independent transition-quality experiment; it is not endorsed by Spotify or proof of superiority to those features.

Public [Spotify developer policy](https://developer.spotify.com/policy) restricts audio analysis and mixing. Authorized DJ partnerships are distinct from a general public Web API integration. This app never obtains, analyzes or mixes Spotify audio.

Optional now-playing metadata:

1. Create a Spotify developer app using your own account.
2. Register `http://127.0.0.1:8765/auth/spotify/callback` exactly.
3. Copy `.env.example` to `.env`, set `SPOTIFY_CLIENT_ID`, and restart with `docker compose up -d`.
4. Use **Connect Spotify** in the app. It requests only `user-read-currently-playing`.

OAuth uses PKCE and browser-bound state. Tokens are kept in server memory, never in client JavaScript or the repository. Restarting signs you out. Spotify account/app access restrictions still apply. Playback and rendering work without these credentials.

## Verification

```sh
docker compose run --rm automix python -m unittest discover -s tests -v
```

Tests cover audio boundaries, confidence fallbacks, cue budgets, pitch preservation, source/timeline continuity across three tracks, silence, clipping, uploaded file validation, range requests and recovered comparison sources. They use synthetic fixtures, not downloaded music. Real-track and browser results are recorded separately in [VERIFICATION.md](VERIFICATION.md).

## Scope and limits

- Queue order is supplied by you; there is no recommendation engine.
- UMX-HQ supplies source-separation evidence, not a calibrated singing probability or lyric-onset transcript. Separation can still leak instruments or miss quiet vocals; the UI labels model, heuristic, and unavailable evidence.
- A beat grid can still be wrong despite high confidence, especially with tempo changes or weak percussion. The head/tail model is tuned for this small demonstration.
- The current planner aligns downbeats and local energy changes; it does not infer full musical phrases or harmonic key.
- WSOLA is suitable for small corrections but can still produce artifacts. Listen to the output before making quality claims.
- The output ceiling measures sample peaks, not intersample true peaks or perceived loudness. A/B exports are not a blinded loudness-matched listening study.
- Memory use grows with track length because this prototype decodes and renders into arrays. Use a small queue; chunked rendering is the upgrade for large libraries.
- Source code was independently written after studying BitChord's behavior; it is not a line-by-line port or a formal legal clean-room process.

See [PITCH.md](PITCH.md) for the engineering proposal and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for code/model/music provenance.
