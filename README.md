# AutoMix Lab

A Python audio workbench for studying musical transitions. Import recordings you have permission to use, or try the included three-track licensed demo. Compare automatic cue selection, beat alignment, gentle tempo correction, bass handoff and EQ with a conventional equal-power crossfade.

## Start

Install/start Docker Desktop, then run in this directory:

```sh
docker compose up --build
```

Open **http://127.0.0.1:8765**. The first build downloads CPU PyTorch and dependencies. No GPU, Python installation, Spotify subscription, or account is needed for the demo.

1. Click **Build demo mix**. The app downloads three tracks from Kevin MacLeod's official catalog, checks their SHA-256 hashes, analyzes them and renders AutoMix plus a plain reference.
2. Click **Hear transition 01** or **02** to start four seconds before a handoff.
3. Switch **AutoMix / Plain crossfade** to compare the corresponding transition.
4. Inspect the incoming cue, overlap, tempo adjustment, waveform and reasoning below the player. Use **WAV** or **Mix plan** to export.
5. Use **Add audio** for 2–6 MP3/WAV/FLAC files in filename-selection order. Each file must be mono/stereo, 2 seconds–10 minutes, at most 100 MB, and sampled between 8 and 192 kHz.

The server binds only to loopback. Audio and model weights stay in `data/`, excluded from Git. Once the demo and model are cached, ordinary mixing works offline. Failed model downloads produce an explicit safe fallback and are retried on a later build. Rendered sessions survive restarts. A partially rendered job is marked interrupted on restart; rebuild it using the same files.

Stop with `docker compose stop`. Uploaded sources and exports accumulate in `data/`; this is a local workbench with manual retention.

## What it does

The engine measures a 250 ms energy envelope, audible boundaries and an approximate voice-like activity curve. Beat This! small0 measures beats/downbeats in the head and tail. It does not invent a beat grid for the unmeasured middle.

The planner ranks entry downbeats by energy rise, voice-like activity and a 12-second audible-intro budget. It aligns the outgoing fade near the audible ending. Both confident grids and an octave-normalized tempo difference within 4% permit beatmatching. Otherwise it chooses DJ-assisted or safe crossfade.

The renderer processes two tracks simultaneously during each overlap, using pitch-preserving WSOLA on the incoming region, a return to native tempo, equal-power volume curves, complementary filter sweeps and a bass handoff. Completed WAVs have a -1 dBFS sample-peak ceiling. Ordinary playback outside transition windows uses the original decoded samples, subject to one global output gain.

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
- Voice-like activity is a **DSP heuristic**, not source separation or reliable singer-onset detection. Centered melodic instruments can trigger it.
- A beat grid can still be wrong despite high confidence, especially with tempo changes or weak percussion. The head/tail model is tuned for this small demonstration.
- The current planner aligns downbeats and local energy changes; it does not infer full musical phrases or harmonic key.
- WSOLA is suitable for small corrections but can still produce artifacts. Listen to the output before making quality claims.
- The output ceiling measures sample peaks, not intersample true peaks or perceived loudness. A/B exports are not a blinded loudness-matched listening study.
- Memory use grows with track length because this prototype decodes and renders into arrays. Use a small queue; chunked rendering is the upgrade for large libraries.
- Source code was independently written after studying BitChord's behavior; it is not a line-by-line port or a formal legal clean-room process.

See [PITCH.md](PITCH.md) for the engineering proposal and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for code/model/music provenance.
