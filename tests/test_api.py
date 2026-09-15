"""API boundaries and lifecycle tested without Spotify credentials or model downloads."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

import app as server


def wav(seconds=2.1, channels=1):
    stream = io.BytesIO()
    sf.write(stream, np.zeros((int(seconds * 8000), channels)), 8000, format="WAV")
    return stream.getvalue()


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.patch = patch.object(server, "DATA", Path(self.temp.name))
        self.patch.start()
        server.JOBS.clear()
        self.client = TestClient(server.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.patch.stop()
        self.temp.cleanup()

    def test_demo_and_optional_spotify_without_credentials(self):
        self.assertEqual(len(self.client.get("/api/demo-tracks").json()["tracks"]), 3)
        self.assertEqual(self.client.get("/api/latest").json(), {"automix": None, "plain": None})
        self.assertFalse(self.client.get("/api/spotify/now-playing").json()["connected"])
        self.assertEqual(self.client.get("/auth/spotify/callback?state=bad&code=x").status_code, 400)

    def test_reject_bad_input_without_scheduling(self):
        cases = [
            ({"mode": "wrong", "demo": "true"}, None, 400),
            ({"demo": "true", "track_ids": '["made-up", "style-funk"]'}, None, 400),
            ({}, [("files", ("../escape.wav", wav(), "audio/wav")),
                   ("files", ("fine.wav", wav(), "audio/wav"))], 400),
            ({}, [("files", ("bad.wav", b"not audio", "audio/wav")),
                   ("files", ("fine.wav", wav(), "audio/wav"))], 400),
            ({}, [("files", ("tiny.wav", wav(.1), "audio/wav")),
                   ("files", ("fine.wav", wav(), "audio/wav"))], 400),
            ({}, [("files", ("surround.wav", wav(channels=3), "audio/wav")),
                   ("files", ("fine.wav", wav(), "audio/wav"))], 400),
        ]
        for data, files, status in cases:
            with self.subTest(data=data, files=bool(files)):
                self.assertEqual(self.client.post("/api/mixes", data=data, files=files).status_code, status)
        self.assertEqual(server.JOBS, {})

    def test_limits_and_cross_origin_requests(self):
        self.assertEqual(self.client.post("/api/mixes", headers={"Content-Length": str(server.MAX_BODY + 1)}).status_code, 413)
        self.assertEqual(self.client.post("/api/demo-tracks/download", headers={"Origin": "https://unrelated.example"}).status_code, 403)
        self.assertEqual(self.client.get("/api/health", headers={"Host": "untrusted.example"}).status_code, 403)
        with patch.object(server, "MAX_BODY", 4):
            self.assertEqual(self.client.post("/api/mixes", content=iter([b"a=1", b"234"]),
                                             headers={"Content-Type": "application/x-www-form-urlencoded"}).status_code, 413)

    def test_valid_upload_and_baseline_use_content_identity(self):
        audio = wav()
        with patch.object(server, "new_job", return_value={"id": "test"}) as schedule:
            result = self.client.post("/api/mixes", data={"mode": "plain"},
                                     files=[("files", ("a.wav", audio, "audio/wav")),
                                            ("files", ("b.wav", audio, "audio/wav"))])
        self.assertEqual(result.status_code, 200)
        self.assertTrue(schedule.called)
        self.assertEqual(len(list((server.DATA / "uploads").glob("*.wav"))), 1)

    def test_audio_byte_ranges_and_result_recovery(self):
        ident = "a" * 32
        folder = server.DATA / "mixes" / ident
        folder.mkdir()
        result = {"id": ident, "mode": "automix", "created": 10, "playlist_key": "x"}
        (folder / "result.json").write_text(json.dumps(result))
        audio = wav()
        (folder / "mix.wav").write_bytes(audio)
        response = self.client.get(f"/api/mixes/{ident}/audio", headers={"Range": "bytes=0-43"})
        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.content, audio[:44])
        self.assertEqual(self.client.get("/api/latest").json()["automix"]["id"], ident)
        self.assertEqual(self.client.get("/api/jobs/not-a-valid-id").status_code, 404)

    def test_recovered_baseline_reuses_original_uploads(self):
        ident = "b" * 32
        folder = server.DATA / "mixes" / ident
        folder.mkdir()
        (folder / "result.json").write_text(json.dumps({"id": ident, "mode": "automix"}))
        source = server.DATA / "uploads" / "owned.wav"
        source.write_bytes(wav())
        tracks = [{"id": "owned", "title": "My recording", "path": str(source)}] * 2
        (folder / "sources.json").write_text(json.dumps(tracks))
        with patch.object(server, "new_job", return_value={"id": "pending"}) as schedule:
            response = self.client.post("/api/mixes", data={"source_mix_id": ident, "mode": "plain"})
        self.assertEqual(response.status_code, 200)
        with patch.object(server, "make_mix") as renderer:
            schedule.call_args.args[0]("pending")
        self.assertEqual(renderer.call_args.args[1], tracks)
        self.assertEqual(renderer.call_args.args[2], "plain")


if __name__ == "__main__":
    unittest.main()
