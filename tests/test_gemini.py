import base64
import io
import json
import wave

import httpx
import pytest
import respx

from promo.config import MissingKeyError
from promo.providers import gemini
from promo.providers.http import ProviderError

URL = gemini.TTS_URL.format(model=gemini.DEFAULT_MODEL)


def _pcm(seconds=0.5, rate=24000) -> bytes:
    return b"\x01\x00" * int(seconds * rate)


def _wav(seconds=0.5, rate=24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(_pcm(seconds, rate))
    return buf.getvalue()


def _ok(audio: bytes, mime="audio/wav"):
    return httpx.Response(200, json={"candidates": [{"content": {"parts": [
        {"inlineData": {"mimeType": mime, "data": base64.b64encode(audio).decode()}}]}}]})


def _fake_key_applies(node) -> bool:
    """Live tests must read the real key from .env, so they never get the fake one."""
    return node.get_closest_marker("live") is None


@pytest.fixture(autouse=True)
def key(request, monkeypatch):
    if _fake_key_applies(request.node):
        monkeypatch.setenv("GEMINI_API_KEY", "g-test")


def test_fake_key_skips_live_tests():
    class Node:
        def __init__(self, marker):
            self.marker = marker

        def get_closest_marker(self, name):
            return self.marker if name == "live" else None

    assert _fake_key_applies(Node(None)) is True
    assert _fake_key_applies(Node(object())) is False


def test_body_shapes():
    b = gemini.build_body("Hi.", "Kore", "warm", "prebuilt")
    assert b["contents"][0]["parts"][0] == {"text": "Hi.", "speech_metadata": {"style": "warm"}}
    assert b["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert b["generationConfig"]["speechConfig"]["voiceConfig"] == {"prebuiltVoiceConfig": {"voiceName": "Kore"}}
    v = gemini.build_body("Hi.", "Kore", "", "voice")
    assert v["contents"][0]["parts"][0] == {"text": "Hi."}
    assert v["generationConfig"]["speechConfig"]["voiceConfig"] == {"voice": "Kore"}


def test_ensure_wav_passthrough_and_wrap():
    w = _wav()
    assert gemini.ensure_wav(w) is w
    wrapped = gemini.ensure_wav(_pcm(0.25, 16000), "audio/L16;codec=pcm;rate=16000")
    with wave.open(io.BytesIO(wrapped)) as r:
        assert r.getframerate() == 16000 and r.getnframes() == 4000 and r.getnchannels() == 1


@respx.mock
def test_synthesize_returns_wav_and_sends_key_header():
    route = respx.post(URL).mock(return_value=_ok(_wav()))
    out = gemini.synthesize("Moving company money just got safer.", voice="Kore", style="confident")
    assert out[:4] == b"RIFF"
    req = route.calls[0].request
    assert req.headers["x-goog-api-key"] == "g-test"
    assert json.loads(req.content)["contents"][0]["parts"][0]["text"] == "Moving company money just got safer."


@respx.mock
def test_raw_pcm_response_is_wrapped():
    respx.post(URL).mock(return_value=_ok(_pcm(), "audio/L16;codec=pcm;rate=24000"))
    out = gemini.synthesize("Hi.", voice="Kore")
    with wave.open(io.BytesIO(out)) as r:
        assert r.getframerate() == 24000


@respx.mock
def test_voice_shape_fallback_is_remembered():
    route = respx.post(URL).mock(side_effect=[
        httpx.Response(400, json={"error": {"message": 'Invalid JSON payload received. Unknown name "prebuiltVoiceConfig"'}}),
        _ok(_wav()),
        _ok(_wav()),
    ])
    gemini.synthesize("One.", voice="Kore")
    first, second = (json.loads(c.request.content) for c in route.calls[:2])
    assert "prebuiltVoiceConfig" in first["generationConfig"]["speechConfig"]["voiceConfig"]
    assert second["generationConfig"]["speechConfig"]["voiceConfig"] == {"voice": "Kore"}
    gemini.synthesize("Two.", voice="Kore")
    third = json.loads(route.calls[2].request.content)
    assert third["generationConfig"]["speechConfig"]["voiceConfig"] == {"voice": "Kore"}
    assert route.call_count == 3


@respx.mock
def test_shape_file_not_rewritten_when_unchanged():
    respx.post(URL).mock(return_value=_ok(_wav()))
    gemini.synthesize("One.", voice="Kore")
    f = gemini._shape_file()
    before = (f.stat().st_ino, f.stat().st_mtime_ns)
    gemini.synthesize("Two.", voice="Kore")
    assert (f.stat().st_ino, f.stat().st_mtime_ns) == before


@respx.mock
def test_unwritable_shape_file_never_loses_the_take(monkeypatch):
    def boom(shape):
        raise OSError("disk full")

    monkeypatch.setattr(gemini, "_remember_shape", boom)
    respx.post(URL).mock(return_value=_ok(_wav()))
    assert gemini.synthesize("Hi.", voice="Kore")[:4] == b"RIFF"


@respx.mock
def test_401_raises_provider_error_with_message():
    route = respx.post(URL).mock(return_value=httpx.Response(401, json={"error": {"message": "API key not valid."}}))
    with pytest.raises(ProviderError) as err:
        gemini.synthesize("Hi.", voice="Kore")
    assert "gemini: HTTP 401: API key not valid." in str(err.value)
    assert route.call_count == 1  # no retry, no shape fallback on auth errors


def test_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setattr("promo.config.load_env", lambda path=None: None)
    with pytest.raises(MissingKeyError):
        gemini.synthesize("Hi.", voice="Kore")


def test_estimate_usd():
    usd = gemini.estimate_usd(gemini.DEFAULT_MODEL, "x" * 400, 10.0)
    assert usd == pytest.approx(10 * 25 * 9.0 / 1e6 + 100 * 0.5 / 1e6)


@pytest.mark.live
def test_live_synthesis():
    from promo.config import load_env
    load_env()
    out = gemini.synthesize("Hello from the promo harness.", voice="Kore", style="warm, friendly")
    with wave.open(io.BytesIO(out)) as r:
        assert r.getnframes() / r.getframerate() > 0.5
