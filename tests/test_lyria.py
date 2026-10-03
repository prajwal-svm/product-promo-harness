import base64
import json
import subprocess

import httpx
import pytest
import respx

from promo.config import MissingKeyError
from promo.providers import lyria
from promo.providers.http import ProviderError


def _ok(audio: bytes, mime="audio/mpeg", text="[Intro] instrumental"):
    return httpx.Response(200, json={"id": "i-1", "status": "completed", "steps": [
        {"type": "model_output", "content": [
            {"type": "text", "text": text},
            {"type": "audio", "mime_type": mime, "data": base64.b64encode(audio).decode()}]}]})


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


@respx.mock
def test_compose_request_shape():
    route = respx.post(lyria.INTERACTIONS_URL).mock(return_value=_ok(b"ID3mp3"))
    assert lyria.compose("a prompt", 12_000) == b"ID3mp3"
    req = route.calls[0].request
    assert json.loads(req.content) == {"model": "lyria-3.5", "input": "a prompt", "store": False}
    assert req.headers["x-goog-api-key"] == "g-test"


@respx.mock
def test_compose_takes_another_model():
    route = respx.post(lyria.INTERACTIONS_URL).mock(return_value=_ok(b"ID3mp3"))
    lyria.compose("p", 30_000, model="lyria-3-clip-preview")
    assert json.loads(route.calls[0].request.content)["model"] == "lyria-3-clip-preview"


def test_extract_audio_takes_the_last_audio_block_and_skips_text():
    first, last = base64.b64encode(b"one").decode(), base64.b64encode(b"two").decode()
    payload = {"steps": [{"type": "thought", "content": [{"type": "text", "text": "plan"}]},
                         {"type": "model_output", "content": [{"type": "audio", "data": first}]},
                         {"type": "model_output", "content": [{"type": "text", "text": "lyrics"},
                                                              {"type": "audio", "data": last, "mime_type": "audio/wav"}]}]}
    assert lyria.extract_audio(payload) == (b"two", "audio/wav")


def test_extract_audio_reads_the_older_outputs_shape():
    payload = {"outputs": [{"type": "text", "text": "x"},
                           {"type": "audio", "data": base64.b64encode(b"mp3").decode(), "mime_type": "audio/mp3"}]}
    assert lyria.extract_audio(payload) == (b"mp3", "audio/mp3")


@pytest.mark.parametrize("payload", [{"steps": []}, {"status": "failed", "steps": [
    {"type": "model_output", "content": [{"type": "text", "text": "I can't make that."}]}]}, {}])
def test_a_reply_without_audio_is_a_provider_error_that_quotes_it(payload):
    with pytest.raises(ProviderError) as err:
        lyria.extract_audio(payload)
    assert err.value.provider == "lyria" and "no audio" in str(err.value)
    if payload.get("status"):
        assert "failed" in str(err.value) and "can't make that" in str(err.value)


@respx.mock
def test_http_errors_name_lyria():
    respx.post(lyria.INTERACTIONS_URL).mock(return_value=httpx.Response(
        403, json={"error": {"code": 403, "message": "Lyria is not available on the free tier"}}))
    with pytest.raises(ProviderError) as err:
        lyria.compose("p", 10_000)
    assert "lyria: HTTP 403: Lyria is not available on the free tier" in str(err.value)


@respx.mock
def test_missing_key_makes_no_request(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setattr("promo.config.load_env", lambda path=None: None)
    route = respx.post(lyria.INTERACTIONS_URL).mock(return_value=_ok(b"x"))
    with pytest.raises(MissingKeyError):
        lyria.compose("p", 10_000)
    assert route.call_count == 0


def test_a_call_is_billed_per_song():
    assert lyria.call_usd("lyria-3.5") == 0.08
    assert lyria.call_usd("lyria-3-clip-preview") == 0.04
    assert lyria.call_usd("lyria-3-pro-preview") == 0.08


@pytest.mark.live
def test_live_shortest_lyria_call(tmp_path):
    """One real call: the shortest bed lyria-3.5 makes. About $0.08, logged to the cache's live/costs.jsonl (outside
    the repo, kept across runs) because a live test has no run folder of its own."""
    from promo.config import cache_dir, load_env
    from promo.providers import costs

    load_env()
    audio = lyria.compose(
        "Instrumental only, no vocals. Minimal electronic groove at 120 BPM in 4/4: kick on every beat, "
        "closed hi-hats, warm bass. Total length 0:10 (10 seconds).", 10_000)
    costs.log_cost(cache_dir("live") / "costs.jsonl", provider="lyria", model=lyria.MUSIC_MODEL,
                   units={"calls": 1, "test": "test_live_shortest_lyria_call"},
                   usd=lyria.call_usd(lyria.MUSIC_MODEL), cached=False)
    p = tmp_path / "m.mp3"
    p.write_bytes(audio)
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True, check=True)
    assert float(out.stdout) >= 5
