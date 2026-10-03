import io
import shutil
import wave
from pathlib import Path

import pytest

from promo.providers import supertonic
from promo.providers.supertonic import SupertonicError, SupertonicNotInstalled


def _duration(wav: bytes) -> float:
    with wave.open(io.BytesIO(wav)) as r:
        return r.getnframes() / r.getframerate()


def test_synthesize_builds_the_argument_list_from_the_options(fake_supertonic):
    wav = supertonic.synthesize("Hello there.", voice="M1", model="supertonic", lang="fr", steps=9, speed=1.1,
                                silence_s=0.5)
    (call,) = fake_supertonic.tts()
    out = Path(fake_supertonic.flag(call, "-o"))
    assert call == ["tts", "-o", str(out), "--model", "supertonic", "--voice", "M1", "--lang", "fr", "--steps", "9",
                    "--speed", "1.1", "--silence-duration", "0.5", "--", "Hello there."]
    assert out.name == "line.wav" and not out.exists()                  # written into a temp dir, gone afterwards
    assert wav[:4] == b"RIFF" and _duration(wav) > 1.0


def test_defaults_are_f2_with_twelve_steps_at_1_25x(fake_supertonic):
    supertonic.synthesize("Hi.")
    (call,) = fake_supertonic.tts()
    flag = fake_supertonic.flag
    assert [flag(call, f) for f in ("--voice", "--model", "--lang", "--steps", "--speed", "--silence-duration")] == [
        "F2", "supertonic-2", "en", "12", "1.25", "0.3"]


@pytest.mark.parametrize("given, passed", [("en-GB", "en"), ("pt_BR", "pt"), ("FR", "fr"), ("ko", "ko")])
def test_the_cli_gets_only_the_primary_language_subtag(fake_supertonic, given, passed):
    supertonic.synthesize("Hi.", lang=given)
    assert fake_supertonic.flag(fake_supertonic.tts()[0], "--lang") == passed
    assert supertonic.primary_lang(given) == passed


def test_the_cache_key_keeps_the_full_language_tag(fake_supertonic):
    base = dict(voice="F2", model="supertonic-2", steps=12, speed=1.25, silence_s=0.3)
    assert supertonic.cache_key("Hi.", lang="en-GB", **base) != supertonic.cache_key("Hi.", lang="en", **base)


def test_text_that_starts_with_a_dash_is_not_read_as_an_option(fake_supertonic):
    supertonic.synthesize("-5 degrees today.")
    assert fake_supertonic.tts()[0][-2:] == ["--", "-5 degrees today."]


def test_missing_binary_is_a_named_error_with_the_install_route(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(SupertonicNotInstalled, match=r"^Supertonic is not installed \(pip install supertonic or "
                                                      r"`uv tool install supertonic`"):
        supertonic.synthesize("Hi.")
    with pytest.raises(SupertonicNotInstalled):
        supertonic.version()


def test_a_failing_cli_reports_its_own_error_line(fake_supertonic, monkeypatch):
    monkeypatch.setenv("FAKE_SUPERTONIC_FAIL", "Voice style 'Kore' not found. Available voice styles: F1, F2")
    with pytest.raises(SupertonicError, match=r"^supertonic failed \(exit 1\): Voice style 'Kore' not found"):
        supertonic.synthesize("Hi.", voice="Kore")


def test_exit_zero_without_a_wav_is_an_error(fake_supertonic, monkeypatch):
    monkeypatch.setenv("FAKE_SUPERTONIC_NO_FILE", "1")
    with pytest.raises(SupertonicError, match="no WAV"):
        supertonic.synthesize("Hi.")


def test_a_silent_draw_is_drawn_again(fake_supertonic, monkeypatch):
    monkeypatch.setenv("FAKE_SUPERTONIC_SILENT", "2")
    wav = supertonic.synthesize("Psst.")
    assert len(fake_supertonic.tts()) == 3 and supertonic._peak(wav) > 0.2


def test_endless_silence_is_an_error_not_a_take(fake_supertonic, monkeypatch):
    monkeypatch.setenv("FAKE_SUPERTONIC_SILENT", "99")
    with pytest.raises(supertonic.SupertonicSilence, match=r"silence 5 times in a row for 'Hm\.'"):
        supertonic.synthesize("Hm.")
    assert len(fake_supertonic.tts()) == supertonic.SILENT_DRAWS


def test_errors_are_tool_errors_so_the_cli_catches_them():
    from promo.proc import ToolError

    assert issubclass(SupertonicError, ToolError) and issubclass(SupertonicNotInstalled, SupertonicError)


def test_cache_key_covers_every_input_and_the_version(fake_supertonic, monkeypatch):
    base = dict(voice="F2", model="supertonic-2", lang="en", steps=12, speed=1.25, silence_s=0.3)
    key = supertonic.cache_key("Hello.", **base)
    assert key == supertonic.cache_key("Hello.", **base)
    assert supertonic.cache_key("Hello!", **base) != key
    for field, other in (("voice", "F1"), ("model", "supertonic"), ("lang", "fr"), ("steps", 16), ("speed", 1.2125),
                         ("silence_s", 0.5)):
        assert supertonic.cache_key("Hello.", **{**base, field: other}) != key, field
    monkeypatch.setattr(supertonic, "version", lambda: "supertonic 10.0")
    assert supertonic.cache_key("Hello.", **base) != key


def test_the_version_is_looked_up_once_per_executable(fake_supertonic):
    assert supertonic.version() == "supertonic 9.9.9"
    supertonic.cache_key("a", voice="F2", model="m", lang="en", steps=1, speed=1.0, silence_s=0.3)
    supertonic.cache_key("b", voice="F2", model="m", lang="en", steps=1, speed=1.0, silence_s=0.3)
    assert [c for c in fake_supertonic.calls() if c == ["version"]] == [["version"]]


@pytest.mark.live
@pytest.mark.skipif(shutil.which("supertonic") is None, reason="supertonic is not installed")
def test_live_supertonic_speaks_a_line_that_asr_hears_back(tmp_path):
    from promo import asr

    text = "Your references stay straight, and the paper writes itself in your own voice."
    wav = supertonic.synthesize(text)
    path = tmp_path / "line.wav"
    path.write_bytes(wav)
    with wave.open(io.BytesIO(wav)) as r:
        assert (r.getframerate(), r.getnchannels()) == (44100, 1)
    assert 3.0 < _duration(wav) < 8.0                                    # 14 words at about 167 wpm
    heard = " ".join(w.text for w in asr.transcribe(path, "en"))
    assert asr.wer(text, heard) <= 0.1, heard
