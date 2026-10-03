import io
import json
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from promo import voice
from promo.cli import app
from promo.config import MissingKeyError
from promo.models import Caption, LineTake
from promo.packs import PackError, load_pack
from promo.providers.http import ProviderError
from promo.run import new_run
from promo.shotlist import Shotlist, VOLine

FIX = Path(__file__).parent / "fixtures" / "shotlist_music.json"


def tone_wav(seconds: float, rate=24000, amp=0.3) -> bytes:
    pad = np.zeros(int(0.2 * rate), dtype=np.float32)
    t = np.arange(int(seconds * rate)) / rate
    tone = np.clip(amp * np.sin(2 * np.pi * 220 * t), -1.0, 1.0).astype(np.float32)  # amp > 1 clips at full scale
    y = np.concatenate([pad, tone, pad])
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes((y * 32767).astype("<i2").tobytes())
    return buf.getvalue()


class FakeSynth:
    """Returns a tone per call. `seconds` and `amp` may be lists: one entry per call, the last one repeats."""

    def __init__(self, seconds=1.0, amp=0.3):
        self.calls = []
        self.seconds = seconds
        self.amp = amp

    @staticmethod
    def _pick(value, i):
        return value[min(i, len(value) - 1)] if isinstance(value, (list, tuple)) else value

    def __call__(self, text, *, voice, style="", model=""):
        i = len(self.calls)
        self.calls.append({"text": text, "voice": voice, "style": style, "model": model})
        return tone_wav(self._pick(self.seconds, i), amp=self._pick(self.amp, i))


def words_for(text):
    return [Caption(text=w, startMs=i * 200, endMs=i * 200 + 180) for i, w in enumerate(text.split())]


def test_spoken_text_rewrites_whole_words_only():
    assert voice.spoken_text("Meet Signet. Signets!", {"Signet": "Sig-nit"}) == "Meet Sig-nit. Signets!"


def test_trim_and_normalize(tmp_path):
    src = tmp_path / "raw.wav"
    src.write_bytes(tone_wav(1.0))
    out = voice.trim_and_normalize(src, tmp_path / "out.wav")
    info = sf.info(str(out))
    assert info.samplerate == 48000 and info.channels == 1
    assert info.duration == pytest.approx(1.0 + 0.12, abs=0.05)


def test_first_take_accepted(tmp_path):
    synth = FakeSynth()
    line = VOLine(id="l1", text="Moving company money just got safer.", style="confident")
    take = voice.synthesize_line(line, voice="Kore", base_style="warm", model="m", pronounce={"Signet": "Sig-nit"},
                                 language="en", vo_dir=tmp_path / "audio" / "vo", synth=synth,
                                 transcribe=lambda p, lang: words_for(line.text))
    assert take.takes == 1 and take.wer == 0.0
    assert synth.calls[0]["style"] == "warm; confident"
    assert (tmp_path / "audio" / "vo" / "l1.wav").exists()
    assert take.file == "audio/vo/l1.wav"
    assert [w.text for w in take.words] == line.text.split()


def test_retakes_until_match_and_keeps_best(tmp_path):
    synth = FakeSynth()
    line = VOLine(id="l2", text="Risk has to sign off.")
    answers = iter([words_for("Risky hats to sign of"), words_for("Risk has to sign off.")])
    take = voice.synthesize_line(line, voice="Kore", base_style="warm", model="m", pronounce={}, language="en",
                                 vo_dir=tmp_path / "vo", synth=synth, transcribe=lambda p, lang: next(answers))
    assert take.takes == 2 and take.wer == 0.0
    assert synth.calls[1]["style"] == "warm; clear diction"
    assert len(list((tmp_path / "vo" / ".takes").glob("l2-*.wav"))) == 2


def test_gives_up_after_three_and_reports_best(tmp_path):
    line = VOLine(id="l3", text="one two three four")
    answers = iter([words_for("one two x y"), words_for("one two three x"), words_for("x x x x")])
    take = voice.synthesize_line(line, voice="Kore", base_style="", model="m", pronounce={}, language="en",
                                 vo_dir=tmp_path / "vo", synth=FakeSynth(), transcribe=lambda p, lang: next(answers))
    assert take.takes == 3 and take.wer == pytest.approx(0.25)


def test_pronunciation_is_spoken_but_wer_uses_script(tmp_path):
    synth = FakeSynth()
    line = VOLine(id="l4", text="Meet Signet.")
    take = voice.synthesize_line(line, voice="Kore", base_style="", model="m", pronounce={"Signet": "Sig-nit"},
                                 language="en", vo_dir=tmp_path / "vo", synth=synth,
                                 transcribe=lambda p, lang: words_for("Meet Signet."))
    assert synth.calls[0]["text"] == "Meet Sig-nit."
    assert take.wer == 0.0


def test_synthesize_all_writes_records_logs_costs_and_is_idempotent(tmp_path):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    run.shotlist.write_text(FIX.read_text())
    sl = Shotlist.model_validate(json.loads(FIX.read_text()))
    pack = load_pack("saas-kinetic")
    synth = FakeSynth(seconds=1.2)
    tr = lambda p, lang: []  # WER 1.0: forces all takes, still records the best
    takes = voice.synthesize_all(run, sl, pack, synth=synth, transcribe=tr)
    assert [t.id for t in takes] == ["c01-l1", "c02-l1", "c02-l2"]
    rec = LineTake.model_validate_json((run.vo_dir / "c01-l1.json").read_text())
    assert rec.voice == "Kore" and rec.style.startswith("warm, confident")
    rows = [json.loads(l) for l in run.costs.read_text().splitlines()]
    assert rows and all(r["provider"] == "gemini" for r in rows)
    n = len(synth.calls)
    # second run: same text/voice/style/model → skipped entirely
    voice.synthesize_all(run, sl, pack, synth=synth, transcribe=tr)
    assert len(synth.calls) == n
    # forced run: every call is a cache hit, so the fake is still not called again
    voice.synthesize_all(run, sl, pack, force=True, synth=synth, transcribe=tr)
    assert len(synth.calls) == n
    assert any(json.loads(l)["cached"] for l in run.costs.read_text().splitlines())


def test_audition_writes_one_file_per_voice(tmp_path):
    synth = FakeSynth()
    paths = voice.audition("Hello there.", ["Kore", "Aoede"], "warm", tmp_path / "aud", synth=synth)
    assert [p.name for p in paths] == ["Kore.wav", "Aoede.wav"]
    assert {c["voice"] for c in synth.calls} == {"Kore", "Aoede"}
    assert all(c["model"] == "gemini-3.8-flash-lite-tts" for c in synth.calls)


def _demo_run(tmp_path):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    run.shotlist.write_text(FIX.read_text())
    return run, Shotlist.model_validate(json.loads(FIX.read_text())), load_pack("saas-kinetic")


def test_vo_dir_json_files_are_only_line_takes(tmp_path):
    from promo.timing import load_measured

    run, sl, pack = _demo_run(tmp_path)
    voice.synthesize_all(run, sl, pack, synth=FakeSynth(), transcribe=lambda p, lang: [])
    assert sorted(f.name for f in run.vo_dir.glob("*.json")) == ["c01-l1.json", "c02-l1.json", "c02-l2.json"]
    assert sorted(load_measured(run.root).takes) == ["c01-l1", "c02-l1", "c02-l2"]
    assert (run.vo_dir / "c01-l1.src.txt").exists() and (run.vo_dir / ".takes").is_dir()


def test_only_limits_the_lines(tmp_path):
    run, sl, pack = _demo_run(tmp_path)
    takes = voice.synthesize_all(run, sl, pack, only=["c02-l2"], synth=FakeSynth(), transcribe=lambda p, lang: [])
    assert [t.id for t in takes] == ["c02-l2"]
    assert not (run.vo_dir / "c01-l1.json").exists()


def test_retaken_line_is_still_skipped_on_rerun(tmp_path):
    run, sl, pack = _demo_run(tmp_path)
    texts = {ln.id: ln.text for ln in sl.all_lines()}

    def tr(path, lang):
        # the take file is named <id>-<n>.wav; the first take of every line mismatches, the retake matches
        stem = Path(path).stem
        line_id, n = stem.rsplit("-", 1)
        return words_for(texts[line_id]) if n == "2" else []

    synth = FakeSynth()
    takes = voice.synthesize_all(run, sl, pack, synth=synth, transcribe=tr)
    assert all(t.takes == 2 and t.wer == 0.0 for t in takes)
    assert all(t.style.endswith("; clear diction") for t in takes)
    calls = len(synth.calls)
    again = voice.synthesize_all(run, sl, pack, synth=synth, transcribe=tr)
    assert len(synth.calls) == calls and [t.id for t in again] == [t.id for t in takes]


def test_changed_inputs_retake(tmp_path):
    run, _, pack = _demo_run(tmp_path)
    doc = json.loads(FIX.read_text())
    doc["pronounce"] = {}
    analysed = []

    def takes_made() -> int:
        """Run c01-l1 against the current doc; how many takes (ASR passes) did it cost, cached or not?"""
        analysed.clear()
        voice.synthesize_all(run, Shotlist.model_validate(doc), pack, only=["c01-l1"], synth=FakeSynth(),
                             transcribe=lambda p, lang: analysed.append(p) or [])
        return len(analysed)

    line = doc["chapters"][0]["vo"][0]
    assert takes_made() == 3 and takes_made() == 0          # first run, then unchanged
    line["text"] = "Signet moves money."
    assert takes_made() == 3 and takes_made() == 0          # text
    doc["pronounce"] = {"Other": "Oh-ther"}
    assert takes_made() == 0                                # a pronunciation this line does not use
    doc["pronounce"] = {"Signet": "Sig-nit"}
    assert takes_made() == 3 and takes_made() == 0          # spoken text changed, script text did not
    line["style"] = ""
    assert takes_made() == 3 and takes_made() == 0          # line style dropped
    doc["voice"] = {"name": "Aoede"}
    assert takes_made() == 3 and takes_made() == 0          # voice
    doc["voice"] = {"name": "Aoede", "model": "other-model"}
    assert takes_made() == 3 and takes_made() == 0          # model
    doc["language"] = "fr"
    assert takes_made() == 3 and takes_made() == 0          # language


def test_fewer_takes_remove_stale_take_files(tmp_path):
    vo = tmp_path / "vo"
    line = VOLine(id="l5", text="one two")
    voice.synthesize_line(line, voice="Kore", base_style="", model="m", pronounce={}, language="en", vo_dir=vo,
                          synth=FakeSynth(), transcribe=lambda p, lang: [])
    assert len(list((vo / ".takes").glob("l5-*.wav"))) == 3
    voice.synthesize_line(line, voice="Kore", base_style="", model="m", pronounce={}, language="en", vo_dir=vo,
                          synth=FakeSynth(), transcribe=lambda p, lang: words_for(line.text))
    assert [p.name for p in (vo / ".takes").glob("l5-*.wav")] == ["l5-1.wav"]
    assert not list((vo / ".takes").glob(".*"))


def test_pace_outside_band_forces_retake(tmp_path):
    synth = FakeSynth(seconds=4.0)
    line = VOLine(id="l6", text="Risk has to sign off.")  # 5 words at 190 wpm = 1.58 s; 4.12 s is far too slow
    take = voice.synthesize_line(line, voice="Kore", base_style="", model="m", pronounce={}, language="en",
                                 vo_dir=tmp_path / "vo", synth=synth, pace_wpm=190,
                                 transcribe=lambda p, lang: words_for(line.text))
    assert take.takes == 3


def test_accepted_take_is_the_one_kept_even_if_a_rejected_take_had_equal_wer(tmp_path):
    vo = tmp_path / "vo"
    line = VOLine(id="p1", text="Risk has to sign off.")  # 1.58 s at 190 wpm
    take = voice.synthesize_line(line, voice="Kore", base_style="", model="m", pronounce={}, language="en",
                                 vo_dir=vo, synth=FakeSynth(seconds=[4.0, 1.4]), pace_wpm=190,
                                 transcribe=lambda p, lang: words_for(line.text))
    assert take.takes == 2 and take.wer == 0.0 and take.style == "clear diction"
    kept = sf.info(str(vo / "p1.wav")).duration
    assert kept == pytest.approx(1.4 + 0.12, abs=0.05) and take.duration_s == pytest.approx(kept, abs=0.001)
    assert (vo / "p1.wav").read_bytes() == (vo / ".takes" / "p1-2.wav").read_bytes()


def test_accepted_take_beats_a_rejected_take_with_lower_wer(tmp_path):
    vo = tmp_path / "vo"
    line = VOLine(id="p2", text="one two three four five six seven eight nine ten")  # 10 words: WER 0.1 is accepted
    answers = iter([words_for("one two three four five six seven eight nine ten"),  # WER 0 but far too slow
                    words_for("one two three four five six seven eight nine tin")])  # WER 0.1 and on pace
    take = voice.synthesize_line(line, voice="Kore", base_style="", model="m", pronounce={}, language="en",
                                 vo_dir=vo, synth=FakeSynth(seconds=[9.0, 3.0]), pace_wpm=190,
                                 transcribe=lambda p, lang: next(answers))
    assert take.takes == 2 and take.wer == pytest.approx(0.1)
    assert (vo / "p2.wav").read_bytes() == (vo / ".takes" / "p2-2.wav").read_bytes()


def test_all_rejected_keeps_lowest_wer_and_earlier_on_ties(tmp_path):
    vo = tmp_path / "vo"
    line = VOLine(id="p3", text="Risk has to sign off.")
    take = voice.synthesize_line(line, voice="Kore", base_style="", model="m", pronounce={}, language="en",
                                 vo_dir=vo, synth=FakeSynth(seconds=[4.0, 5.0, 6.0]), pace_wpm=190,
                                 transcribe=lambda p, lang: words_for(line.text))
    assert take.takes == 3 and take.wer == 0.0
    assert (vo / "p3.wav").read_bytes() == (vo / ".takes" / "p3-1.wav").read_bytes()


def test_raw_clipping_is_detected_by_run_length(tmp_path):
    def raw(samples):
        path = tmp_path / "x.wav"
        sf.write(str(path), np.asarray(samples, dtype=np.float32), 24000, subtype="PCM_16")
        return path

    quiet = 0.2 * np.ones(1000)
    assert voice._clipped(raw(quiet)) is False
    assert voice._clipped(raw(np.concatenate([quiet, [1.0], quiet]))) is False          # one stray peak
    assert voice._clipped(raw(np.concatenate([quiet, [1.0, 0.2, 1.0, 0.2, 1.0], quiet]))) is False  # not consecutive
    assert voice._clipped(raw(np.concatenate([quiet, [1.0, 1.0, 1.0], quiet]))) is True
    assert voice._clipped(raw(np.concatenate([quiet, [-1.0] * 5, quiet]))) is True
    assert voice._clipped(raw([])) is False


def test_clipped_raw_take_forces_a_retake(tmp_path):
    vo = tmp_path / "vo"
    line = VOLine(id="l7", text="Risk has to sign off.")
    # the first take is a sine driven past full scale; normalisation would hide that, the raw WAV shows it
    take = voice.synthesize_line(line, voice="Kore", base_style="", model="m", pronounce={}, language="en",
                                 vo_dir=vo, synth=FakeSynth(amp=[1.6, 0.3]),
                                 transcribe=lambda p, lang: words_for(line.text))
    assert take.takes == 2 and take.style == "clear diction"
    assert (vo / "l7.wav").read_bytes() == (vo / ".takes" / "l7-2.wav").read_bytes()
    assert voice._clipped(vo / ".takes" / "l7-1.wav") is False  # the normalised copy of the clipped take looks clean


def test_clean_raw_take_is_accepted_first_time(tmp_path):
    line = VOLine(id="l7b", text="Risk has to sign off.")
    take = voice.synthesize_line(line, voice="Kore", base_style="", model="m", pronounce={}, language="en",
                                 vo_dir=tmp_path / "vo", synth=FakeSynth(amp=0.9),
                                 transcribe=lambda p, lang: words_for(line.text))
    assert take.takes == 1


def test_always_clipped_gives_up_after_three_and_keeps_the_first(tmp_path):
    vo = tmp_path / "vo"
    line = VOLine(id="l7c", text="Risk has to sign off.")
    take = voice.synthesize_line(line, voice="Kore", base_style="", model="m", pronounce={}, language="en",
                                 vo_dir=vo, synth=FakeSynth(amp=1.6), transcribe=lambda p, lang: words_for(line.text))
    assert take.takes == 3
    assert (vo / "l7c.wav").read_bytes() == (vo / ".takes" / "l7c-1.wav").read_bytes()


def test_final_wav_is_written_atomically(tmp_path, monkeypatch):
    vo = tmp_path / "vo"
    line = VOLine(id="a1", text="Risk has to sign off.")
    kwargs = dict(voice="Kore", base_style="", model="m", pronounce={}, language="en", vo_dir=vo)
    voice.synthesize_line(line, synth=FakeSynth(), transcribe=lambda p, lang: words_for(line.text), **kwargs)
    before = (vo / "a1.wav").read_bytes()
    assert not list(vo.glob(".*.tmp.wav"))

    def broken_copy(src, dst):
        Path(dst).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(voice.shutil, "copyfile", broken_copy)
    with pytest.raises(OSError):
        voice.synthesize_line(line, synth=FakeSynth(seconds=2.0), transcribe=lambda p, lang: words_for(line.text),
                              **dict(kwargs, model="m2"))
    assert (vo / "a1.wav").read_bytes() == before
    assert not list(vo.glob(".*.tmp.wav"))


def test_spoken_text_keeps_backslashes_in_replacement():
    assert voice.spoken_text("Meet Signet.", {"Signet": r"Sig\1nit"}) == r"Meet Sig\1nit."


def test_pronunciation_map_keeps_script_spelling_in_word_texts(tmp_path):
    line = VOLine(id="l8", text="Meet Signet.")
    take = voice.synthesize_line(line, voice="Kore", base_style="", model="m", pronounce={"Signet": "Sig-nit"},
                                 language="en", vo_dir=tmp_path / "vo", synth=FakeSynth(),
                                 transcribe=lambda p, lang: words_for("meet signet"))
    assert take.wer == 0.0 and [w.text for w in take.words] == ["Meet", "Signet."]


def test_currency_narration_does_not_loop_retakes(tmp_path):
    line = VOLine(id="l9", text="Reversals over $10,000?")
    take = voice.synthesize_line(line, voice="Kore", base_style="", model="m", pronounce={}, language="en",
                                 vo_dir=tmp_path / "vo", synth=FakeSynth(),
                                 transcribe=lambda p, lang: words_for("Reversals over ten thousand dollars"))
    assert take.takes == 1 and take.wer == 0.0


# --- the `promo voice` commands -------------------------------------------------------------------------------

runner = CliRunner()


def _in_run(tmp_path, monkeypatch):
    run, _, _ = _demo_run(tmp_path)
    monkeypatch.chdir(run.root)
    return run


def _take(line_id, wer=0.0, takes=1):
    return LineTake(id=line_id, file=f"audio/vo/{line_id}.wav", duration_s=1.5, wer=wer, takes=takes)


def test_voice_needs_all_or_line(tmp_path, monkeypatch):
    _in_run(tmp_path, monkeypatch)
    res = runner.invoke(app, ["voice"])
    assert res.exit_code == 2 and "--all" in res.output


def test_voice_all_reports_each_line(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch)
    seen = {}

    def fake(r, sl, pack, only=None, force=False):
        seen.update(run=r.root, only=only, force=force)
        return [_take("c01-l1"), _take("c02-l1", wer=0.0, takes=2)]

    monkeypatch.setattr("promo.voice.synthesize_all", fake)
    res = runner.invoke(app, ["voice", "--all", "--force"])
    assert res.exit_code == 0, res.output
    assert "c01-l1" in res.output and "takes 2" in res.output and "verified by speech recognition" in res.output
    assert seen == {"run": run.root, "only": None, "force": True}


def test_voice_line_passes_ids_and_flags_mismatch(tmp_path, monkeypatch):
    _in_run(tmp_path, monkeypatch)
    seen = {}

    def fake(r, sl, pack, only=None, force=False):
        seen["only"] = only
        return [_take("c01-l1", wer=0.4, takes=3)]

    monkeypatch.setattr("promo.voice.synthesize_all", fake)
    res = runner.invoke(app, ["voice", "--line", "c01-l1"])
    assert res.exit_code == 1 and "CHECK" in res.output and "still mismatch" in res.output
    assert seen["only"] == ["c01-l1"]


def test_voice_unknown_line_is_a_usage_error(tmp_path, monkeypatch):
    _in_run(tmp_path, monkeypatch)
    res = runner.invoke(app, ["voice", "--line", "nope"])
    assert res.exit_code == 2 and "nope" in res.output


@pytest.mark.parametrize("exc, text", [
    (MissingKeyError("GEMINI_API_KEY"), "GEMINI_API_KEY is missing"),
    (ProviderError("gemini", 401, "API key not valid"), "HTTP 401"),
])
def test_voice_provider_failure_is_one_line(tmp_path, monkeypatch, exc, text):
    _in_run(tmp_path, monkeypatch)

    def fake(*a, **k):
        raise exc

    monkeypatch.setattr("promo.voice.synthesize_all", fake)
    res = runner.invoke(app, ["voice", "--all"])
    assert res.exit_code == 1 and text in res.output
    assert res.output.count("promo:") == 1 and "Traceback" not in res.output


def test_voice_outside_a_run_is_a_clean_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = runner.invoke(app, ["voice", "--all"])
    assert res.exit_code == 2 and "no run folder" in res.output


def test_voice_audition_defaults_to_the_packs_voices(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch)
    seen = {}

    def fake(text, voices, style, out_dir, costs_path=None, **k):
        seen.update(text=text, voices=voices, style=style, out_dir=out_dir, costs_path=costs_path)
        return [out_dir / f"{v}.wav" for v in voices]

    monkeypatch.setattr("promo.voice.audition", fake)
    res = runner.invoke(app, ["voice", "audition", "Hello there."])
    assert res.exit_code == 0, res.output
    assert seen["voices"] == ["Kore", "Aoede", "Leda", "Autonoe", "Callirrhoe"]
    assert seen["style"].startswith("warm, confident") and seen["out_dir"] == run.auditions_dir
    assert seen["costs_path"] == run.costs
    assert "Kore.wav" in res.output and "voice.name" in res.output


def test_voice_audition_takes_voices_and_style(tmp_path, monkeypatch):
    _in_run(tmp_path, monkeypatch)
    seen = {}

    def fake(text, voices, style, out_dir, **k):
        seen.update(voices=voices, style=style)
        return []

    monkeypatch.setattr("promo.voice.audition", fake)
    res = runner.invoke(app, ["voice", "audition", "Hi.", "--voices", "Puck, Zephyr", "--style", "calm"])
    assert res.exit_code == 0, res.output
    assert seen == {"voices": ["Puck", "Zephyr"], "style": "calm"}


def test_voice_audition_missing_key_is_one_line(tmp_path, monkeypatch):
    _in_run(tmp_path, monkeypatch)

    def fake(*a, **k):
        raise MissingKeyError("GEMINI_API_KEY")

    monkeypatch.setattr("promo.voice.audition", fake)
    res = runner.invoke(app, ["voice", "audition", "Hi."])
    assert res.exit_code == 1 and "GEMINI_API_KEY is missing" in res.output and "Traceback" not in res.output


def test_trim_and_normalize_in_path_with_spaces_and_emoji(tmp_path):
    d = tmp_path / "Codex Projects" / "promo 🎬"
    d.mkdir(parents=True)
    (d / "raw take.wav").write_bytes(tone_wav(1.0))
    out = voice.trim_and_normalize(d / "raw take.wav", d / "out take.wav")
    assert out.exists() and sf.info(str(out)).samplerate == 48000


def test_inline_audio_tags_are_spoken_but_not_scored(tmp_path):
    synth = FakeSynth(seconds=1.0)
    text = "Moving money <short pause> just got safer <breath>."
    line = VOLine(id="t1", text=text)
    # ASR never transcribes the tags; 5 real words at 190 wpm is 1.58 s, so the 1.12 s take is on pace
    # (counting the tags as words would make it 2.2 s and off pace)
    take = voice.synthesize_line(line, voice="Kore", base_style="", model="m", pronounce={}, language="en",
                                 vo_dir=tmp_path / "vo", synth=synth, pace_wpm=190,
                                 transcribe=lambda p, lang: words_for("Moving money just got safer."))
    assert take.takes == 1 and take.wer == 0.0
    assert synth.calls[0]["text"] == text                                   # tags still go to TTS
    assert [w.text for w in take.words] == ["Moving", "money", "just", "got", "safer."]
    assert [w.startMs for w in take.words] == [0, 200, 400, 600, 800]     # timings untouched


@pytest.mark.parametrize("text, plain", [
    ("Moving money <short pause> just got safer.", "Moving money just got safer."),
    ("<breath> Hello there.", "Hello there."),
    ("Wait<long pause>... really?", "Wait... really?"),
    ("Safer <breath>. Faster <breath>!", "Safer. Faster!"),
    ("No tags here.", "No tags here."),
    ("A  spaced\n line.", "A spaced line."),
])
def test_script_text_strips_inline_tags(text, plain):
    assert voice._script_text(text) == plain


def test_pronunciation_and_tags_together(tmp_path):
    synth = FakeSynth()
    line = VOLine(id="t2", text="Meet Signet <breath>.")
    take = voice.synthesize_line(line, voice="Kore", base_style="", model="m", pronounce={"Signet": "Sig-nit"},
                                 language="en", vo_dir=tmp_path / "vo", synth=synth,
                                 transcribe=lambda p, lang: words_for("Meet Signet."))
    assert synth.calls[0]["text"] == "Meet Sig-nit <breath>."
    assert take.takes == 1 and [w.text for w in take.words] == ["Meet", "Signet."]


def test_audition_logs_costs_and_marks_cache_hits(tmp_path):
    synth = FakeSynth()
    costs_path = tmp_path / "costs.jsonl"
    voice.audition("Hello there.", ["Kore", "Aoede"], "warm", tmp_path / "aud", synth=synth, costs_path=costs_path)
    rows = [json.loads(l) for l in costs_path.read_text().splitlines()]
    assert [r["cached"] for r in rows] == [False, False]
    assert all(r["provider"] == "gemini" and r["model"] == "gemini-3.8-flash-lite-tts" and r["usd"] > 0 for r in rows)
    assert all(r["units"]["chars"] == len("Hello there.") and r["units"]["audio_s"] > 1 for r in rows)
    voice.audition("Hello there.", ["Kore", "Aoede"], "warm", tmp_path / "aud", synth=synth, costs_path=costs_path)
    rows = [json.loads(l) for l in costs_path.read_text().splitlines()]
    assert [r["cached"] for r in rows] == [False, False, True, True] and len(synth.calls) == 2


def test_audition_without_costs_path_logs_nothing(tmp_path):
    voice.audition("Hello there.", ["Kore"], "", tmp_path / "aud", synth=FakeSynth())
    assert not list(tmp_path.glob("**/costs.jsonl"))


def test_voice_audition_drops_empty_voice_names(tmp_path, monkeypatch):
    _in_run(tmp_path, monkeypatch)
    seen = {}

    def fake(text, voices, style, out_dir, **k):
        seen["voices"] = voices
        return []

    monkeypatch.setattr("promo.voice.audition", fake)
    res = runner.invoke(app, ["voice", "audition", "Hi.", "--voices", "Kore, ,Aoede,"])
    assert res.exit_code == 0, res.output
    assert seen["voices"] == ["Kore", "Aoede"]


@pytest.mark.parametrize("value", ["", " , ,", ","])
def test_voice_audition_with_no_voice_names_is_a_usage_error(tmp_path, monkeypatch, value):
    _in_run(tmp_path, monkeypatch)
    monkeypatch.setattr("promo.voice.audition", lambda *a, **k: pytest.fail("must not synthesise"))
    res = runner.invoke(app, ["voice", "audition", "Hi.", "--voices", value])
    assert res.exit_code == 2 and res.output.count("promo:") == 1 and "voice name" in res.output


# --- Supertonic: the local provider ---------------------------------------------------------------------------

def _st_run(tmp_path, voice_block=None, **line_fields):
    """A demo run whose shot list speaks with Supertonic; `line_fields` are added to c01-l1."""
    doc = json.loads(FIX.read_text())
    doc["voice"] = {"provider": "supertonic", **(voice_block or {})}
    doc["chapters"][0]["vo"][0].update(line_fields)
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    run.shotlist.write_text(json.dumps(doc))
    return run, Shotlist.model_validate(doc), load_pack("saas-kinetic")


def _st_line(line_id="s1", text="Risk has to sign off.", **kw):
    return VOLine(id=line_id, text=text, **kw)


def _st_take(tmp_path, line=None, transcribe=None, **kw):
    line = line or _st_line()
    return voice.synthesize_line(line, voice="F2", base_style="", model="supertonic-2", pronounce={}, language="en",
                                 vo_dir=tmp_path / "vo", provider="supertonic",
                                 transcribe=transcribe or (lambda p, lang: words_for(line.text)), **kw)


def _calls(fake):
    return [(int(fake.flag(c, "--steps")), float(fake.flag(c, "--speed"))) for c in fake.tts()]


def test_provider_dispatch_runs_the_supertonic_cli_not_gemini(tmp_path, fake_supertonic, monkeypatch):
    monkeypatch.setattr(voice.gemini, "synthesize", lambda *a, **k: pytest.fail("gemini must not be called"))
    take = _st_take(tmp_path)
    (call,) = fake_supertonic.tts()
    assert call[-2:] == ["--", "Risk has to sign off."]
    assert [fake_supertonic.flag(call, f) for f in ("--voice", "--model", "--lang", "--steps", "--speed")] == [
        "F2", "supertonic-2", "en", "12", "1.25"]
    assert (take.provider, take.voice, take.model, take.style, take.takes, take.verified) == (
        "supertonic", "F2", "supertonic-2", "", 1, True)
    assert take.params == {"steps": 12, "speed": 1.25, "silence_s": 0.3}
    assert sf.info(str(tmp_path / "vo" / "s1.wav")).samplerate == 48000    # trimmed and normalised like any take


def test_gemini_stays_the_default_provider_and_never_touches_supertonic(tmp_path, fake_supertonic):
    synth = FakeSynth()
    line = _st_line("g1")
    take = voice.synthesize_line(line, voice="Kore", base_style="warm", model="m", pronounce={}, language="en",
                                 vo_dir=tmp_path / "vo", synth=synth, transcribe=lambda p, lang: words_for(line.text))
    assert take.provider == "gemini" and take.params == {} and len(synth.calls) == 1 and fake_supertonic.calls() == []


def test_supertonic_options_default_from_the_provider_and_pass_through(tmp_path, fake_supertonic):
    _st_take(tmp_path, options={"steps": 20, "speed": 1.4})
    (call,) = fake_supertonic.tts()
    assert (fake_supertonic.flag(call, "--steps"), fake_supertonic.flag(call, "--speed"),
            fake_supertonic.flag(call, "--silence-duration")) == ("20", "1.4", "0.3")


def test_supertonic_cost_row_is_free_and_local(tmp_path, fake_supertonic):
    costs_path = tmp_path / "costs.jsonl"
    _st_take(tmp_path, costs_path=costs_path)
    (row,) = [json.loads(l) for l in costs_path.read_text().splitlines()]
    assert row["provider"] == "supertonic" and row["model"] == "supertonic-2" and row["usd"] == 0.0
    assert row["units"] == {"chars": len("Risk has to sign off."), "local": True} and row["cached"] is False


def test_supertonic_takes_are_cached_by_content_hash(tmp_path, fake_supertonic):
    costs_path = tmp_path / "costs.jsonl"
    _st_take(tmp_path, costs_path=costs_path)
    _st_take(tmp_path, costs_path=costs_path)
    assert len(fake_supertonic.tts()) == 1
    assert [json.loads(l)["cached"] for l in costs_path.read_text().splitlines()] == [False, True]
    _st_take(tmp_path, line=_st_line(text="Risk has to sign off now."), costs_path=costs_path)  # new text: new call
    assert len(fake_supertonic.tts()) == 2


def test_supertonic_retakes_change_steps_and_speed(tmp_path, fake_supertonic):
    answers = iter([words_for("x"), words_for("Risk has"), words_for("Risk has to sign off.")])
    take = _st_take(tmp_path, transcribe=lambda p, lang: next(answers))
    assert _calls(fake_supertonic) == [(12, 1.25), (16, 1.2125), (20, 1.2875)]    # +0/x1, +4/x0.97, +8/x1.03
    assert take.takes == 3 and take.wer == 0.0 and take.params == {"steps": 20, "speed": 1.2875, "silence_s": 0.3}
    assert take.style == ""                                                     # no style prompt to vary


def test_supertonic_retake_speed_stays_inside_the_cli_range(tmp_path, fake_supertonic):
    _st_take(tmp_path, transcribe=lambda p, lang: [], options={"steps": 12, "speed": 2.0})
    assert [s for _, s in _calls(fake_supertonic)] == [2.0, 1.94, 2.0]


def test_supertonic_speaks_the_respelling_without_inline_audio_tags(tmp_path, fake_supertonic):
    line = _st_line("s2", "Meet Signet <breath>. It <short pause> just works.")
    voice.synthesize_line(line, voice="F2", base_style="", model="supertonic-2", pronounce={"Signet": "Sig-nit"},
                          language="en", vo_dir=tmp_path / "vo", provider="supertonic",
                          transcribe=lambda p, lang: words_for("Meet Signet. It just works."))
    assert fake_supertonic.tts()[0][-1] == "Meet Sig-nit. It just works."


def test_supertonic_word_timings_come_from_asr_like_any_take(tmp_path, fake_supertonic):
    take = _st_take(tmp_path)
    assert [w.text for w in take.words] == ["Risk", "has", "to", "sign", "off."]
    assert [w.startMs for w in take.words] == [0, 200, 400, 600, 800]


def test_supertonic_audition_writes_one_wav_per_voice(tmp_path, fake_supertonic):
    settings = voice.VoiceSettings("supertonic", "F2", "supertonic-2", options={"steps": 12, "speed": 1.25, "silence_s": 0.3})
    costs_path = tmp_path / "costs.jsonl"
    paths = voice.audition("Hello there.", ["F1", "F2", "M1"], "ignored", tmp_path / "aud", settings=settings,
                           costs_path=costs_path)
    assert [p.name for p in paths] == ["F1.wav", "F2.wav", "M1.wav"] and all(p.exists() for p in paths)
    assert [fake_supertonic.flag(c, "--voice") for c in fake_supertonic.tts()] == ["F1", "F2", "M1"]
    rows = [json.loads(l) for l in costs_path.read_text().splitlines()]
    assert [r["provider"] for r in rows] == ["supertonic"] * 3 and all(r["units"]["local"] for r in rows)
    voice.audition("Hello there.", ["F1"], "", tmp_path / "aud", settings=settings)    # same inputs: cache hit
    assert len(fake_supertonic.tts()) == 3


# --- verify: false -----------------------------------------------------------------------------------------

def _no_asr(path, lang):
    pytest.fail("ASR must not run for a verify: false line")


def test_verify_false_skips_asr_and_splits_the_duration_by_word_length(tmp_path, fake_supertonic, monkeypatch):
    monkeypatch.setenv("FAKE_SUPERTONIC_SECONDS", "1.0")
    line = _st_line("psst", "Oh wonderful friend.", verify=False)
    take = _st_take(tmp_path, line=line, transcribe=_no_asr)
    assert take.takes == 1 and take.wer == 0.0 and take.verified is False
    assert [w.text for w in take.words] == ["Oh", "wonderful", "friend."]
    spans = [(w.startMs, w.endMs) for w in take.words]
    assert spans[0][0] == 0 and spans[-1][1] == round(take.duration_s * 1000)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))                  # contiguous
    lengths = [end - start for start, end in spans]                             # 2, 9 and 6 letters
    assert lengths[1] > lengths[2] > lengths[0]
    assert lengths[1] / lengths[0] == pytest.approx(9 / 2, rel=0.02)


def test_verify_false_still_runs_the_pace_check(tmp_path, fake_supertonic, monkeypatch):
    monkeypatch.setenv("FAKE_SUPERTONIC_SECONDS", "6.0")
    line = _st_line("psst", "Psst, over here.", verify=False)                   # 3 words: 0.95 s at 190 wpm
    take = _st_take(tmp_path, line=line, transcribe=_no_asr, pace_wpm=190)
    assert take.takes == 3 and take.verified is False
    monkeypatch.setenv("FAKE_SUPERTONIC_SECONDS", "1.0")                        # on pace (new text: not a cache hit)
    assert _st_take(tmp_path, line=_st_line("psst2", "Psst, over there.", verify=False), transcribe=_no_asr,
                    pace_wpm=190).takes == 1


def test_verify_false_still_runs_the_clip_check(tmp_path):
    line = _st_line("clip", "Psst, over here.", verify=False)
    take = voice.synthesize_line(line, voice="Kore", base_style="", model="m", pronounce={}, language="en",
                                 vo_dir=tmp_path / "vo", synth=FakeSynth(amp=[1.6, 0.3]), transcribe=_no_asr)
    assert take.takes == 2 and take.verified is False


def test_a_silent_take_is_rejected_even_when_nothing_checks_the_words(tmp_path):
    line = _st_line("hm", "Hm.", verify=False)
    take = voice.synthesize_line(line, voice="Kore", base_style="", model="m", pronounce={}, language="en",
                                 vo_dir=tmp_path / "vo", synth=FakeSynth(amp=[0.0, 0.3]), transcribe=_no_asr)
    assert take.takes == 2 and take.duration_s > voice.MIN_TAKE_S
    assert sf.info(str(tmp_path / "vo" / "hm.wav")).duration == pytest.approx(take.duration_s, abs=0.001)


def test_a_gemini_line_whose_every_take_is_silent_comes_back_as_a_problem_not_an_error(tmp_path):
    line = _st_line("hm", "Hm.", verify=False)
    take = voice.synthesize_line(line, voice="Kore", base_style="", model="m", pronounce={}, language="en",
                                 vo_dir=tmp_path / "vo", synth=FakeSynth(amp=0.0), transcribe=_no_asr)
    assert take.takes == 3 and take.problem == voice.SILENT and take.wer == 1.0 and take.duration_s < voice.MIN_TAKE_S
    assert (tmp_path / "vo" / "hm.wav").exists()                                  # the best (silent) take, as before


def test_a_verified_line_still_goes_through_asr(tmp_path, fake_supertonic):
    seen = []
    _st_take(tmp_path, transcribe=lambda p, lang: seen.append(p) or words_for("Risk has to sign off."))
    assert len(seen) == 1


# --- idempotence and settings --------------------------------------------------------------------------------

def test_supertonic_takes_are_reused_until_an_input_changes(tmp_path, fake_supertonic):
    run, sl, pack = _st_run(tmp_path)
    texts = {ln.id: ln.text for ln in sl.all_lines()}
    asr_passes = []

    def tr(path, lang):
        asr_passes.append(path)
        return words_for(texts[Path(path).stem.rsplit("-", 1)[0]])

    voice_block = {}

    def takes_made(**change):
        """Apply `change` to the shot list's voice block (they accumulate); how many ASR passes did the run cost?"""
        voice_block.update(change)
        asr_passes.clear()
        voice.synthesize_all(run, sl.model_copy(update={"voice": sl.voice.model_copy(update=voice_block)}), pack,
                             only=["c01-l1"], transcribe=tr)
        return len(asr_passes)

    assert takes_made() == 1 and takes_made() == 0                              # first run, then unchanged
    assert takes_made(speed=1.4) == 1 and takes_made() == 0
    assert takes_made(steps=20) == 1 and takes_made() == 0
    assert takes_made(silence_s=0.5) == 1 and takes_made() == 0
    assert takes_made(name="F1") == 1 and takes_made() == 0
    assert takes_made(model="supertonic") == 1 and takes_made() == 0


def test_a_retaken_supertonic_line_is_still_skipped_on_rerun(tmp_path, fake_supertonic):
    run, sl, pack = _st_run(tmp_path)
    texts = {ln.id: ln.text for ln in sl.all_lines()}

    def tr(path, lang):
        line_id, n = Path(path).stem.rsplit("-", 1)
        return words_for(texts[line_id]) if n == "2" else []                    # take 2 is the first to match

    takes = voice.synthesize_all(run, sl, pack, transcribe=tr)
    assert all(t.takes == 2 and t.params["steps"] == 16 for t in takes)
    made = len(fake_supertonic.tts())
    again = voice.synthesize_all(run, sl, pack, transcribe=tr)
    assert len(fake_supertonic.tts()) == made and [t.id for t in again] == [t.id for t in takes]


def test_switching_provider_retakes_the_line(tmp_path, fake_supertonic):
    run, sl, pack = _st_run(tmp_path)
    voice.synthesize_all(run, sl, pack, only=["c01-l1"], transcribe=lambda p, lang: words_for(sl.all_lines()[0].text))
    synth = FakeSynth()
    gem = sl.model_copy(update={"voice": sl.voice.model_copy(update={"provider": "gemini", "name": "Kore"})})
    (take,) = voice.synthesize_all(run, gem, pack, only=["c01-l1"], synth=synth,
                                   transcribe=lambda p, lang: words_for(sl.all_lines()[0].text))
    assert take.provider == "gemini" and len(synth.calls) == 1


def test_toggling_verify_retakes_the_line(tmp_path, fake_supertonic):
    run, sl, pack = _st_run(tmp_path)
    passes = []
    tr = lambda p, lang: passes.append(p) or words_for("Moving company money just got safer.")
    voice.synthesize_all(run, sl, pack, only=["c01-l1"], transcribe=tr)
    assert len(passes) == 1
    unverified = sl.model_copy(deep=True)
    unverified.chapters[0].vo[0].verify = False
    (take,) = voice.synthesize_all(run, unverified, pack, only=["c01-l1"], transcribe=_no_asr)
    assert take.verified is False and len(passes) == 1
    assert voice.synthesize_all(run, unverified, pack, only=["c01-l1"], transcribe=_no_asr)[0].verified is False
    (back,) = voice.synthesize_all(run, sl, pack, only=["c01-l1"], transcribe=tr)
    assert back.verified is True and len(passes) == 2


def test_voice_settings_resolution_order(tmp_path):
    _, sl, pack = _st_run(tmp_path)
    s = voice.voice_settings(sl, pack)                                          # pack's supertonic block
    assert (s.provider, s.voice, s.model, dict(s.options)) == (
        "supertonic", "F2", "supertonic-2", {"steps": 12, "speed": 1.25, "silence_s": 0.3})
    assert s.pace_wpm == 170                                                    # Supertonic's own pace, not the pack's 190
    over = sl.model_copy(update={"voice": sl.voice.model_copy(update={"name": "M1", "steps": 8, "speed": 1.1})})
    s = voice.voice_settings(over, pack)                                        # the shot list wins
    assert (s.voice, dict(s.options)) == ("M1", {"steps": 8, "speed": 1.1, "silence_s": 0.3})
    pack.voice.pop("supertonic")                                                # no pack block: provider defaults
    s = voice.voice_settings(sl, pack)
    assert (s.voice, s.pace_wpm) == ("F2", 190)                                 # no Supertonic pace: the pack's
    g = voice.voice_settings(Shotlist.model_validate(json.loads(FIX.read_text())), load_pack("saas-kinetic"))
    assert (g.provider, g.voice, g.model) == ("gemini", "Kore", "gemini-3.8-flash-tts")
    assert g.style.startswith("warm, confident") and g.pace_wpm == 190


def test_a_pack_can_make_supertonic_its_provider_and_a_typo_is_an_error(tmp_path):
    sl = Shotlist.model_validate(json.loads(FIX.read_text()))
    pack = load_pack("saas-kinetic")
    pack.voice["provider"] = "supertonic"
    assert voice.voice_settings(sl, pack).provider == "supertonic"
    pack.voice["provider"] = "superton1c"
    with pytest.raises(PackError, match="superton1c.*gemini, supertonic"):
        voice.voice_settings(sl, pack)


def test_audition_voices_follow_the_provider(tmp_path):
    _, sl, pack = _st_run(tmp_path)
    assert voice.audition_voices(voice.voice_settings(sl, pack), pack) == ["F2", "F1", "F3", "M1", "M2"]
    gem = Shotlist.model_validate(json.loads(FIX.read_text()))
    assert voice.audition_voices(voice.voice_settings(gem, pack), pack) == [
        "Kore", "Aoede", "Leda", "Autonoe", "Callirrhoe"]


def test_line_take_records_from_before_supertonic_still_load():
    old = '{"id": "c01-l1", "file": "audio/vo/c01-l1.wav", "duration_s": 1.5, "wer": 0.0, "takes": 2, ' \
          '"voice": "Kore", "model": "m", "style": "warm; clear diction"}'
    take = LineTake.model_validate_json(old)
    assert (take.provider, take.params, take.verified) == ("gemini", {}, True)


# --- the commands ----------------------------------------------------------------------------------------------

def test_voice_audition_with_supertonic_writes_the_named_voices(tmp_path, monkeypatch, fake_supertonic):
    run, _, _ = _st_run(tmp_path)
    monkeypatch.chdir(run.root)
    res = runner.invoke(app, ["voice", "audition", "Hi there.", "--voices", "F1,F2,M1"])
    assert res.exit_code == 0, res.output
    assert [Path(l).name for l in res.output.splitlines() if l.endswith(".wav")] == ["F1.wav", "F2.wav", "M1.wav"]
    assert all((run.auditions_dir / f"{v}.wav").exists() for v in ("F1", "F2", "M1"))
    assert [fake_supertonic.flag(c, "--voice") for c in fake_supertonic.tts()] == ["F1", "F2", "M1"]


def test_voice_audition_with_supertonic_defaults_to_its_voices(tmp_path, monkeypatch, fake_supertonic):
    run, _, _ = _st_run(tmp_path)
    monkeypatch.chdir(run.root)
    assert runner.invoke(app, ["voice", "audition", "Hi."]).exit_code == 0
    assert [fake_supertonic.flag(c, "--voice") for c in fake_supertonic.tts()] == ["F2", "F1", "F3", "M1", "M2"]


@pytest.mark.parametrize("args", [["voice", "--all"], ["voice", "audition", "Hi."]])
def test_missing_supertonic_is_a_one_line_prerequisite_error(tmp_path, monkeypatch, args):
    run, _, _ = _st_run(tmp_path)
    monkeypatch.chdir(run.root)
    monkeypatch.setenv("PATH", str(tmp_path))                                   # no supertonic anywhere
    res = runner.invoke(app, args)
    assert res.exit_code == 2, res.output
    assert res.output.count("promo:") == 1 and "Supertonic is not installed (pip install supertonic" in res.output
    assert "Traceback" not in res.output and len(res.output.strip().splitlines()) == 1


def test_a_failing_supertonic_is_a_one_line_exit_1(tmp_path, monkeypatch, fake_supertonic):
    run, _, _ = _st_run(tmp_path)
    monkeypatch.chdir(run.root)
    monkeypatch.setenv("FAKE_SUPERTONIC_FAIL", "Voice style 'Kore' not found")
    res = runner.invoke(app, ["voice", "--all"])
    assert res.exit_code == 1 and "Voice style 'Kore' not found" in res.output and "Traceback" not in res.output
    assert res.output.count("promo:") == 1


def test_voice_all_marks_unverified_lines(tmp_path, monkeypatch):
    _in_run(tmp_path, monkeypatch)
    unverified = LineTake(id="c02-l1", file="audio/vo/c02-l1.wav", duration_s=0.9, verified=False)
    monkeypatch.setattr("promo.voice.synthesize_all", lambda *a, **k: [_take("c01-l1"), unverified])
    res = runner.invoke(app, ["voice", "--all"])
    assert res.exit_code == 0, res.output
    assert "WER  n/a" in res.output and "Not verified (verify: false): c02-l1" in res.output
    assert "verified by speech recognition" in res.output


# --- ASR aliases -----------------------------------------------------------------------------------------------

def test_an_asr_alias_scores_a_known_mishearing_as_the_scripts_word(tmp_path):
    line = VOLine(id="a1", text="Oleafly puts all of it in one place.")
    heard = lambda p, lang: words_for("Oh Leafly puts all of it in one place.")
    kwargs = dict(voice="Kore", base_style="", model="m", pronounce={"Oleafly": "oh-LEEF-lee"}, language="en",
                  transcribe=heard)
    synth = FakeSynth()
    take = voice.synthesize_line(line, vo_dir=tmp_path / "plain", synth=synth, **kwargs)
    assert take.takes == 3 and take.wer > voice.MAX_WER                    # without the alias it retakes and fails
    assert synth.calls[0]["text"].startswith("oh-LEEF-lee puts")            # TTS gets the respelling either way
    take = voice.synthesize_line(line, vo_dir=tmp_path / "vo", synth=synth,
                                 asr_aliases={"Oleafly": ["oh leafly"]}, **kwargs)
    assert take.takes == 1 and take.wer == 0.0
    assert [w.text for w in take.words] == line.text.split()                # the script's words, on ASR's clock
    assert take.words[0].startMs == 0 and take.words[0].endMs == 380        # "Oh Leafly" became one word


def test_an_asr_alias_does_not_hide_a_real_error(tmp_path):
    line = VOLine(id="a2", text="Oleafly puts all of it in one place.")
    take = voice.synthesize_line(line, voice="Kore", base_style="", model="m", pronounce={}, language="en",
                                 vo_dir=tmp_path / "vo", synth=FakeSynth(), asr_aliases={"Oleafly": ["oh leafly"]},
                                 transcribe=lambda p, lang: words_for("Oh Lee Flee puts all of it in one place."))
    assert take.takes == 3 and take.wer > voice.MAX_WER                    # "oh lee flee" is not a listed mishearing


def test_voice_settings_merge_pack_and_shot_list_aliases_with_the_shot_list_winning():
    doc = json.loads(FIX.read_text())
    doc["asr_aliases"] = {"Oleafly": ["oh leafly"], "Typst": ["tipst"]}
    pack = load_pack("saas-kinetic")
    pack.voice["asr_aliases"] = {"Typst": ["tips"], "CV": ["c v"]}
    assert voice.voice_settings(Shotlist.model_validate(doc), pack).aliases == {
        "Typst": ["tipst"], "CV": ["c v"], "Oleafly": ["oh leafly"]}
    assert voice.voice_settings(Shotlist.model_validate(json.loads(FIX.read_text())), load_pack("saas-kinetic")
                                ).aliases == {}


def test_changing_an_alias_rescores_only_the_lines_it_applies_to(tmp_path):
    run, sl, pack = _demo_run(tmp_path)
    heard = {"c01-l1": "Moving company money just got safe her", "c02-l1": "Reversals over ten thousand dollars",
             "c02-l2": "Risk has to sign off"}
    passes = []

    def tr(path, lang):
        passes.append(Path(path).stem.rsplit("-", 1)[0])
        return words_for(heard[passes[-1]])

    synth = FakeSynth()

    def rerun(**update):
        passes.clear()
        voice.synthesize_all(run, sl.model_copy(update=update), pack, synth=synth, transcribe=tr)
        return sorted(set(passes))

    assert rerun() == ["c01-l1", "c02-l1", "c02-l2"]
    first = voice.synthesize_all(run, sl, pack, synth=synth, transcribe=tr)
    assert {t.id: t.takes for t in first}["c01-l1"] == 3 and first[0].wer > voice.MAX_WER   # "safe her" is a miss
    assert rerun() == []                                                                    # unchanged
    calls = len(synth.calls)
    assert rerun(asr_aliases={"Typst": ["tips"]}) == []                                    # no line says Typst
    assert rerun(asr_aliases={"safer": ["safe her"]}) == ["c01-l1"]                        # only the line that says safer
    assert len(synth.calls) == calls                                                       # same audio, from the cache
    (take,) = [t for t in voice.synthesize_all(run, sl.model_copy(update={"asr_aliases": {"safer": ["safe her"]}}),
                                               pack, synth=synth, transcribe=tr) if t.id == "c01-l1"]
    assert take.wer == 0.0 and take.takes == 1
    assert rerun(asr_aliases={"safer": ["safe her"]}) == []
    assert rerun(asr_aliases={"safer": ["safe her", "save her"]}) == ["c01-l1"]            # the alias list changed


def test_aliases_do_not_change_the_fingerprint_of_a_line_they_do_not_apply_to():
    line = VOLine(id="f1", text="Risk has to sign off.")
    plain = voice._fingerprint(line, "en", {}, {})
    assert voice._fingerprint(line, "en", {}, {"Typst": ["tips"]}) == plain
    assert voice._fingerprint(line, "en", {}, {"sign": ["sine"]}) != plain
    assert voice._fingerprint(VOLine(id="f1", text=line.text, verify=False), "en", {}, {"sign": ["sine"]}) == plain


# --- silent draws fall through, they do not abort the run ----------------------------------------------------------

def test_five_silent_draws_fall_through_to_the_next_retake_variant(tmp_path, fake_supertonic, monkeypatch):
    draws = voice.supertonic.SILENT_DRAWS
    monkeypatch.setenv("FAKE_SUPERTONIC_SILENT", str(draws))
    take = _st_take(tmp_path, line=_st_line("hm", "Hm.", verify=False), transcribe=_no_asr)
    assert len(fake_supertonic.tts()) == draws + 1
    assert take.takes == 2 and take.problem == "" and take.params["steps"] == 16 and take.duration_s > 0.1
    assert (tmp_path / "vo" / "hm.wav").exists()


def test_a_line_that_is_silent_in_every_variant_is_stored_as_a_check_and_the_run_moves_on(tmp_path, fake_supertonic,
                                                                                         monkeypatch):
    run, sl, pack = _st_run(tmp_path)
    monkeypatch.setenv("FAKE_SUPERTONIC_SILENT", "15")                           # 3 variants x 5 draws: the first line only
    takes = voice.synthesize_all(run, sl, pack, transcribe=lambda p, lang: words_for(
        {ln.id: ln.text for ln in sl.all_lines()}[Path(p).stem.rsplit("-", 1)[0]]))
    first, *rest = takes
    assert (first.id, first.problem, first.wer, first.duration_s, first.takes) == ("c01-l1", voice.SILENT, 1.0, 0.0, 3)
    assert not (run.vo_dir / "c01-l1.wav").exists() and (run.vo_dir / "c01-l1.json").exists()
    assert [t.id for t in rest] == ["c02-l1", "c02-l2"] and all(t.problem == "" for t in rest)
    # a line with a problem is never "unchanged": the next run tries again, and now the draws speak
    calls = len(fake_supertonic.tts())
    again = voice.synthesize_all(run, sl, pack, transcribe=lambda p, lang: words_for(
        {ln.id: ln.text for ln in sl.all_lines()}[Path(p).stem.rsplit("-", 1)[0]]))
    assert again[0].problem == "" and again[0].duration_s > 0.1 and (run.vo_dir / "c01-l1.wav").exists()
    assert len(fake_supertonic.tts()) > calls


def test_voice_all_reports_a_silent_line_as_check_with_its_reason(tmp_path, monkeypatch):
    _in_run(tmp_path, monkeypatch)
    silent = LineTake(id="c02-l1", file="audio/vo/c02-l1.wav", duration_s=0.0, wer=1.0, problem=voice.SILENT)
    monkeypatch.setattr("promo.voice.synthesize_all", lambda *a, **k: [_take("c01-l1"), silent])
    res = runner.invoke(app, ["voice", "--all"])
    assert res.exit_code == 1, res.output
    assert "CHECK (every take came back silent)" in res.output
    text = " ".join(res.output.split())
    assert "still mismatch or are silent" in text and "asr_aliases" in text and "--force" in text


def test_voice_help_says_supertonic_draws_vary_and_the_cache_pins_them():
    help_text = " ".join(runner.invoke(app, ["voice", "--help"]).output.split())
    assert "varies from draw to draw" in help_text and "fresh draw" in help_text and "--force" in help_text


# --- the take's words are the script's own ----------------------------------------------------------------------

def _asr(spec):
    """'word:start-end ...' as ASR words."""
    out = []
    for part in spec.split():
        text, span = part.rsplit(":", 1)
        a, b = span.split("-")
        out.append(Caption(text=text, startMs=int(a), endMs=int(b), timestampMs=int(a)))
    return out


def _words_of(text, heard, aliases=None, tmp_path=None):
    line = VOLine(id="w1", text=text)
    return voice.synthesize_line(line, voice="Kore", base_style="", model="m", pronounce={}, language="en",
                                 vo_dir=tmp_path / "vo", synth=FakeSynth(seconds=3.0), asr_aliases=aliases,
                                 transcribe=lambda p, lang: _asr(heard)).words


def test_a_hyphenated_script_word_spans_its_asr_parts(tmp_path):
    words = _words_of("They stay read-only.", "They:0-300 stay:300-700 read:800-1100 only.:1100-1500", tmp_path=tmp_path)
    assert [(w.text, w.startMs, w.endMs) for w in words] == [("They", 0, 300), ("stay", 300, 700), ("read-only.", 800, 1500)]


def test_a_script_word_keeps_its_spelling_when_asr_wrote_the_other_one(tmp_path):
    heard = "Your:0-200 own:200-400 color:400-800 and:800-900 twenty:900-1200 one:1200-1400 languages.:1400-2000"
    words = _words_of("Your own colour, and twenty-one languages.", heard, tmp_path=tmp_path)
    assert [(w.text, w.startMs, w.endMs) for w in words] == [
        ("Your", 0, 200), ("own", 200, 400), ("colour,", 400, 800), ("and", 800, 900), ("twenty-one", 900, 1400),
        ("languages.", 1400, 2000)]


def test_a_merged_asr_token_is_shared_between_the_script_words_it_stands_for(tmp_path):
    heard = "The:0-100 tools:100-300 beta:300-500 handles:500-800 the:800-900 chores,22:900-1800 converters,:1800-2300"
    words = _words_of("The Tools beta handles the chores: twenty-two converters,", heard, tmp_path=tmp_path)
    by_text = {w.text: (w.startMs, w.endMs) for w in words}
    assert [w.text for w in words] == "The Tools beta handles the chores: twenty-two converters,".split()
    assert by_text["chores:"][0] == 900 and by_text["chores:"][1] == by_text["twenty-two"][0]    # shares 900-1800
    assert by_text["twenty-two"][1] == 1800 and by_text["converters,"] == (1800, 2300)


def test_alias_and_hyphen_together(tmp_path):
    heard = "On:0-100 the:100-200 ATS:200-500 friendly:500-900 C:900-1000 V:1000-1100 templates:1200-1700"
    words = _words_of("On the ATS-friendly CV templates", heard, {"CV": ["c v"]}, tmp_path)
    assert [(w.text, w.startMs, w.endMs) for w in words] == [
        ("On", 0, 100), ("the", 100, 200), ("ATS-friendly", 200, 900), ("CV", 900, 1100), ("templates", 1200, 1700)]


def test_a_word_asr_dropped_gets_a_share_of_the_gap_and_an_empty_take_is_spread(tmp_path):
    words = _words_of("Ask Oleafly then wait", "Ask:0-300 then:900-1100 wait:1100-1400", tmp_path=tmp_path)
    assert [w.text for w in words] == ["Ask", "Oleafly", "then", "wait"]
    assert (words[1].startMs, words[1].endMs) == (300, 900)
    words = _words_of("Hello there friend", "", tmp_path=tmp_path)                 # ASR heard nothing
    assert [w.text for w in words] == ["Hello", "there", "friend"] and words[0].startMs == 0
    assert words[-1].endMs == round(words[-1].endMs) > 2000                          # spread over the whole take


def _stale_run(tmp_path):
    """A demo run synthesised once, then c01-l1's record rewritten the way an older harness stored it: ASR's words."""
    run, sl, pack = _demo_run(tmp_path)
    texts = {ln.id: ln.text for ln in sl.all_lines()}
    passes = []

    def tr(path, lang):
        passes.append(Path(path))
        stem = Path(path).stem                                                    # c01-l1 (the kept take) or c01-l1-2
        return words_for(texts[stem if stem in texts else stem.rsplit("-", 1)[0]])

    synth = FakeSynth()
    voice.synthesize_all(run, sl, pack, synth=synth, transcribe=tr)
    passes.clear()
    record = run.vo_dir / "c01-l1.json"
    old = LineTake.model_validate_json(record.read_text())
    old.words = [w.model_copy(update={"text": w.text.lower().strip(".")}) for w in old.words]
    record.write_text(old.model_dump_json())
    return run, sl, pack, texts, passes, tr, synth, record


def test_a_stored_take_with_raw_asr_words_is_rescored_not_skipped(tmp_path):
    run, sl, pack, texts, passes, tr, synth, record = _stale_run(tmp_path)
    voice.synthesize_all(run, sl, pack, synth=synth, transcribe=tr)
    assert passes == [run.vo_dir / "c01-l1.wav"]                                  # one ASR pass, on the kept take
    assert [w.text for w in LineTake.model_validate_json(record.read_text()).words] == texts["c01-l1"].split()
    passes.clear()
    voice.synthesize_all(run, sl, pack, synth=synth, transcribe=tr)
    assert passes == []                                                           # and now it is unchanged


def test_rescoring_a_stale_take_never_synthesises_even_when_the_cache_is_gone(tmp_path, monkeypatch):
    run, sl, pack, texts, passes, tr, synth, record = _stale_run(tmp_path)
    before, wav = len(synth.calls), (run.vo_dir / "c01-l1.wav").read_bytes()
    monkeypatch.setenv("PROMO_CACHE_DIR", str(tmp_path / "an-empty-cache"))        # every cached synthesis is gone
    (take,) = voice.synthesize_all(run, sl, pack, only=["c01-l1"], transcribe=tr,
                                   synth=lambda *a, **k: pytest.fail("must not synthesise"))
    assert len(synth.calls) == before and (run.vo_dir / "c01-l1.wav").read_bytes() == wav   # same audio, no new call
    assert [w.text for w in take.words] == texts["c01-l1"].split() and take.takes == 1


def test_a_stale_take_whose_wav_is_gone_is_synthesised_again(tmp_path, monkeypatch):
    run, sl, pack, texts, passes, tr, synth, record = _stale_run(tmp_path)
    (run.vo_dir / "c01-l1.wav").unlink()
    monkeypatch.setenv("PROMO_CACHE_DIR", str(tmp_path / "an-empty-cache"))        # neither the wav nor the cache
    before = len(synth.calls)
    voice.synthesize_all(run, sl, pack, only=["c01-l1"], synth=synth, transcribe=tr)
    assert len(synth.calls) == before + 1 and (run.vo_dir / "c01-l1.wav").exists()
    assert passes == [run.vo_dir / ".takes" / "c01-l1-1.wav"]


# --- the anchors that broke on the real voice -----------------------------------------------------------------------

def _anchor_offsets(tmp_path, text, heard, words, aliases=None):
    """Seconds from the line's start at which each word anchor in `words` fires, after voice and timing."""
    from promo.timing import Measured, resolve

    doc = json.loads(FIX.read_text())
    doc["chapters"][0]["vo"][0]["text"] = text
    events = doc["chapters"][0]["shots"][0]["events"]
    events[1:] = [{"id": f"w{i}", "kind": "click", "at": {"line": "c01-l1", "word": w}} for i, w in enumerate(words)]
    sl = Shotlist.model_validate(doc)
    line = sl.all_lines()[0]
    take = voice.synthesize_line(line, voice="Kore", base_style="", model="m", pronounce={}, language="en",
                                 vo_dir=tmp_path / "vo", synth=FakeSynth(seconds=3.0), asr_aliases=aliases,
                                 transcribe=lambda p, lang: _asr(heard))
    timing = resolve(sl, Measured(takes={"c01-l1": take}), load_pack("saas-kinetic").limits())
    assert [f for f in timing.findings if f.code == "word_not_found"] == []
    start = timing.line("c01-l1").start_s
    return {e.id: round(e.t - start, 3) for e in timing.events if e.id.startswith("w")}


def test_anchor_on_a_hyphenated_word_or_either_of_its_parts(tmp_path):
    heard = "Link:0-200 your:200-350 data:350-600 folders.:600-1000 They:1100-1300 stay:1300-1600 read:1700-1950 only.:1950-2300"
    at = _anchor_offsets(tmp_path, "Link your data folders. They stay read-only.", heard, ["read-only", "read", "only"])
    assert at == {"w0": 1.7, "w1": 1.7, "w2": 1.95}                      # "only" starts where ASR put it


def test_anchor_on_a_hyphenated_word_that_asr_split_into_parts_and_a_letter_pair(tmp_path):
    heard = "On:0-100 the:100-200 ATS:200-500 friendly:500-900 C:900-1000 V:1000-1100 templates:1200-1700"
    at = _anchor_offsets(tmp_path, "On the ATS-friendly CV templates", heard,
                         ["ATS-friendly", "ATS", "friendly", "CV", "templates"], {"CV": ["c v"]})
    assert at["w0"] == at["w1"] == 0.2 and at["w2"] == 0.5               # "friendly" where ASR put it, not by letters
    assert at["w3"] == 0.9 and at["w4"] == 1.2


def test_anchor_on_a_british_spelling_asr_wrote_the_american_way(tmp_path):
    heard = "Dark:0-300 mode:300-600 your:700-900 own:900-1100 color:1100-1500 and:1500-1600 twenty:1600-1900 one:1900-2100 languages.:2100-2700"
    at = _anchor_offsets(tmp_path, "Dark mode, your own colour, and twenty-one languages.", heard,
                         ["colour,", "colour", "and", "twenty-one", "languages."])
    assert at == {"w0": 1.1, "w1": 1.1, "w2": 1.5, "w3": 1.6, "w4": 2.1}


def test_anchor_on_a_word_asr_merged_with_a_number(tmp_path):
    heard = "The:0-100 tools:100-300 beta:300-500 handles:500-800 the:800-900 chores,22:900-1800 converters,:1800-2300 statistics:2300-2900"
    at = _anchor_offsets(tmp_path, "The Tools beta handles the chores: twenty-two converters, statistics",
                         heard, ["chores:", "chores", "converters,", "statistics"])
    assert at == {"w0": 0.9, "w1": 0.9, "w2": 1.8, "w3": 2.3}


# --- provider details ------------------------------------------------------------------------------------------------

def test_supertonic_gets_the_primary_language_subtag_and_the_cache_keeps_the_full_tag(tmp_path, fake_supertonic):
    line = _st_line("g1", "Risk has to sign off.")
    for lang in ("en-GB", "en"):
        voice.synthesize_line(line, voice="F2", base_style="", model="supertonic-2", pronounce={}, language=lang,
                              vo_dir=tmp_path / "vo", provider="supertonic",
                              transcribe=lambda p, l: words_for(line.text))
    assert [fake_supertonic.flag(c, "--lang") for c in fake_supertonic.tts()] == ["en", "en"]
    assert len(fake_supertonic.tts()) == 2                                       # en-GB and en are two cache entries


def test_pack_asr_aliases_are_validated_like_the_shot_list(tmp_path):
    sl = Shotlist.model_validate(json.loads(FIX.read_text()))
    pack = load_pack("saas-kinetic")
    for bad in ({"Typst": "tips"}, {"Typst": []}, {"Typst": [""]}, {"": ["tips"]}, ["Typst"]):
        pack.voice["asr_aliases"] = bad
        with pytest.raises(PackError, match=r"pack saas-kinetic: voice\.asr_aliases"):
            voice.voice_settings(sl, pack)
    pack.voice["asr_aliases"] = {"Typst": ["tips"]}
    assert voice.voice_settings(sl, pack).aliases == {"Typst": ["tips"]}


# --- a bad language, and a line with a problem, as the commands see them ------------------------------------------------

def test_voice_all_prints_a_value_error_as_one_line(tmp_path, monkeypatch):
    _in_run(tmp_path, monkeypatch)

    def refuse(*a, **k):
        raise ValueError("speech recognition (Whisper) cannot transcribe language 'xx': Unsupported language: xx")

    monkeypatch.setattr("promo.voice.synthesize_all", refuse)
    res = runner.invoke(app, ["voice", "--all"])
    assert res.exit_code == 1 and "Unsupported language: xx" in res.output
    assert res.output.count("promo:") == 1 and "Traceback" not in res.output and len(res.output.strip().splitlines()) == 1


def test_a_british_english_shot_list_runs_through_parakeet_and_supertonic(tmp_path, monkeypatch, fake_supertonic):
    """language en-GB: Parakeet (not Whisper), the CLI gets `en`, and the spelling rules apply."""
    from promo import asr

    seen = []
    monkeypatch.setattr(asr, "_transcribe_parakeet", lambda p: seen.append("parakeet") or words_for("Pick a color."))
    monkeypatch.setattr(asr, "_transcribe_whisper", lambda p, lang: pytest.fail("en-GB must not go to Whisper"))
    line = _st_line("b1", "Pick a colour.")
    take = voice.synthesize_line(line, voice="F2", base_style="", model="supertonic-2", pronounce={}, language="en-GB",
                                 vo_dir=tmp_path / "vo", provider="supertonic")          # the default transcribe
    assert seen == ["parakeet"] and take.wer == 0.0 and [w.text for w in take.words] == ["Pick", "a", "colour."]
    assert fake_supertonic.flag(fake_supertonic.tts()[0], "--lang") == "en"


def _timing_with_a_problem_line(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch)
    bad = LineTake(id="c01-l1", file="audio/vo/c01-l1.wav", duration_s=0.0, wer=1.0, problem=voice.SILENT)
    (run.vo_dir / "c01-l1.json").write_text(bad.model_dump_json())
    res = runner.invoke(app, ["timing"])
    assert res.exit_code == 0, res.output
    return run


def test_promo_captions_sees_a_problem_line_as_estimated(tmp_path, monkeypatch):
    from promo.timing import load_timing

    run = _timing_with_a_problem_line(tmp_path, monkeypatch)
    (line,) = [l for l in load_timing(run.timing).lines if l.id == "c01-l1"]
    assert line.estimated and line.file is None
    res = runner.invoke(app, ["captions"])
    assert res.exit_code == 2 and "no voice take yet" in res.output


def test_assemble_leaves_a_problem_line_out_of_the_vo_track(tmp_path, monkeypatch):
    from promo.timeline import collect_media
    from promo.timing import load_timing

    run = _timing_with_a_problem_line(tmp_path, monkeypatch)
    (run.vo_dir / "c01-l1.wav").write_bytes(b"")                       # even a file left behind is not used
    assert "c01-l1" not in collect_media(run, load_timing(run.timing)).vo
