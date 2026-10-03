import inspect
import json
import subprocess
from pathlib import Path

import httpx
import numpy as np
import pytest
import respx
import soundfile as sf
from typer.testing import CliRunner

from promo import sfx, sfx_synth
from promo.cli import app
from promo.config import MissingKeyError
from promo.packs import Limits, load_pack
from promo.proc import ToolError
from promo.providers import cache, elevenlabs
from promo.providers.http import ProviderError
from promo.run import new_run
from promo.shotlist import Shotlist
from promo.timing import Measured, resolve, write_timing

FIX = Path(__file__).parent / "fixtures" / "shotlist_music.json"
runner = CliRunner()
ONE_SAMPLE = 1 / 48000  # a sound peak lands on its event to within one sample


def _fake_key_applies(node) -> bool:
    """Live tests must read the real key from .env, so they never get the fake one."""
    return node.get_closest_marker("live") is None


@pytest.fixture(autouse=True)
def key(request, monkeypatch):
    if _fake_key_applies(request.node):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test")


def test_fake_key_skips_live_tests():
    class Node:
        def __init__(self, marker):
            self.marker = marker

        def get_closest_marker(self, name):
            return self.marker if name == "live" else None

    assert _fake_key_applies(Node(None)) is True
    assert _fake_key_applies(Node(object())) is False


def _no_key(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY")
    monkeypatch.setattr("promo.config.load_env", lambda path=None: None)
    monkeypatch.setattr("promo.cli.load_env", lambda path=None: None)


def _data():
    return json.loads(FIX.read_text())


def _timing(data=None):
    return resolve(Shotlist.model_validate(data or _data()), Measured(), Limits())


def _mp3(tmp_path, name="g") -> bytes:
    wav = tmp_path / f"{name}.wav"
    sf.write(wav, (np.hanning(4800) * 0.5).astype(np.float32), 48000)
    mp3 = tmp_path / f"{name}.mp3"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), str(mp3)], check=True)
    return mp3.read_bytes()


def _dominant_hz(x):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / sfx_synth.SR)[np.argmax(spec)]


def _hidden(folder: Path) -> list[str]:
    return [p.name for p in folder.iterdir() if p.name.startswith(".")]


# --- the synthesiser ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", sfx_synth.KINDS)
def test_every_kind_is_deterministic_and_bounded(kind):
    a = sfx_synth.synth(kind, seed=3)
    b = sfx_synth.synth(kind, seed=3)
    assert a.dtype == np.float32 and len(a) > 100
    assert np.array_equal(a, b)
    assert np.max(np.abs(a)) == pytest.approx(0.9, abs=1e-3)


@pytest.mark.parametrize("kind", ["click", "key", "whoosh", "riser", "impact"])
def test_the_seed_changes_the_noise(kind):
    assert not np.array_equal(sfx_synth.synth(kind, seed=1), sfx_synth.synth(kind, seed=2))


def test_tick_pitch_rises_with_value():
    low = _dominant_hz(sfx_synth.synth("tick", value=0))
    high = _dominant_hz(sfx_synth.synth("tick", value=100))
    assert low == pytest.approx(1000, rel=0.05) and high == pytest.approx(4400, rel=0.05)


def test_value_maps_to_pitch_on_a_log_scale_and_clamps():
    mid = _dominant_hz(sfx_synth.synth("tick", value=50))
    assert mid == pytest.approx(1000 * 4.4 ** 0.5, rel=0.05)  # the geometric middle, not the arithmetic 2700
    assert _dominant_hz(sfx_synth.synth("tick", value=-40)) == pytest.approx(1000, rel=0.05)
    assert _dominant_hz(sfx_synth.synth("tick", value=900)) == pytest.approx(4400, rel=0.05)


def test_value_range_and_pitch_range_are_honoured():
    x = sfx_synth.synth("tick", value=1.0, value_range=(0.0, 1.0), pitch_range=(500.0, 2000.0))
    assert _dominant_hz(x) == pytest.approx(2000, rel=0.05)


def test_an_explicit_pitch_beats_value():
    assert _dominant_hz(sfx_synth.synth("tick", value=100, pitch=1500)) == pytest.approx(1500, rel=0.05)


@pytest.mark.parametrize("kwargs, text", [
    ({"kind": "kazoo"}, "unknown sfx kind 'kazoo'"),
    ({"kind": "tick", "duration": 0}, "duration"),
    ({"kind": "tick", "duration": -1}, "duration"),
    ({"kind": "tick", "duration": float("nan")}, "duration"),
    ({"kind": "tick", "duration": 600}, "duration"),
    ({"kind": "tick", "pitch": 0}, "pitch"),
    ({"kind": "tick", "pitch": 30000}, "pitch"),
    ({"kind": "tick", "value": 5, "pitch_range": (0.0, 4400.0)}, "pitch_range"),
])
def test_impossible_parameters_are_a_named_error(kwargs, text):
    with pytest.raises(sfx_synth.SfxError, match=text):
        sfx_synth.synth(**kwargs)
    assert issubclass(sfx_synth.SfxError, ValueError)


def test_write_wav_is_24_bit_mono_48k_and_byte_identical(tmp_path):
    x = sfx_synth.synth("pop", seed=5)
    a = sfx_synth.write_wav(x, tmp_path / "deep" / "a.wav")
    b = sfx_synth.write_wav(sfx_synth.synth("pop", seed=5), tmp_path / "b.wav")
    info = sf.info(str(a))
    assert (info.samplerate, info.channels, info.subtype) == (48000, 1, "PCM_24")
    assert a.read_bytes() == b.read_bytes()
    assert not _hidden(tmp_path) and not _hidden(tmp_path / "deep")


def test_write_wav_to_an_unwritable_place_is_an_oserror_and_leaves_nothing(tmp_path):
    (tmp_path / "f").write_text("a file, not a folder")
    with pytest.raises(OSError):
        sfx_synth.write_wav(sfx_synth.synth("tick"), tmp_path / "f" / "x.wav")
    assert not _hidden(tmp_path)


# --- the stem -------------------------------------------------------------------------------------------------

def test_render_stem_puts_peaks_on_events(tmp_path):
    sl = Shotlist.model_validate(json.loads(FIX.read_text()))
    t = resolve(sl, Measured(), Limits())
    stem, placements, warnings = sfx.render_stem(t, load_pack("saas-kinetic"), tmp_path / "track.wav")
    data, sr = sf.read(str(stem), dtype="float32")
    assert sr == 48000 and len(data) == round(t.duration_s * sr)
    assert warnings == []
    assert [p.event for p in placements] == [e.id for e in t.events]
    for e in t.events:
        assert sfx.local_peak_s(data, sr, e.t) == pytest.approx(e.t, abs=ONE_SAMPLE)


def test_render_stem_writes_placements_next_to_it_and_leaves_no_temp_files(tmp_path):
    t = _timing()
    stem, placements, _ = sfx.render_stem(t, load_pack("saas-kinetic"), tmp_path / "out" / "track.wav")
    rows = json.loads((stem.parent / "placements.json").read_text())
    assert [r["event"] for r in rows] == [e.id for e in t.events]
    assert [r["sfx"] for r in rows] == ["headline", "click", "card", "tick"]
    for r, e in zip(rows, t.events):
        assert r["t"] == e.t and 0 <= r["start_s"] <= e.t
    assert not _hidden(stem.parent)


def test_two_renders_are_byte_identical(tmp_path):
    t = _timing()
    pack = load_pack("saas-kinetic")
    sfx.render_stem(t, pack, tmp_path / "a" / "track.wav")
    sfx.render_stem(t, pack, tmp_path / "b" / "track.wav")
    assert (tmp_path / "a" / "track.wav").read_bytes() == (tmp_path / "b" / "track.wav").read_bytes()
    assert (tmp_path / "a" / "placements.json").read_bytes() == (tmp_path / "b" / "placements.json").read_bytes()


def test_sfx_gain_db_scales_the_whole_stem(tmp_path):
    t = _timing()
    peaks = {}
    for db in (-6, -18):
        pack = load_pack("saas-kinetic")
        pack.sfx["sfx_gain_db"] = db
        stem, _, _ = sfx.render_stem(t, pack, tmp_path / f"{db}.wav")
        peaks[db] = float(np.max(np.abs(sf.read(str(stem), dtype="float32")[0])))
    assert peaks[-6] == pytest.approx(0.9 * 10 ** (-6 / 20), rel=0.01)
    assert peaks[-18] == pytest.approx(0.9 * 10 ** (-18 / 20), rel=0.01)


@pytest.mark.parametrize("bad", ["loud", None, True])
def test_a_non_numeric_gain_is_a_named_error(tmp_path, bad):
    pack = load_pack("saas-kinetic")
    pack.sfx["sfx_gain_db"] = bad
    with pytest.raises(sfx.SfxError, match="sfx_gain_db"):
        sfx.render_stem(_timing(), pack, tmp_path / "t.wav")


def test_unknown_sfx_is_warned_and_skipped(tmp_path):
    data = json.loads(FIX.read_text())
    data["chapters"][1]["shots"][1]["events"][0]["sfx"] = "kazoo"
    data["chapters"][1]["shots"][1]["events"][0]["kind"] = "kazoo"
    t = resolve(Shotlist.model_validate(data), Measured(), Limits())
    _, placements, warnings = sfx.render_stem(t, load_pack("saas-kinetic"), tmp_path / "track.wav")
    assert "c02-e2" not in [p.event for p in placements]
    assert any("kazoo" in w for w in warnings)


def test_an_event_without_an_sfx_name_uses_its_kind(tmp_path):
    data = _data()
    del data["chapters"][0]["shots"][0]["events"][1]["sfx"]  # kind "click" stays
    _, placements, warnings = sfx.render_stem(_timing(data), load_pack("saas-kinetic"), tmp_path / "t.wav")
    assert [p.sfx for p in placements][1] == "click" and warnings == []


def test_an_event_with_no_sfx_and_no_matching_kind_is_silent_not_a_warning(tmp_path):
    data = _data()
    ev = data["chapters"][0]["shots"][0]["events"][1]
    del ev["sfx"]
    ev["kind"] = "scribble"
    _, placements, warnings = sfx.render_stem(_timing(data), load_pack("saas-kinetic"), tmp_path / "t.wav")
    assert "c01-e2" not in [p.event for p in placements] and warnings == []


def test_event_value_drives_the_tick_pitch_in_the_stem(tmp_path):
    data = _data()
    data["chapters"][1]["shots"][0]["events"][0].update({"kind": "tick", "sfx": "tick", "value": 0})
    data["chapters"][1]["shots"][1]["events"][0]["value"] = 100
    stem, placements, _ = sfx.render_stem(_timing(data), load_pack("saas-kinetic"), tmp_path / "t.wav")
    x, sr = sf.read(str(stem), dtype="float32")
    start = {p.event: round(p.start_s * sr) for p in placements}
    n = round(0.04 * sr)
    assert _dominant_hz(x[start["c02-e1"]:start["c02-e1"] + n]) == pytest.approx(1000, rel=0.08)
    assert _dominant_hz(x[start["c02-e2"]:start["c02-e2"] + n]) == pytest.approx(4400, rel=0.08)


def test_synth_params_in_a_pack_entry_reach_the_synth(tmp_path):
    pack = load_pack("saas-kinetic")
    pack.sfx["tick"] = {"synth": "tick", "pitch": 3000}
    stem, placements, _ = sfx.render_stem(_timing(), pack, tmp_path / "t.wav")
    x, sr = sf.read(str(stem), dtype="float32")
    p = next(p for p in placements if p.event == "c02-e2")
    assert _dominant_hz(x[round(p.start_s * sr):][: round(0.04 * sr)]) == pytest.approx(3000, rel=0.08)


def test_a_sound_that_peaks_after_its_event_is_cropped_not_shifted(tmp_path):
    """Whoosh peaks late in its body; an event at t=0.1 s leaves no room for the lead-in, so it starts cut."""
    t = _timing()
    early = t.events[0].model_copy(update={"id": "early", "t": 0.1, "sfx": "riser", "kind": "riser"})
    pack = load_pack("saas-kinetic")
    pack.sfx["riser"] = {"synth": "riser", "duration": 1.5}
    stem, placements, _ = sfx.render_stem(t.model_copy(update={"events": [early]}), pack, tmp_path / "t.wav")
    data, sr = sf.read(str(stem), dtype="float32")
    assert placements[0].start_s == 0.0
    assert sfx.local_peak_s(data, sr, 0.1, before=0.1, after=0.1) == pytest.approx(0.1, abs=ONE_SAMPLE)


def test_simultaneous_sounds_are_scaled_down_with_a_warning_and_keep_their_peak_time(tmp_path):
    t = _timing()
    pile = [t.events[0].model_copy(update={"id": f"x{i}", "t": 1.0, "sfx": "tick", "kind": "tick"}) for i in range(6)]
    stem, _, warnings = sfx.render_stem(t.model_copy(update={"events": pile}), load_pack("saas-kinetic"),
                                        tmp_path / "t.wav")
    data, sr = sf.read(str(stem), dtype="float32")
    assert float(np.max(np.abs(data))) <= 0.99 + 1e-4
    assert any("scaled down" in w for w in warnings)
    assert sfx.local_peak_s(data, sr, 1.0) == pytest.approx(1.0, abs=ONE_SAMPLE)


def _write_hit(path: Path, peak_at_s: float = 0.2, sr: int = 44100) -> None:
    k = np.arange(int(0.5 * sr)) / sr
    x = 0.05 * np.sin(2 * np.pi * 300 * k) * np.exp(-k * 20)
    x[int(peak_at_s * sr)] = 0.8  # the one sample that is clearly the loudest
    sf.write(path, x.astype(np.float32), sr)


def test_a_file_spec_is_resampled_and_peak_aligned(tmp_path):
    _write_hit(tmp_path / "hit.wav", peak_at_s=0.2)
    pack = load_pack("saas-kinetic").model_copy(update={"root": tmp_path})
    pack.sfx["click"] = {"file": "hit.wav"}
    t = _timing()
    stem, placements, warnings = sfx.render_stem(t, pack, tmp_path / "t.wav")
    data, sr = sf.read(str(stem), dtype="float32")
    ev = t.events[1]
    assert ev.id == "c01-e2" and warnings == []
    assert sfx.local_peak_s(data, sr, ev.t) == pytest.approx(ev.t, abs=ONE_SAMPLE)
    assert placements[1].start_s == pytest.approx(ev.t - 0.2, abs=0.002)


def test_a_missing_sample_file_is_a_named_error(tmp_path):
    pack = load_pack("saas-kinetic").model_copy(update={"root": tmp_path})
    pack.sfx["click"] = {"file": "nope.wav"}
    with pytest.raises(sfx.SfxError, match=r"c01-e2.*'click'.*nope\.wav"):
        sfx.render_stem(_timing(), pack, tmp_path / "t.wav")


def test_an_undecodable_sample_file_is_a_named_error_not_a_tool_error(tmp_path):
    (tmp_path / "bad.wav").write_bytes(b"not audio at all")
    pack = load_pack("saas-kinetic").model_copy(update={"root": tmp_path})
    pack.sfx["click"] = {"file": "bad.wav"}
    with pytest.raises(sfx.SfxError, match=r"bad\.wav"):
        sfx.render_stem(_timing(), pack, tmp_path / "t.wav")


@pytest.mark.parametrize("spec, text", [
    ({"synth": "kazoo"}, "unknown sfx kind 'kazoo'"),
    ({"synth": "tick", "duration": -1}, "duration"),
    ({"volume": 3}, "needs synth, file or gen"),
    ({"gen": "", "duration": 0.5}, "prompt"),
    ({"gen": "a click", "duration": "long"}, "duration"),
    ({"gen": "a click", "duration": 0}, "duration"),
    ({"file": 3}, "file"),
])
def test_a_broken_pack_entry_is_a_named_error(tmp_path, spec, text):
    pack = load_pack("saas-kinetic")
    pack.sfx["click"] = spec

    def boom(prompt, duration):
        raise AssertionError("a broken entry must not reach the paid provider")

    with pytest.raises(sfx.SfxError, match=text) as err:
        sfx.render_stem(_timing(), pack, tmp_path / "t.wav", gen=boom)
    assert "c01-e2" in str(err.value) and "click" in str(err.value)
    assert not (tmp_path / "t.wav").exists()


def test_gen_spec_uses_elevenlabs_once(tmp_path):
    wav = tmp_path / "g.wav"
    sf.write(wav, (np.hanning(4800) * 0.5).astype(np.float32), 48000)
    mp3 = tmp_path / "g.mp3"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), str(mp3)], check=True)
    calls = []

    def fake_gen(text, duration_s):
        calls.append(text)
        return mp3.read_bytes()

    pack = load_pack("saas-kinetic")
    pack.sfx["click"] = {"gen": "soft glassy UI click", "duration": 0.5}
    sl = Shotlist.model_validate(json.loads(FIX.read_text()))
    t = resolve(sl, Measured(), Limits())
    sfx.render_stem(t, pack, tmp_path / "a.wav", gen=fake_gen)
    sfx.render_stem(t, pack, tmp_path / "b.wav", gen=fake_gen)
    assert calls == ["soft glassy UI click"]


def test_a_generated_sound_is_peak_aligned_logged_once_per_render_and_costed_as_cached_after_the_first(tmp_path):
    mp3 = _mp3(tmp_path)
    calls = []
    costs = tmp_path / "costs.jsonl"
    pack = load_pack("saas-kinetic")
    pack.sfx["click"] = pack.sfx["card"] = {"gen": "soft glassy UI click", "duration": 0.5}  # two events, one spec
    t = _timing()
    for name in ("a", "b"):
        stem, _, _ = sfx.render_stem(t, pack, tmp_path / f"{name}.wav", costs_path=costs,
                                     gen=lambda text, d: calls.append((text, d)) or mp3)
    assert calls == [("soft glassy UI click", 0.5)]  # one paid call for both events and both renders
    rows = [json.loads(l) for l in costs.read_text().splitlines()]
    assert [r["cached"] for r in rows] == [False, True]
    assert rows[0]["provider"] == "elevenlabs" and rows[0]["units"] == {"sfx_s": 0.5, "unpriced": True}
    data, sr = sf.read(str(stem), dtype="float32")
    for e in t.events:
        if e.sfx in ("click", "card"):
            assert sfx.local_peak_s(data, sr, e.t) == pytest.approx(e.t, abs=ONE_SAMPLE)
    assert (tmp_path / "a.wav").read_bytes() == (tmp_path / "b.wav").read_bytes()


def test_a_paid_sound_that_cannot_be_decoded_is_logged_before_the_error(tmp_path):
    costs = tmp_path / "costs.jsonl"
    pack = load_pack("saas-kinetic")
    pack.sfx["click"] = {"gen": "a click", "duration": 0.5}
    with pytest.raises(sfx.SfxError, match="a click"):
        sfx.render_stem(_timing(), pack, tmp_path / "t.wav", costs_path=costs, gen=lambda text, d: b"not audio")
    assert json.loads(costs.read_text().splitlines()[-1])["cached"] is False
    assert not (tmp_path / "t.wav").exists() and not _hidden(tmp_path)


def test_a_failed_provider_call_logs_nothing_and_leaves_no_stem(tmp_path):
    costs = tmp_path / "costs.jsonl"
    pack = load_pack("saas-kinetic")
    pack.sfx["click"] = {"gen": "a click", "duration": 0.5}

    def rejected(text, d):
        raise ProviderError("elevenlabs", 401, "Invalid API key")

    with pytest.raises(ProviderError):
        sfx.render_stem(_timing(), pack, tmp_path / "t.wav", costs_path=costs, gen=rejected)
    assert not costs.exists() and not (tmp_path / "t.wav").exists()


def _gen_pack(prompt="a click"):
    pack = load_pack("saas-kinetic")
    pack.sfx["click"] = {"gen": prompt, "duration": 0.5}
    return pack


def test_the_cost_row_marks_the_sound_as_unpriced(tmp_path):
    costs = tmp_path / "costs.jsonl"
    sfx.generate("a click", 0.5, gen=lambda text, d: b"ID3x", costs_path=costs)
    row = json.loads(costs.read_text())
    assert row["units"] == {"sfx_s": 0.5, "unpriced": True} and row["usd"] == 0.0 and row["cached"] is False


def test_garbage_audio_is_discarded_from_the_cache_so_a_rerun_regenerates_it(tmp_path):
    costs, good = tmp_path / "costs.jsonl", _mp3(tmp_path)
    bodies, calls = [b"not audio", good], []

    def provider(text, d):
        calls.append(text)
        return bodies[len(calls) - 1]

    key = sfx.gen_key("a click", 0.5)
    with pytest.raises(sfx.SfxError, match=r"c01-e2.*discarded.*regenerate"):
        sfx.render_stem(_timing(), _gen_pack(), tmp_path / "t.wav", costs_path=costs, gen=provider)
    assert cache.cache_get("elevenlabs", key) is None  # the bad body is gone, not served forever
    stem, placements, _ = sfx.render_stem(_timing(), _gen_pack(), tmp_path / "t.wav", costs_path=costs, gen=provider)
    assert len(calls) == 2 and stem.exists() and len(placements) == 4
    assert cache.cache_get("elevenlabs", key) == good
    assert [json.loads(l)["cached"] for l in costs.read_text().splitlines()] == [False, False]


def test_a_cached_garbage_body_is_discarded_too(tmp_path):
    key = sfx.gen_key("a click", 0.5)
    cache.cache_put("elevenlabs", key, b"not audio")  # an entry from an earlier run
    calls, good = [], _mp3(tmp_path)
    provider = lambda text, d: calls.append(text) or good  # noqa: E731
    with pytest.raises(sfx.SfxError, match="discarded"):
        sfx.render_stem(_timing(), _gen_pack(), tmp_path / "t.wav", gen=provider)
    assert calls == [] and cache.cache_get("elevenlabs", key) is None
    sfx.render_stem(_timing(), _gen_pack(), tmp_path / "t.wav", gen=provider)
    assert calls == ["a click"]


def test_a_cache_entry_that_cannot_be_removed_is_named_not_claimed_discarded(tmp_path, monkeypatch):
    key = sfx.gen_key("a click", 0.5)
    cache.cache_put("elevenlabs", key, b"not audio")
    real_unlink = Path.unlink

    def deny(self, *args, **kwargs):
        if self.name == key:
            raise PermissionError("denied")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", deny)
    with pytest.raises(sfx.SfxError, match=r"could not be removed.*denied") as err:
        sfx.render_stem(_timing(), _gen_pack(), tmp_path / "t.wav", gen=lambda text, d: pytest.fail("cached"))
    assert "discarded" not in str(err.value) and key in str(err.value)
    assert cache.cache_get("elevenlabs", key) == b"not audio"


def test_good_cached_audio_is_never_discarded(tmp_path):
    good = _mp3(tmp_path)
    sfx.render_stem(_timing(), _gen_pack(), tmp_path / "a.wav", gen=lambda text, d: good)
    sfx.render_stem(_timing(), _gen_pack(), tmp_path / "b.wav", gen=lambda text, d: pytest.fail("cached"))
    assert cache.cache_get("elevenlabs", sfx.gen_key("a click", 0.5)) == good


def test_an_empty_body_is_rejected_before_it_is_cached_or_logged(tmp_path):
    costs = tmp_path / "costs.jsonl"
    with pytest.raises(ProviderError, match="empty sound"):
        sfx.generate("a click", 0.5, gen=lambda text, d: b"", costs_path=costs)
    assert cache.cache_get("elevenlabs", sfx.gen_key("a click", 0.5)) is None
    assert not costs.exists()


def test_the_cache_key_follows_the_providers_output_format_and_prompt_influence(monkeypatch):
    calls = []
    gen = lambda text, d: calls.append(text) or b"ID3x"  # noqa: E731
    sfx.generate("a click", 0.5, gen=gen)
    sfx.generate("a click", 0.5, gen=gen)
    assert len(calls) == 1
    monkeypatch.setattr(elevenlabs, "SFX_PROMPT_INFLUENCE", 0.9)
    sfx.generate("a click", 0.5, gen=gen)
    assert len(calls) == 2
    monkeypatch.setattr(elevenlabs, "SFX_OUTPUT_FORMAT", "mp3_44100_192")
    sfx.generate("a click", 0.5, gen=gen)
    assert len(calls) == 3


def test_the_provider_defaults_are_the_values_the_cache_key_uses():
    params = inspect.signature(elevenlabs.sound_generation).parameters
    assert params["prompt_influence"].default == elevenlabs.SFX_PROMPT_INFLUENCE
    assert params["output_format"].default == elevenlabs.SFX_OUTPUT_FORMAT


def test_inserting_an_event_does_not_change_the_other_sounds(tmp_path):
    """The noise seed comes from the event id, not its position, so an earlier event leaves later sounds alone."""
    t = _timing()
    pack = load_pack("saas-kinetic")
    a, _, _ = sfx.render_stem(t, pack, tmp_path / "a.wav")
    extra = t.events[0].model_copy(update={"id": "x0", "t": 0.05, "sfx": "tick", "kind": "tick"})
    b, _, _ = sfx.render_stem(t.model_copy(update={"events": [extra, *t.events]}), pack, tmp_path / "b.wav")
    xa, sr = sf.read(str(a), dtype="float32")
    xb, _ = sf.read(str(b), dtype="float32")
    changed = np.flatnonzero(xa != xb)
    assert len(changed) > 0 and changed[0] >= round(0.048 * sr) and changed[-1] <= round(0.092 * sr)  # the new tick only
    # the whoosh and the click are the noisy kinds; both are untouched outside the new tick's own span
    assert np.array_equal(xa[round(0.1 * sr):], xb[round(0.1 * sr):])


# --- the commands ---------------------------------------------------------------------------------------------

def _in_run(tmp_path, monkeypatch, *, timing=True, data=None):
    run = new_run(tmp_path, "demo", "saas-kinetic", "16:9", None, "en")
    run.shotlist.write_text(json.dumps(data) if data is not None else FIX.read_text())
    if timing:
        write_timing(_timing(data), run.timing)
    monkeypatch.chdir(run.root)
    return run


def test_render_command_writes_the_stem_and_placements(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch)
    res = runner.invoke(app, ["sfx", "render"])
    assert res.exit_code == 0, res.output
    assert str(run.sfx_dir / "track.wav") in res.output and "(4 sounds)" in res.output
    data, sr = sf.read(str(run.sfx_dir / "track.wav"), dtype="float32")
    assert sr == 48000 and len(data) == round(_timing().duration_s * sr)
    assert len(json.loads((run.sfx_dir / "placements.json").read_text())) == 4
    again = (run.sfx_dir / "track.wav").read_bytes()
    assert runner.invoke(app, ["sfx", "render"]).exit_code == 0
    assert (run.sfx_dir / "track.wav").read_bytes() == again
    assert not run.costs.exists()  # nothing paid, nothing logged


def test_render_command_prints_warnings(tmp_path, monkeypatch):
    data = _data()
    data["chapters"][1]["shots"][1]["events"][0]["sfx"] = "kazoo"
    _in_run(tmp_path, monkeypatch, data=data)
    res = runner.invoke(app, ["sfx", "render"])
    assert res.exit_code == 0 and "warning" in res.output and "kazoo" in res.output and "(3 sounds)" in res.output


def test_render_outside_a_run_is_a_clean_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = runner.invoke(app, ["sfx", "render"])
    assert res.exit_code == 2 and "no run folder" in res.output and "Traceback" not in res.output


def test_render_without_timing_json_is_one_line(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch, timing=False)
    res = runner.invoke(app, ["sfx", "render"])
    assert res.exit_code == 2 and "promo timing" in res.output  # a missing prerequisite, like scenes and beats
    assert res.output.count("promo:") == 1 and "Traceback" not in res.output
    assert not run.sfx_dir.joinpath("track.wav").exists()


@pytest.mark.parametrize("body", ["{not json", '{"version": 1}', "\xff\xfe"])
def test_render_with_a_corrupt_timing_json_is_one_line(tmp_path, monkeypatch, body):
    run = _in_run(tmp_path, monkeypatch)
    run.timing.write_bytes(body.encode("latin-1"))
    res = runner.invoke(app, ["sfx", "render"])
    assert res.exit_code == 1 and "timing.json" in res.output and "promo timing" in res.output
    assert res.output.count("promo:") == 1 and "Traceback" not in res.output


def test_render_with_a_broken_shotlist_is_one_line(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch)
    run.shotlist.write_text("{not json")
    res = runner.invoke(app, ["sfx", "render"])
    assert res.exit_code == 1 and res.output.count("promo:") == 1 and "Traceback" not in res.output


@pytest.mark.parametrize("exc, text", [
    (MissingKeyError("ELEVENLABS_API_KEY"), "ELEVENLABS_API_KEY is missing"),
    (ProviderError("elevenlabs", 401, "Invalid API key"), "HTTP 401"),
    (ToolError("command failed (1): ffmpeg -i x.mp3"), "ffmpeg"),
    (sfx.SfxError("c01-e2: sfx 'click' in pack saas-kinetic: file not found"), "file not found"),
    (OSError("No space left on device"), "No space left"),
])
def test_render_failures_are_one_line(tmp_path, monkeypatch, exc, text):
    _in_run(tmp_path, monkeypatch)

    def fake(*a, **k):
        raise exc

    monkeypatch.setattr("promo.sfx.render_stem", fake)
    res = runner.invoke(app, ["sfx", "render"])
    assert res.exit_code == 1 and text in res.output
    assert res.output.count("promo:") == 1 and "Traceback" not in res.output


def test_render_lets_a_genuine_bug_propagate(tmp_path, monkeypatch):
    _in_run(tmp_path, monkeypatch)

    def fake(*a, **k):
        raise ValueError("a bug, not a user error")

    monkeypatch.setattr("promo.sfx.render_stem", fake)
    res = runner.invoke(app, ["sfx", "render"])
    assert res.exit_code == 1 and isinstance(res.exception, ValueError)


@respx.mock
def test_render_command_with_a_generated_sound_pays_once_across_reruns(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch)
    pack = load_pack("saas-kinetic")  # the repo's pack stays untouched: the command is given a patched copy
    pack.sfx["click"] = {"gen": "soft glassy UI click", "duration": 0.5}
    monkeypatch.setattr("promo.packs.load_pack", lambda name: pack)
    route = respx.post(elevenlabs.SFX_URL).mock(return_value=httpx.Response(200, content=_mp3(tmp_path)))
    assert runner.invoke(app, ["sfx", "render"]).exit_code == 0
    assert runner.invoke(app, ["sfx", "render"]).exit_code == 0
    assert route.call_count == 1
    assert json.loads(route.calls[0].request.content) == {
        "text": "soft glassy UI click", "duration_seconds": 0.5, "prompt_influence": 0.5}
    assert [json.loads(l)["cached"] for l in run.costs.read_text().splitlines()] == [False, True]


@respx.mock
def test_render_command_without_a_key_for_a_generated_sound_is_one_line(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch)
    pack = load_pack("saas-kinetic")
    pack.sfx["click"] = {"gen": "soft glassy UI click", "duration": 0.5}
    monkeypatch.setattr("promo.packs.load_pack", lambda name: pack)
    _no_key(monkeypatch)
    route = respx.post(elevenlabs.SFX_URL).mock(return_value=httpx.Response(200, content=b"x"))
    res = runner.invoke(app, ["sfx", "render"])
    assert res.exit_code == 1 and "ELEVENLABS_API_KEY is missing" in res.output
    assert res.output.count("promo:") == 1 and "Traceback" not in res.output
    assert route.call_count == 0 and not run.costs.exists()


def test_synth_command_writes_a_deterministic_wav(tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "sub" / "b.wav"
    for out in (a, b):
        res = runner.invoke(app, ["sfx", "synth", "tick", "--out", str(out), "--value", "100"])
        assert res.exit_code == 0, res.output
        assert str(out) in res.output
    assert a.read_bytes() == b.read_bytes()
    x, sr = sf.read(str(a), dtype="float32")
    assert sr == 48000 and _dominant_hz(x) == pytest.approx(4400, rel=0.05)


def test_synth_command_takes_pitch_and_duration(tmp_path):
    out = tmp_path / "t.wav"
    res = runner.invoke(app, ["sfx", "synth", "tick", "--out", str(out), "--pitch", "2000", "--duration", "0.1"])
    assert res.exit_code == 0, res.output
    x, sr = sf.read(str(out), dtype="float32")
    assert len(x) == int(0.1 * sr) and _dominant_hz(x) == pytest.approx(2000, rel=0.05)


@pytest.mark.parametrize("args, text", [
    (["kazoo", "--out", "k.wav"], "unknown sfx kind 'kazoo'"),
    (["tick", "--out", "k.wav", "--duration", "0"], "duration"),
    (["tick", "--out", "k.wav", "--pitch", "-5"], "pitch"),
    (["tick", "--out", "k.mp3"], ".wav"),
    (["tick", "--out", "k"], ".wav"),
])
def test_synth_usage_errors_are_one_line_and_write_nothing(tmp_path, monkeypatch, args, text):
    monkeypatch.chdir(tmp_path)
    res = runner.invoke(app, ["sfx", "synth", *args])
    assert res.exit_code == 2 and text in res.output
    assert res.output.count("promo:") == 1 and "Traceback" not in res.output
    assert not list(tmp_path.iterdir())


def test_synth_into_an_unwritable_place_is_one_line(tmp_path):
    (tmp_path / "f").write_text("a file")
    res = runner.invoke(app, ["sfx", "synth", "tick", "--out", str(tmp_path / "f" / "x.wav")])
    assert res.exit_code == 1 and res.output.count("promo:") == 1 and "Traceback" not in res.output


@respx.mock
def test_gen_command_writes_the_mp3_and_a_second_run_is_free(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch)
    route = respx.post(elevenlabs.SFX_URL).mock(return_value=httpx.Response(200, content=b"ID3sfx"))
    out = run.sfx_dir / "click.mp3"
    args = ["sfx", "gen", "soft glassy UI click", "--duration", "0.8", "--out", str(out)]
    res = runner.invoke(app, args)
    assert res.exit_code == 0, res.output
    assert str(out) in res.output and out.read_bytes() == b"ID3sfx"
    req = route.calls[0].request
    assert json.loads(req.content) == {"text": "soft glassy UI click", "duration_seconds": 0.8, "prompt_influence": 0.5}
    assert req.headers["xi-api-key"] == "el-test"
    out.unlink()
    assert runner.invoke(app, args).exit_code == 0
    assert route.call_count == 1 and out.read_bytes() == b"ID3sfx"  # the cache served it
    rows = [json.loads(l) for l in run.costs.read_text().splitlines()]
    assert [r["cached"] for r in rows] == [False, True]
    assert rows[0]["provider"] == "elevenlabs" and rows[0]["units"] == {"sfx_s": 0.8, "unpriced": True}
    assert not _hidden(run.sfx_dir)


@respx.mock
def test_gen_command_outside_a_run_works_and_logs_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    route = respx.post(elevenlabs.SFX_URL).mock(return_value=httpx.Response(200, content=b"ID3sfx"))
    res = runner.invoke(app, ["sfx", "gen", "a click", "--out", "c.mp3"])
    assert res.exit_code == 0, res.output
    assert (tmp_path / "c.mp3").read_bytes() == b"ID3sfx" and route.call_count == 1
    assert json.loads(route.calls[0].request.content)["duration_seconds"] == 0.5  # the default
    assert not list(tmp_path.glob("**/costs.jsonl"))


@respx.mock
def test_gen_command_logs_the_paid_call_before_a_write_can_fail(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch)
    route = respx.post(elevenlabs.SFX_URL).mock(return_value=httpx.Response(200, content=b"ID3sfx"))
    (tmp_path / "f").write_text("a file, not a folder")
    res = runner.invoke(app, ["sfx", "gen", "a click", "--out", str(tmp_path / "f" / "c.mp3")])
    assert res.exit_code == 1 and res.output.count("promo:") == 1 and "Traceback" not in res.output
    assert [json.loads(l)["cached"] for l in run.costs.read_text().splitlines()] == [False]
    out = tmp_path / "c.mp3"
    assert runner.invoke(app, ["sfx", "gen", "a click", "--out", str(out)]).exit_code == 0
    assert route.call_count == 1 and out.read_bytes() == b"ID3sfx"  # the paid audio was cached, not lost


@respx.mock
def test_gen_command_without_a_key_is_one_line_and_pays_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _no_key(monkeypatch)
    route = respx.post(elevenlabs.SFX_URL).mock(return_value=httpx.Response(200, content=b"x"))
    res = runner.invoke(app, ["sfx", "gen", "a click", "--out", "c.mp3"])
    assert res.exit_code == 1 and "ELEVENLABS_API_KEY is missing" in res.output
    assert res.output.count("promo:") == 1 and "Traceback" not in res.output
    assert route.call_count == 0 and not (tmp_path / "c.mp3").exists()


@respx.mock
def test_gen_command_with_a_rejected_key_is_one_line(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch)
    route = respx.post(elevenlabs.SFX_URL).mock(return_value=httpx.Response(
        401, json={"detail": {"status": "invalid_api_key", "message": "Invalid API key"}}))
    res = runner.invoke(app, ["sfx", "gen", "a click", "--out", str(tmp_path / "c.mp3")])
    assert res.exit_code == 1 and "elevenlabs: HTTP 401: Invalid API key" in res.output
    assert res.output.count("promo:") == 1 and "Traceback" not in res.output
    assert route.call_count == 1 and not (tmp_path / "c.mp3").exists() and not run.costs.exists()


@respx.mock
def test_gen_command_with_an_empty_body_is_one_line_and_keeps_nothing(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch)
    route = respx.post(elevenlabs.SFX_URL).mock(return_value=httpx.Response(200, content=b""))
    out = tmp_path / "c.mp3"
    res = runner.invoke(app, ["sfx", "gen", "a click", "--out", str(out)])
    assert res.exit_code == 1 and "empty sound" in res.output
    assert res.output.count("promo:") == 1 and "Traceback" not in res.output
    assert route.call_count == 1 and not out.exists() and not run.costs.exists()
    assert cache.cache_get("elevenlabs", sfx.gen_key("a click", 0.5)) is None


@pytest.mark.parametrize("args, text", [
    (["a click", "--out", "c.wav"], ".mp3"),
    (["a click", "--out", "c"], ".mp3"),
    (["a click", "--out", "c.mp3", "--duration", "0"], "duration"),
    (["a click", "--out", "c.mp3", "--duration", "-2"], "duration"),
    (["   ", "--out", "c.mp3"], "prompt"),
])
@respx.mock
def test_gen_usage_errors_are_one_line_and_make_no_request(tmp_path, monkeypatch, args, text):
    monkeypatch.chdir(tmp_path)
    route = respx.post(elevenlabs.SFX_URL).mock(return_value=httpx.Response(200, content=b"x"))
    res = runner.invoke(app, ["sfx", "gen", *args])
    assert res.exit_code == 2 and text in res.output
    assert res.output.count("promo:") == 1 and "Traceback" not in res.output
    assert route.call_count == 0 and not list(tmp_path.iterdir())


def test_sfx_without_a_subcommand_shows_help():
    res = runner.invoke(app, ["sfx"])
    assert "render" in res.output and "synth" in res.output and "gen" in res.output


@pytest.mark.live
def test_live_half_a_second_of_sound(tmp_path):
    from promo.config import load_env
    load_env()
    audio = elevenlabs.sound_generation("soft glassy UI click", 0.5)
    p = tmp_path / "s.mp3"
    p.write_bytes(audio)
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True, check=True)
    assert 0.2 < float(out.stdout) < 2.0
