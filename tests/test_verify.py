import json
import math
import re
import shutil
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from promo import verify
from promo.cli import app
from promo.hyperframes import ensure_shot, scaffold_scenes
from promo.packs import load_pack
from promo.run import new_run
from promo.shotlist import Shotlist
from promo.timing import Measured, resolve, write_timing

FIX = Path(__file__).parent / "fixtures" / "shotlist_music.json"
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "promo" / "oleafly-30s"
runner = CliRunner()


def _fake_key_applies(node) -> bool:
    """Live tests must read the real key from .env, so they never get the fake one."""
    return node.get_closest_marker("live") is None


@pytest.fixture(autouse=True)
def keys(request, monkeypatch):
    if _fake_key_applies(request.node):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test")
        monkeypatch.setenv("GEMINI_API_KEY", "gm-test")


def test_fake_key_skips_live_tests():
    class Node:
        def __init__(self, marker):
            self.marker = marker

        def get_closest_marker(self, name):
            return self.marker if name == "live" else None

    assert _fake_key_applies(Node(None)) is True
    assert _fake_key_applies(Node(object())) is False


def _setup(tmp_path):
    run = new_run(tmp_path, "d", "saas-kinetic", "16:9", None, "en")
    run.shotlist.write_text(FIX.read_text())
    pack = load_pack("saas-kinetic")
    return run, pack, resolve(Shotlist.model_validate(json.loads(FIX.read_text())), Measured(), pack.limits())


def test_pick_times_are_inside_the_shot(tmp_path):
    _, _, t = _setup(tmp_path)
    times = verify.pick_times(t, "c02-s1", n=6)
    assert times == sorted(times) and all(0 <= x < 2.0 for x in times) and len(times) <= 6


def test_pick_times_covers_the_first_and_last_frame_and_every_event_and_can_ask_for_one(tmp_path):
    _, _, t = _setup(tmp_path)
    shot = t.shot("c01-s1")
    times = verify.pick_times(t, "c01-s1", n=20)
    assert times[0] == 0.0 and times[-1] == pytest.approx(shot.duration_s - 1 / t.fps, abs=0.001)
    for e in t.events_in("c01-s1"):
        assert round(e.t_local, 3) in times
    assert len(verify.pick_times(t, "c01-s1", n=1)) == 1
    assert len(verify.pick_times(t, "c01-s1", n=3)) == 3


def _fake_snap(colour_for):
    def snap(project, times, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, t in enumerate(times):
            p = out_dir / f"frame-{i:02d}-at-{t:.3f}s.png"
            Image.new("RGB", (8, 8), colour_for(t, out_dir.name)).save(p)
            paths.append(p)
        return paths
    return snap


def test_verify_passes_when_frames_match(tmp_path):
    run, pack, t = _setup(tmp_path)
    res = verify.verify(run, t, pack, only=["c01-s1"], snap=_fake_snap(lambda tt, d: (int(tt * 10), 0, 0)))
    assert res.ok and res.mismatches == []
    assert json.loads((run.renders / "verify.json").read_text())["ok"] is True


def test_verify_catches_order_dependent_frames(tmp_path):
    run, pack, t = _setup(tmp_path)
    bad = _fake_snap(lambda tt, d: (255, 0, 0) if d == "b" else (int(tt * 10), 0, 0))
    res = verify.verify(run, t, pack, only=["c01-s1"], snap=bad)
    assert not res.ok and res.mismatches[0]["shot"] == "c01-s1"


def test_a_mismatch_says_how_big_it_is(tmp_path):
    run, pack, t = _setup(tmp_path)
    res = verify.verify(run, t, pack, only=["c01-s1"],
                        snap=_fake_snap(lambda tt, d: (40, 0, 0) if d == "b" else (30, 0, 0)))
    assert res.mismatches[0]["reason"] == "differs between seek orders"
    assert res.mismatches[0]["max_diff"] == 10 and res.mismatches[0]["pixels"] == 64


def test_a_snapshot_that_captures_nothing_is_a_mismatch_not_a_pass(tmp_path):
    run, pack, t = _setup(tmp_path)
    res = verify.verify(run, t, pack, only=["c01-s1"], snap=lambda project, times, out_dir: [])
    assert res.ok is False
    assert {m["reason"] for m in res.mismatches} == {"no frame was captured"}
    assert len(res.mismatches) == len(verify.pick_times(t, "c01-s1", 6))


def test_snapshots_are_kept_only_for_a_shot_that_mismatches_and_never_mixed_with_an_earlier_run(tmp_path):
    run, pack, t = _setup(tmp_path)
    work = run.renders / "verify"
    verify.verify(run, t, pack, only=["c01-s1"], snap=_fake_snap(lambda tt, d: (1, 2, 3)))
    assert not work.exists()                                       # nothing to inspect when everything matched
    bad = _fake_snap(lambda tt, d: (255, 0, 0) if d == "b" else (1, 2, 3))
    verify.verify(run, t, pack, only=["c01-s1"], snap=bad)
    assert (work / "c01-s1" / "a").is_dir()
    stale = work / "c01-s1" / "a" / "frame-99-at-0.000s.png"       # a frame from this run's predecessor
    Image.new("RGB", (8, 8), "white").save(stale)
    res = verify.verify(run, t, pack, only=["c01-s1"], snap=_fake_snap(lambda tt, d: (1, 2, 3)))
    assert res.ok and not work.exists()


def test_an_unknown_shot_is_a_named_error_and_writes_nothing(tmp_path):
    run, pack, t = _setup(tmp_path)
    with pytest.raises(verify.MissingScenes, match="nope"):
        verify.verify(run, t, pack, only=["nope"], snap=_fake_snap(lambda tt, d: (1, 2, 3)))
    assert not (run.renders / "verify.json").exists()


def test_a_shot_of_no_frames_is_not_verified(tmp_path):
    _, _, t = _setup(tmp_path)
    flat = t.model_copy(update={"shots": [s.model_copy(update={"end_s": s.start_s}) if s.id == "c02-s2" else s
                                          for s in t.shots]})
    assert verify.verifiable_shots(flat) == ["c01-s1", "c02-s1"]


def test_an_unreadable_design_is_a_named_error(tmp_path):
    run, pack, t = _setup(tmp_path)
    run.design.write_text("{not json")
    with pytest.raises(verify.VerifyError, match="design.json"):
        verify.verify(run, t, pack, snap=_fake_snap(lambda tt, d: (1, 2, 3)))


def _with_scenes(tmp_path):
    run, pack, t = _setup(tmp_path)
    scaffold_scenes(run.scenes)
    for s in t.shots:
        ensure_shot(run.scenes, s.id)
    return run, pack, t


def test_each_verified_shot_records_the_hash_promo_scenes_would_make_and_an_edit_makes_it_stale(tmp_path):
    from promo.scenes import design_for, shot_hash

    run, pack, t = _with_scenes(tmp_path)
    res = verify.verify(run, t, pack, only=["c01-s1"], snap=_fake_snap(lambda tt, d: (1, 2, 3)))
    assert res.fingerprints == {"c01-s1": shot_hash(run.scenes, t, "c01-s1", design_for(run, pack), "high", t.fps)}
    recorded = json.loads((run.renders / "verify.json").read_text())["fingerprints"]
    assert verify.stale_shots(run, t, pack, recorded) == []
    scene = run.shots_dir / "c01-s1.html"
    scene.write_text(scene.read_text() + "\n<!-- edited -->\n")
    assert verify.stale_shots(run, t, pack, recorded) == ["c01-s1"]
    scene.unlink()
    assert verify.stale_shots(run, t, pack, recorded) == ["c01-s1"]


def test_missing_scenes_names_the_shots_with_nothing_to_snapshot(tmp_path):
    run, pack, t = _setup(tmp_path)
    assert verify.missing_scenes(run, t) == ["c01-s1", "c02-s1", "c02-s2"]
    assert verify.missing_scenes(run, t, ["c02-s1"]) == ["c02-s1"]
    ensure_shot(run.scenes, "c01-s1")
    (run.render_dir / "c02-s1").mkdir(parents=True)
    assert verify.missing_scenes(run, t) == ["c02-s2"]


def test_verify_captures_from_its_own_project_with_the_settle_hook_on_and_leaves_the_scenes_alone(tmp_path):
    from promo.config import packs_dir

    run, pack, t = _with_scenes(tmp_path)
    custom = run.scenes / "lib" / "stage.js"
    custom.write_text("// the scene's own stage, without the hook\n")
    seen = {}

    def snap(project, times, out_dir):
        seen.setdefault("project", project)
        seen["design"] = json.loads((project / "design.json").read_text())
        seen["stage"] = (project / "lib" / "stage.js").read_text()
        return _fake_snap(lambda tt, d: (1, 2, 3))(project, times, out_dir)

    verify.verify(run, t, pack, only=["c01-s1"], snap=snap)
    assert seen["project"] == run.renders / "verify" / "c01-s1" / "project"
    assert seen["design"]["capture"] == {"settle": True} and seen["design"]["tokens"]       # the run's own design is kept
    assert seen["stage"] == (packs_dir() / "_core" / "scene-template" / "lib" / "stage.js").read_text()
    assert "capture-settle:begin" in seen["stage"]
    assert custom.read_text() == "// the scene's own stage, without the hook\n"            # never written through the link
    assert not (run.render_dir / "c01-s1").exists()                                         # never a running render's directory


def test_a_mismatch_keeps_the_project_it_was_captured_from_and_the_run_design_has_no_flag(tmp_path):
    run, pack, t = _with_scenes(tmp_path)
    bad = _fake_snap(lambda tt, d: (255, 0, 0) if d == "b" else (1, 2, 3))
    verify.verify(run, t, pack, only=["c01-s1"], snap=bad)
    kept = run.renders / "verify" / "c01-s1" / "project"
    assert (kept / "index.html").exists() and json.loads((kept / "design.json").read_text())["capture"]["settle"] is True
    assert not run.design.exists() or "capture" not in json.loads(run.design.read_text())


def test_without_a_scene_file_the_project_an_earlier_render_left_is_copied_and_armed(tmp_path):
    run, pack, t = _setup(tmp_path)
    old = run.render_dir / "c01-s1"
    (old / "lib").mkdir(parents=True)
    (old / "lib" / "stage.js").write_text("// old\n")
    (old / "design.json").write_text(json.dumps({"tokens": {"x": 1}}))
    seen = {}

    def snap(project, times, out_dir):
        seen["design"] = json.loads((project / "design.json").read_text())
        return _fake_snap(lambda tt, d: (1, 2, 3))(project, times, out_dir)

    verify.verify(run, t, pack, only=["c01-s1"], snap=snap)
    assert seen["design"] == {"tokens": {"x": 1}, "capture": {"settle": True}}
    assert (old / "lib" / "stage.js").read_text() == "// old\n" and json.loads((old / "design.json").read_text()) == {"tokens": {"x": 1}}


# --- the state gate, the pixel warning and the overlap audit -------------------------------------------------------------

def _digest_png(path, state, audit=(), colour=(10, 20, 30), width=48, body_rows=6, patch=None):
    """A capture as the stage makes it: the deflated digest as RGB pixels in the top rows, a flat colour below."""
    payload = {"v": 1, "shot": "c01-s1", "state": state, "audit": list(audit)}
    packed = zlib.compress(json.dumps(payload).encode())
    flat = verify.DIGEST_MAGIC + len(packed).to_bytes(4, "big") + packed
    rows = math.ceil(len(flat) / (3 * width))
    flat = flat + bytes(rows * width * 3 - len(flat))
    pixels = np.zeros((rows + body_rows, width, 3), dtype=np.uint8)
    pixels[:rows] = np.frombuffer(flat, dtype=np.uint8).reshape(rows, width, 3)
    pixels[rows:] = colour
    if patch:
        pixels[rows + patch[1]:rows + patch[1] + patch[3], patch[0]:patch[0] + patch[2]] = patch[4]
    Image.fromarray(pixels).save(path)


def _state_snap(state_for, *, colour_for=lambda t, d: (10, 20, 30), audit=(), patch_for=lambda t, d: None):
    def snap(project, times, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, t in enumerate(times):
            p = out_dir / f"frame-{i:02d}-at-{t:.3f}s.png"
            _digest_png(p, state_for(t, out_dir.name), audit, colour_for(t, out_dir.name), patch=patch_for(t, out_dir.name))
            paths.append(p)
        return paths
    return snap


STATE = [["#root", "width: 1920px;", "", [0, 0, 1920, 1080]],
         ["#pL2", "transform: translate(0px, 3px) rotate(4.29049deg); opacity: 1;", "", [100, 200, 130, 130]],
         ["#card", "opacity: 0.5;", "", [300, 300, 500, 200]]]


def _with(state, element, style):
    return [[e[0], style if e[0] == element else e[1], *e[2:]] for e in state]


def test_digest_round_trips_through_a_png_and_a_damaged_one_is_not_a_digest(tmp_path):
    _digest_png(tmp_path / "d.png", STATE)
    payload, rows = verify.decode_digest(tmp_path / "d.png")
    assert payload["state"] == STATE and payload["shot"] == "c01-s1" and rows >= 1
    Image.new("RGB", (48, 8), (9, 9, 9)).save(tmp_path / "plain.png")
    assert verify.decode_digest(tmp_path / "plain.png") == (None, 0)
    pixels = np.asarray(Image.open(tmp_path / "d.png").convert("RGB")).copy()
    pixels[1, 3] ^= 0xFF                                            # a bit of the deflated bytes
    Image.fromarray(pixels).save(tmp_path / "bad.png")
    assert verify.decode_digest(tmp_path / "bad.png") == (None, 0)


SAME_VALUES = [
    ("transform", "scale(1.11)", "scale(1.11, 1.11)"),                                    # one notation, then the other
    ("transform", "translate(0px, 30px) rotate(5.64341deg) scale(0.9, 0.9)", "translate3d(0px, 30px, 0px) rotate(5.6434deg) scale(0.9)"),
    ("transform", "translate(0px, 0px)", "translate3d(0px, 0px, 0px)"),
    ("transform", "translate(12px)", "translateX(12px)"),
    ("transform", "rotateZ(5deg)", "rotate(5deg)"),
    ("transform", "scaleX(2) scaleY(3)", "scale3d(2, 1, 1) scale3d(1, 3, 1)"),
    ("transform", "translate(0, 0)", "translate(0px, 0px)"),                              # a zero has no unit
    ("transform", "rotate(4.29049deg)", "rotate(4.2905deg)"),                              # the same value to 4 decimals
    ("transform", "rotate(5.64345deg)", "rotate(5.64344deg)"),                            # ...however the rounding falls
    ("transform", "ROTATE( 5deg )", "rotate(5deg)"),
    ("opacity", "0.5", "0.50001"),
    ("opacity", ".5", "0.50"),
    ("filter", "blur(0px)", "blur(0.00004px)"),
    ("box-shadow", "0 24px 60px rgba(22, 24, 26, 0.18)", "0 24px 60px rgba(22,24,26,0.18)"),
]
DIFFERENT_VALUES = [
    ("transform", "rotate(5.6434deg)", "rotate(5.6454deg)"),                              # 0.002 degrees: a real difference
    ("transform", "rotate(4.29deg)", "rotate(5.29deg)"),                                  # a whole degree
    ("transform", "scale(1.11)", "scale(1.12)"),
    ("transform", "translate(0px, 30px)", "translate(0px, 30.001px)"),
    ("transform", "rotate(5deg)", "rotate(5rad)"),                                        # a different unit
    ("transform", "rotate(5deg)", "scale(5)"),
    ("opacity", "0.5", "0.501"),
    ("filter", "blur(0px)", "blur(2px)"),
]


@pytest.mark.parametrize("prop,forward,reverse", SAME_VALUES)
def test_one_value_written_two_ways_is_the_same_value(prop, forward, reverse):
    assert verify._same_value(prop, forward, reverse) is True
    assert verify._same_value(prop, reverse, forward) is True


@pytest.mark.parametrize("prop,forward,reverse", DIFFERENT_VALUES)
def test_a_real_difference_is_a_different_value(prop, forward, reverse):
    assert verify._same_value(prop, forward, reverse) is False and verify._same_value(prop, reverse, forward) is False


def test_the_computed_matrix_settles_a_notation_the_canonicaliser_does_not_know():
    # rotate3d(0, 0, 1, 5deg) is rotate(5deg): the browser's matrix says so, to 1e-5
    m5 = "matrix(0.996195, 0.0871557, -0.0871557, 0.996195, 0, 0)"
    assert verify._same_value("transform", "rotate3d(0, 0, 1, 5deg)", "rotate(5deg)", m5, m5) is True
    m6 = "matrix(0.994522, 0.104528, -0.104528, 0.994522, 0, 0)"
    assert verify._same_value("transform", "rotate3d(0, 0, 1, 5deg)", "rotate(6deg)", m5, m6) is False
    assert verify._same_value("transform", "rotate3d(0, 0, 1, 5deg)", "rotate(5deg)", "", "") is False   # no matrix, no proof
    assert verify._same_value("opacity", "0.5", "0.5", "x", "y") is True and verify._same_value("opacity", None, "1") is False


ROTATED = "matrix(0.996195, 0.0871557, -0.0871557, 0.996195, 0, 0)"


@pytest.mark.parametrize("prop,forward,reverse", [
    ("transform-origin", "0% 0%", "100% 100%"),
    ("transform-origin", "50% 50%", "0px 0px"),
    ("transform-style", "flat", "preserve-3d"),
    ("transform-box", "fill-box", "view-box"),
])
def test_the_matrix_cannot_vouch_for_the_other_transform_properties(prop, forward, reverse):
    # the matrix of a rotated element is the same whatever its origin was when the matrix was computed: it is evidence
    # about `transform` and the SVG transform attribute only
    assert verify._same_value(prop, forward, reverse, ROTATED, ROTATED) is False
    assert verify._same_value("transform", "rotate(5deg)", "rotate3d(0, 0, 1, 5deg)", ROTATED, ROTATED) is True
    assert verify._same_value("transform (attribute)", "rotate(5)", "rotate3d(0, 0, 1, 5)", ROTATED, ROTATED) is True


def test_a_transform_origin_that_differs_on_a_rotated_element_fails_a_shot_with_the_same_matrix(tmp_path):
    run, pack, t = _setup(tmp_path)
    spun = lambda origin: [["#root", "width: 1920px;", "", [0, 0, 1920, 1080], ""],
                           ["#pL2", f"transform: rotate(5deg); transform-origin: {origin};", "", [100, 200, 130, 130], ROTATED]]
    res = verify.verify(run, t, pack, only=["c01-s1"], snap=_state_snap(
        lambda tt, d: spun("100% 100%") if d == "b" else spun("0% 0%")))
    assert res.ok is False
    m = res.mismatches[0]
    assert (m["kind"], m["element"], m["property"], m["forward"], m["reverse"]) == \
        ("state", "#pL2", "transform-origin", "0% 0%", "100% 100%")
    assert "#pL2 transform-origin: 0% 0% forward vs 100% 100% reverse" in m["message"]
    same = verify.verify(run, t, pack, only=["c01-s1"], snap=_state_snap(lambda tt, d: spun("0% 0%")))
    assert same.ok is True


def test_values_written_two_ways_in_the_digest_do_not_fail_a_shot(tmp_path):
    run, pack, t = _setup(tmp_path)
    forward = _with(STATE, "#pL2", "transform: translate(0px, 3px) rotate(4.29049deg) scale(1.11); opacity: 1;")
    reverse = _with(STATE, "#pL2", "transform: translate3d(0px, 3px, 0px) rotate(4.2905deg) scale(1.11, 1.11); opacity: 1.00001;")
    res = verify.verify(run, t, pack, only=["c01-s1"], snap=_state_snap(lambda tt, d: reverse if d == "b" else forward))
    assert res.ok is True and res.mismatches == [] and res.warnings == []


def test_a_real_state_difference_fails_and_says_values_differ_naming_both(tmp_path):
    run, pack, t = _setup(tmp_path)
    other = _with(STATE, "#pL2", "transform: translate(0px, 3px) rotate(4.2915deg); opacity: 1;")
    res = verify.verify(run, t, pack, only=["c01-s1"], snap=_state_snap(lambda tt, d: other if d == "b" else STATE))
    assert res.ok is False and res.warnings == []
    m = res.mismatches[0]
    assert (m["kind"], m["shot"], m["element"], m["property"]) == ("state", "c01-s1", "#pL2", "transform")
    assert m["forward"] == "translate(0px, 3px) rotate(4.29049deg)" and m["reverse"] == "translate(0px, 3px) rotate(4.2915deg)"
    assert m["hint"].startswith("values differ between the orders") and "overlap audit" in m["hint"]
    assert "#pL2 transform: translate(0px, 3px) rotate(4.29049deg) forward vs translate(0px, 3px) rotate(4.2915deg) reverse" \
        in m["message"]
    for words in ("serialis", "serializ", "roundings"):
        assert words not in m["message"]                       # a same value in two notations is never the cause now
    assert m["elements_differing"] == 1 and m["t_local"] == 0.0


def test_a_different_value_a_missing_element_and_a_style_present_in_one_order_each_get_their_own_message(tmp_path):
    run, pack, t = _setup(tmp_path)
    other = _with(STATE, "#card", "opacity: 0.9;")
    res = verify.verify(run, t, pack, only=["c01-s1"], snap=_state_snap(
        lambda tt, d: other if d == "b" else STATE, colour_for=lambda tt, d: (40, 20, 30) if d == "b" else (10, 20, 30)))
    m = res.mismatches[0]
    assert (m["element"], m["property"], m["forward"], m["reverse"]) == ("#card", "opacity", "0.5", "0.9")
    assert m["hint"].startswith("values differ") and "overlap audit" in m["hint"]
    assert m["max_diff"] == 30 and m["pixels"] > 0                                   # the pixel diff rides along
    # an element with an id is always in the digest: when it is not, it is not in the page
    gone = verify.verify(run, t, pack, only=["c01-s1"], snap=_state_snap(lambda tt, d: STATE[:2] if d == "b" else STATE))
    g = gone.mismatches[0]
    assert (g["element"], g["property"], g["forward"], g["reverse"]) == ("#card", "(the element)", "present", "absent")
    assert g["hint"] == "the element exists in one order only"
    # #aiTyped: there in both orders, but an inline style only in one: that is what the message says, not "absent"
    typed = STATE + [["#aiTyped", "", "", [10, 10, 5, 5]]]
    styled = STATE + [["#aiTyped", "opacity: 1;", "", [10, 10, 5, 5]]]
    one = verify.verify(run, t, pack, only=["c01-s1"], snap=_state_snap(lambda tt, d: styled if d == "b" else typed))
    o = one.mismatches[0]
    assert (o["element"], o["property"], o["forward"], o["reverse"]) == ("#aiTyped", "opacity", None, "1")
    assert "#aiTyped opacity: (unset) forward vs 1 reverse" in o["message"] and "absent" not in o["message"]
    assert "there in both orders" in o["hint"] and "set in one only" in o["hint"]
    # an element with no id that has an inline style in one order only: verify cannot tell whether it exists in the other
    anon = STATE + [["#root>span:4", "opacity: 1;", "", [1, 1, 2, 2]]]
    no_id = verify.verify(run, t, pack, only=["c01-s1"], snap=_state_snap(lambda tt, d: anon if d == "b" else STATE))
    n = no_id.mismatches[0]
    assert (n["element"], n["property"], n["forward"], n["reverse"]) == ("#root>span:4", "(inline style)", None, "opacity: 1")
    assert "no id" in n["hint"] and "absent" not in n["message"]


def test_the_order_the_properties_were_written_in_is_not_a_state_difference(tmp_path):
    run, pack, t = _setup(tmp_path)
    flipped = _with(STATE, "#pL2", "opacity: 1; transform: translate(0px, 3px) rotate(4.29049deg);")
    res = verify.verify(run, t, pack, only=["c01-s1"], snap=_state_snap(lambda tt, d: flipped if d == "b" else STATE))
    assert res.ok is True and res.mismatches == [] and res.warnings == []


def test_the_one_by_one_capture_is_compared_too(tmp_path):
    run, pack, t = _setup(tmp_path)
    other = _with(STATE, "#pL2", "transform: translate(0px, 3px) rotate(4.3905deg); opacity: 1;")
    res = verify.verify(run, t, pack, only=["c01-s1"], snap=_state_snap(lambda tt, d: other if d.startswith("c") else STATE))
    assert res.ok is False and "reverse" not in res.mismatches[0]["message"].split("—")[0].split("forward vs")[1]
    assert " single" in res.mismatches[0]["message"]


def test_identical_state_with_different_pixels_is_a_warning_naming_the_layer_and_the_region(tmp_path):
    run, pack, t = _setup(tmp_path)
    res = verify.verify(run, t, pack, only=["c01-s1"], snap=_state_snap(
        lambda tt, d: STATE, patch_for=lambda tt, d: (20, 2, 4, 3, (200, 0, 0)) if d == "b" else None))
    assert res.ok is True and res.mismatches == []
    w = res.warnings[0]
    assert (w["kind"], w["reason"]) == ("pixels", "DOM identical; Chrome reused raster from an earlier frame")
    assert w["max_diff"] == 190 and w["pixels"] == 12                                  # 4 x 3 px, red 10 -> 200
    assert w["region"][0] == 20 and w["region"][2] == 24 and w["region"][3] - w["region"][1] == 3
    assert w["layer"] == "#root"                                                       # the smallest box holding it
    assert "DOM identical; Chrome reused raster from an earlier frame on #root" in w["message"] and "region" in w["message"]
    assert "190 of 255 on 12 pixels" in w["message"] and "tolerance" not in w["message"]
    assert (run.renders / "verify" / "c01-s1" / "b").is_dir()                          # the captures are kept for a look


def test_the_layer_is_the_smallest_element_under_the_region_and_none_when_nothing_is(tmp_path):
    assert verify._layer_at(STATE, [310, 310, 320, 320]) == "#card"
    assert verify._layer_at(STATE, [110, 210, 120, 220]) == "#pL2"
    assert verify._layer_at(STATE, [1500, 900, 1510, 910]) == "#root"
    assert verify._layer_at(STATE, None) is None and verify._layer_at([], [1, 1, 2, 2]) is None


def test_the_digest_rows_are_masked_out_of_the_pixel_comparison(tmp_path):
    run, pack, t = _setup(tmp_path)
    moved = [[e[0], e[1], e[2], [e[3][0] + 1, *e[3][1:]]] for e in STATE]            # only the boxes differ: bytes in the strip
    res = verify.verify(run, t, pack, only=["c01-s1"], snap=_state_snap(lambda tt, d: moved if d == "b" else STATE))
    assert res.ok is True and res.warnings == []


def test_a_capture_without_a_digest_among_digests_fails_by_name(tmp_path):
    run, pack, t = _setup(tmp_path)
    real = _state_snap(lambda tt, d: STATE)

    def snap(project, times, out_dir):
        paths = real(project, times, out_dir)
        if out_dir.name == "b":
            for p in paths:
                Image.new("RGB", (48, 10), (10, 20, 30)).save(p)
        return paths

    res = verify.verify(run, t, pack, only=["c01-s1"], snap=snap)
    assert res.ok is False and res.mismatches[0]["reason"] == "no state digest" and "reverse capture" in res.mismatches[0]["message"]


AUDIT_HIT = [{"target": "#curRing", "property": "scaleX, scaleY, opacity",
              "a": {"start": 2.4, "end": 3.1, "site": ["lib/demo-world.js:306", "lib/demo-world.js:393"]},
              "b": {"start": 3.0, "end": 3.0, "site": ["lib/demo-world.js:310", "lib/demo-world.js:392"]}}]


def test_an_overlap_in_the_audit_fails_with_the_sites_unless_allowed(tmp_path):
    run, pack, t = _setup(tmp_path)
    res = verify.verify(run, t, pack, only=["c01-s1"], snap=_state_snap(lambda tt, d: STATE, audit=AUDIT_HIT))
    assert res.ok is False and [m["kind"] for m in res.mismatches] == ["overlap"]      # reported once, not once per frame
    m = res.mismatches[0]
    assert (m["shot"], m["target"], m["property"]) == ("c01-s1", "#curRing", "scaleX, scaleY, opacity")
    for part in ("c01-s1: #curRing scaleX, scaleY, opacity is tweened twice at once", "2.400-3.100s (lib/demo-world.js:306 < "
                 "lib/demo-world.js:393)", "3.000-3.000s (lib/demo-world.js:310 < lib/demo-world.js:392)",
                 "backward step", "earlier one wins"):
        assert part in m["message"], part
    allowed = verify.verify(run, t, pack, only=["c01-s1"], snap=_state_snap(lambda tt, d: STATE, audit=AUDIT_HIT),
                            allow_overlaps=True)
    assert allowed.ok is True and [w["kind"] for w in allowed.warnings] == ["overlap"] and allowed.mismatches == []


def test_sites_in_the_served_page_are_found_again_in_the_shots_own_files(tmp_path):
    source = {"lib/demo-world.js": ["const go = (el, t) => {", "  W.fromTo(e, from, to, t);", "  click(t);", "  click(t);", "};",
                                    "go('#ring', 2.5, { scale: 1 }, \"def\", 0.7);", "go('#ring', 3.0, { scale: 2 }, \"def\", 0.7);"],
              "scenes/shots/c01-s1.html": ["<script>", "  const S = await PromoStage.load(ID);", "  window.__timelines[ID] = tl;"]}
    # the served page puts lib/demo-world.js at an offset of 385 lines: its text is found once, so the offset is known,
    # and `click(t);` (found twice in the file) is settled by it
    audit = [{"target": "#ring", "property": "scaleX", "a": {"start": 0, "end": 1, "site": [
                 "L386: const go = (el, t) => {", "L388: click(t);", "L391: go('#ring', 2.5, { scale: 1 }, \"def\", 0.7);"]},
              "b": {"start": 0.5, "end": 1.5, "site": ["L387: W.fromTo(e, from, to, t);", "L1000: nothing like this anywhere",
                                                       "L2: const S = await PromoStage.load(ID);"]}}]
    out = verify.resolve_sites(audit, source)[0]
    assert out["a"]["site"] == ["lib/demo-world.js:1", "lib/demo-world.js:3", "lib/demo-world.js:6"]
    assert out["b"]["site"] == ["lib/demo-world.js:2", "line 1000 of the served page", "scenes/shots/c01-s1.html:2"]
    assert audit[0]["a"]["site"][0] == "L386: const go = (el, t) => {"                     # the input is not changed


def test_a_site_text_matches_whether_or_not_the_served_page_kept_its_trailing_comment(tmp_path):
    source = {"scenes/shots/c01-s1.html": ["  const a = 1;", "  if (i === 2) tl.to(c, { x: 60 }, 0.3);   // on the entrance's x"]}
    audit = [{"target": "#k", "property": "x", "a": {"start": 0, "end": 1, "site": ["L9: if (i === 2) tl.to(c, { x: 60 }, 0.3);"]},
              "b": {"start": 0, "end": 1, "site": ["L9: if (i === 2) tl.to(c, { x: 60 }, 0.3);   // on the entrance's x"]}}]
    out = verify.resolve_sites(audit, source)[0]
    assert out["a"]["site"] == out["b"]["site"] == ["scenes/shots/c01-s1.html:2"]


def test_a_site_text_that_contains_the_separator_survives(tmp_path):
    line = "for (let j = 0; j < n; j++) go(s, 1.6 + j * 4.4, { y: 1 });"
    out = verify.resolve_sites([{"target": "#a", "property": "y", "a": {"start": 0, "end": 1, "site": [f"L9: {line}"]},
                                 "b": {"start": 0, "end": 1, "site": []}}], {"lib/x.js": ["", line]})[0]
    assert out["a"]["site"] == ["lib/x.js:2"] and out["b"]["site"] == []


def test_an_overlap_without_a_known_site_still_says_target_property_and_times(tmp_path):
    run, pack, t = _setup(tmp_path)
    bare = [{**AUDIT_HIT[0], "a": {"start": 1.0, "end": 2.0, "site": []}, "b": {"start": 1.5, "end": 2.5, "site": []}}]
    res = verify.verify(run, t, pack, only=["c01-s1"], snap=_state_snap(lambda tt, d: STATE, audit=bare))
    assert "1.000-2.000s and 1.500-2.500s;" in res.mismatches[0]["message"]


def test_the_failures_are_what_the_critic_reads_so_overlaps_and_state_both_make_the_result_not_ok(tmp_path):
    run, pack, t = _setup(tmp_path)
    verify.verify(run, t, pack, only=["c01-s1"], snap=_state_snap(lambda tt, d: STATE, audit=AUDIT_HIT))
    saved = json.loads((run.renders / "verify.json").read_text())
    assert saved["ok"] is False and saved["shots"] == ["c01-s1"] and saved["mismatches"][0]["kind"] == "overlap"
    assert saved["warnings"] == [] and "fingerprints" in saved


# --- the command ----------------------------------------------------------------------------------------------------

def _in_run(tmp_path, monkeypatch, *, timing=True, scenes=True):
    run, pack, t = _with_scenes(tmp_path) if scenes else _setup(tmp_path)
    if timing:
        write_timing(t, run.timing)
    monkeypatch.chdir(run.root)
    return run


REAL_VERIFY = verify.verify


def _use_snap(monkeypatch, snap):
    monkeypatch.setattr(verify, "verify", lambda run, t, pack, **kw: REAL_VERIFY(run, t, pack, snap=snap, **kw))


def test_verify_command_ok(tmp_path, monkeypatch):
    run = _in_run(tmp_path, monkeypatch)
    _use_snap(monkeypatch, _fake_snap(lambda tt, d: (1, 2, 3)))
    res = runner.invoke(app, ["verify", "--shot", "c01-s1", "--frames", "3"])
    assert res.exit_code == 0, res.output
    assert "ok: 1 shots checked" in res.output
    assert json.loads((run.renders / "verify.json").read_text())["shots"] == ["c01-s1"]


def test_verify_command_mismatch_is_exit_1_and_says_where_and_how_much(tmp_path, monkeypatch):
    _in_run(tmp_path, monkeypatch)
    _use_snap(monkeypatch, _fake_snap(lambda tt, d: (40, 0, 0) if d == "b" else (30, 0, 0)))
    res = runner.invoke(app, ["verify", "--shot", "c01-s1"])
    assert res.exit_code == 1
    assert "FAIL: 1 shots checked" in res.output and "c01-s1 at 0.0s: pixels differ between seek orders" in res.output
    assert "largest difference 10 of 255 on 64" in res.output and "renders/verify" in res.output


def test_verify_command_prints_state_failures_overlaps_and_warnings_and_allow_overlaps_demotes_the_overlaps(tmp_path, monkeypatch):
    _in_run(tmp_path, monkeypatch)
    rounded = _with(STATE, "#pL2", "transform: translate(0px, 3px) rotate(4.3905deg); opacity: 1;")
    _use_snap(monkeypatch, _state_snap(lambda tt, d: rounded if d == "b" else STATE, audit=AUDIT_HIT))
    res = runner.invoke(app, ["verify", "--shot", "c01-s1", "--frames", "2"])
    assert res.exit_code == 1
    assert "FAIL: 1 shots checked, 3 failures, 0 warnings" in res.output
    assert "  FAIL  c01-s1 at 0.0s: #pL2 transform: translate(0px, 3px) rotate(4.29049deg) forward vs" in res.output
    assert "values differ between the orders" in res.output
    assert "  FAIL  c01-s1: #curRing scaleX, scaleY, opacity is tweened twice at once" in res.output
    allowed = runner.invoke(app, ["verify", "--shot", "c01-s1", "--frames", "2", "--allow-overlaps"])
    assert allowed.exit_code == 1 and "  warn  c01-s1: #curRing" in allowed.output       # the state failures still fail it
    _use_snap(monkeypatch, _state_snap(lambda tt, d: STATE, audit=AUDIT_HIT))
    ok = runner.invoke(app, ["verify", "--shot", "c01-s1", "--frames", "2", "--allow-overlaps"])
    assert ok.exit_code == 0 and "ok: 1 shots checked, 0 failures, 1 warnings" in ok.output


def test_verify_command_pixel_only_differences_are_warnings_and_exit_0(tmp_path, monkeypatch):
    _in_run(tmp_path, monkeypatch)
    _use_snap(monkeypatch, _state_snap(lambda tt, d: STATE, patch_for=lambda tt, d: (20, 2, 4, 3, (200, 0, 0)) if d == "b" else None))
    res = runner.invoke(app, ["verify", "--shot", "c01-s1", "--frames", "2"])
    assert res.exit_code == 0 and "ok: 1 shots checked, 0 failures, 2 warnings" in res.output
    assert "  warn  c01-s1 at 0.0s: DOM identical; Chrome reused raster from an earlier frame on #root" in res.output


def test_verify_command_without_timing_is_exit_2_and_one_line(tmp_path, monkeypatch):
    _in_run(tmp_path, monkeypatch, timing=False)
    res = runner.invoke(app, ["verify"])
    assert res.exit_code == 2 and "promo timing" in res.output and "Traceback" not in res.output


def test_verify_command_unknown_shot_and_missing_scenes_are_exit_2(tmp_path, monkeypatch):
    _in_run(tmp_path, monkeypatch)
    _use_snap(monkeypatch, _fake_snap(lambda tt, d: (1, 2, 3)))
    res = runner.invoke(app, ["verify", "--shot", "nope"])
    assert res.exit_code == 2 and "nope" in res.output and "Traceback" not in res.output


def test_verify_command_with_no_scene_file_is_exit_2(tmp_path, monkeypatch):
    _in_run(tmp_path, monkeypatch, scenes=False)
    res = runner.invoke(app, ["verify"])
    assert res.exit_code == 2 and "no scene file for c01-s1" in res.output and "promo scenes --scaffold" in res.output


def test_verify_command_with_a_failing_tool_is_one_line_exit_1(tmp_path, monkeypatch):
    from promo.proc import ToolError

    _in_run(tmp_path, monkeypatch)

    def boom(project, times, out_dir):
        raise ToolError("command failed (1): npx hyperframes snapshot\nChrome crashed")

    _use_snap(monkeypatch, boom)
    res = runner.invoke(app, ["verify", "--shot", "c01-s1"])
    assert res.exit_code == 1 and "command failed" in res.output and "Traceback" not in res.output
    assert (Path.cwd() / "logs" / "last-error.log").read_text().count("Chrome crashed") == 1


def test_verify_command_without_node_is_exit_2(tmp_path, monkeypatch):
    _in_run(tmp_path, monkeypatch)

    def no_node(project, times, out_dir):
        raise FileNotFoundError(2, "No such file or directory", "npx")

    _use_snap(monkeypatch, no_node)
    res = runner.invoke(app, ["verify", "--shot", "c01-s1"])
    assert res.exit_code == 2 and "Node.js" in res.output


def test_verify_command_rejects_zero_frames(tmp_path, monkeypatch):
    _in_run(tmp_path, monkeypatch)
    assert runner.invoke(app, ["verify", "--frames", "0"]).exit_code == 2


# --- live: the real HyperFrames snapshot -------------------------------------------------------------------------------

def _copy_example(tmp_path):
    from promo.run import RunPaths

    run_dir = tmp_path / "run"
    shutil.copytree(EXAMPLE / "scenes", run_dir / "scenes", ignore=shutil.ignore_patterns(".render"))
    for name in ("shotlist.json", "timing.json", "design.json"):
        shutil.copy(EXAMPLE / name, run_dir / name)
    return RunPaths(run_dir)


CLEAN_SCENE = """<!doctype html>
<html lang="en"><head><meta charset="UTF-8" /><meta name="viewport" content="width=1920, height=1080" />
<script src="assets/vendor/gsap.min.js"></script><script src="lib/motion.js"></script><script src="lib/stage.js"></script>
<style>html, body { margin: 0; width: 100%; height: 100%; background: #101114; }
#root { position: relative; width: 100%; height: 100%; overflow: hidden; background: #f2f3ec; }
.card { position: absolute; width: 300px; height: 180px; border-radius: 20px; background: #fff; box-shadow: 0 24px 60px rgba(22,24,26,0.18); }</style></head>
<body><div id="root" data-composition-id="c01-s1" data-start="0" data-duration="1" data-width="1920" data-height="1080" data-fps="60"></div>
<script>
(async () => {
  const S = await PromoStage.load("c01-s1");
  const root = document.getElementById("root");
  const r = PromoMotion.rng(3);
  const tl = gsap.timeline({ paused: true });
  for (let i = 0; i < 6; i++) {
    const c = document.createElement("div"); c.className = "card"; c.id = "k" + i;
    c.style.left = (100 + i * 280) + "px"; c.style.top = (200 + (i % 2) * 260) + "px"; root.appendChild(c);
    const rot = Math.round((r() - 0.5) * 20 * 1e4) / 1e4;
    tl.fromTo(c, { x: -200, y: 40, rotation: 0, opacity: 0 }, { x: 0, y: 0, rotation: rot, opacity: 1, duration: 0.8, ease: PromoMotion.ease("default") }, 0.1 * i);
    /*OVERLAP*/
  }
  window.__timelines = window.__timelines || {}; window.__timelines["c01-s1"] = tl;
})();
</script></body></html>
"""


@pytest.mark.live
def test_live_a_clean_scene_passes_every_gate_and_an_overlapping_tween_is_found_with_its_line(tmp_path):
    """A small scene of the harness's own making, with the pinned HyperFrames: rounded generated values, explicit start
    values and no overlaps give the same state in every order (nothing to fail, nothing to warn about that is not pixels);
    adding a tween on the same property as the entrance fails the audit and names the line it was written on."""
    from promo.hyperframes import ensure_shot, scaffold_scenes

    run, pack, t = _setup(tmp_path)
    scaffold_scenes(run.scenes)
    ensure_shot(run.scenes, "c01-s1")
    (run.shots_dir / "c01-s1.html").write_text(CLEAN_SCENE)
    res = verify.verify(run, t, pack, only=["c01-s1"], n=6)
    assert res.ok is True and res.mismatches == [], res.mismatches
    assert all(w["reason"].startswith("DOM identical") for w in res.warnings)          # pixels only, never state

    bad = CLEAN_SCENE.replace("/*OVERLAP*/", "if (i === 2) tl.to(c, { x: 60, duration: 0.5 }, 0.3);   // on the entrance's x")
    (run.shots_dir / "c01-s1.html").write_text(bad)
    res = verify.verify(run, t, pack, only=["c01-s1"], n=6)
    assert res.ok is False
    overlaps = [m for m in res.mismatches if m["kind"] == "overlap"]
    assert [m["target"] for m in overlaps] == ["#k2"] and overlaps[0]["property"] == "x"
    line = next(i for i, text in enumerate(bad.splitlines(), 1) if "on the entrance's x" in text)
    assert f"scenes/shots/c01-s1.html:{line}" in overlaps[0]["b"]["site"]


@pytest.mark.live
def test_live_without_the_hook_a_real_shots_pixels_depend_on_the_seek_order(tmp_path, monkeypatch):
    """c03-s1 of the Oleafly 30 s run (a copy; the film is frozen) with a stage that has no hook: no digest, so the
    pixels are all there is, and they differ between forward and reverse capture."""
    if not (EXAMPLE / "scenes" / "shots" / "c03-s1.html").exists():
        pytest.skip("examples/promo/oleafly-30s is not here")
    from promo.config import packs_dir
    from promo.shotlist import load_shotlist
    from promo.timing import load_timing

    run = _copy_example(tmp_path)
    template = packs_dir() / "_core" / "scene-template" / "lib" / "stage.js"
    bare_dir = tmp_path / "bare" / "lib"
    bare_dir.mkdir(parents=True)
    bare = re.sub(r"// capture-settle:begin.*?// capture-settle:end", "", template.read_text(), flags=re.S)
    bare = bare.replace("if ((design.capture || {}).settle) installCaptureSettle(shotId);", "")
    bare = bare.replace("capture: { stateOf, auditTimeline }", "capture: {}")
    (bare_dir / "stage.js").write_text(bare)
    monkeypatch.setattr(verify, "TEMPLATE_DIR", tmp_path / "bare")
    res = verify.verify(run, load_timing(run.timing), load_pack(load_shotlist(run.shotlist).pack), only=["c03-s1"], n=12)
    assert res.ok is False and {m["reason"] for m in res.mismatches} == {"differs between seek orders"}
    assert not (run.scenes / ".render").exists()


@pytest.mark.live
def test_live_equal_values_printed_two_ways_pass_and_the_audit_names_the_overlaps_in_a_real_shot(tmp_path):
    """c01-s1 of the Oleafly 30 s run (a copy; the film is frozen): its generated rotations print with two roundings
    (rotate(5.64341deg) in one order, rotate(5.6434deg) in the other) and its dust scales are written scale(1.11) and
    scale(1.11, 1.11): equal values, so the state gate passes. Its click ring, filter pulses and drifting props overlap
    their own tweens, which the audit finds with their lines."""
    if not (EXAMPLE / "scenes" / "shots" / "c01-s1.html").exists():
        pytest.skip("examples/promo/oleafly-30s is not here")
    from promo.shotlist import load_shotlist
    from promo.timing import load_timing

    run = _copy_example(tmp_path)
    res = verify.verify(run, load_timing(run.timing), load_pack(load_shotlist(run.shotlist).pack), only=["c01-s1"], n=6)
    assert res.ok is False
    assert [m for m in res.mismatches if m["kind"] == "state"] == []
    overlaps = [m for m in res.mismatches if m["kind"] == "overlap"]
    assert {m["target"] for m in overlaps} >= {"#c0r", "#slotStrip"}
    assert all(any("lib/demo-world.js:" in part for part in m["a"]["site"]) for m in overlaps)
    assert all(any("lib/demo-world.js:" in part for part in m["b"]["site"]) for m in overlaps)


@pytest.mark.live
def test_live_an_inline_style_that_exists_in_one_order_only_is_named_as_that(tmp_path):
    """c04-s1 (a copy): #aiTyped is in the page in both orders, but has no inline style in one and `opacity: 1` in the
    other, and the message says exactly that instead of calling the element absent."""
    if not (EXAMPLE / "scenes" / "shots" / "c04-s1.html").exists():
        pytest.skip("examples/promo/oleafly-30s is not here")
    from promo.shotlist import load_shotlist
    from promo.timing import load_timing

    run = _copy_example(tmp_path)
    res = verify.verify(run, load_timing(run.timing), load_pack(load_shotlist(run.shotlist).pack), only=["c04-s1"], n=6)
    state = [m for m in res.mismatches if m["kind"] == "state"]
    assert state and all(m["element"] == "#aiTyped" and m["property"] == "opacity" for m in state)
    assert all("(unset) forward vs 1 reverse" in m["message"] and "absent" not in m["message"] for m in state)
    assert all("there in both orders" in m["hint"] for m in state)
