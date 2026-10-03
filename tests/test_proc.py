import json
import sys

import pytest

from promo import proc


def test_run_returns_stdout():
    out = proc.run([sys.executable, "-c", "print('hi')"])
    assert out.stdout.strip() == "hi"


def test_run_failure_includes_command_and_stderr_tail():
    code = "import sys; [print(f'line{i}', file=sys.stderr) for i in range(60)]; sys.exit(3)"
    with pytest.raises(proc.ToolError) as err:
        proc.run([sys.executable, "-c", code])
    msg = str(err.value)
    assert "failed (3)" in msg
    assert "line59" in msg and "line19" not in msg  # only the last 40 lines


def test_atomic_path_renames_on_success(tmp_path):
    final = tmp_path / "out dir" / "clip 🎬.wav"
    with proc.atomic_path(final) as tmp:
        assert tmp != final
        assert tmp.suffix == ".wav"
        tmp.write_bytes(b"data")
    assert final.read_bytes() == b"data"
    assert not tmp.exists()


def test_atomic_path_cleans_up_on_error(tmp_path):
    final = tmp_path / "x.txt"
    with pytest.raises(RuntimeError):
        with proc.atomic_path(final) as tmp:
            tmp.write_text("partial")
            raise RuntimeError("boom")
    assert not final.exists()
    assert not tmp.exists()


def test_sha256_is_stable_and_order_sensitive():
    assert proc.sha256_hex("a", b"b") == proc.sha256_hex("a", b"b")
    assert proc.sha256_hex("a", "b") != proc.sha256_hex("b", "a")


def test_write_sidecar(tmp_path):
    out = tmp_path / "shot.mp4"
    out.write_bytes(b"x")
    side = proc.write_sidecar(out, tool="hyperframes", version="0.8.98", flags=["--fps", "60"], inputs={"a": "1"})
    data = json.loads(side.read_text())
    assert side.name == "shot.mp4.render.json"
    assert data["tool"] == "hyperframes" and data["inputs"] == {"a": "1"}
    assert data["output_sha256"] == proc.sha256_file(out)
