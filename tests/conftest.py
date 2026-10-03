import json
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_cache(request, tmp_path, monkeypatch):
    """Keep every test's API cache out of ~/.cache, except live tests.

    Live tests use the real cache so paid results stay cached and inspectable.
    """
    if not request.node.get_closest_marker("live"):
        monkeypatch.setenv("PROMO_CACHE_DIR", str(tmp_path / "cache"))
    yield


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


FAKE_SUPERTONIC = r'''#!/usr/bin/env python3
"""A stand-in for the supertonic CLI: records its argv, writes a sine WAV (44.1 kHz mono, 0.2 s of silence each end)."""
import json, math, os, struct, sys, wave

args = sys.argv[1:]
runs = 0
if os.environ.get("FAKE_SUPERTONIC_LOG"):
    with open(os.environ["FAKE_SUPERTONIC_LOG"], "a") as f:
        f.write(json.dumps(args) + "\n")
    runs = sum(1 for line in open(os.environ["FAKE_SUPERTONIC_LOG"]) if line.startswith('["tts"'))
if args[:1] == ["version"]:
    print("supertonic 9.9.9")
    sys.exit(0)
if os.environ.get("FAKE_SUPERTONIC_FAIL"):
    print("❌ Error: " + os.environ["FAKE_SUPERTONIC_FAIL"])
    sys.exit(1)
if os.environ.get("FAKE_SUPERTONIC_NO_FILE"):
    sys.exit(0)

def value(flag, default):
    return args[args.index(flag) + 1] if flag in args else default

text, speed = args[-1], float(value("--speed", 1.05))
seconds = float(os.environ.get("FAKE_SUPERTONIC_SECONDS") or max(0.5, len(text.split()) * 0.4 / speed))
rate, pad = 44100, [0] * 8820
silent = runs <= int(os.environ.get("FAKE_SUPERTONIC_SILENT") or 0)      # the first N draws are digital silence
tone = [0 if silent else int(0.3 * 32767 * math.sin(2 * math.pi * 220 * i / rate)) for i in range(int(seconds * rate))]
with wave.open(value("-o", "out.wav"), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
    w.writeframes(struct.pack(f"<{len(pad) * 2 + len(tone)}h", *pad, *tone, *pad))
'''


class FakeSupertonic:
    def __init__(self, exe: Path, log: Path):
        self.exe, self.log = exe, log

    def calls(self) -> list[list[str]]:
        """Every invocation's arguments, in order (`version` calls included)."""
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def tts(self) -> list[list[str]]:
        return [c for c in self.calls() if c[0] == "tts"]

    @staticmethod
    def flag(call: list[str], name: str) -> str:
        return call[call.index(name) + 1]


@pytest.fixture
def fake_supertonic(tmp_path, monkeypatch) -> FakeSupertonic:
    """A fake `supertonic` first on PATH. FAKE_SUPERTONIC_SECONDS, _FAIL, _NO_FILE and _SILENT=N steer it."""
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    exe = bin_dir / "supertonic"
    exe.write_text(FAKE_SUPERTONIC, encoding="utf-8")
    exe.chmod(0o755)
    log = tmp_path / "supertonic-calls.jsonl"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SUPERTONIC_LOG", str(log))
    for var in ("FAKE_SUPERTONIC_SECONDS", "FAKE_SUPERTONIC_FAIL", "FAKE_SUPERTONIC_NO_FILE", "FAKE_SUPERTONIC_SILENT"):
        monkeypatch.delenv(var, raising=False)
    return FakeSupertonic(exe, log)
