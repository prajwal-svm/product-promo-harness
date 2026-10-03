"""Subprocess runner, atomic outputs, hashing and render sidecars."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shlex
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class ToolError(RuntimeError):
    """An external tool (ffmpeg, npx hyperframes, ...) failed."""


def run(
    cmd: list,
    *,
    cwd: Path | None = None,
    env: dict | None = None,
    timeout: float | None = None,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess:
    args = [str(c) for c in cmd]
    proc = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=text, timeout=timeout)
    if check and proc.returncode != 0:
        stderr = proc.stderr if text else proc.stderr.decode("utf-8", "replace")
        tail = "\n".join(stderr.splitlines()[-40:])
        raise ToolError(f"command failed ({proc.returncode}): {shlex.join(args)}\n{tail}")
    return proc


@contextmanager
def atomic_path(final: Path) -> Iterator[Path]:
    """Yield a temp path next to `final`; rename it into place only if the block succeeds."""
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f".{final.stem}.tmp{final.suffix}")
    if tmp.exists():
        tmp.unlink()
    try:
        yield tmp
        os.replace(tmp, final)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def sha256_hex(*parts: bytes | str) -> str:
    h = hashlib.sha256()
    for part in parts:
        data = part.encode("utf-8") if isinstance(part, str) else part
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sidecar(output: Path, *, tool: str, version: str, flags: list[str], inputs: dict[str, str]) -> Path:
    side = output.with_name(output.name + ".render.json")
    payload = {
        "output": output.name,
        "output_sha256": sha256_file(output),
        "tool": tool,
        "version": version,
        "flags": flags,
        "inputs": inputs,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    with atomic_path(side) as tmp:
        tmp.write_text(json.dumps(payload, indent=2))
    return side
