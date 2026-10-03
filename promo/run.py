"""Run folders: locating them, their paths, and scaffolding new ones."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from promo.config import templates_dir
from promo.packs import load_pack

RUN_MARKER = "shotlist.json"
ASPECTS = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}
_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class RunNotFound(RuntimeError):
    pass


@dataclass(frozen=True)
class RunPaths:
    root: Path

    def _p(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    shotlist = property(lambda self: self._p("shotlist.json"))
    timing = property(lambda self: self._p("timing.json"))
    script_md = property(lambda self: self._p("script.md"))
    brief = property(lambda self: self._p("brief.md"))
    story = property(lambda self: self._p("story.md"))
    brand = property(lambda self: self._p("brand.json"))
    design = property(lambda self: self._p("design.json"))
    guide = property(lambda self: self._p("ANIMATION_GUIDE.md"))
    scenes = property(lambda self: self._p("scenes"))
    shots_dir = property(lambda self: self._p("scenes", "shots"))
    render_dir = property(lambda self: self._p("scenes", ".render"))
    vo_dir = property(lambda self: self._p("audio", "vo"))
    music_dir = property(lambda self: self._p("audio", "music"))
    sfx_dir = property(lambda self: self._p("audio", "sfx"))
    mix_dir = property(lambda self: self._p("audio", "mix"))
    auditions_dir = property(lambda self: self._p("audio", "auditions"))
    captions_dir = property(lambda self: self._p("captions"))
    clips_dir = property(lambda self: self._p("clips"))
    renders = property(lambda self: self._p("renders"))
    critic = property(lambda self: self._p("critic"))
    timeline = property(lambda self: self._p("timeline.otio"))
    costs = property(lambda self: self._p("costs.jsonl"))
    attribution = property(lambda self: self._p("ATTRIBUTION.md"))


def find_run(start: Path | None = None) -> RunPaths:
    here = Path(start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        if (d / RUN_MARKER).is_file():
            return RunPaths(d)
    raise RunNotFound(
        f"promo: no run folder here (no {RUN_MARKER} in {here} or its parents). Run `promo new <slug>` first."
    )


def _skeleton(pack_name: str, aspect: str, lang: str) -> dict:
    pack = load_pack(pack_name)
    w, h = ASPECTS[aspect]
    music_mode = pack.timing_mode() == "music"
    # Two events, so the skeleton passes its own first `promo timing` (one event leaves a dead gap).
    shot = {"id": "c01-s1", "events": [
        {"id": "c01-e1", "kind": "headline",
         "at": {"beat": 0} if music_mode else {"shot": "c01-s1", "edge": "start", "offset": 0.2}},
        {"id": "c01-e2", "kind": "tick",
         "at": {"beat": 4} if music_mode else {"shot": "c01-s1", "edge": "start", "offset": 2.0}}]}
    chapter = {"id": "c01", "purpose": "Hook: replace with the one-line problem",
               "headline": {"text": "Replace this headline", "key": "headline"}, "vo": [], "shots": [shot]}
    if music_mode:
        chapter["bars"] = 2
    else:
        shot["seconds"] = 3.0
    return {
        "version": 2, "pack": pack_name, "language": lang,
        "canvas": {"aspect": aspect, "width": w, "height": h, "fps": pack.tokens.get("fps", 60)},
        "timing_mode": pack.timing_mode(), "preroll_s": 0.3,
        "music": {"bpm": pack.music.get("bpm"), "beats_per_bar": 4},
        "voice": {"name": pack.voice.get("voice")},
        "claims": [], "chapters": [chapter],
    }


def new_run(project_dir: Path, slug: str, pack: str, aspect: str, length_s: float | None, lang: str) -> RunPaths:
    if not _SLUG.match(slug):
        raise ValueError(f"slug '{slug}' must be lowercase letters, digits and dashes")
    if aspect not in ASPECTS:
        raise ValueError(f"aspect '{aspect}' must be one of {', '.join(ASPECTS)}")
    run = RunPaths(Path(project_dir).resolve() / "promo" / slug)
    if run.root.exists():
        raise FileExistsError(f"{run.root} already exists")
    skeleton = _skeleton(pack, aspect, lang)
    run.root.mkdir(parents=True)
    for d in (run.vo_dir, run.music_dir, run.sfx_dir, run.mix_dir, run.auditions_dir, run.captions_dir,
              run.clips_dir, run.renders, run.critic, run.shots_dir, run.root / "assets" / "capture"):
        d.mkdir(parents=True, exist_ok=True)
    subs = {"{{SLUG}}": slug, "{{PACK}}": pack, "{{ASPECT}}": aspect, "{{LANG}}": lang,
            "{{LENGTH}}": f"{length_s:g} s" if length_s else "(ask)"}
    for name, dest in (("brief.md", run.brief), ("story.md", run.story), ("README.md", run.root / "README.md"),
                       ("gitignore.txt", run.root / ".gitignore")):
        text = (templates_dir() / "run" / name).read_text()
        for k, v in subs.items():
            text = text.replace(k, v)
        dest.write_text(text)
    run.shotlist.write_text(json.dumps(skeleton, indent=2) + "\n")
    return run
