#!/usr/bin/env bash
# Install the promo harness: the `promo` CLI on PATH and its skills linked into ~/.claude/skills. Safe to run again.
# PROMO_SKILLS_DIR overrides where the skills are linked (default: $HOME/.claude/skills).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILLS_DIR="${PROMO_SKILLS_DIR:-$HOME/.claude/skills}"
cd "$ROOT"

uv sync --extra dev
uv tool install --editable "$ROOT" --python 3.12 --force

mkdir -p "$SKILLS_DIR"
blocked=0
for dir in "$ROOT"/skills/*/; do
  [ -f "${dir}SKILL.md" ] || continue
  name="$(basename "$dir")"
  dest="$SKILLS_DIR/$name"
  if [ -e "$dest" ] && [ ! -L "$dest" ]; then
    echo "not linked: $dest exists and is not a link; move it away and run this again" >&2
    blocked=$((blocked + 1))
    continue
  fi
  ln -sfn "${dir%/}" "$dest"
  echo "linked skill $name"
done

if [ ! -f "$ROOT/.env" ]; then
  cp "$ROOT/.env.example" "$ROOT/.env"
  chmod 600 "$ROOT/.env"        # it will hold API keys
  echo "created $ROOT/.env from .env.example: fill in the keys you need (Supertonic voice needs none)"
fi

echo "promo: $(command -v promo || echo 'not on PATH yet: run uv tool update-shell, or add ~/.local/bin to PATH')"
echo "next: promo doctor"
if [ "$blocked" -gt 0 ]; then
  echo "$blocked skill(s) not linked; see above" >&2
  exit 1
fi
