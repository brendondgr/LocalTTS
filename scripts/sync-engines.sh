#!/usr/bin/env bash
# Refresh engines/ (the TTS engine installer) and workers/align.py from video-maker's tts/,
# which is the canonical copy. LocalTTS keeps its own copy so it installs without the skill.
#   scripts/sync-engines.sh [path/to/video-maker/tts]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:-$HOME/Projects/bdgrClaudeSkills/video-maker/tts}"
[ -f "$SRC/setup.py" ] || { echo "no engine installer at $SRC (pass the video-maker/tts path)" >&2; exit 1; }
mkdir -p "$ROOT/engines/breeze/patches"
for f in setup.sh setup.py probe.py registry.py install.sh install.ps1 kokoro_tts.py \
         breeze/install.sh breeze/breeze_tts.py breeze/align.py breeze/UPSTREAM_COMMIT; do
  cp "$SRC/$f" "$ROOT/engines/$f"
done
rm -f "$ROOT/engines/breeze/patches/"*.patch
cp "$SRC"/breeze/patches/*.patch "$ROOT/engines/breeze/patches/"
cp "$SRC/breeze/align.py" "$ROOT/workers/align.py"
echo "synced engines/ and workers/align.py from $SRC"
