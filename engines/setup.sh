#!/usr/bin/env bash
# Set up the TTS engines (Kokoro, Breeze) for this machine. Looks first, explains, then installs.
#   bash tts/setup.sh --plan        # just look and explain
#   bash tts/setup.sh               # look, explain, ask, install
# All options: bash tts/setup.sh --help. Linux and macOS (Windows: tts/install.ps1, Kokoro only).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Inside a toolbox/distrobox container the GPU stack lives on the host: look and install there.
if [ -z "${TTS_NO_HOST:-}" ] && { [ -e /run/.toolboxenv ] || [ -e /run/.containerenv ]; }; then
  if command -v flatpak-spawn >/dev/null 2>&1; then
    exec flatpak-spawn --host --directory="$PWD" bash "$HERE/setup.sh" "$@"
  elif command -v distrobox-host-exec >/dev/null 2>&1; then
    exec distrobox-host-exec bash "$HERE/setup.sh" "$@"
  fi
fi

# Any Python 3.8+ will do for the planner (standard library only); uv can supply one.
for py in python3 python; do
  if command -v "$py" >/dev/null 2>&1 && "$py" -c 'import sys; sys.exit(sys.version_info < (3, 8))' 2>/dev/null; then
    exec "$py" "$HERE/setup.py" "$@"
  fi
done
if command -v uv >/dev/null 2>&1; then
  exec uv run --no-project --python 3.12 python "$HERE/setup.py" "$@"
fi
echo "✖ need Python 3.8+ or uv (curl -LsSf https://astral.sh/uv/install.sh | sh)" >&2
exit 1
