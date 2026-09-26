#!/usr/bin/env bash
# Install LocalTTS: the API server, the `localtts` command, and a service that starts at boot
# (systemd on Linux). Re-run any time; it is idempotent.
#
#   ./install.sh                 # server + service; offers to set up TTS engines if some are missing
#   ./install.sh --engines       # also run the engine setup (inspects this machine first)
#   ./install.sh --no-engines    # never touch the engines
#   ./install.sh --no-service    # no boot service (run it with `localtts start`)
#
# The TTS engines (Kokoro, Breeze) are installed by engines/setup.sh, which looks at this
# machine's GPU/OS and picks the right PyTorch build. It is the same installer video-maker uses,
# so engines installed by either are shared (~/.config/tts-engines/engines.json).
set -euo pipefail
if [ -z "${LOCALTTS_NO_HOST:-}" ] && { [ -e /run/.toolboxenv ] || [ -e /run/.containerenv ]; }; then
  if command -v flatpak-spawn >/dev/null 2>&1; then exec flatpak-spawn --host bash "$(readlink -f "$0")" "$@"
  elif command -v distrobox-host-exec >/dev/null 2>&1; then exec distrobox-host-exec bash "$(readlink -f "$0")" "$@"; fi
fi
ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
CONF_DIR="$HOME/.config/localtts"
ENGINES=ask SERVICE=1
for a in "$@"; do
  case "$a" in
    --engines) ENGINES=yes ;;
    --no-engines) ENGINES=no ;;
    --no-service) SERVICE=0 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done
command -v uv >/dev/null || { echo "uv is required: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }

echo "== server venv ($ROOT/.venv)"
[[ -x "$ROOT/.venv/bin/python" ]] || uv venv -q "$ROOT/.venv" --python 3.12
uv pip install -q --python "$ROOT/.venv/bin/python" -r "$ROOT/requirements.txt"
PY="$ROOT/.venv/bin/python"

echo "== TTS engines"
missing=0
for e in kokoro breeze; do
  p="$(cd "$ROOT" && "$PY" -c "from localtts.config import settings; print(settings.${e}_python)")"
  if [[ -x "$p" ]]; then echo "  ok      $e  $p"; else echo "  missing $e"; missing=$((missing + 1)); fi
done
if [[ "$ENGINES" == ask && "$missing" -gt 0 ]]; then
  if [[ -t 0 ]]; then
    echo; bash "$ROOT/engines/setup.sh" --plan || true; echo
    read -r -p "Run engines/setup.sh now to install what is missing? [Y/n] " ans
    [[ "${ans:-y}" =~ ^[Yy] ]] && ENGINES=yes
  else
    echo "  set them up with: bash $ROOT/engines/setup.sh --plan"
  fi
fi
[[ "$ENGINES" == yes ]] && bash "$ROOT/engines/setup.sh" --engines kokoro,breeze
command -v ffmpeg >/dev/null || echo "  WARNING: ffmpeg not on PATH; voice uploads need it"

echo "== config"
mkdir -p "$CONF_DIR" "$ROOT/data/voices"
[[ -f "$CONF_DIR/localtts.env" ]] || cp "$ROOT/localtts.env.example" "$CONF_DIR/localtts.env"
echo "  $CONF_DIR/localtts.env"
# breeze-tts (and video-maker) can then use the voices saved here.
"$PY" "$ROOT/engines/registry.py" add-voices-dir "$ROOT/data/voices"

echo "== command"
mkdir -p "$HOME/.local/bin"
ln -sf "$ROOT/bin/localtts" "$HOME/.local/bin/localtts"
echo "  $HOME/.local/bin/localtts"

[[ "$SERVICE" == 1 ]] && "$ROOT/bin/localtts" enable
"$ROOT/bin/localtts" restart
