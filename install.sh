#!/usr/bin/env bash
# Install LocalTTS as a systemd user service that starts at boot, plus the `localtts` command.
# Re-run any time; it is idempotent. Run on the host (it hops out of a toolbox by itself).
set -euo pipefail
if [[ -f /run/.containerenv ]] && command -v flatpak-spawn >/dev/null 2>&1; then
  exec flatpak-spawn --host bash "$(readlink -f "$0")" "$@"
fi
ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
UNIT_DIR="$HOME/.config/systemd/user"
CONF_DIR="$HOME/.config/localtts"

echo "== server venv ($ROOT/.venv)"
[[ -x "$ROOT/.venv/bin/python" ]] || uv venv -q "$ROOT/.venv" --python 3.12
uv pip install -q --python "$ROOT/.venv/bin/python" -r "$ROOT/requirements.txt"

echo "== engine venvs (used as-is, never modified)"
for py in "${LOCALTTS_BREEZE_PYTHON:-$HOME/venvs/breeze-next/bin/python}" "${LOCALTTS_KOKORO_PYTHON:-$HOME/venvs/kokoro/bin/python}"; do
  [[ -x "$py" ]] && echo "  ok      $py" || echo "  MISSING $py (that engine will fail to load)"
done
command -v ffmpeg >/dev/null || echo "  WARNING: ffmpeg not on PATH; voice uploads need it"

echo "== config"
mkdir -p "$CONF_DIR" "$ROOT/data/voices" "$ROOT/data/outputs"
[[ -f "$CONF_DIR/localtts.env" ]] || cp "$ROOT/localtts.env.example" "$CONF_DIR/localtts.env"
echo "  $CONF_DIR/localtts.env"

echo "== systemd user unit"
mkdir -p "$UNIT_DIR"
sed "s|@ROOT@|$ROOT|g" "$ROOT/systemd/localtts.service" > "$UNIT_DIR/localtts.service"
systemctl --user daemon-reload
systemctl --user enable localtts >/dev/null
# Linger lets user services start at boot without anyone logging in.
if [[ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != yes ]]; then
  loginctl enable-linger "$USER" || echo "  could not enable linger; the service will start at login instead of boot"
fi

echo "== command"
mkdir -p "$HOME/.local/bin"
ln -sf "$ROOT/bin/localtts" "$HOME/.local/bin/localtts"
echo "  $HOME/.local/bin/localtts"

systemctl --user restart localtts
"$ROOT/bin/localtts" start
