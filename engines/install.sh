#!/usr/bin/env bash
# Install kokoro-tts into one self-contained environment and put a `kokoro-tts` launcher on PATH.
# Linux and macOS. (Windows: tts/install.ps1.)
#
#   bash tts/install.sh                         # auto-detect the GPU
#   bash tts/install.sh --gpu rocm --rocm-arch gfx1151
#   bash tts/install.sh --gpu cuda --cuda cu130
#   bash tts/install.sh --gpu cpu
#
# Options (env var in brackets):
#   --gpu auto|rocm|cuda|xpu|mps|cpu   torch backend (default auto)            [KOKORO_GPU]
#   --rocm-arch gfxNNNN                AMD target for the ROCm nightly index    [KOKORO_ROCM_ARCH]
#   --rocm-index URL                   override the ROCm wheel index            [KOKORO_ROCM_INDEX]
#   --cuda cuNNN                       CUDA wheel flavour (default cu130)       [KOKORO_CUDA]
#   --prefix DIR                       environment dir (default ~/venvs/kokoro) [KOKORO_HOME]
#   --bin DIR                          launcher dir (default ~/.local/bin)      [KOKORO_BIN]
#   --force-torch                      reinstall torch even if present
#
# Re-running is safe: it reuses the environment and only fixes what is missing.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Inside a toolbox/distrobox container the GPU stack lives on the host: install there.
if [ -z "${KOKORO_NO_HOST:-}" ] && { [ -e /run/.toolboxenv ] || [ -e /run/.containerenv ]; }; then
  if command -v flatpak-spawn >/dev/null 2>&1; then
    echo "» container detected — running the installer on the host (set KOKORO_NO_HOST=1 to stop this)"
    exec flatpak-spawn --host --directory="$PWD" bash "$HERE/install.sh" "$@"
  elif command -v distrobox-host-exec >/dev/null 2>&1; then
    echo "» container detected — running the installer on the host (set KOKORO_NO_HOST=1 to stop this)"
    exec distrobox-host-exec bash "$HERE/install.sh" "$@"
  fi
fi

GPU="${KOKORO_GPU:-auto}"
ROCM_ARCH="${KOKORO_ROCM_ARCH:-}"
ROCM_INDEX="${KOKORO_ROCM_INDEX:-}"
CUDA="${KOKORO_CUDA:-cu130}"
PREFIX="${KOKORO_HOME:-$HOME/venvs/kokoro}"
BIN="${KOKORO_BIN:-$HOME/.local/bin}"
FORCE_TORCH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --rocm-arch) ROCM_ARCH="$2"; shift 2 ;;
    --rocm-index) ROCM_INDEX="$2"; shift 2 ;;
    --cuda) CUDA="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --bin) BIN="$2"; shift 2 ;;
    --force-torch) FORCE_TORCH=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\033[1m» %s\033[0m\n' "$*"; }
warn() { printf '\033[33m! %s\033[0m\n' "$*" >&2; }
die() { printf '\033[31m✖ %s\033[0m\n' "$*" >&2; exit 1; }
OS="$(uname -s)"

# ---- prerequisites ---------------------------------------------------------------------------
command -v uv >/dev/null 2>&1 || die "uv not found. Install it:  curl -LsSf https://astral.sh/uv/install.sh | sh   (or: dnf/brew/pacman install uv)"
if ! command -v espeak-ng >/dev/null 2>&1; then
  warn "espeak-ng not on PATH. Kokoro bundles a copy through espeakng-loader, but the system package is the"
  warn "reliable fallback for unknown words:  Fedora: sudo dnf install espeak-ng | Debian/Ubuntu: sudo apt install espeak-ng"
  warn "                                     Arch: sudo pacman -S espeak-ng     | macOS: brew install espeak-ng"
fi

# ---- pick the torch backend ------------------------------------------------------------------
detect_rocm_arch() {
  local v
  for f in /sys/class/kfd/kfd/topology/nodes/*/properties; do
    v="$(awk '/gfx_target_version/ {print $2}' "$f" 2>/dev/null || true)"
    if [ -n "$v" ] && [ "$v" != "0" ]; then
      printf 'gfx%d%d%x\n' $((v / 10000)) $(((v / 100) % 100)) $((v % 100)); return
    fi
  done
  command -v rocminfo >/dev/null 2>&1 && rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | grep -v gfx000 | head -1
}
if [ "$GPU" = auto ]; then
  if [ "$OS" = Darwin ]; then GPU=mps
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then GPU=cuda
  elif [ -e /dev/kfd ] && [ -n "$(detect_rocm_arch)" ]; then GPU=rocm
  else GPU=cpu
  fi
  say "auto-detected backend: $GPU"
fi

# AMD: TheRock nightly indexes are per GPU family. Map the chip to its index directory.
rocm_family() {
  case "$1" in
    gfx1150|gfx1151|gfx1152|gfx1153|gfx900|gfx906|gfx908|gfx90a) echo "$1" ;;
    gfx110*) echo gfx110X-all ;;     # RX 7000 / Radeon 700M
    gfx120*) echo gfx120X-all ;;     # RX 9000
    gfx103*) echo gfx103X-all ;;     # RX 6000
    gfx101*) echo gfx101X-dgpu ;;    # RX 5000
    gfx94*)  echo gfx94X-dcgpu ;;    # MI300
    gfx950)  echo gfx950-dcgpu ;;    # MI350
    *) echo "" ;;
  esac
}

TORCH_ARGS=()
case "$GPU" in
  rocm)
    [ "$OS" = Linux ] || die "--gpu rocm is Linux-only here (Windows: use install.ps1)"
    [ -n "$ROCM_ARCH" ] || ROCM_ARCH="$(detect_rocm_arch || true)"
    if [ -z "$ROCM_INDEX" ]; then
      fam="$(rocm_family "${ROCM_ARCH:-}")"
      [ -n "$fam" ] || die "unknown AMD target '${ROCM_ARCH:-none}'. Pass --rocm-arch, or --rocm-index https://download.pytorch.org/whl/rocm7.2 for an officially supported card"
      ROCM_INDEX="https://rocm.nightlies.amd.com/v2/$fam/"
    fi
    say "AMD ${ROCM_ARCH:-?} → $ROCM_INDEX"
    [ -r /dev/kfd ] && [ -w /dev/kfd ] || warn "/dev/kfd not accessible: sudo usermod -aG render,video \$USER, then log out and back in"
    TORCH_ARGS=(--pre --index-url "$ROCM_INDEX" torch torchaudio) ;;
  cuda) TORCH_ARGS=(--index-url "https://download.pytorch.org/whl/$CUDA" torch torchaudio) ;;
  xpu)  TORCH_ARGS=(--index-url "https://download.pytorch.org/whl/xpu" torch torchaudio) ;;
  cpu)
    if [ "$OS" = Darwin ]; then TORCH_ARGS=(torch torchaudio)
    else TORCH_ARGS=(--index-url "https://download.pytorch.org/whl/cpu" torch torchaudio); fi ;;
  mps)  TORCH_ARGS=(torch torchaudio) ;;   # the default PyPI wheels include Metal (MPS) on Apple Silicon
  *) die "--gpu must be auto, rocm, cuda, xpu, mps or cpu" ;;
esac

# ---- environment -----------------------------------------------------------------------------
PY="$PREFIX/bin/python"
if [ ! -x "$PY" ]; then
  say "creating Python 3.12 environment at $PREFIX"
  uv venv "$PREFIX" --python 3.12
fi
export VIRTUAL_ENV="$PREFIX"

backend_ok() {   # does the installed torch match the requested backend?
  "$PY" - "$GPU" <<'EOF' 2>/dev/null
import sys, torch
want = sys.argv[1]
ok = {"rocm": bool(torch.version.hip), "cuda": bool(torch.version.cuda) and not torch.version.hip,
      "xpu": hasattr(torch, "xpu"), "mps": True, "cpu": True}[want]
sys.exit(0 if ok else 1)
EOF
}

# Torch first: installing kokoro first would pull the default (NVIDIA/CPU) torch from PyPI.
if [ "$FORCE_TORCH" = 1 ] || ! backend_ok; then
  say "installing torch ($GPU)"
  if [ "$FORCE_TORCH" = 1 ]; then uv pip install --reinstall "${TORCH_ARGS[@]}"; else uv pip install "${TORCH_ARGS[@]}"; fi
else
  say "torch already matches backend $GPU — keeping it"
fi

say "installing kokoro, soundfile and the spaCy English model"
uv pip install "kokoro>=0.9.4" soundfile \
  "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

if ! backend_ok; then
  warn "kokoro's install replaced the $GPU torch — reinstalling it"
  uv pip install --reinstall "${TORCH_ARGS[@]}"
fi

# ---- the program and its launcher ------------------------------------------------------------
cp "$HERE/kokoro_tts.py" "$PREFIX/kokoro_tts.py"
mkdir -p "$BIN"
cat > "$BIN/kokoro-tts" <<EOF
#!/bin/sh
# kokoro-tts launcher (installed by video-maker/tts/install.sh). Environment: $PREFIX
KOKORO_HOME="\${KOKORO_HOME:-$PREFIX}"
# In a toolbox/distrobox container, run on the host, where the GPU libraries were installed.
if [ -z "\$KOKORO_NO_HOST" ] && { [ -e /run/.toolboxenv ] || [ -e /run/.containerenv ]; }; then
  if command -v flatpak-spawn >/dev/null 2>&1; then
    exec flatpak-spawn --host --directory="\$PWD" "\$KOKORO_HOME/bin/python" "\$KOKORO_HOME/kokoro_tts.py" "\$@"
  elif command -v distrobox-host-exec >/dev/null 2>&1; then
    exec distrobox-host-exec "\$KOKORO_HOME/bin/python" "\$KOKORO_HOME/kokoro_tts.py" "\$@"
  fi
fi
exec "\$KOKORO_HOME/bin/python" "\$KOKORO_HOME/kokoro_tts.py" "\$@"
EOF
chmod +x "$BIN/kokoro-tts"
# Record it for breeze-tts / video-maker / LocalTTS (see registry.py).
"$PY" "$HERE/registry.py" set kokoro python="$PY" home="$PREFIX" backend="$GPU" launcher="$BIN/kokoro-tts" \
  torch="$("$PY" -c 'import torch; print(torch.__version__)')"
case ":$PATH:" in *":$BIN:"*) ;; *) warn "$BIN is not on PATH — add it (fish: fish_add_path $BIN; bash/zsh: export PATH=\"$BIN:\$PATH\")" ;; esac

say "smoke test (first run downloads the model, ~330 MB, and compiles GPU kernels)"
"$BIN/kokoro-tts" --check
say "done. Try:  kokoro-tts \"Hello from Kokoro.\" -o hello.wav"
