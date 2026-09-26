#!/usr/bin/env bash
# Install Breeze TTS 2 into its own environment and put a `breeze-tts` launcher on PATH.
# Normally run by tts/setup.sh, which inspects the machine and passes the right options;
# every choice below can also be given by hand.
#
#   --backend cuda|rocm|cpu      torch build family
#   --torch-index URL            wheel index for torch (none = PyPI)
#   --torch-spec "SPEC ..."      torch packages, e.g. "torch==2.9.1 torchaudio==2.9.1"
#   --device cuda|cpu            where Breeze runs (ROCm counts as cuda)
#   --dtype bf16|fp16|fp32       model precision (fp16 for NVIDIA older than sm_80)
#   --fast "a,b"                 fast (CUDA/HIP-graph) stages, empty for eager
#   --whisper MODEL              Whisper model for word timings
#   --prefix DIR                 environment (default ~/venvs/breeze-tts)
#   --src DIR                    Breeze source checkout (default ~/.local/share/tts-engines/breeze-tts)
#   --weights DIR                weights (default ~/.local/share/tts-engines/breeze-tts-2; reused if present)
#   --bin DIR                    launcher dir (default ~/.local/bin)
#   --accept-license             required: the weights and outputs are research / non-commercial only
#
# Re-running is safe: finished steps are skipped.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TTS="$(cd "$HERE/.." && pwd)"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}/tts-engines"

BACKEND="" TORCH_INDEX="" TORCH_SPEC="torch==2.9.1 torchaudio==2.9.1" DEVICE="cuda" DTYPE="bf16" FAST=""
WHISPER="openai/whisper-large-v3-turbo" PREFIX="$HOME/venvs/breeze-tts" SRC="$DATA/breeze-tts"
WEIGHTS="$DATA/breeze-tts-2" BIN="$HOME/.local/bin" ACCEPT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --backend) BACKEND="$2"; shift 2 ;;
    --torch-index) TORCH_INDEX="$2"; shift 2 ;;
    --torch-spec) TORCH_SPEC="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --fast) FAST="$2"; shift 2 ;;
    --whisper) WHISPER="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --src) SRC="$2"; shift 2 ;;
    --weights) WEIGHTS="$2"; shift 2 ;;
    --bin) BIN="$2"; shift 2 ;;
    --accept-license) ACCEPT=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\033[1m» %s\033[0m\n' "$*"; }
warn() { printf '\033[33m! %s\033[0m\n' "$*" >&2; }
die() { printf '\033[31m✖ %s\033[0m\n' "$*" >&2; exit 1; }

[ -n "$BACKEND" ] || die "--backend is required (run tts/setup.sh, which works it out for this machine)"
[ "$ACCEPT" = 1 ] || die "Breeze TTS 2 weights and outputs are licensed for research and non-commercial use only.
  Read https://huggingface.co/BreezeBlue/Breeze-TTS-2 and re-run with --accept-license."
command -v uv >/dev/null 2>&1 || die "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh"
command -v git >/dev/null 2>&1 || die "git not found"

# ---- 1. environment + torch ------------------------------------------------------------------
PY="$PREFIX/bin/python"
if [ ! -x "$PY" ]; then
  say "creating Python 3.12 environment at $PREFIX"
  uv venv "$PREFIX" --python 3.12
fi
export VIRTUAL_ENV="$PREFIX"

backend_ok() {
  "$PY" - "$BACKEND" <<'EOF' 2>/dev/null
import sys, torch
want = sys.argv[1]
ok = {"rocm": bool(torch.version.hip), "cuda": bool(torch.version.cuda) and not torch.version.hip, "cpu": True}[want]
sys.exit(0 if ok else 1)
EOF
}
read -r -a SPEC <<< "$TORCH_SPEC"
install_torch() {
  local idx=()
  [ -n "$TORCH_INDEX" ] && idx=(--pre --index-url "$TORCH_INDEX")
  uv pip install "$@" "${idx[@]}" "${SPEC[@]}"
}
if backend_ok; then
  say "torch already matches backend $BACKEND — keeping it"
else
  say "installing torch for $BACKEND (${TORCH_INDEX:-PyPI})"
  install_torch
fi

# ---- 2. Breeze source: upstream at the pinned commit + this skill's patch -------------------
COMMIT="$(cat "$HERE/UPSTREAM_COMMIT")"
if [ ! -d "$SRC/.git" ]; then
  say "fetching Breeze source → $SRC"
  mkdir -p "$(dirname "$SRC")"
  git clone --quiet https://github.com/breezeblue-ai/breeze-tts.git "$SRC"
fi
if [ "$(git -C "$SRC" rev-parse HEAD)" != "$COMMIT" ] && ! git -C "$SRC" merge-base --is-ancestor "$COMMIT" HEAD 2>/dev/null; then
  git -C "$SRC" fetch --quiet origin
  git -C "$SRC" checkout --quiet "$COMMIT"
fi
for p in "$HERE"/patches/*.patch; do
  if git -C "$SRC" apply --reverse --check "$p" 2>/dev/null; then continue; fi
  say "applying $(basename "$p")"
  git -C "$SRC" apply "$p" || die "patch $(basename "$p") does not apply to $SRC (local changes?)"
done

# ---- 3. Python packages (upstream's list, minus torch, plus what breeze-tts needs) ----------
say "installing Breeze's Python packages"
CONSTRAINTS="$(mktemp)"; trap 'rm -f "$CONSTRAINTS"' EXIT
"$PY" -c "import torch, torchaudio; print(f'torch=={torch.__version__}'); print(f'torchaudio=={torchaudio.__version__}')" > "$CONSTRAINTS"
uv pip install --constraint "$CONSTRAINTS" "qwen-tts==0.1.1" "transformers==4.57.3" "numpy>=2.0" "soundfile>=0.13" \
  librosa "huggingface_hub>=0.30" accelerate
if ! backend_ok; then
  warn "a dependency replaced the $BACKEND torch — reinstalling it"
  install_torch --reinstall
fi

# ---- 4. weights + Whisper --------------------------------------------------------------------
if [ -f "$WEIGHTS/config.json" ] && ls "$WEIGHTS"/*.safetensors >/dev/null 2>&1; then
  say "weights already at $WEIGHTS"
else
  say "downloading Breeze-TTS-2 weights (~7.2 GB) → $WEIGHTS"
  "$PY" -c "from huggingface_hub import snapshot_download as d; d('BreezeBlue/Breeze-TTS-2', local_dir='$WEIGHTS')"
fi
say "fetching Whisper ($WHISPER) for word timings"
"$PY" -c "from huggingface_hub import snapshot_download as d; d('$WHISPER', allow_patterns=['*.json','*.safetensors','*.txt','*.model'])" >/dev/null

# ---- 5. program, launcher, registry ----------------------------------------------------------
cp "$HERE/breeze_tts.py" "$HERE/align.py" "$PREFIX/"
mkdir -p "$BIN"
cat > "$BIN/breeze-tts" <<EOF
#!/bin/sh
# breeze-tts launcher (installed by video-maker/tts/breeze/install.sh). Environment: $PREFIX
BREEZE_HOME="\${BREEZE_HOME:-$PREFIX}"
if [ -z "\$BREEZE_NO_HOST" ] && { [ -e /run/.toolboxenv ] || [ -e /run/.containerenv ]; }; then
  if command -v flatpak-spawn >/dev/null 2>&1; then
    # flatpak-spawn does not pass the environment through; forward the BREEZE_* overrides.
    set -- \$(env | sed -n 's/^\\(BREEZE_[A-Z_]*=.*\\)/--env=\\1/p') "\$BREEZE_HOME/bin/python" "\$BREEZE_HOME/breeze_tts.py" "\$@"
    exec flatpak-spawn --host --directory="\$PWD" "\$@"
  elif command -v distrobox-host-exec >/dev/null 2>&1; then
    exec distrobox-host-exec "\$BREEZE_HOME/bin/python" "\$BREEZE_HOME/breeze_tts.py" "\$@"
  fi
fi
exec "\$BREEZE_HOME/bin/python" "\$BREEZE_HOME/breeze_tts.py" "\$@"
EOF
chmod +x "$BIN/breeze-tts"
case ":$PATH:" in *":$BIN:"*) ;; *) warn "$BIN is not on PATH — add it (fish: fish_add_path $BIN; bash/zsh: export PATH=\"$BIN:\$PATH\")" ;; esac

# Test with these exact settings before recording them, so a broken install is never registered.
say "smoke test (loads the model; the first run on a GPU also compiles kernels)"
BREEZE_REPO="$SRC" BREEZE_MODEL="$WEIGHTS" BREEZE_DEVICE="$DEVICE" BREEZE_DTYPE="$DTYPE" BREEZE_FAST="$FAST" \
  BREEZE_WHISPER="$WHISPER" BREEZE_HOME="$PREFIX" "$BIN/breeze-tts" --check \
  || die "the smoke test failed (output above). Nothing was recorded; fix and re-run."
TORCH_V="$("$PY" -c 'import torch; print(torch.__version__)')"
"$PY" "$TTS/registry.py" set breeze python="$PY" home="$PREFIX" repo="$SRC" weights="$WEIGHTS" backend="$BACKEND" \
  device="$DEVICE" dtype="$DTYPE" fast="$FAST" whisper="$WHISPER" torch="$TORCH_V" launcher="$BIN/breeze-tts" \
  license_accepted="$(date +%Y-%m-%dT%H:%M:%S%z)"
say "done. Try:  breeze-tts \"Hello from Breeze.\" --instruction \"a warm, calm narrator\" -o hello.wav"
