#!/usr/bin/env bash
# ATLAS installer for Linux (Ubuntu/Debian). Safe to re-run: finished steps are skipped.
#   ./install.sh               full install (desktop app + speech models)
#   ./install.sh --headless    no desktop shell; open http://localhost:8000 in a browser
#   ./install.sh --skip-models models download on first launch instead
# Needs sudo only for the apt packages in step 1. No accounts, no API keys.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG="$ROOT/install.log"
HEADLESS=0; SKIP_MODELS=0; FORCE_CPU=0
for a in "$@"; do case "$a" in
  --headless) HEADLESS=1 ;; --skip-models) SKIP_MODELS=1 ;; --cpu) FORCE_CPU=1 ;;
  *) echo "unknown option: $a"; exit 2 ;; esac; done
T=6
step() { printf '\n\033[36m[%s/%s] %s\033[0m\n' "$1" "$T" "$2"; echo "[$1/$T] $2" >>"$LOG"; }
ok()   { printf '  \033[32mok\033[0m  %s\n' "$1"; }
warn() { printf '  \033[33m!!\033[0m  %s\n' "$1"; echo "WARN $1" >>"$LOG"; }
fail() { printf '  \033[31mInstall stopped:\033[0m %s\n  Details: %s\n' "$1" "$LOG"; echo "FAIL $1" >>"$LOG"; exit 1; }
run()  { "$@" >>"$LOG" 2>&1 || fail "command failed: $*"; }
echo "ATLAS install $(date -Is)" >"$LOG"

step 1 "System packages (python3.12, ffmpeg, PortAudio, espeak-ng, build tools)"
need=()
command -v python3.12 >/dev/null || need+=(python3.12)
python3.12 -c "import venv, ensurepip" 2>/dev/null || need+=(python3.12-venv)
command -v ffmpeg >/dev/null || need+=(ffmpeg)
compgen -G "/usr/lib/*/libportaudio.so*" >/dev/null || compgen -G "/usr/lib/libportaudio.so*" >/dev/null || need+=(libportaudio2)
command -v espeak-ng >/dev/null || need+=(espeak-ng)
# PyAudio has no Linux wheel: it builds against the PortAudio headers.
[ -f /usr/include/portaudio.h ] || need+=(portaudio19-dev)
[ -f /usr/include/python3.12/Python.h ] || need+=(python3.12-dev)
command -v gcc >/dev/null || need+=(build-essential)
if [ $HEADLESS -eq 0 ] && ! command -v npm >/dev/null; then need+=(nodejs npm); fi
if [ ${#need[@]} -gt 0 ]; then
  echo "  installing: ${need[*]} (sudo)"
  if ! sudo -n true 2>/dev/null && ! sudo -v 2>/dev/null; then
    fail "these system packages are missing and this account can't use sudo. Ask an admin to run: sudo apt-get install -y ${need[*]}"
  fi
  run sudo apt-get update -qq
  run sudo apt-get install -y -qq "${need[@]}"
fi
ok "python: $(python3.12 --version)"

step 2 "Private Python environment (.venv)"
VPY="$ROOT/.venv/bin/python"
[ -x "$VPY" ] || run python3.12 -m venv "$ROOT/.venv"
run "$VPY" -m pip install --upgrade pip wheel --quiet
ok ".venv ready"

step 3 "PyTorch"
if "$VPY" -c "import torch" 2>/dev/null; then ok "already installed ($("$VPY" -c 'import torch;print(torch.__version__)'))"
elif [ "$FORCE_CPU" = 0 ] && command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  ok "NVIDIA GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  run "$VPY" -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
elif [ "$FORCE_CPU" = 1 ]; then
  warn "--cpu: test install only. The voice engine needs an NVIDIA GPU, so ATLAS will not be able to speak."
  run "$VPY" -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
else
  fail "No NVIDIA GPU found. ATLAS's voice engine needs an NVIDIA GPU (8 GB+ VRAM) with a CUDA 12 driver.
If you have one, install the NVIDIA driver (sudo ubuntu-drivers install), reboot, and run this again."
fi

step 4 "ATLAS packages (the long one)"
run "$VPY" -m pip install -r "$ROOT/requirements-public.txt"
run "$VPY" -m pip install --no-deps realtimestt==0.3.104
run "$VPY" -m pip install --no-deps openwakeword==0.6.0
run "$VPY" -c "import fastapi, faster_whisper, RealtimeTTS, RealtimeSTT, speechbrain; print('imports ok')"
ok "packages ready"

step 5 "Desktop shell and speech models"
if [ $HEADLESS -eq 1 ]; then ok "headless: skipping desktop shell"
else (cd "$ROOT/desktop" && run npm install --no-audit --no-fund --loglevel=error); ok "desktop shell ready"; fi
if [ $SKIP_MODELS -eq 1 ]; then warn "model download skipped; ATLAS fetches them on first launch"
elif "$VPY" "$ROOT/tools/prefetch_models.py" >>"$LOG" 2>&1; then ok "speech models downloaded"
else warn "some models will download on first launch instead"; fi
mkdir -p "$ROOT"/{user,voices,personas,logs}

step 6 "Launcher"
cat >"$ROOT/atlas" <<EOF
#!/usr/bin/env bash
cd "$ROOT"
if [ -x desktop/node_modules/.bin/electron ] && [ -n "\${DISPLAY:-}\${WAYLAND_DISPLAY:-}" ]; then
  ATLAS_PYTHON="$VPY" exec desktop/node_modules/.bin/electron desktop
fi
echo "ATLAS: open http://localhost:8000 in your browser (Ctrl+C to stop)"
exec "$VPY" -u server.py
EOF
chmod +x "$ROOT/atlas"
if [ $HEADLESS -eq 0 ] && [ -d "$HOME/.local/share/applications" -o -d "$HOME/.local/share" ]; then
  mkdir -p "$HOME/.local/share/applications"
  printf '[Desktop Entry]\nType=Application\nName=ATLAS\nExec=%s\nIcon=%s\nTerminal=false\nCategories=AudioVideo;\n' \
    "$ROOT/atlas" "$ROOT/static/favicon.ico" >"$HOME/.local/share/applications/atlas.desktop"
fi
ok "run: $ROOT/atlas"
printf '\n\033[32mATLAS is installed.\033[0m First launch opens the setup wizard.\n'
