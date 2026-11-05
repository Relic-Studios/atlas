#!/usr/bin/env bash
# ATLAS uninstaller for Linux: undoes install.sh.
#   bash uninstall.sh               asks before deleting your data
#   bash uninstall.sh --keep-data   keeps settings, voices, agents, memories and call recordings in this folder
#   bash uninstall.sh --remove-data deletes them too, then the whole folder
# Removes: a running ATLAS from this folder, the app menu entry, the private Python environment (.venv), the
# desktop shell's packages, logs and the app files. System packages, Python 3.12, Node.js and the downloaded
# speech models (shared Hugging Face cache) are left alone; other programs may use them.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA=(user private voices personas agent_state memory_db recordings)
MODE=""
for a in "$@"; do case "$a" in
  --keep-data) MODE=keep ;; --remove-data) MODE=remove ;; -y|--yes) ;;
  *) echo "unknown option: $a"; exit 2 ;; esac; done
case "$ROOT" in /|"$HOME") echo "refusing: $ROOT doesn't look like an ATLAS folder"; exit 1 ;; esac
[ -f "$ROOT/server.py" ] || { echo "refusing: no server.py in $ROOT"; exit 1; }
# Never the development folder (AGENTS.md and dev_pack are left out of every public build and install).
[ -e "$ROOT/AGENTS.md" ] || [ -e "$ROOT/dev_pack" ] && { echo "refusing: $ROOT is the ATLAS development folder"; exit 1; }

if [ -z "$MODE" ]; then
  read -r -p "Uninstall ATLAS from $ROOT? (y/n) " a; [[ "$a" =~ ^[yY] ]] || { echo "Nothing was changed."; exit 1; }
  read -r -p "Also delete your ATLAS data (settings, voices, agents, memories, call recordings)? (y/n) " a
  if [[ "$a" =~ ^[yY] ]]; then MODE=remove; else MODE=keep; fi
fi

# 1. Stop ATLAS if it's running from this folder.
for pid in $(pgrep -f "$ROOT/" 2>/dev/null); do
  [ "$pid" = "$$" ] && continue
  echo "  stopping $pid"; kill "$pid" 2>/dev/null
done
sleep 1

# 2. The app menu entry install.sh wrote (only if it points here).
ENTRY="$HOME/.local/share/applications/atlas.desktop"
if [ -f "$ENTRY" ] && grep -qF "Exec=$ROOT/atlas" "$ENTRY"; then rm -f "$ENTRY"; echo "  removed the app menu entry"; fi

# 3. What install.sh and running ATLAS created.
for rel in .venv desktop/node_modules logs; do
  [ -e "$ROOT/$rel" ] && { rm -rf "${ROOT:?}/$rel"; echo "  removed $rel"; }
done
find "$ROOT" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null
rm -f "$ROOT"/install*.log "$ROOT/server.log" "$ROOT/server.err.log" "$ROOT/boot.pid" "$ROOT/atlas"

# 4. Your data and the app files.
if [ "$MODE" = remove ]; then
  cd / && rm -rf "${ROOT:?}"
  printf '\nATLAS is uninstalled and %s is removed.\n' "$ROOT"
else
  for p in "$ROOT"/* "$ROOT"/.[!.]*; do
    [ -e "$p" ] || continue
    n="$(basename "$p")"
    [[ " ${DATA[*]} " == *" $n "* ]] && continue
    rm -rf "$p"
  done
  printf '\nATLAS is uninstalled. Your data is kept in %s (delete that folder to remove it).\n' "$ROOT"
fi
