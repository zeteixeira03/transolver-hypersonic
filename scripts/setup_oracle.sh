#!/usr/bin/env bash
# Provision an Oracle Cloud Always Free A1 box (Ubuntu 22.04, aarch64, 4 cores)
# and launch the Phase 3 dataset generation pipeline.
#
# What it does, idempotently:
#   1. apt build deps
#   2. build SU2 v8 from source for aarch64, install to ~/SU2, add to PATH
#   3. create a Python venv with the generation-only deps (no torch)
#   4. launch scripts/phase3_generate.py under nohup, 1 worker per core
#
# Usage (from the cloned repo root, or anywhere -- it finds itself):
#   bash scripts/setup_oracle.sh                 # full setup + launch
#   bash scripts/setup_oracle.sh --no-launch     # set up only
#   bash scripts/setup_oracle.sh --systemd       # also install a user service that
#                                                # restarts the pipeline on reboot
#   HF_REPO=owner/name HF_TOKEN=hf_xxx bash scripts/setup_oracle.sh   # also push to HF
#
# The pipeline is resumable: if the box reboots, just re-run this script (or, with
# --systemd, it comes back on its own). Progress lives in the SQLite ledger.

set -euo pipefail

SU2_VERSION="${SU2_VERSION:-v8.0.1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$HOME/re-entry-venv"
SU2_PREFIX="$HOME/SU2"
SU2_SRC="$HOME/SU2-src"
WORKERS="${WORKERS:-$(nproc)}"
WORKDIR="$REPO/data/raw/phase3"

DO_LAUNCH=1
DO_SYSTEMD=0
for arg in "$@"; do
    case "$arg" in
        --no-launch) DO_LAUNCH=0 ;;
        --systemd)   DO_SYSTEMD=1 ;;
        *) echo "unknown arg: $arg" >&2; exit 1 ;;
    esac
done

banner() { echo; echo "==== $* ===="; echo; }

# ---------------------------------------------------------------------------
banner "1/4  apt build dependencies"
sudo apt-get update -y
sudo apt-get install -y build-essential git python3 python3-venv python3-pip ca-certificates

# ---------------------------------------------------------------------------
banner "2/4  SU2 $SU2_VERSION (aarch64 build)"
if [ -x "$SU2_PREFIX/bin/SU2_CFD" ]; then
    echo "SU2_CFD already present at $SU2_PREFIX/bin; skipping build"
else
    rm -rf "$SU2_SRC"
    git clone --depth 1 -b "$SU2_VERSION" https://github.com/su2code/SU2.git "$SU2_SRC"
    cd "$SU2_SRC"
    # bundled meson/ninja; MPI off (we run one rank per case), no python wrapper
    ./meson.py setup build --prefix="$SU2_PREFIX" -Dwith-mpi=disabled \
        -Denable-pywrapper=false --buildtype=release
    ./ninja -C build install
    cd "$REPO"
fi
# put SU2 on PATH for this shell and future logins
export SU2_RUN="$SU2_PREFIX/bin"
export PATH="$SU2_RUN:$PATH"
if ! grep -q "SU2_RUN=$SU2_PREFIX/bin" "$HOME/.bashrc" 2>/dev/null; then
    {
        echo ""
        echo "# SU2 (added by re-entry/scripts/setup_oracle.sh)"
        echo "export SU2_RUN=$SU2_PREFIX/bin"
        echo 'export PATH=$SU2_RUN:$PATH'
    } >> "$HOME/.bashrc"
fi
"$SU2_RUN/SU2_CFD" --help >/dev/null 2>&1 && echo "SU2_CFD OK: $("$SU2_RUN/SU2_CFD" --help 2>&1 | head -n1)" \
    || { echo "SU2_CFD build looks broken"; exit 1; }

# ---------------------------------------------------------------------------
banner "3/4  Python venv with generation deps"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install -q -U pip wheel
# generation-only: numpy/scipy/matplotlib/gmsh/pyvista(+vtk)/huggingface_hub -- no torch
"$VENV/bin/pip" install -q numpy scipy matplotlib gmsh pyvista huggingface-hub
"$VENV/bin/python" -c "import numpy, scipy, matplotlib, gmsh, pyvista; print('python deps OK')"

# ---------------------------------------------------------------------------
banner "4/4  launch the pipeline"
mkdir -p "$WORKDIR"
HF_ARGS=()
if [ -n "${HF_REPO:-}" ]; then
    HF_ARGS=(--hf-repo "$HF_REPO")
    echo "will push each tensor + the final manifest to HF dataset: $HF_REPO"
fi
CMD=("$VENV/bin/python" "$REPO/scripts/phase3_generate.py" --workdir "$WORKDIR" --workers "$WORKERS" "${HF_ARGS[@]}")

if [ "$DO_SYSTEMD" -eq 1 ]; then
    mkdir -p "$HOME/.config/systemd/user"
    cat > "$HOME/.config/systemd/user/phase3.service" <<EOF
[Unit]
Description=re-entry Phase 3 dataset generation
After=network.target

[Service]
Type=simple
WorkingDirectory=$REPO
Environment=PATH=$SU2_RUN:/usr/local/bin:/usr/bin:/bin
${HF_REPO:+Environment=HF_REPO=$HF_REPO}
${HF_TOKEN:+Environment=HF_TOKEN=$HF_TOKEN}
ExecStart=${CMD[*]}
Restart=on-failure
RestartSec=30

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable --now phase3.service
    loginctl enable-linger "$USER" >/dev/null 2>&1 || true
    echo "systemd user service 'phase3' enabled and started."
    echo "  journalctl --user -u phase3 -f      # follow logs"
    echo "  systemctl --user status phase3      # status"
elif [ "$DO_LAUNCH" -eq 1 ]; then
    nohup "${CMD[@]}" > "$WORKDIR/pipeline.log" 2>&1 &
    echo $! > "$WORKDIR/pipeline.pid"
    echo "pipeline launched, PID $(cat "$WORKDIR/pipeline.pid")"
    echo "  tail -f $WORKDIR/pipeline.log       # follow progress"
    echo "  python $REPO/scripts/phase3_generate.py --package-only --workdir $WORKDIR   # when done"
    echo
    echo "stage 2 (5-case diverse validation: blunt and sharp cones, low and high"
    echo "Mach, small nose) runs first at ~3 h per case sequentially -- ~15 h total"
    echo "before the 780-case sweep begins. Pass --no-validate to phase3_generate.py"
    echo "to skip stage 2 if the recipe has already been confirmed elsewhere."
else
    echo "setup complete; not launched. To start it:"
    echo "  cd $REPO && ${CMD[*]}"
fi
