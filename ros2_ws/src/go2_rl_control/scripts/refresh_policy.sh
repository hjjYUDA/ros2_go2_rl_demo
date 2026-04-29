#!/usr/bin/env bash
# Refresh the deployed policy from the latest checkpoint of an
# unitree_rl_lab training run. Produces ./policy/policy.onnx and ./policy/deploy.yaml,
# then rebuilds the ROS 2 package.
#
# Usage:
#   refresh_policy.sh                                    # auto-pick latest run + ckpt
#   refresh_policy.sh /path/to/model_5000.pt             # specific checkpoint
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
WS_DIR="$(cd "$PKG_DIR"/../.. && pwd)"
RL_LAB_DIR="${RL_LAB_DIR:-$HOME/unitree_rl_lab}"
EXPORTER="$RL_LAB_DIR/scripts/rsl_rl/export_onnx_standalone.py"
PY="${PY:-$HOME/miniconda3/envs/env_isaaclab/bin/python}"

if [[ -n "${1:-}" ]]; then
    CKPT="$1"
else
    LATEST_RUN=$(ls -td "$RL_LAB_DIR"/logs/rsl_rl/unitree_go2_velocity_robust/*/ 2>/dev/null | head -1)
    if [[ -z "$LATEST_RUN" ]]; then
        echo "[err] no run found under $RL_LAB_DIR/logs/rsl_rl/unitree_go2_velocity_robust/" >&2
        exit 1
    fi
    CKPT=$(ls -t "$LATEST_RUN"/model_*.pt 2>/dev/null | head -1)
fi

if [[ ! -f "$CKPT" ]]; then
    echo "[err] checkpoint not found: $CKPT" >&2
    exit 1
fi

echo "[info] checkpoint = $CKPT"
"$PY" "$EXPORTER" --checkpoint "$CKPT"

CKPT_DIR=$(dirname "$CKPT")
cp "$CKPT_DIR/exported/policy.onnx" "$PKG_DIR/policy/policy.onnx"
cp "$CKPT_DIR/params/deploy.yaml"   "$PKG_DIR/policy/deploy.yaml"
echo "[ok] refreshed $PKG_DIR/policy/{policy.onnx,deploy.yaml}"

(cd "$WS_DIR" && colcon build --symlink-install --packages-select go2_rl_control)
echo "[ok] colcon build completed; remember to 'source $WS_DIR/install/setup.bash'"
