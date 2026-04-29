#!/usr/bin/env bash
# Wait until unitree_rl_lab train.py exits, then run refresh_policy.sh.
# Usage:
#   nohup ~/ros2_ws/src/go2_rl_control/scripts/wait_train_then_refresh.sh >> /tmp/wait_train.log 2>&1 &
#
# Optional: after refresh, start ROS 2 demo (headless) for remote topic viewing:
#   RUN_LAUNCH=1 nohup ... &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVAL="${WAIT_INTERVAL_SEC:-120}"

echo "[$(date -Is)] watching for train.py ..."

while pgrep -f "scripts/rsl_rl/train.py" >/dev/null 2>&1; do
  echo "[$(date -Is)] still training, sleep ${INTERVAL}s"
  sleep "${INTERVAL}"
done

echo "[$(date -Is)] train.py finished, running refresh_policy.sh"
bash "${SCRIPT_DIR}/refresh_policy.sh"

if [[ "${RUN_LAUNCH:-0}" == "1" ]]; then
  WS_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
  # shellcheck disable=SC1091
  source "${WS_DIR}/install/setup.bash"
  echo "[$(date -Is)] starting sim.launch.py visualize:=false (set RUN_VISUALIZE=true for viewer)"
  VIS="${RUN_VISUALIZE:-false}"
  exec ros2 launch go2_rl_control sim.launch.py visualize:="${VIS}"
fi

echo "[$(date -Is)] done. Source workspace and run: ros2 launch go2_rl_control sim.launch.py"
