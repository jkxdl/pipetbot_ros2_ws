#!/usr/bin/env bash
set -eo pipefail

WORKSPACE_ROOT="/home/robot/pipetbot_ros2_ws"

set +u
source "/opt/ros/humble/setup.bash"
if [ ! -f "${WORKSPACE_ROOT}/install/setup.bash" ]; then
  echo "Workspace install/setup.bash not found. Please build the workspace first."
  exit 1
fi

source "${WORKSPACE_ROOT}/install/setup.bash"
set -u

exec ros2 launch pipettingrobot_gui pipetting_operator.launch.py
