#!/bin/bash
# Thin wrapper for the Python launcher

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

source ~/Desktop/LIFT/R5/ROS/R5_ws/devel/setup.bash
exec python3 deploy/client_lift2.py --profile default "$@"
