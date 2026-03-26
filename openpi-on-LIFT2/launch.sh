#!/bin/bash
# Launch script for OpenPI LIFT2 client

# Get script directory and change to it
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Default values
HOST="192.168.101.101"
PORT=7777
PUBLISH_RATE=60  # 上采样后的控制频率
EXECUTE_HORIZON=19  # 执行全部上采样后的帧（10 帧 @ 30Hz → 19 帧 @ 60Hz = 317ms）
ACTION_CHUNK_SIZE=10  # 从预测中取 10 帧（模型预测的全部）
MAX_PUBLISH_STEP=1000  # 最大步数，0为无限模式
LANGUAGE_INSTRUCTION="perform task"
ENABLE_UPSAMPLE="--enable_upsample"  # 默认启用上采样
TARGET_HZ=60  # 目标控制频率
SOURCE_HZ=30  # 模型预测频率
VERBOSE=""
DEBUG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --language_instruction)
            LANGUAGE_INSTRUCTION="$2"
            shift 2
            ;;
        --infinite)
            MAX_PUBLISH_STEP=0
            shift
            ;;
        --no_upsample)
            ENABLE_UPSAMPLE=""
            PUBLISH_RATE=30
            EXECUTE_HORIZON=10
            shift
            ;;
        --action_chunk_size)
            ACTION_CHUNK_SIZE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--host IP] [--port PORT] [--language_instruction TEXT] [--infinite] [--no_upsample] [--action_chunk_size N] [--verbose] [--debug]"
            exit 1
            ;;
    esac
done

echo "=================================================="
echo "OpenPI LIFT2 Client Launcher"
echo "=================================================="
echo "Policy Server: $HOST:$PORT"
echo "Control Rate: $PUBLISH_RATE Hz"
echo "Execute Horizon: $EXECUTE_HORIZON frames"
echo "Max Steps: $([ "$MAX_PUBLISH_STEP" -le 0 ] && echo "Infinite" || echo "$MAX_PUBLISH_STEP")"
echo "Action Upsampling: $([ -n "$ENABLE_UPSAMPLE" ] && echo "Enabled (${SOURCE_HZ}Hz -> ${TARGET_HZ}Hz, chunk=$ACTION_CHUNK_SIZE)" || echo "Disabled")"
echo "Language Instruction: $LANGUAGE_INSTRUCTION"
if [ -n "$DEBUG" ]; then
    echo "Debug Mode: ON (press Enter each step)"
fi
echo "=================================================="
echo ""

# Check if policy server is reachable
echo "Checking policy server connectivity..."
if timeout 2 bash -c "echo > /dev/tcp/$HOST/$PORT" 2>/dev/null; then
    echo "✓ Policy server is reachable"
else
    echo "✗ Cannot reach policy server at $HOST:$PORT"
    echo "  Please ensure the policy server is running:"
    echo "  uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_lift2_lora --policy.dir=<checkpoint_dir>"
    exit 1
fi

# Source ROS workspace
source ~/Desktop/LIFT/R5/ROS/R5_ws/devel/setup.bash

echo ""
echo "Starting OpenPI client..."
echo ""

# Run the client
python3 deploy/client_lift2.py \
    --host "$HOST" \
    --port "$PORT" \
    --publish_rate "$PUBLISH_RATE" \
    --execute_horizon "$EXECUTE_HORIZON" \
    --max_publish_step "$MAX_PUBLISH_STEP" \
    --language_instruction "$LANGUAGE_INSTRUCTION" \
    $ENABLE_UPSAMPLE \
    --action_chunk_size "$ACTION_CHUNK_SIZE" \
    --target_hz "$TARGET_HZ" \
    --source_hz "$SOURCE_HZ" \
    $VERBOSE \
    $DEBUG
