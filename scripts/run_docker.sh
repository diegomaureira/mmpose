#!/usr/bin/env bash
set -euo pipefail

# =============================
# Configuration
# =============================
DATA_DIR="${DATA_DIR:-/media/disk/dm_test}"
MMPPOSE_DIR="${MMPPOSE_DIR:-/media/disk/mmpose}"
IMAGE_NAME="${IMAGE_NAME:-mmpose}"
CONTAINER_NAME="${CONTAINER_NAME:-mmpose}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"

# =============================
# Pre-flight checks
# =============================
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

if [ ! -d "$MMPPOSE_DIR" ]; then
    echo "❌ ERROR: MMPose directory not found: $MMPPOSE_DIR"
    exit 1
fi

mkdir -p "$PIP_CACHE_DIR"

# =============================
# Startup info
# =============================
echo "🚀 Starting MMPose container..."
echo "   🧩 Image:       $IMAGE_NAME"
echo "   📦 Container:   $CONTAINER_NAME"
echo "   💻 Mount code:  $MMPPOSE_DIR → /mmpose"
echo "   📂 Mount data:  $DATA_DIR → /mmpose/data"
echo "   🧰 Pip cache:   $PIP_CACHE_DIR → /root/.cache/pip"
echo ""

# =============================
# Container entry command
# =============================
SETUP_CMD=$(cat <<'EOF'
set -e
cd /mmpose
echo "🔍 Installing MMPose..."
pip install -r requirements.txt
pip install -v -e .
echo ""
exec bash
EOF
)

# =============================
# Run container
# =============================
docker run --gpus all \
    --name "$CONTAINER_NAME" \
    --rm \
    --shm-size=8g \
    -it \
    -v "${MMPPOSE_DIR}:/mmpose" \
    -v "${DATA_DIR}:/mmpose/data" \
    -v "${PIP_CACHE_DIR}:/root/.cache/pip" \
    --network host \
    -e PYTHONPATH=/mmpose:${PYTHONPATH:-} \
    -w /mmpose \
    "$IMAGE_NAME" \
    bash -c "$SETUP_CMD"
