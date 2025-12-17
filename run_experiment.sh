#!/bin/bash
# Run a complete experiment: train + QA predictions + report generation
#
# Usage: ./run_experiment.sh [experiment_name]
#
# Environment variables for experiment configuration:
#   MODEL_NAME      - timm model name (default: efficientnet_b0)
#   BATCH_SIZE      - batch size (default: 32)
#   NUM_EPOCHS      - number of epochs (default: 10)
#   LEARNING_RATE   - learning rate (default: 1e-4)
#   IMAGE_SIZE      - input image size (default: 512)
#   LABEL_SMOOTHING - positive label smoothing (default: 0.0)
#   POS_WEIGHT      - positive class weight (default: auto)
#   NUM_WORKERS     - data loading workers (default: 4)

set -e  # Exit on error

# Source experiment config if exists
if [ -f "experiment.env" ]; then
    echo "Loading experiment config from experiment.env"
    source experiment.env
fi

# Get experiment name from argument or git branch
if [ -n "$1" ]; then
    EXP_NAME="$1"
else
    BRANCH=$(git rev-parse --abbrev-ref HEAD)
    DATE=$(date +%Y-%m-%d)
    EXP_NAME="${DATE}-${BRANCH}"
fi

# Default configuration (can be overridden by environment variables)
MODEL_NAME="${MODEL_NAME:-efficientnet_b0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"
POS_WEIGHT="${POS_WEIGHT:-}"  # Empty = auto-compute
NUM_WORKERS="${NUM_WORKERS:-4}"

# Directories
DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
EXP_DIR="experiments/${EXP_NAME}"

echo "=========================================="
echo "Running experiment: ${EXP_NAME}"
echo "=========================================="
echo "Configuration:"
echo "  MODEL_NAME:      ${MODEL_NAME}"
echo "  BATCH_SIZE:      ${BATCH_SIZE}"
echo "  NUM_EPOCHS:      ${NUM_EPOCHS}"
echo "  LEARNING_RATE:   ${LEARNING_RATE}"
echo "  IMAGE_SIZE:      ${IMAGE_SIZE}"
echo "  LABEL_SMOOTHING: ${LABEL_SMOOTHING}"
echo "  POS_WEIGHT:      ${POS_WEIGHT:-auto}"
echo "  NUM_WORKERS:     ${NUM_WORKERS}"
echo "=========================================="

# Create experiment directory
mkdir -p "${EXP_DIR}"

# Build training command
TRAIN_CMD="uv run python src/train.py \
    --data-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --model-name ${MODEL_NAME} \
    --batch-size ${BATCH_SIZE} \
    --num-epochs ${NUM_EPOCHS} \
    --learning-rate ${LEARNING_RATE} \
    --image-size ${IMAGE_SIZE} \
    --num-workers ${NUM_WORKERS}"

# Add optional parameters
if [ -n "${LABEL_SMOOTHING}" ] && [ "${LABEL_SMOOTHING}" != "0.0" ]; then
    TRAIN_CMD="${TRAIN_CMD} --label-smoothing ${LABEL_SMOOTHING}"
fi

if [ -n "${POS_WEIGHT}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --pos-weight ${POS_WEIGHT}"
fi

# Step 1: Train
echo ""
echo "[1/4] Training model..."
echo "Command: ${TRAIN_CMD}"
echo ""

START_TIME=$(date +%s)
eval ${TRAIN_CMD}
END_TIME=$(date +%s)
TRAIN_TIME=$((END_TIME - START_TIME))

echo ""
echo "Training completed in ${TRAIN_TIME} seconds"

# Find best checkpoint
BEST_CKPT=$(ls -t ${OUTPUT_DIR}/models/best-*.ckpt 2>/dev/null | head -1)
if [ -z "${BEST_CKPT}" ]; then
    echo "ERROR: No checkpoint found!"
    exit 1
fi
echo "Best checkpoint: ${BEST_CKPT}"

# Step 2: Run QA predictions
echo ""
echo "[2/4] Running QA predictions..."
QA_OUTPUT="${EXP_DIR}/qa_predictions.csv"

uv run python src/predict_qa.py \
    --data-dir ${DATA_DIR} \
    --checkpoint "${BEST_CKPT}" \
    --output "${QA_OUTPUT}" \
    --image-size ${IMAGE_SIZE}

# Step 3: Generate HTML report
echo ""
echo "[3/4] Generating HTML report..."

uv run python src/generate_report.py \
    --qa-predictions "${QA_OUTPUT}" \
    --data-dir ${DATA_DIR} \
    --output "${EXP_DIR}/report.html" \
    --experiment-name "${EXP_NAME}"

# Step 4: Save config
echo ""
echo "[4/4] Saving experiment config..."

cat > "${EXP_DIR}/config.json" << EOFCONFIG
{
    "experiment_name": "${EXP_NAME}",
    "timestamp": "$(date -Iseconds)",
    "git_branch": "$(git rev-parse --abbrev-ref HEAD)",
    "git_commit": "$(git rev-parse HEAD)",
    "hyperparameters": {
        "model_name": "${MODEL_NAME}",
        "batch_size": ${BATCH_SIZE},
        "num_epochs": ${NUM_EPOCHS},
        "learning_rate": ${LEARNING_RATE},
        "image_size": ${IMAGE_SIZE},
        "label_smoothing": ${LABEL_SMOOTHING},
        "pos_weight": ${POS_WEIGHT:-null},
        "num_workers": ${NUM_WORKERS}
    },
    "training_time_seconds": ${TRAIN_TIME},
    "checkpoint": "${BEST_CKPT}"
}
EOFCONFIG

# Copy checkpoint to experiment directory
cp "${BEST_CKPT}" "${EXP_DIR}/"
# Also copy last checkpoint if exists
LAST_CKPT="${OUTPUT_DIR}/models/last.ckpt"
if [ -f "${LAST_CKPT}" ]; then
    cp "${LAST_CKPT}" "${EXP_DIR}/"
fi

echo ""
echo "=========================================="
echo "Experiment complete!"
echo "Results saved to: ${EXP_DIR}/"
echo "  - config.json"
echo "  - qa_predictions.csv"
echo "  - report.html"
echo "  - *.ckpt"
echo "=========================================="
