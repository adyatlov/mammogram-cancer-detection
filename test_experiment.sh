#!/bin/bash
# Quick test of experiment code with mock data
# Usage: ./test_experiment.sh

set -e

echo "=========================================="
echo "Testing experiment code with mock data"
echo "=========================================="

# Source experiment config if exists
if [ -f "experiment.env" ]; then
    echo "Loading experiment config from experiment.env"
    source experiment.env
fi

# Override for test
DATA_DIR="${DATA_DIR:-data/mock}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/test}"
NUM_EPOCHS=1
BATCH_SIZE=4
NUM_WORKERS=0

echo ""
echo "Configuration:"
echo "  DATA_DIR: ${DATA_DIR}"
echo "  MODEL_NAME: ${MODEL_NAME:-efficientnet_b0}"
echo "  LABEL_SMOOTHING: ${LABEL_SMOOTHING:-0.0}"
echo "  POS_WEIGHT: ${POS_WEIGHT:-auto}"
echo ""

# Clean up old test outputs
rm -rf "${OUTPUT_DIR}"

# Build training command
TRAIN_CMD="uv run python src/train.py \
    --data-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --model-name ${MODEL_NAME:-efficientnet_b0} \
    --batch-size ${BATCH_SIZE} \
    --num-epochs ${NUM_EPOCHS} \
    --learning-rate ${LEARNING_RATE:-1e-4} \
    --image-size ${IMAGE_SIZE:-512} \
    --num-workers ${NUM_WORKERS} \
    --accelerator cpu \
    --qa-patients 2 \
    --qa-positive 0"

# Add optional parameters
if [ -n "${LABEL_SMOOTHING}" ] && [ "${LABEL_SMOOTHING}" != "0.0" ]; then
    TRAIN_CMD="${TRAIN_CMD} --label-smoothing ${LABEL_SMOOTHING}"
fi

if [ -n "${POS_WEIGHT}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --pos-weight ${POS_WEIGHT}"
fi

echo "[1/3] Testing training..."
echo "Command: ${TRAIN_CMD}"
echo ""

eval ${TRAIN_CMD}

echo ""
echo "[2/3] Testing QA predictions..."

# Find checkpoint
BEST_CKPT=$(ls -t ${OUTPUT_DIR}/models/best-*.ckpt 2>/dev/null | head -1)
if [ -z "${BEST_CKPT}" ]; then
    BEST_CKPT=$(ls -t ${OUTPUT_DIR}/models/last.ckpt 2>/dev/null | head -1)
fi

if [ -z "${BEST_CKPT}" ]; then
    echo "ERROR: No checkpoint found!"
    exit 1
fi
echo "Using checkpoint: ${BEST_CKPT}"

QA_OUTPUT="${OUTPUT_DIR}/qa_predictions.csv"

uv run python src/predict_qa.py \
    --data-dir ${DATA_DIR} \
    --checkpoint "${BEST_CKPT}" \
    --output "${QA_OUTPUT}" \
    --image-size ${IMAGE_SIZE:-512} \
    --device cpu

echo ""
echo "[3/3] Testing report generation..."

uv run python src/generate_report.py \
    --qa-predictions "${QA_OUTPUT}" \
    --data-dir ${DATA_DIR} \
    --output "${OUTPUT_DIR}/report.html" \
    --experiment-name "test"

echo ""
echo "=========================================="
echo "All tests passed!"
echo "=========================================="
echo "Test outputs:"
ls -la ${OUTPUT_DIR}/
