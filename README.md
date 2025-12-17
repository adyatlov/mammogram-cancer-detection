# Breast Cancer Detection from Mammograms

Machine learning pipeline for detecting breast cancer in mammography images using the RSNA Screening Mammography Breast Cancer Detection dataset.

## Project Overview

- **Task**: Binary classification (cancer vs. healthy)
- **Model**: EfficientNet-B0 (pretrained on ImageNet)
- **Framework**: PyTorch + PyTorch Lightning
- **Dataset**: RSNA Screening Mammography (~54,000 images, 12,000 patients)

## Results (Baseline Model)

| Metric | Value |
|--------|-------|
| Validation pF1 | 0.067 |
| QA AUC-ROC | 0.86 |
| QA Recall | 75% |
| Training Time | ~17 min (10 epochs on H200) |

## Project Structure

```
bc/
├── src/                    # Source code
│   ├── dataset.py          # PyTorch Dataset and transforms
│   ├── model.py            # EfficientNet model definition
│   ├── preprocess.py       # DICOM to PNG conversion
│   ├── train.py            # Training script
│   ├── predict.py          # Inference script
│   └── predict_qa.py       # QA holdout predictions
├── data/                   # Data directory (not in git)
│   ├── train.csv           # Training labels
│   ├── train_processed.csv # Preprocessed metadata
│   └── qa_holdout.csv      # QA holdout set
├── outputs/
│   ├── models/             # Trained model checkpoints
│   ├── qa_review/          # QA review HTML and images
│   └── qa_predictions.csv  # QA predictions
├── EXPERIMENT_LOG.md       # Detailed experiment journal
├── pyproject.toml          # Python dependencies
└── README.md               # This file
```

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- CUDA-compatible GPU (for training)

### Installation

```bash
# Clone repository
git clone <repo-url>
cd bc

# Install dependencies
uv sync

# Download dataset from Kaggle
kaggle competitions download -c rsna-breast-cancer-detection
unzip rsna-breast-cancer-detection.zip -d data/
```

### Data Preprocessing

Convert DICOM images to PNG (512x512):

```bash
uv run python src/preprocess.py \
    --data-dir data \
    --output-dir data \
    --split train \
    --num-workers 16
```

### Training

```bash
uv run python src/train.py \
    --data-dir data \
    --output-dir outputs \
    --batch-size 32 \
    --num-epochs 10
```

### Inference

```bash
# On test set
uv run python src/predict.py \
    --data-dir data \
    --checkpoint outputs/models/best-*.ckpt \
    --output-dir outputs/submissions

# On QA holdout
uv run python src/predict_qa.py \
    --data-dir data \
    --checkpoint outputs/models/best-*.ckpt \
    --output outputs/qa_predictions.csv
```

## Key Learnings

### DICOM Processing
- RSNA mammograms use JPEG Lossless compression
- Required: `pylibjpeg>=2.0` and `pylibjpeg-libjpeg>=2.1`
- OpenCV on headless Linux needs: `libgl1`, `libglib2.0-0`

### Performance
- Preprocessing is CPU-bound (JPEG Lossless decoding)
- Use `ripunzip` for 3x faster parallel extraction of large archives
- Copy data locally on GPU node to avoid network I/O bottleneck

## Potential Improvements

- [ ] Larger model (EfficientNet-B4/B5)
- [ ] More aggressive augmentation
- [ ] Multi-view fusion (combine predictions per patient)
- [ ] Grad-CAM visualization for interpretability
- [ ] Train longer with learning rate scheduling

## References

- [RSNA Competition](https://www.kaggle.com/competitions/rsna-breast-cancer-detection)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [timm Library](https://github.com/huggingface/pytorch-image-models)
