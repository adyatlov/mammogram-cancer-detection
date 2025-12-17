# Breast Cancer Detection from Mammograms - Project Plan

## Goal
Build an ML model to detect breast cancer from screening mammograms using the RSNA Screening Mammography Breast Cancer Detection dataset.

---

## Dataset

- **Source**: Kaggle RSNA Screening Mammography Breast Cancer Detection
- **Size**: 270GB zip file
- **Images**: ~54,000 DICOM mammograms (3518×2800 pixels, 16-bit)
- **Patients**: ~11,913
- **Cancer rate**: ~2% (highly imbalanced)
- **Target**: Binary classification (cancer yes/no per breast)
- **Metric**: Probabilistic F1 score (pF1)

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.11 |
| Package manager | uv |
| Deep learning | PyTorch + PyTorch Lightning |
| Model | EfficientNet-B0 (via timm) |
| Image processing | pydicom, OpenCV, albumentations |
| Data | pandas, scikit-learn |
| Cloud | Nebius (CPU for preprocessing, H200 for training) |

---

## Project Structure

```
bc/
├── pyproject.toml           # uv dependencies
├── src/
│   ├── preprocess.py        # DICOM → 512x512 PNG (supports zip input)
│   ├── dataset.py           # PyTorch Dataset + augmentations
│   ├── model.py             # EfficientNet classifier + Lightning module
│   ├── train.py             # Training script
│   └── predict.py           # Inference + submission generation
├── tests/
│   └── create_mock_data.py  # Generate synthetic data for testing
├── data/                    # Local data (gitignored)
├── outputs/                 # Models and submissions
├── NEBIUS_SETUP.md          # Cloud deployment guide
└── PROJECT_PLAN.md          # This file
```

---

## Pipeline

### Step 1: Preprocessing
- Read DICOM directly from zip (no extraction needed)
- Apply VOI LUT windowing
- Handle MONOCHROME1/MONOCHROME2
- Resize to 512×512 with aspect ratio preservation
- Save as 8-bit PNG
- Output: ~5-10GB of PNGs + metadata CSV

### Step 2: Dataset
- PyTorch Dataset loading preprocessed PNGs
- Convert grayscale → RGB (for pretrained models)
- Train augmentations: flip, rotation, brightness/contrast
- Patient-level train/val split (prevent data leakage)

### Step 3: Model
- EfficientNet-B0 backbone (pretrained on ImageNet)
- Binary classifier head
- Weighted BCE loss (pos_weight = neg/pos ratio ≈ 50)
- AdamW optimizer + cosine annealing LR

### Step 4: Training
- Mixed precision (fp16/bf16)
- Early stopping on val_pf1
- Save best checkpoint

### Step 5: Inference
- Load best checkpoint
- Predict probabilities on test set
- Aggregate predictions per breast (prediction_id)
- Generate submission.csv

---

## Cloud Deployment (Nebius)

### Infrastructure
| Resource | Spec | Cost |
|----------|------|------|
| Shared filesystem | 300GB (or 50GB after deleting zip) | $4.50/month |
| CPU VM (preprocessing) | Intel Ice Lake, 8 vCPU, 32GB RAM | $0.05/hour |
| GPU VM (training) | H200, 80GB VRAM | $2.30/hour |

### Workflow
1. Upload zip to shared filesystem
2. Preprocess on cheap CPU VM (~4 hours, ~$0.20)
3. Train on H200 (~4 hours, ~$9.20)
4. Download results

**Total cost: ~$14**

---

## Commands

### Local Development
```bash
# Install dependencies
uv sync

# Create mock data for testing
uv run python tests/create_mock_data.py --data-dir data/mock

# Preprocess mock data
uv run python src/preprocess.py --data-dir data/mock --output-dir data/mock --split train

# Train on mock data
uv run python src/train.py --data-dir data/mock --output-dir outputs/mock --num-epochs 2

# Predict
uv run python src/predict.py --checkpoint outputs/mock/models/best-*.ckpt --data-dir data/mock --on-train
```

### Production (Nebius)
```bash
# Preprocess from zip
uv run python src/preprocess.py \
    --zip /mnt/shared/rsna-breast-cancer-detection.zip \
    --output-dir /mnt/shared

# Train on H200
uv run python src/train.py \
    --data-dir /mnt/shared \
    --output-dir /mnt/shared/outputs \
    --batch-size 64 \
    --num-epochs 20

# Generate submission
uv run python src/predict.py \
    --checkpoint /mnt/shared/outputs/models/best-*.ckpt \
    --data-dir /mnt/shared
```

---

## Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image size | 512×512 |
| Batch size | 64 (H200) / 16 (local) |
| Learning rate | 1e-4 |
| Epochs | 10-20 |
| Optimizer | AdamW |
| Scheduler | Cosine annealing |
| Loss | Weighted BCE (pos_weight ≈ 50) |
| Precision | 16-mixed |
| Early stopping | patience=5 on val_pf1 |

---

## Expected Results

- **Baseline pF1**: 0.1-0.2 (with class imbalance handling)
- **Good pF1**: 0.3-0.4 (with proper training)
- **Competition winners**: 0.5+ (with ensembles, larger models, TTA)

---

## Future Improvements

1. **Larger model**: EfficientNet-B4/B5, ConvNeXt, Vision Transformer
2. **Higher resolution**: 1024×1024 or 2048×2048
3. **Multi-view fusion**: Combine CC and MLO views
4. **Ensemble**: Multiple models with different seeds
5. **Test-time augmentation (TTA)**: Flip/rotate at inference
6. **External data**: Pretrain on other mammography datasets

---

## Status

- [x] Project structure
- [x] Dependencies (pyproject.toml)
- [x] Preprocessing script (with zip support)
- [x] Dataset class
- [x] Model (EfficientNet + Lightning)
- [x] Training script
- [x] Prediction script
- [x] Mock data testing
- [x] End-to-end pipeline verified
- [ ] Download real dataset (in progress - 270GB)
- [ ] Preprocess real dataset
- [ ] Train on real dataset
- [ ] Generate submission
