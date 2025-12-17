# Breast Cancer Detection - Nebius Deployment Plan

## Overview

Train an EfficientNet model on the RSNA Screening Mammography dataset using Nebius cloud infrastructure.

## Infrastructure

### Shared Filesystem: 300GB
- **Purpose**: Persistent storage accessible by all VMs
- **Contents**:
  - `rsna-breast-cancer-detection.zip` (270GB)
  - `train_processed/` (~5-10GB of 512x512 PNGs)
  - `test_processed/` (~2GB)
  - `outputs/models/` (~1GB checkpoints)
- **Cost**: ~$4.50/month

### Option A: Keep zip (300GB)
### Option B: Delete zip after preprocessing (50GB) - ~$0.75/month

---

## Step-by-Step Instructions

### 1. Create Nebius Shared Filesystem

1. Go to Nebius Console → Storage → Shared filesystems
2. Create new filesystem:
   - Name: `bc-data`
   - Size: 300GB (or 50GB if deleting zip)
   - Zone: Same as your VMs

### 2. Upload Dataset

**Option A: From local machine**
```bash
# Mount shared filesystem locally (if supported) or use a cheap VM
scp rsna-breast-cancer-detection.zip user@cpu-vm:/mnt/shared/
```

**Option B: From cheap CPU VM**
```bash
# SSH into CPU VM
ssh user@cpu-vm

# Mount shared filesystem
sudo mkdir -p /mnt/shared
sudo mount -t nfs <filesystem-ip>:/export /mnt/shared

# Download from Kaggle (requires kaggle.json credentials)
pip install kaggle
kaggle competitions download -c rsna-breast-cancer-detection -p /mnt/shared/
```

### 3. Preprocess on CPU VM ($0.05/hour)

**Create CPU VM:**
- Type: Intel Ice Lake
- vCPUs: 8
- RAM: 32GB
- Local disk: Not needed (using shared filesystem)

**Run preprocessing:**
```bash
# SSH into CPU VM
ssh user@cpu-vm

# Mount shared filesystem
sudo mkdir -p /mnt/shared
sudo mount -t nfs <filesystem-ip>:/export /mnt/shared

# Clone repo and install dependencies
git clone <your-repo> /home/user/bc
cd /home/user/bc
pip install uv
uv sync

# Preprocess directly from zip (no extraction needed!)
uv run python src/preprocess.py \
    --zip /mnt/shared/rsna-breast-cancer-detection.zip \
    --output-dir /mnt/shared \
    --split both

# Verify output
ls -lh /mnt/shared/train_processed/ | head
wc -l /mnt/shared/train_processed.csv

# Optional: Delete zip to save space
rm /mnt/shared/rsna-breast-cancer-detection.zip
```

**Expected output:**
- `train_processed/` - ~54,000 PNG files (~5-10GB)
- `train_processed.csv` - metadata with labels
- `test_processed/` - ~8,000 PNG files (~2GB)
- `test_processed.csv` - metadata

**Time estimate:** ~2-4 hours for 54,000 images

**Cost:** ~$0.20 (4 hours × $0.05/hour)

### 4. Train on H200 GPU ($2.30/hour)

**Create H200 VM:**
- Type: NVIDIA H200
- vCPUs: 16
- RAM: 200GB
- GPU Memory: 80GB

**Run training:**
```bash
# SSH into H200 VM
ssh user@h200-vm

# Mount shared filesystem
sudo mkdir -p /mnt/shared
sudo mount -t nfs <filesystem-ip>:/export /mnt/shared

# Clone repo and install dependencies
git clone <your-repo> /home/user/bc
cd /home/user/bc
pip install uv
uv sync

# Train model
uv run python src/train.py \
    --data-dir /mnt/shared \
    --output-dir /mnt/shared/outputs \
    --batch-size 64 \
    --num-epochs 20 \
    --num-workers 8

# Model saved to /mnt/shared/outputs/models/best-*.ckpt
```

**Training parameters for H200 (80GB VRAM):**
| Parameter | Value |
|-----------|-------|
| batch-size | 64-128 |
| num-workers | 8 |
| num-epochs | 10-20 |
| image-size | 512 (default) |

**Time estimate:** ~2-4 hours depending on epochs

**Cost:** ~$5-10 (2-4 hours × $2.30/hour)

### 5. Generate Predictions

```bash
# Still on H200 VM
uv run python src/predict.py \
    --checkpoint /mnt/shared/outputs/models/best-*.ckpt \
    --data-dir /mnt/shared \
    --output-dir /mnt/shared/outputs/submissions

# Download submission
scp user@h200-vm:/mnt/shared/outputs/submissions/submission.csv ./
```

### 6. Cleanup

```bash
# Delete H200 VM (expensive!)
# Keep shared filesystem if you want to iterate

# Optional: Delete CPU VM
# Shared filesystem persists independently
```

---

## Cost Summary

| Resource | Duration | Cost |
|----------|----------|------|
| Shared filesystem (300GB) | 1 month | ~$4.50 |
| CPU VM (preprocessing) | 4 hours | ~$0.20 |
| H200 VM (training) | 4 hours | ~$9.20 |
| **Total** | | **~$14** |

If you delete the zip and use 50GB filesystem: **~$10 total**

---

## File Structure on Shared Filesystem

```
/mnt/shared/
├── rsna-breast-cancer-detection.zip  (270GB, delete after preprocessing)
├── train_processed/                   (~5-10GB)
│   ├── 10006_100060.png
│   ├── 10006_100061.png
│   └── ... (54,000 files)
├── train_processed.csv
├── test_processed/                    (~2GB)
│   └── ... (8,000 files)
├── test_processed.csv
└── outputs/
    ├── models/
    │   ├── best-epoch=XX-val_pf1=X.XXXX.ckpt
    │   └── last.ckpt
    └── submissions/
        └── submission.csv
```

---

## Commands Quick Reference

```bash
# Preprocess from zip
uv run python src/preprocess.py --zip /mnt/shared/rsna-breast-cancer-detection.zip --output-dir /mnt/shared

# Train
uv run python src/train.py --data-dir /mnt/shared --output-dir /mnt/shared/outputs --batch-size 64

# Predict on test set
uv run python src/predict.py --checkpoint /mnt/shared/outputs/models/best-*.ckpt --data-dir /mnt/shared

# Predict on train set (for evaluation)
uv run python src/predict.py --checkpoint /mnt/shared/outputs/models/best-*.ckpt --data-dir /mnt/shared --on-train
```

---

## Troubleshooting

### "CUDA out of memory"
Reduce batch size: `--batch-size 32` or `--batch-size 16`

### Slow data loading
Increase workers: `--num-workers 16`

### Mount issues
```bash
# Check if mounted
df -h | grep shared

# Remount
sudo umount /mnt/shared
sudo mount -t nfs <filesystem-ip>:/export /mnt/shared
```
