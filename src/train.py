"""
Training script for breast cancer detection model.
"""

import argparse
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from dataset import (
    MammogramDataset,
    create_qa_holdout,
    create_patient_split,
    get_train_transforms,
    get_val_transforms,
)
from model import MammogramLightningModule


def compute_pos_weight(df: pd.DataFrame) -> float:
    """Compute positive class weight for imbalanced data."""
    n_pos = df["cancer"].sum()
    n_neg = len(df) - n_pos
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"Class distribution: {n_neg} negative, {n_pos} positive")
    print(f"Using pos_weight: {pos_weight:.2f}")
    return pos_weight


def train(
    data_dir: Path,
    output_dir: Path,
    model_name: str = "efficientnet_b0",
    batch_size: int = 16,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    num_workers: int = 4,
    val_ratio: float = 0.2,
    image_size: int = 512,
    accelerator: str = "auto",
    devices: int = 1,
    label_smoothing: float = 0.0,
    pos_weight: float | None = None,
    qa_patients: int = 100,
    qa_positive: int = 4,
):
    """
    Train the mammogram classification model.

    Args:
        data_dir: Path to data directory with train_processed.csv
        output_dir: Path to save checkpoints
        model_name: Name of the timm model
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        num_workers: Number of data loading workers
        val_ratio: Fraction of data for validation
        image_size: Input image size
        accelerator: PyTorch Lightning accelerator
        devices: Number of devices (GPUs)
        label_smoothing: Smoothing for positive labels (0.2 means 1.0 -> 0.8)
        pos_weight: Weight for positive class (None = auto-compute from data)
        qa_patients: Number of patients for QA holdout (0 = skip QA holdout)
        qa_positive: Number of cancer-positive patients in QA holdout
    """
    # Load preprocessed metadata
    train_csv = data_dir / "train_processed.csv"
    if not train_csv.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found at {train_csv}. "
            "Run preprocess.py first."
        )

    df = pd.read_csv(train_csv)
    print(f"Loaded {len(df)} images from {train_csv}")

    # Create QA holdout if requested
    if qa_patients > 0:
        remaining_df, qa_df = create_qa_holdout(df, n_patients=qa_patients, n_positive=qa_positive)
        # Save QA holdout for manual review
        qa_csv = data_dir / "qa_holdout.csv"
        qa_df.to_csv(qa_csv, index=False)
        print(f"QA holdout saved to {qa_csv}")
    else:
        remaining_df = df
        print("Skipping QA holdout (qa_patients=0)")

    # Split remaining data by patient
    train_df, val_df = create_patient_split(remaining_df, val_ratio=val_ratio)

    # Compute class weight (auto or specified)
    if pos_weight is None:
        computed_pos_weight = compute_pos_weight(train_df)
    else:
        print(f"Using specified pos_weight: {pos_weight:.2f}")
        computed_pos_weight = pos_weight

    # Log label smoothing
    if label_smoothing > 0:
        print(f"Using label smoothing: {label_smoothing} (positive labels: 1.0 -> {1.0 - label_smoothing})")

    # Create datasets
    train_dataset = MammogramDataset(
        df=train_df,
        data_dir=data_dir,
        transform=get_train_transforms(image_size),
        is_test=False,
    )

    val_dataset = MammogramDataset(
        df=val_df,
        data_dir=data_dir,
        transform=get_val_transforms(image_size),
        is_test=False,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Create model
    model = MammogramLightningModule(
        model_name=model_name,
        pretrained=True,
        learning_rate=learning_rate,
        pos_weight=computed_pos_weight,
        num_epochs=num_epochs,
        label_smoothing=label_smoothing,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "models",
        filename="best-{epoch:02d}-{val_pf1:.4f}",
        monitor="val_pf1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_pf1",
        mode="max",
        patience=5,
        verbose=True,
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=10,
        precision="16-mixed",  # Use mixed precision for faster training
    )

    # Train
    print(f"\nStarting training for {num_epochs} epochs...")
    trainer.fit(model, train_loader, val_loader)

    print(f"\nBest model saved to: {checkpoint_callback.best_model_path}")
    print(f"Best val_pf1: {checkpoint_callback.best_model_score:.4f}")

    return checkpoint_callback.best_model_path


def main():
    parser = argparse.ArgumentParser(description="Train mammogram classifier")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Path to output directory",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="efficientnet_b0",
        help="Name of timm model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Input image size",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="PyTorch Lightning accelerator",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for positive class (0.2 means 1.0 -> 0.8)",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help="Weight for positive class (default: auto-compute from data)",
    )
    parser.add_argument(
        "--qa-patients",
        type=int,
        default=100,
        help="Number of patients for QA holdout (0 = skip QA holdout)",
    )
    parser.add_argument(
        "--qa-positive",
        type=int,
        default=4,
        help="Number of cancer-positive patients in QA holdout",
    )
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        image_size=args.image_size,
        accelerator=args.accelerator,
        devices=args.devices,
        label_smoothing=args.label_smoothing,
        pos_weight=args.pos_weight,
        qa_patients=args.qa_patients,
        qa_positive=args.qa_positive,
    )


if __name__ == "__main__":
    main()
