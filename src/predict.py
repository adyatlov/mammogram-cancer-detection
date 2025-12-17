"""
Inference script for breast cancer detection model.

Generates predictions on test set and creates submission file.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MammogramDataset, get_val_transforms
from model import MammogramLightningModule


def predict(
    data_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 512,
    device: str = "cuda",
):
    """
    Generate predictions on test set.

    Args:
        data_dir: Path to data directory with test_processed.csv
        checkpoint_path: Path to model checkpoint
        output_dir: Path to save submission file
        batch_size: Batch size for inference
        num_workers: Number of data loading workers
        image_size: Input image size
        device: Device to run inference on
    """
    # Check device availability
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Load preprocessed test metadata
    test_csv = data_dir / "test_processed.csv"
    if not test_csv.exists():
        raise FileNotFoundError(
            f"Preprocessed test data not found at {test_csv}. "
            "Run preprocess.py first."
        )

    df_test = pd.read_csv(test_csv)
    print(f"Loaded {len(df_test)} test images")

    # Create dataset
    test_dataset = MammogramDataset(
        df=df_test,
        data_dir=data_dir,
        transform=get_val_transforms(image_size),
        is_test=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Load model
    print(f"Loading model from {checkpoint_path}")
    model = MammogramLightningModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        weights_only=False,  # Required for PyTorch 2.6+
    )
    model = model.to(device)
    model.eval()

    # Generate predictions
    all_predictions = []
    all_prediction_ids = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch["image"].to(device)
            prediction_ids = batch.get("prediction_id", batch["patient_id"])

            logits = model(images)
            probs = torch.sigmoid(logits.squeeze(-1))

            all_predictions.extend(probs.cpu().numpy().tolist())
            all_prediction_ids.extend(prediction_ids.numpy().tolist())

    # Create predictions DataFrame
    df_preds = pd.DataFrame({
        "prediction_id": all_prediction_ids,
        "cancer": all_predictions,
    })

    # Aggregate predictions by prediction_id (average multiple images)
    df_submission = df_preds.groupby("prediction_id")["cancer"].mean().reset_index()

    # Save submission
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "submission.csv"
    df_submission.to_csv(submission_path, index=False)

    print(f"\nSubmission saved to {submission_path}")
    print(f"Total predictions: {len(df_submission)}")
    print(f"\nPrediction statistics:")
    print(df_submission["cancer"].describe())

    return df_submission


def predict_on_train(
    data_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 512,
    device: str = "cuda",
):
    """
    Generate predictions on training set for analysis.

    This is useful for evaluating the model on the full training data
    and analyzing prediction errors.
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Load preprocessed train metadata
    train_csv = data_dir / "train_processed.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Preprocessed data not found at {train_csv}")

    df_train = pd.read_csv(train_csv)
    print(f"Loaded {len(df_train)} training images")

    # Create dataset (without augmentation)
    train_dataset = MammogramDataset(
        df=df_train,
        data_dir=data_dir,
        transform=get_val_transforms(image_size),
        is_test=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Load model
    print(f"Loading model from {checkpoint_path}")
    model = MammogramLightningModule.load_from_checkpoint(
        checkpoint_path,
        map_location=device,
        weights_only=False,  # Required for PyTorch 2.6+
    )
    model = model.to(device)
    model.eval()

    # Generate predictions
    all_predictions = []
    all_labels = []
    all_patient_ids = []
    all_image_ids = []

    print("Generating predictions on training set...")
    with torch.no_grad():
        for batch in tqdm(train_loader):
            images = batch["image"].to(device)
            labels = batch["label"]
            patient_ids = batch["patient_id"]
            image_ids = batch["image_id"]

            logits = model(images)
            probs = torch.sigmoid(logits.squeeze(-1))

            all_predictions.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_patient_ids.extend(patient_ids.numpy().tolist())
            all_image_ids.extend(image_ids.numpy().tolist())

    # Create predictions DataFrame
    df_preds = pd.DataFrame({
        "patient_id": all_patient_ids,
        "image_id": all_image_ids,
        "cancer": all_labels,
        "prediction": all_predictions,
    })

    # Save predictions
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_path = output_dir / "train_predictions.csv"
    df_preds.to_csv(preds_path, index=False)

    # Compute metrics
    from sklearn.metrics import roc_auc_score, f1_score

    labels = df_preds["cancer"].values
    preds = df_preds["prediction"].values

    auc = roc_auc_score(labels, preds)
    f1 = f1_score(labels, preds > 0.5)

    print(f"\nTrain predictions saved to {preds_path}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"F1 Score (threshold=0.5): {f1:.4f}")

    return df_preds


def main():
    parser = argparse.ArgumentParser(description="Generate predictions")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to data directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/submissions"),
        help="Path to output directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Input image size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--on-train",
        action="store_true",
        help="Run predictions on training set instead of test",
    )
    args = parser.parse_args()

    if args.on_train:
        predict_on_train(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            device=args.device,
        )
    else:
        predict(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            device=args.device,
        )


if __name__ == "__main__":
    main()
