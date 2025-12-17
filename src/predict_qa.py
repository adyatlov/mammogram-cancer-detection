"""
Predict on QA holdout set for manual review.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MammogramDataset, get_val_transforms
from model import MammogramLightningModule


def predict_qa(
    data_dir: Path,
    checkpoint_path: Path,
    output_path: Path,
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 512,
    device: str = "cuda",
):
    """Generate predictions on QA holdout set."""
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Load QA holdout
    qa_csv = data_dir / "qa_holdout.csv"
    if not qa_csv.exists():
        raise FileNotFoundError(f"QA holdout not found at {qa_csv}")

    df_qa = pd.read_csv(qa_csv)
    print(f"Loaded {len(df_qa)} QA images from {df_qa['patient_id'].nunique()} patients")
    print(f"Cancer positive images: {df_qa['cancer'].sum()}")

    # Create dataset
    qa_dataset = MammogramDataset(
        df=df_qa,
        data_dir=data_dir,
        transform=get_val_transforms(image_size),
        is_test=False,
    )

    qa_loader = DataLoader(
        qa_dataset,
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
        weights_only=False,
    )
    model = model.to(device)
    model.eval()

    # Generate predictions
    all_predictions = []
    all_labels = []
    all_patient_ids = []
    all_image_ids = []

    print("Generating predictions on QA holdout...")
    with torch.no_grad():
        for batch in tqdm(qa_loader):
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
        "actual_cancer": [int(x) for x in all_labels],
        "predicted_prob": all_predictions,
    })

    # Add predicted label
    df_preds["predicted_cancer"] = (df_preds["predicted_prob"] > 0.5).astype(int)

    # Sort by predicted probability descending (most suspicious first)
    df_preds = df_preds.sort_values("predicted_prob", ascending=False)

    # Save predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_preds.to_csv(output_path, index=False)

    # Print summary
    print(f"\nPredictions saved to {output_path}")
    print(f"\n{'='*60}")
    print("QA HOLDOUT SUMMARY")
    print(f"{'='*60}")
    print(f"Total images: {len(df_preds)}")
    print(f"Actual cancer: {df_preds['actual_cancer'].sum()}")
    print(f"Predicted cancer (p>0.5): {df_preds['predicted_cancer'].sum()}")

    # Top 20 most suspicious
    print(f"\n{'='*60}")
    print("TOP 20 MOST SUSPICIOUS (sorted by predicted probability)")
    print(f"{'='*60}")
    top20 = df_preds.head(20)
    for i, row in top20.iterrows():
        actual = "CANCER" if row["actual_cancer"] else "healthy"
        match = "✓" if row["actual_cancer"] == row["predicted_cancer"] else "✗"
        print(f"  Patient {int(row['patient_id']):5d} | Image {int(row['image_id']):10d} | "
              f"Prob: {row['predicted_prob']:.3f} | Actual: {actual:7s} | {match}")

    # Cancer cases detection
    print(f"\n{'='*60}")
    print("ACTUAL CANCER CASES (model's predictions)")
    print(f"{'='*60}")
    cancer_cases = df_preds[df_preds["actual_cancer"] == 1]
    for i, row in cancer_cases.iterrows():
        detected = "DETECTED" if row["predicted_prob"] > 0.5 else "MISSED"
        print(f"  Patient {int(row['patient_id']):5d} | Image {int(row['image_id']):10d} | "
              f"Prob: {row['predicted_prob']:.3f} | {detected}")

    # Metrics
    from sklearn.metrics import roc_auc_score, precision_score, recall_score

    labels = df_preds["actual_cancer"].values
    preds = df_preds["predicted_prob"].values

    if labels.sum() > 0:
        auc = roc_auc_score(labels, preds)
        precision = precision_score(labels, preds > 0.5, zero_division=0)
        recall = recall_score(labels, preds > 0.5, zero_division=0)

        print(f"\n{'='*60}")
        print("METRICS")
        print(f"{'='*60}")
        print(f"  AUC-ROC: {auc:.4f}")
        print(f"  Precision (p>0.5): {precision:.4f}")
        print(f"  Recall (p>0.5): {recall:.4f}")

    return df_preds


def main():
    parser = argparse.ArgumentParser(description="Predict on QA holdout")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/qa_predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    predict_qa(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
