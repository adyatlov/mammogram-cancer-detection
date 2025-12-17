"""
PyTorch Dataset for breast cancer mammogram classification.
"""

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def get_train_transforms(image_size: int = 512) -> A.Compose:
    """Get augmentation transforms for training."""
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 512) -> A.Compose:
    """Get transforms for validation/test (no augmentation)."""
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class MammogramDataset(Dataset):
    """
    Dataset for loading preprocessed mammogram images.

    Args:
        df: DataFrame with columns: png_path, cancer (optional for test)
        data_dir: Base directory containing the processed images
        transform: Albumentations transform to apply
        is_test: If True, don't return labels
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_dir: Path,
        transform: A.Compose | None = None,
        is_test: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_test = is_test

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Load image
        img_path = self.data_dir / row["png_path"]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")

        # Convert to 3-channel (for pretrained models expecting RGB)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        result = {
            "image": img,
            "patient_id": row["patient_id"],
            "image_id": row["image_id"],
        }

        if not self.is_test and "cancer" in row:
            result["label"] = torch.tensor(row["cancer"], dtype=torch.float32)

        # Include prediction_id for test set aggregation
        if "prediction_id" in row:
            result["prediction_id"] = row["prediction_id"]

        return result


def create_qa_holdout(
    df: pd.DataFrame,
    n_patients: int = 100,
    n_positive: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a QA holdout set with stratified sampling.

    Args:
        df: DataFrame with patient_id and cancer columns
        n_patients: Number of patients for QA holdout
        n_positive: Number of cancer-positive patients in holdout
        random_state: Random seed for reproducibility

    Returns:
        remaining_df, qa_df
    """
    np.random.seed(random_state)

    # Get patient-level cancer labels (patient has cancer if any image is positive)
    patient_cancer = df.groupby("patient_id")["cancer"].max().reset_index()

    positive_patients = patient_cancer[patient_cancer["cancer"] == 1]["patient_id"].values
    negative_patients = patient_cancer[patient_cancer["cancer"] == 0]["patient_id"].values

    np.random.shuffle(positive_patients)
    np.random.shuffle(negative_patients)

    # Select stratified QA patients
    n_negative = n_patients - n_positive
    qa_positive = positive_patients[:n_positive]
    qa_negative = negative_patients[:n_negative]
    qa_patients = set(qa_positive) | set(qa_negative)

    # Split DataFrame
    qa_df = df[df["patient_id"].isin(qa_patients)].copy()
    remaining_df = df[~df["patient_id"].isin(qa_patients)].copy()

    qa_pos = qa_df["cancer"].sum()
    print(f"QA holdout: {len(qa_df)} images from {len(qa_patients)} patients")
    print(f"QA positive: {int(qa_pos)} images from {n_positive} patients")

    return remaining_df, qa_df


def create_patient_split(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by patient to prevent data leakage.

    Args:
        df: DataFrame with patient_id column
        val_ratio: Fraction of patients for validation
        random_state: Random seed for reproducibility

    Returns:
        train_df, val_df
    """
    # Get unique patients
    patients = df["patient_id"].unique()
    np.random.seed(random_state)
    np.random.shuffle(patients)

    # Split patients
    n_val = int(len(patients) * val_ratio)
    val_patients = set(patients[:n_val])
    train_patients = set(patients[n_val:])

    # Split DataFrame
    train_df = df[df["patient_id"].isin(train_patients)].copy()
    val_df = df[df["patient_id"].isin(val_patients)].copy()

    print(f"Train: {len(train_df)} images from {len(train_patients)} patients")
    print(f"Val: {len(val_df)} images from {len(val_patients)} patients")

    # Print class distribution
    if "cancer" in df.columns:
        train_pos = train_df["cancer"].sum()
        val_pos = val_df["cancer"].sum()
        print(f"Train positive: {train_pos} ({100*train_pos/len(train_df):.2f}%)")
        print(f"Val positive: {val_pos} ({100*val_pos/len(val_df):.2f}%)")

    return train_df, val_df
