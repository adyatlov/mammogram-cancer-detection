"""
Preprocess DICOM mammograms to PNG images.

This script converts DICOM files to PNG images with proper
windowing and normalization for training.

By default, saves at native resolution to preserve detail.
Resize to target size during training via transforms.

Supports reading directly from zip files to avoid extraction.
"""

import argparse
import io
import zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm


def read_dicom_bytes(data: bytes) -> np.ndarray:
    """Read DICOM from bytes and return the pixel array with proper windowing."""
    dicom = pydicom.dcmread(io.BytesIO(data))

    # Apply VOI LUT (Value of Interest Look Up Table) for proper windowing
    img = apply_voi_lut(dicom.pixel_array, dicom)

    # Handle photometric interpretation
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        img = img.max() - img

    # Normalize to 0-255 range
    img = img - img.min()
    if img.max() != 0:
        img = img / img.max()
    img = (img * 255).astype(np.uint8)

    return img


def read_dicom(path: Path) -> np.ndarray:
    """Read a DICOM file and return the pixel array with proper windowing."""
    dicom = pydicom.dcmread(path)

    # Apply VOI LUT (Value of Interest Look Up Table) for proper windowing
    img = apply_voi_lut(dicom.pixel_array, dicom)

    # Handle photometric interpretation
    # MONOCHROME1: white = low values, black = high values (inverted)
    # MONOCHROME2: white = high values, black = low values (normal)
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        img = img.max() - img

    # Normalize to 0-255 range
    img = img - img.min()
    if img.max() != 0:
        img = img / img.max()
    img = (img * 255).astype(np.uint8)

    return img


def resize_and_pad(img: np.ndarray, target_size: int | None = None) -> np.ndarray:
    """Resize image maintaining aspect ratio and pad to square.

    Args:
        img: Input image
        target_size: Target size (square). If None or 0, return original image.
    """
    if target_size is None or target_size == 0:
        return img

    h, w = img.shape[:2]

    # Calculate scale to fit within target_size
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to make square
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2

    img_padded = np.zeros((target_size, target_size), dtype=np.uint8)
    img_padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img_resized

    return img_padded


def process_single_image(args: tuple) -> dict | None:
    """Process a single DICOM image. Returns metadata dict or None on error."""
    dicom_path, output_dir, target_size = args

    try:
        # Read and process DICOM
        img = read_dicom(dicom_path)
        img = resize_and_pad(img, target_size)

        # Create output path: data/processed/{patient_id}_{image_id}.png
        patient_id = dicom_path.parent.name
        image_id = dicom_path.stem
        output_path = output_dir / f"{patient_id}_{image_id}.png"

        # Save as PNG
        cv2.imwrite(str(output_path), img)

        return {
            "patient_id": int(patient_id),
            "image_id": int(image_id),
            "png_path": str(output_path.relative_to(output_dir.parent)),
        }
    except Exception as e:
        print(f"Error processing {dicom_path}: {e}")
        return None


def preprocess_dataset(
    data_dir: Path,
    output_dir: Path,
    split: str = "train",
    target_size: int = 512,
    num_workers: int = 8,
) -> pd.DataFrame:
    """
    Preprocess all DICOM files in a dataset split.

    Args:
        data_dir: Path to data directory containing train_images/test_images
        output_dir: Path to output directory for processed images
        split: 'train' or 'test'
        target_size: Target image size (square)
        num_workers: Number of parallel workers

    Returns:
        DataFrame with processed image metadata
    """
    input_dir = data_dir / f"{split}_images"
    output_dir = output_dir / f"{split}_processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all DICOM files
    dicom_files = list(input_dir.glob("*/*.dcm"))
    print(f"Found {len(dicom_files)} DICOM files in {split} set")

    if not dicom_files:
        print(f"No DICOM files found in {input_dir}")
        return pd.DataFrame()

    # Prepare arguments for parallel processing
    args_list = [(f, output_dir, target_size) for f in dicom_files]

    # Process in parallel with progress bar
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_image, args): args for args in args_list}

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {split}"):
            result = future.result()
            if result is not None:
                results.append(result)

    # Create DataFrame with processed metadata
    df_processed = pd.DataFrame(results)

    # Merge with original metadata
    csv_path = data_dir / f"{split}.csv"
    if csv_path.exists():
        df_meta = pd.read_csv(csv_path)
        df_processed = df_processed.merge(df_meta, on=["patient_id", "image_id"], how="left")

    return df_processed


def preprocess_from_zip(
    zip_path: Path,
    output_dir: Path,
    csv_path: Path | None = None,
    split: str = "train",
    target_size: int = 512,
) -> pd.DataFrame:
    """
    Preprocess DICOM files directly from a zip archive without extracting.

    Args:
        zip_path: Path to the zip file
        output_dir: Path to output directory for processed images
        csv_path: Path to metadata CSV (if outside zip)
        split: 'train' or 'test'
        target_size: Target image size (square)

    Returns:
        DataFrame with processed image metadata
    """
    output_images_dir = output_dir / f"{split}_processed"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    results = []
    prefix = f"{split}_images/"

    print(f"Opening zip file: {zip_path}")

    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Find all DICOM files in the zip
        dicom_files = [f for f in zf.namelist() if f.startswith(prefix) and f.endswith('.dcm')]
        print(f"Found {len(dicom_files)} DICOM files in {split} set")

        if not dicom_files:
            print(f"No DICOM files found with prefix {prefix}")
            return pd.DataFrame()

        # Process each file
        for zip_name in tqdm(dicom_files, desc=f"Processing {split}"):
            try:
                # Extract patient_id and image_id from path like "train_images/12345/67890.dcm"
                parts = zip_name.split('/')
                if len(parts) < 3:
                    continue
                patient_id = parts[1]
                image_id = parts[2].replace('.dcm', '')

                # Read DICOM directly from zip
                dicom_bytes = zf.read(zip_name)
                img = read_dicom_bytes(dicom_bytes)
                img = resize_and_pad(img, target_size)

                # Save as PNG
                output_path = output_images_dir / f"{patient_id}_{image_id}.png"
                cv2.imwrite(str(output_path), img)

                results.append({
                    "patient_id": int(patient_id),
                    "image_id": int(image_id),
                    "png_path": str(output_path.relative_to(output_dir)),
                })
            except Exception as e:
                print(f"Error processing {zip_name}: {e}")
                continue

    # Create DataFrame with processed metadata
    df_processed = pd.DataFrame(results)

    # Merge with original metadata
    if csv_path and csv_path.exists():
        df_meta = pd.read_csv(csv_path)
        df_processed = df_processed.merge(df_meta, on=["patient_id", "image_id"], how="left")
    else:
        # Try to find CSV in the zip
        csv_name = f"{split}.csv"
        with zipfile.ZipFile(zip_path, 'r') as zf:
            if csv_name in zf.namelist():
                csv_bytes = zf.read(csv_name)
                df_meta = pd.read_csv(io.BytesIO(csv_bytes))
                df_processed = df_processed.merge(df_meta, on=["patient_id", "image_id"], how="left")

    return df_processed


def main():
    parser = argparse.ArgumentParser(description="Preprocess DICOM mammograms to PNG")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to data directory (used when not using --zip)",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=None,
        help="Path to zip file (reads directly without extracting)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Path to output directory",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=0,
        help="Target image size (square). 0 = native resolution (recommended)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers (only for non-zip mode)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "both"],
        default="both",
        help="Which split to process",
    )
    args = parser.parse_args()

    splits = ["train", "test"] if args.split == "both" else [args.split]

    for split in splits:
        print(f"\n{'='*50}")
        print(f"Processing {split} set")
        print(f"{'='*50}")

        if args.zip:
            # Process directly from zip file
            df = preprocess_from_zip(
                zip_path=args.zip,
                output_dir=args.output_dir,
                split=split,
                target_size=args.target_size,
            )
        else:
            # Process from extracted directory
            df = preprocess_dataset(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                split=split,
                target_size=args.target_size,
                num_workers=args.num_workers,
            )

        if not df.empty:
            # Save processed metadata
            output_csv = args.output_dir / f"{split}_processed.csv"
            df.to_csv(output_csv, index=False)
            print(f"Saved metadata to {output_csv}")
            print(f"Processed {len(df)} images")


if __name__ == "__main__":
    main()
