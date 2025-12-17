"""
Create mock DICOM data for testing the pipeline.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Try to use pydicom to create proper DICOM files
try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import generate_uid
    import datetime
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

# Also support PNG-only mock data
import cv2


def create_mock_dicom(path: Path, patient_id: int, image_id: int, has_cancer: bool = False):
    """Create a mock DICOM file with a simple synthetic mammogram."""

    # Create synthetic mammogram-like image
    img = np.zeros((2048, 1024), dtype=np.uint16)

    # Add breast-like shape (semi-ellipse)
    center_x, center_y = 512, 1024
    for y in range(2048):
        for x in range(1024):
            # Elliptical distance
            dx = (x - center_x) / 400
            dy = (y - center_y) / 800
            dist = dx**2 + dy**2
            if dist < 1 and x < center_x + 300:
                # Inside breast area
                intensity = int(20000 * (1 - dist * 0.5) + np.random.randint(0, 1000))
                img[y, x] = min(65535, intensity)

    # Add a "lesion" for cancer cases
    if has_cancer:
        lesion_x = np.random.randint(300, 600)
        lesion_y = np.random.randint(600, 1400)
        lesion_r = np.random.randint(20, 50)
        for y in range(max(0, lesion_y - lesion_r), min(2048, lesion_y + lesion_r)):
            for x in range(max(0, lesion_x - lesion_r), min(1024, lesion_x + lesion_r)):
                if (x - lesion_x)**2 + (y - lesion_y)**2 < lesion_r**2:
                    img[y, x] = min(65535, img[y, x] + 15000)

    # Downsample to make it faster
    img = cv2.resize(img, (256, 512), interpolation=cv2.INTER_AREA)

    if HAS_PYDICOM:
        # Create DICOM dataset
        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.DigitalMammographyXRayImageStorageForPresentation
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)

        # Patient info
        ds.PatientID = str(patient_id)
        ds.PatientName = f"Patient_{patient_id}"

        # Image info
        ds.SOPInstanceUID = generate_uid()
        ds.SOPClassUID = pydicom.uid.DigitalMammographyXRayImageStorageForPresentation
        ds.Modality = "MG"
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows, ds.Columns = img.shape
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.SamplesPerPixel = 1
        ds.PixelRepresentation = 0

        # Study/Series info
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()

        # Date/Time
        dt = datetime.datetime.now()
        ds.StudyDate = dt.strftime('%Y%m%d')
        ds.ContentDate = dt.strftime('%Y%m%d')

        # Pixel data
        ds.PixelData = img.tobytes()

        # Save
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_as(str(path))
    else:
        # Fallback: save as PNG instead
        path = path.with_suffix('.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        img_8bit = (img / 256).astype(np.uint8)
        cv2.imwrite(str(path), img_8bit)

    return path


def create_mock_dataset(
    data_dir: Path,
    n_patients: int = 10,
    images_per_patient: int = 4,
    cancer_rate: float = 0.2,
):
    """
    Create a mock dataset for testing.

    Args:
        data_dir: Where to create the mock data
        n_patients: Number of mock patients
        images_per_patient: Images per patient
        cancer_rate: Fraction of patients with cancer
    """
    data_dir = Path(data_dir)
    train_images_dir = data_dir / "train_images"

    records = []

    np.random.seed(42)

    for patient_id in range(10000, 10000 + n_patients):
        has_cancer = np.random.random() < cancer_rate

        for img_idx in range(images_per_patient):
            image_id = patient_id * 10 + img_idx
            laterality = "L" if img_idx < 2 else "R"
            view = "CC" if img_idx % 2 == 0 else "MLO"

            # Create DICOM file
            dicom_path = train_images_dir / str(patient_id) / f"{image_id}.dcm"
            create_mock_dicom(dicom_path, patient_id, image_id, has_cancer and laterality == "L")

            records.append({
                "site_id": 1,
                "patient_id": patient_id,
                "image_id": image_id,
                "laterality": laterality,
                "view": view,
                "age": np.random.randint(40, 80),
                "cancer": int(has_cancer and laterality == "L"),
                "biopsy": int(has_cancer and laterality == "L"),
                "invasive": int(has_cancer and laterality == "L" and np.random.random() > 0.5),
                "BIRADS": 0 if has_cancer else np.random.choice([1, 2]),
                "implant": 0,
                "density": np.random.choice(["A", "B", "C", "D"]),
                "machine_id": np.random.randint(1, 5),
                "difficult_negative_case": False,
            })

    # Create DataFrame and save
    df = pd.DataFrame(records)
    df.to_csv(data_dir / "train.csv", index=False)

    print(f"Created mock dataset:")
    print(f"  - {len(records)} images from {n_patients} patients")
    print(f"  - Cancer cases: {df['cancer'].sum()} ({100*df['cancer'].mean():.1f}%)")
    print(f"  - Saved to: {data_dir}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create mock data for testing")
    parser.add_argument("--data-dir", type=Path, default=Path("data/mock"))
    parser.add_argument("--n-patients", type=int, default=10)
    parser.add_argument("--images-per-patient", type=int, default=4)
    parser.add_argument("--cancer-rate", type=float, default=0.2)
    args = parser.parse_args()

    create_mock_dataset(
        data_dir=args.data_dir,
        n_patients=args.n_patients,
        images_per_patient=args.images_per_patient,
        cancer_rate=args.cancer_rate,
    )
