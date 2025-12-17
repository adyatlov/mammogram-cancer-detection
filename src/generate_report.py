"""
Generate HTML report for QA predictions.
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score


def generate_report(
    qa_predictions_path: Path,
    data_dir: Path,
    output_path: Path,
    experiment_name: str = "experiment",
    config: dict | None = None,
):
    """Generate HTML report for QA predictions."""
    # Load predictions
    df = pd.read_csv(qa_predictions_path)

    # Load train_processed.csv to get png paths
    train_csv = data_dir / "train_processed.csv"
    if train_csv.exists():
        train_df = pd.read_csv(train_csv)
        # Join to get png_path for each image
        path_map = train_df.set_index(["patient_id", "image_id"])["png_path"].to_dict()
        df["png_path"] = df.apply(
            lambda row: path_map.get((row["patient_id"], row["image_id"]), None), axis=1
        )
    else:
        # Fall back to constructing path from patient_id/image_id
        df["png_path"] = df.apply(
            lambda row: f"train_processed/{int(row['patient_id'])}/{int(row['image_id'])}.png", axis=1
        )

    # Compute metrics
    labels = df["actual_cancer"].values
    preds = df["predicted_prob"].values

    n_images = len(df)
    n_patients = df["patient_id"].nunique()
    n_cancer = int(df["actual_cancer"].sum())

    auc_roc = roc_auc_score(labels, preds) if labels.sum() > 0 else 0.0
    precision = precision_score(labels, preds > 0.5, zero_division=0)
    recall = recall_score(labels, preds > 0.5, zero_division=0)

    # Cancer cases
    cancer_df = df[df["actual_cancer"] == 1].copy()
    cancer_df = cancer_df.sort_values("predicted_prob", ascending=False)
    n_detected = int((cancer_df["predicted_prob"] > 0.5).sum())
    n_missed = n_cancer - n_detected

    # Top false positives (healthy but high prediction)
    fp_df = df[(df["actual_cancer"] == 0) & (df["predicted_prob"] > 0.5)].copy()
    fp_df = fp_df.sort_values("predicted_prob", ascending=False).head(8)

    # Output directory for images
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy images and build image cards
    cancer_cards = []
    for idx, row in enumerate(cancer_df.itertuples(), 1):
        patient_id = int(row.patient_id)
        image_id = int(row.image_id)
        prob = row.predicted_prob
        detected = prob > 0.5

        # Source image path (use png_path if available)
        png_path = getattr(row, 'png_path', None)
        if png_path:
            src_img = data_dir / png_path
        else:
            src_img = data_dir / "train_processed" / f"{patient_id}" / f"{image_id}.png"

        # Destination image name
        status = "DETECTED" if detected else "MISSED"
        dst_name = f"{idx:02d}_CANCER_{status}_{patient_id}_{image_id}_prob{prob:.3f}.png"
        dst_img = output_dir / dst_name

        # Copy image if exists
        if src_img.exists():
            shutil.copy(src_img, dst_img)

        # Build card HTML
        badge_class = "detected" if detected else "missed"
        border_style = "" if detected else ' style="border: 3px solid #dc3545;"'
        fill_class = "high" if prob >= 0.5 else ("medium" if prob >= 0.3 else "low")

        cancer_cards.append(f'''
        <div class="image-card"{border_style}>
            <img src="{dst_name}" alt="Cancer case">
            <div class="info">
                <h4>Patient {patient_id}</h4>
                <div class="meta">
                    <span class="badge cancer">CANCER</span>
                    <span class="badge {badge_class}">{status}</span>
                    <span>Prob: <strong>{prob:.3f}</strong></span>
                </div>
                <div class="prob-bar"><div class="fill {fill_class}" style="width: {prob*100:.1f}%"></div></div>
            </div>
        </div>
''')

    # False positive cards
    fp_cards = []
    for idx, row in enumerate(fp_df.itertuples(), len(cancer_df) + 1):
        patient_id = int(row.patient_id)
        image_id = int(row.image_id)
        prob = row.predicted_prob

        # Source image path (use png_path if available)
        png_path = getattr(row, 'png_path', None)
        if png_path:
            src_img = data_dir / png_path
        else:
            src_img = data_dir / "train_processed" / f"{patient_id}" / f"{image_id}.png"

        # Destination image name
        dst_name = f"{idx:02d}_FALSE_POS_{patient_id}_{image_id}_prob{prob:.3f}.png"
        dst_img = output_dir / dst_name

        # Copy image if exists
        if src_img.exists():
            shutil.copy(src_img, dst_img)

        fill_class = "high" if prob >= 0.5 else ("medium" if prob >= 0.3 else "low")

        fp_cards.append(f'''
        <div class="image-card">
            <img src="{dst_name}" alt="False Positive">
            <div class="info">
                <h4>Patient {patient_id}</h4>
                <div class="meta">
                    <span class="badge healthy">HEALTHY</span>
                    <span class="badge false-pos">FALSE POS</span>
                    <span>Prob: <strong>{prob:.3f}</strong></span>
                </div>
                <div class="prob-bar"><div class="fill {fill_class}" style="width: {prob*100:.1f}%"></div></div>
            </div>
        </div>
''')

    # Build config info
    config_html = ""
    if config:
        config_items = " | ".join([f"{k}: {v}" for k, v in config.items() if k != "experiment_name"])
        config_html = f"<p>{config_items}</p>"

    # Generate HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QA Report - {experiment_name}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .stat-card .value {{
            font-size: 36px;
            font-weight: bold;
            color: #333;
        }}
        .stat-card.good .value {{ color: #28a745; }}
        .stat-card.warning .value {{ color: #ffc107; }}
        .stat-card.danger .value {{ color: #dc3545; }}

        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .image-card {{
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .image-card img {{
            width: 100%;
            height: 300px;
            object-fit: contain;
            background: #000;
        }}
        .image-card .info {{
            padding: 15px;
        }}
        .image-card h4 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .image-card .meta {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }}
        .badge.cancer {{ background: #dc3545; color: white; }}
        .badge.healthy {{ background: #28a745; color: white; }}
        .badge.detected {{ background: #28a745; color: white; }}
        .badge.missed {{ background: #dc3545; color: white; }}
        .badge.false-pos {{ background: #ffc107; color: #333; }}

        .prob-bar {{
            width: 100%;
            height: 8px;
            background: #e9ecef;
            border-radius: 4px;
            margin-top: 10px;
            overflow: hidden;
        }}
        .prob-bar .fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s;
        }}
        .prob-bar .fill.high {{ background: #dc3545; }}
        .prob-bar .fill.medium {{ background: #ffc107; }}
        .prob-bar .fill.low {{ background: #28a745; }}

        .section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .legend {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <h1>QA Report - {experiment_name}</h1>
    {config_html}

    <div class="summary">
        <div class="stat-card">
            <h3>QA Images</h3>
            <div class="value">{n_images}</div>
        </div>
        <div class="stat-card">
            <h3>QA Patients</h3>
            <div class="value">{n_patients}</div>
        </div>
        <div class="stat-card {"good" if auc_roc >= 0.8 else "warning" if auc_roc >= 0.7 else "danger"}">
            <h3>AUC-ROC</h3>
            <div class="value">{auc_roc:.2f}</div>
        </div>
        <div class="stat-card {"good" if recall >= 0.7 else "warning" if recall >= 0.5 else "danger"}">
            <h3>Recall</h3>
            <div class="value">{recall*100:.0f}%</div>
        </div>
        <div class="stat-card {"good" if precision >= 0.3 else "warning" if precision >= 0.1 else "danger"}">
            <h3>Precision</h3>
            <div class="value">{precision*100:.1f}%</div>
        </div>
        <div class="stat-card">
            <h3>Cancer Cases</h3>
            <div class="value">{n_cancer}</div>
        </div>
    </div>

    <div class="legend">
        <div class="legend-item"><span class="badge cancer">CANCER</span> Actual cancer case</div>
        <div class="legend-item"><span class="badge detected">DETECTED</span> Correctly identified (prob > 0.5)</div>
        <div class="legend-item"><span class="badge missed">MISSED</span> Cancer not detected (prob < 0.5)</div>
        <div class="legend-item"><span class="badge false-pos">FALSE POS</span> Healthy but flagged as suspicious</div>
    </div>

    <h2>Cancer Cases ({n_cancer} images)</h2>
    <p>The model detected <strong>{n_detected} of {n_cancer}</strong> cancer images ({recall*100:.0f}% recall). {n_missed} images were missed.</p>

    <div class="image-grid">
        {"".join(cancer_cards)}
    </div>

    <h2>Top False Positives</h2>
    <p>Healthy cases flagged as highly suspicious (prob > 0.5).</p>

    <div class="image-grid">
        {"".join(fp_cards) if fp_cards else "<p>No false positives with prob > 0.5</p>"}
    </div>

    <footer>
        Generated on {datetime.now().strftime("%B %d, %Y %H:%M")} | Experiment: {experiment_name}
    </footer>
</body>
</html>
'''

    # Write HTML
    with open(output_path, "w") as f:
        f.write(html)

    print(f"Report generated: {output_path}")
    print(f"  - AUC-ROC: {auc_roc:.4f}")
    print(f"  - Recall: {recall*100:.1f}%")
    print(f"  - Precision: {precision*100:.1f}%")
    print(f"  - Cancer detected: {n_detected}/{n_cancer}")


def main():
    parser = argparse.ArgumentParser(description="Generate HTML QA report")
    parser.add_argument("--qa-predictions", type=Path, required=True, help="Path to qa_predictions.csv")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Path to data directory")
    parser.add_argument("--output", type=Path, required=True, help="Output HTML path")
    parser.add_argument("--experiment-name", type=str, default="experiment", help="Experiment name for report title")
    args = parser.parse_args()

    generate_report(
        qa_predictions_path=args.qa_predictions,
        data_dir=args.data_dir,
        output_path=args.output,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
