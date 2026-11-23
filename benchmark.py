"""Class-aware detection precision/recall/F1 at IoU 0.5 on annotated images.

This is a simple reproducible diagnostic, NOT COCO mAP. For proper mAP,
use Ultralytics model.val on a YOLO dataset and report mAP50 and mAP50-95.
"""
import argparse
import csv
from pathlib import Path
from PIL import Image
from visionlab.core import CONDITIONS, degrade, load_detector, detect, read_yolo_labels, matched_counts, metrics_from_counts

p = argparse.ArgumentParser()
p.add_argument("--images", required=True, help="Directory containing .jpg/.jpeg/.png files")
p.add_argument("--labels", required=True, help="Directory with corresponding YOLO .txt bounding-box labels")
p.add_argument("--weights", default="yolo11n.pt")
p.add_argument("--severity", type=float, default=0.6)
p.add_argument("--conf", type=float, default=0.25)
p.add_argument("--output", default="benchmark_results.csv")
p.add_argument("--classes", default="person,car,bicycle,motorcycle,bus,truck", help="COCO class names to evaluate")
a = p.parse_args()
model = load_detector(a.weights)
image_paths = sorted([x for x in Path(a.images).iterdir() if x.suffix.lower() in {".png", ".jpg", ".jpeg"}])
if not image_paths:
    raise SystemExit("No images found. Add labeled images before running benchmark.")
# IMPORTANT: Benchmark label files must use standard COCO class indices aligned with YOLO class names.
id_to_name = {int(i): n for i, n in model.names.items()}
allowed = set(x.strip() for x in a.classes.split(",") if x.strip())
rows = []
for condition in CONDITIONS:
    tp = fp = fn = 0
    for i, path in enumerate(image_paths):
        img = Image.open(path).convert("RGB")
        gt = read_yolo_labels(Path(a.labels) / f"{path.stem}.txt", img.width, img.height, id_to_name)
        gt = [g for g in gt if g.label in allowed]
        pred, _ = detect(model, degrade(img, condition, a.severity, seed=i), a.conf)
        pred = [d for d in pred if d.label in allowed]
        scores = matched_counts(pred, gt)
        tp += scores["tp"]; fp += scores["fp"]; fn += scores["fn"]
    row = {"condition": condition, "images": len(image_paths), "tp": tp, "fp": fp, "fn": fn,
           **metrics_from_counts(tp, fp, fn)}
    rows.append(row)
    print(row)
with Path(a.output).open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
print(f"Saved {a.output}")
