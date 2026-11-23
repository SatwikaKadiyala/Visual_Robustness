"""Optional detector fine-tuning on your annotated YOLO-format dataset.

Run only after you have valid train/val splits, labels and a dataset.yaml.
See Ultralytics documentation for dataset YAML requirements and licensing.
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="YOLO dataset.yaml with valid train and val paths")
p.add_argument("--weights", default="yolo11n.pt")
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--name", default="robustness_baseline")
a = p.parse_args()
if not Path(a.data).exists():
    raise SystemExit(f"Cannot find dataset YAML: {a.data}")
model = YOLO(a.weights)
model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
            project="training_runs", name=a.name)
print("Training completed. Compare the saved best.pt against baseline using a held-out test set.")
