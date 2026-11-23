from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

CONDITIONS = ["original", "low_light", "blur", "fog", "rain"]


def degrade(image: Image.Image, condition: str, severity: float = 0.5, seed: int = 13) -> Image.Image:
    """Controlled synthetic perturbations; not a substitute for real weather images.

    severity is between 0 and 1. Input and output images have equal dimensions.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition {condition}. Options: {CONDITIONS}")
    severity = float(np.clip(severity, 0.0, 1.0))
    a = np.asarray(image.convert("RGB")).copy()
    if condition == "original" or severity == 0:
        return Image.fromarray(a)
    if condition == "low_light":
        a = np.clip(a.astype(np.float32) * (1 - severity * 0.82), 0, 255).astype(np.uint8)
    elif condition == "blur":
        sigma = severity * 5.0
        a = cv2.GaussianBlur(a, (0, 0), sigmaX=max(0.001, sigma))
    elif condition == "fog":
        a = np.clip(a.astype(np.float32) * (1 - severity * 0.78) + 235 * severity * 0.78, 0, 255).astype(np.uint8)
        a = cv2.GaussianBlur(a, (0, 0), sigmaX=1.2 * severity + 0.001)
    elif condition == "rain":
        rng = np.random.default_rng(seed)
        h, w = a.shape[:2]
        overlay = a.copy()
        count = int(max(1, w * h * severity / 1600))
        for _ in range(count):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            length = int(8 + severity * 22)
            cv2.line(overlay, (x, y), (min(w - 1, x + 3), min(h - 1, y + length)), (190, 198, 205), 1)
        a = cv2.addWeighted(a, 0.8, overlay, 0.2, 0)
    return Image.fromarray(a)


@dataclass
class Detection:
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]


def load_detector(weights: str = "yolo11n.pt"):
    """Ultralytics downloads pretrained weights on first use."""
    from ultralytics import YOLO
    return YOLO(weights)


def detect(model, image: Image.Image, confidence: float = 0.25):
    result = model.predict(source=np.asarray(image.convert("RGB")), conf=confidence, verbose=False)[0]
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            cls = int(box.cls.item())
            detections.append(Detection(str(result.names[cls]), float(box.conf.item()),
                                        tuple(float(v) for v in box.xyxy[0].tolist())))
    # model.plot() is RGB array when pil=True and BGR array otherwise.
    
    plotted = result.plot(pil=True)

    if isinstance(plotted, Image.Image):
        annotated = plotted.convert("RGB")
    else:
        annotated = Image.fromarray(np.asarray(plotted))

    return detections, annotated

    


def summarize(detections: list[Detection]) -> dict:
    if not detections:
        return {"number_detected": 0, "mean_confidence": None, "categories": {}}
    categories = {}
    for d in detections:
        categories[d.label] = categories.get(d.label, 0) + 1
    return {"number_detected": len(detections), "mean_confidence": round(float(np.mean([d.confidence for d in detections])), 4), "categories": categories}


def iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union else 0.0


def matched_counts(pred: list[Detection], gold: list[Detection], iou_threshold: float = 0.5):
    """Greedy confidence-sorted class-aware one-to-one detection matching."""
    used = set()
    tp = 0
    for p in sorted(pred, key=lambda x: x.confidence, reverse=True):
        candidates = [(iou(p.xyxy, g.xyxy), j) for j, g in enumerate(gold)
                      if j not in used and g.label == p.label]
        if candidates:
            best, idx = max(candidates)
            if best >= iou_threshold:
                tp += 1
                used.add(idx)
    return {"tp": tp, "fp": len(pred) - tp, "fn": len(gold) - tp}


def metrics_from_counts(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def read_yolo_labels(label_file: Path, image_width: int, image_height: int, id_to_name: dict[int, str]):
    """Read YOLO normalized XYWH boxes and map class IDs to detector class names."""
    output = []
    if not label_file.exists():
        return output
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, xc, yc, bw, bh = map(float, parts[:5])
        x1, y1 = (xc - bw / 2) * image_width, (yc - bh / 2) * image_height
        x2, y2 = (xc + bw / 2) * image_width, (yc + bh / 2) * image_height
        label = id_to_name.get(int(cls))
        if label is not None:
            output.append(Detection(label, 1.0, (x1, y1, x2, y2)))
    return output
