import argparse
import json
from pathlib import Path
from PIL import Image
from visionlab.core import CONDITIONS, degrade, load_detector, detect, summarize
p = argparse.ArgumentParser()
p.add_argument("--image", required=True)
p.add_argument("--condition", choices=CONDITIONS, default="low_light")
p.add_argument("--severity", type=float, default=0.6)
p.add_argument("--weights", default="yolo11n.pt")
p.add_argument("--output", default="annotated.png")
p.add_argument("--no-detector", action="store_true", help="Only transform input; works without model download")
a = p.parse_args()
image = degrade(Image.open(a.image).convert("RGB"), a.condition, a.severity)
if a.no_detector:
    image.save(a.output)
    print(f"Saved perturbation to {a.output}")
else:
    detections, annotated = detect(load_detector(a.weights), image)
    annotated.save(a.output)
    print(json.dumps(summarize(detections), indent=2))
    print(f"Saved {a.output}")
