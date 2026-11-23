import numpy as np
from PIL import Image
from visionlab.core import CONDITIONS, degrade, Detection, matched_counts, metrics_from_counts, iou, summarize

def test_degradations_keep_shape_and_non_mutating():
    original = Image.fromarray(np.full((60, 80, 3), 160, dtype=np.uint8))
    for c in CONDITIONS:
        result = degrade(original, c, severity=0.7)
        assert result.size == original.size
        assert result.mode == "RGB"
    assert np.asarray(original).mean() == 160

def test_iou_matching():
    box = (10, 10, 30, 30)
    p = Detection("person", 0.9, box)
    g = Detection("person", 1.0, box)
    assert iou(box, box) == 1
    assert matched_counts([p], [g]) == {"tp": 1, "fp": 0, "fn": 0}
    assert metrics_from_counts(1, 0, 0)["f1"] == 1.0
    assert matched_counts([Detection("car", 0.9, box)], [g]) == {"tp": 0, "fp": 1, "fn": 1}

def test_summary():
    assert summarize([])["number_detected"] == 0
