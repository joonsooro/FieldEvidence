# src/model/interface.py
from typing import List, Tuple, Literal
import numpy as np

Detection = Tuple[str, float, int, int, int, int]  # (cls, conf, x, y, w, h)
DetectorName = Literal["rule", "yolo"]

class Detector:
    def __init__(self, name: DetectorName = "rule", **kwargs):
        self.name = name
        if name == "rule":
            from .rule_based import RuleBased
            self.impl = RuleBased(**kwargs)
        elif name == "yolo":
            from .yolo_adapter import YoloAdapter
            self.impl = YoloAdapter(**kwargs)
        else:
            raise ValueError(f"unknown detector: {name}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Return list of detections on a single frame."""
        return self.impl.detect(frame)
