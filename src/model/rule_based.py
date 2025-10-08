# src/model/rule_based.py
from typing import List, Tuple
import numpy as np

Detection = Tuple[str, float, int, int, int, int]

class RuleBased:
    """
    Minimum interface to use in the Label base/Centroid base mode
    Currently do not use frames, read CSV label from detech_launch while only keeping calling format when re-forming bboxes.
    """
    def __init__(self, **kwargs):
        pass

    def detect(self, frame: np.ndarray) -> List[Detection]:
        # In the rule-based lightweight path we do not always detect boxes per frame;
        # CSV is read elsewhere, so returning an empty list here is fine.
        # (Boxes are injected by detect_launch.py in labels mode)
        return []
