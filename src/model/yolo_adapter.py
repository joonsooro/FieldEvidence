# src/model/yolo_adapter.py
from typing import List, Tuple
import numpy as np

Detection = Tuple[str, float, int, int, int, int]

class YoloAdapter:
    """
    나중에 ONNXRuntime/Ultralytics로 교체할 어댑터.
    지금은 스텁으로 두고, --detector yolo일 때 경고만 출력.
    """
    def __init__(self, **kwargs):
        self.warned = False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if not self.warned:
            print("[yolo_adapter] WARNING: YOLO adapter not wired yet. Using empty detections.")
            self.warned = True
        return []
