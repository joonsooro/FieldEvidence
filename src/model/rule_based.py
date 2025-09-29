# src/model/rule_based.py
from typing import List, Tuple
import numpy as np

Detection = Tuple[str, float, int, int, int, int]

class RuleBased:
    """
    라벨 기반/센트로이드 기반에서 사용할 최소 인터페이스.
    지금은 T3/T4에서 frame 자체를 쓰지 않고, 이후 detect_launch에서
    CSV 라벨을 읽어 박스를 재구성할 때 호출 형태만 유지하도록 설계.
    """
    def __init__(self, **kwargs):
        pass

    def detect(self, frame: np.ndarray) -> List[Detection]:
        # 룰 기반 경량 경로에서는 프레임 당 박스 검출을 항상 하는 게 아니라,
        # CSV를 읽어오므로 여기서는 빈 리스트를 돌려도 무방.
        # (detect_launch.py에서 labels 모드로 박스가 주입됨)
        return []
