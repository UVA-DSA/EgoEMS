from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Detection:
    label: str
    class_id: int
    score: float
    box_xyxy: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceResult:
    model_name: str
    frame_id: Optional[str]
    timestamp: Optional[str]
    image_width: int
    image_height: int
    inference_ms: float
    preprocess_ms: float
    postprocess_ms: float
    detections: List[Detection]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["detections"] = [d.to_dict() for d in self.detections]
        return data
