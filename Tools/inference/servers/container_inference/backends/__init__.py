from .base import InferenceBackend
from .detr_trt import DETRTensorRTBackend

__all__ = ["InferenceBackend", "DETRTensorRTBackend"]
