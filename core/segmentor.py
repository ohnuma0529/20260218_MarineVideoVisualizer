import torch
import numpy as np
import cv2
from ultralytics import YOLO, SAM
from utils.image_utils import postprocess_mask

try:
    import onnxruntime as ort
except ImportError:
    ort = None

class BaseSegmentor:
    def segment(self, crop_rgb, bbox_rel=None):
        raise NotImplementedError

class YoloSegmentor(BaseSegmentor):
    def __init__(self, model_path, device=None):
        print(f"Initializing YOLO Segmentor: {model_path}")
        self.model = YOLO(model_path)

    def segment(self, crop_rgb, bbox_rel=None):
        # crop_rgb to BGR for YOLO
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        h, w = crop_bgr.shape[:2]
        
        results = self.model.predict(source=crop_bgr, conf=0.1, verbose=False)
        if len(results) > 0 and results[0].masks is not None:
            mask_data = results[0].masks.data[0].cpu().numpy()
            if mask_data.shape != (h, w):
                mask_data = cv2.resize(mask_data, (w, h))
            mask_bool = mask_data > 0.5
            return postprocess_mask(mask_bool)
        return None

class SamSegmentor(BaseSegmentor):
    def __init__(self, model_path):
        print(f"Initializing SAM 2: {model_path}")
        self.model = SAM(model_path)

    def segment(self, crop_rgb, bbox_rel=None):
        if bbox_rel is None:
            return None
        
        results = self.model.predict(source=crop_rgb, bboxes=[bbox_rel], verbose=False)
        if len(results) > 0 and results[0].masks is not None:
            mask_data = results[0].masks.data[0].cpu().numpy()
            mask_bool = mask_data > 0.5
            return postprocess_mask(mask_bool)
        return None
