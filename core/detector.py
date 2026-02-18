import torch
import numpy as np
from ultralytics import YOLO
import sys
import os
from pathlib import Path

# GroundingDINO imports
try:
    from groundingdino.util.inference import load_model as gdino_load_model
    from groundingdino.util.inference import predict as gdino_predict
    import groundingdino.datasets.transforms as T
    from PIL import Image
except ImportError:
    pass

class BaseDetector:
    def detect(self, frame):
        raise NotImplementedError

class YoloDetector(BaseDetector):
    def __init__(self, model_path, conf=0.25):
        print(f"Initializing YOLO detector: {model_path}")
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        results = self.model.predict(source=frame, conf=self.conf, verbose=False)
        detections = []
        if len(results) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            for box, score in zip(boxes, scores):
                detections.append({
                    'box': box.tolist(),
                    'score': float(score),
                    'class': 'fish'
                })
        return detections

class GDinoDetector(BaseDetector):
    def __init__(self, config_path, model_path, device='cuda', caption="fish"):
        print(f"Initializing GroundingDINO: {model_path}")
        self.model = gdino_load_model(config_path, model_path, device=device)
        self.device = device
        self.caption = caption
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def detect(self, frame):
        # frame is BGR
        image_pil = Image.fromarray(frame[..., ::-1]) 
        image_transformed, _ = self.transform(image_pil, None)
        
        boxes, logits, phrases = gdino_predict(
            model=self.model,
            image=image_transformed,
            caption=self.caption,
            box_threshold=0.3,
            text_threshold=0.25,
            device=self.device
        )
        
        h, w = frame.shape[:2]
        detections = []
        for box, logit in zip(boxes, logits):
            # box is cx, cy, w, h normalized
            cx, cy, bw, bh = box.tolist()
            x1 = (cx - bw/2) * w
            y1 = (cy - bh/2) * h
            x2 = (cx + bw/2) * w
            y2 = (cy + bh/2) * h
            detections.append({
                'box': [x1, y1, x2, y2],
                'score': float(logit),
                'class': 'fish'
            })
        return detections
