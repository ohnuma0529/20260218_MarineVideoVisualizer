import torch
import numpy as np
import cv2
import os
import sys
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    ort = None

class FishClassifier:
    def __init__(self, model_path, target_list_path=None, device='cpu'):
        print(f"Initializing Classifier: {model_path}")
        self.device = device
        self.is_onnx = model_path.endswith('.onnx')
        self.session = None
        self.model = None
        
        if self.is_onnx:
            if ort is None:
                raise ImportError("onnxruntime is required for .onnx models.")
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device != 'cpu' else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
        else:
            # PyTorch fallback if needed (Requires orgmodels etc. to be in path)
            pass

        # Load class names
        self.class_names = []
        if target_list_path and os.path.exists(target_list_path):
            import pandas as pd
            df = pd.read_csv(target_list_path)
            # Assuming 'species_jp' column exists
            if 'species_jp' in df.columns:
                self.class_names = df['species_jp'].tolist()

    def classify(self, crop_rgb):
        """
        Classify a single crop (3, H, W) RGB tensor or ndarray.
        Returns label string.
        """
        # Preprocessing... (Needs to match training: resize, normalize)
        # Simplified for now assuming input is already correct shape or resized here
        input_size = (224, 224)
        img_resized = cv2.resize(crop_rgb, input_size)
        img_input = img_resized.astype(np.float32) / 255.0
        # Normalize (ImageNet)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_input = (img_input - mean) / std
        img_input = np.transpose(img_input, (2, 0, 1)) # HWC -> CHW
        img_input = np.expand_dims(img_input, axis=0) # CHW -> BCHW

        if self.is_onnx:
            inputs = {self.session.get_inputs()[0].name: img_input}
            logits = self.session.run(None, inputs)[0]
            idx = np.argmax(logits, axis=1)[0]
            if idx < len(self.class_names):
                return self.class_names[idx]
            return f"ID_{idx}"
        
        return "Unknown"
