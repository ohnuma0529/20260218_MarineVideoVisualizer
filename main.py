import os
import sys
import argparse
import cv2
import time
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from core.detector import YoloDetector, GDinoDetector
from core.segmentor import YoloSegmentor, SamSegmentor
from core.classifier import FishClassifier
from core.visualizer import FrameVisualizer
from utils.config_loader import load_config

def main():
    parser = argparse.ArgumentParser(description="Marine Video Visualizer")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, help="Output filename (optional)")
    args = parser.parse_args()

    # 1. Load Config
    config = load_config(args.config)
    
    # 2. Initialize Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Detector
    det_cfg = config.get("detector", {})
    if det_cfg.get("backend") == "yolo":
        detector = YoloDetector(det_cfg["model_path"], conf=det_cfg.get("conf_thresh", 0.6))
    else:
        detector = GDinoDetector(det_cfg["gdino_config"], det_cfg["gdino_model"], device=device)

    # Segmentor
    seg_cfg = config.get("segmentor", {})
    if seg_cfg.get("backend") == "yoloseg":
        segmentor = YoloSegmentor(seg_cfg["model_path"], device=device)
    else:
        segmentor = SamSegmentor(seg_cfg["sam_model"])

    # Classifier
    cls_cfg = config.get("classifier", {})
    classifier = FishClassifier(cls_cfg["model_path"], target_list_path=cls_cfg.get("target_list_path"), device=device)

    # Visualizer
    vis_cfg = config.get("visualization", {})
    visualizer = FrameVisualizer(
        mask_alpha=vis_cfg.get("mask_alpha", 120),
        highlight_species=vis_cfg.get("highlight_species")
    )

    # 3. Process Video
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Improved Naming
    video_stem = Path(args.video).stem
    date_str = time.strftime("%Y%m%d")
    det_name = det_cfg.get("backend", "det")
    cls_name = Path(cls_cfg.get("model_path", "cls")).stem
    suffix = f"{det_name}_{cls_name}_{date_str}"
    
    if not args.output:
        output_video_name = f"{video_stem}_{suffix}.mp4"
    else:
        output_video_name = args.output

    out_cfg = config.get("output", {})
    out_dir = out_cfg.get("base_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    # Frames Output Directory
    frames_dir = os.path.join(out_dir, f"frames_{video_stem}_{suffix}")
    save_frames = out_cfg.get("save_frames", True)
    if save_frames:
        os.makedirs(frames_dir, exist_ok=True)

    out_path = os.path.join(out_dir, output_video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"Processing: {args.video}")
    print(f"Output Video: {out_path}")
    if save_frames:
        print(f"Output Frames: {frames_dir}")
    
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret: break
        
        # 3.1 Detection
        detections = detector.detect(frame)
        
        # 3.2 Segment & Classify
        results = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det['box'])
            crop_bgr = frame[y1:y2, x1:x2]
            if crop_bgr.size == 0: continue
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            
            # Segmentation
            mask_crop = segmentor.segment(crop_rgb, bbox_rel=[0, 0, x2-x1, y2-y1])
            if mask_crop is not None:
                full_mask = np.zeros((height, width), dtype=bool)
                full_mask[y1:y2, x1:x2] = mask_crop
                det['mask'] = full_mask
            
            # Classification
            label = classifier.classify(crop_rgb)
            det['label'] = label
            results.append(det)

        # 3.3 Visualization
        vis_frame = visualizer.draw_detections(frame, results)
        out_writer.write(vis_frame)
        
        # Save frame to image
        if save_frames:
            frame_path = os.path.join(frames_dir, f"frame_{i:06d}.jpg")
            cv2.imwrite(frame_path, vis_frame)

    cap.release()
    out_writer.release()
    print("\nProcessing Complete!")

if __name__ == "__main__":
    import torch # Delayed import inside __main__ to avoid issues if needed
    main()
