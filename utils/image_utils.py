import cv2
import numpy as np

def postprocess_mask(mask_bool):
    """
    Remove small noise from boolean mask.
    """
    if mask_bool is None: return None
    mask_uint8 = (mask_bool.astype(np.uint8)) * 255
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened > 0

def expand_box(box, ratio, max_w, max_h):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx, cy = x1 + w/2, y1 + h/2
    nw, nh = w * ratio, h * ratio
    nx1 = max(0, cx - nw/2)
    ny1 = max(0, cy - nh/2)
    nx2 = min(max_w, cx + nw/2)
    ny2 = min(max_h, cy + nh/2)
    return [nx1, ny1, nx2, ny2]
