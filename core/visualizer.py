import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

class FrameVisualizer:
    def __init__(self, mask_alpha=120, highlight_species=None):
        self.mask_alpha = mask_alpha
        self.highlight_species = highlight_species
        self.highlight_color = (255, 0, 0) # Red
        self.default_color = (0, 0, 255)   # Blue
        
        # Japanese Fonts
        self.font_paths = [
            "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
            "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        ]
        self.font = None

    def _get_font(self, size):
        for p in self.font_paths:
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, size)
                except: continue
        return ImageFont.load_default()

    def draw_detections(self, frame, detections):
        """
        frame: BGR numpy array
        detections: list of dicts { 'box': [x1, y1, x2, y2], 'label': str, 'mask': bool_array, 'score': float }
        """
        h, w = frame.shape[:2]
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGBA')
        overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(pil_img)
        overlay_draw = ImageDraw.Draw(overlay)
        
        font_size = max(12, int(h * 0.03))
        font = self._get_font(font_size)
        
        for det in detections:
            box = det['box']
            label = det.get('label', 'Unknown')
            mask = det.get('mask')
            score = det.get('score', 0.0)
            
            # Color Selection
            if self.highlight_species and label == self.highlight_species:
                color = self.highlight_color
            else:
                color = self.default_color
                
            x1, y1, x2, y2 = map(int, box)
            
            # Draw Mask
            if mask is not None:
                mask_color = (*color, self.mask_alpha)
                mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).convert('L')
                temp_mask = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
                temp_mask_draw = ImageDraw.Draw(temp_mask)
                temp_mask_draw.bitmap((0, 0), mask_pil, fill=mask_color)
                pil_img = Image.alpha_composite(pil_img, temp_mask)
                draw = ImageDraw.Draw(pil_img) # Reset draw object

            # Draw Bounding Box
            draw.rectangle([x1, y1, x2, y2], outline=tuple(color), width=3)
            
            # Draw Label
            display_text = f"{label} ({score:.2f})"
            tw, th = draw.textbbox((x1, y1), display_text, font=font)[2:]
            draw.rectangle([x1, y1 - (th - y1), tw, y1], fill=tuple(color))
            draw.text((x1, y1 - (th - y1)), display_text, font=font, fill=(255, 255, 255))

        return cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
