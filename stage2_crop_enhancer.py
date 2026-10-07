import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Any, Optional

from config import CROP_BUFFER_PCT, ENHANCE_UPSCALE_FACTOR, SHARPEN_IMAGE

class TargetCropEnhancer:
    def __init__(self, buffer_pct: float = CROP_BUFFER_PCT, upscale_factor: float = ENHANCE_UPSCALE_FACTOR):
        self.buffer_pct = buffer_pct
        self.upscale_factor = upscale_factor

    def extract_and_enhance_crop(
        self, 
        frame: np.ndarray, 
        detections: List[Dict[str, Any]]
    ) -> Tuple[Optional[Image.Image], Optional[np.ndarray], Dict[str, Any]]:
        """
        Crops target region of interest (ROI) with context padding buffer and person inclusion.
        Applies super-resolution upscaling and image sharpening.
        """
        if frame is None or not detections:
            return None, None, {}

        img_h, img_w = frame.shape[:2]

        # Get trigger boxes (weapons/guns) AND any person boxes in the scene to include full context
        trigger_boxes = [d["box"] for d in detections if d.get("is_trigger", False)]
        person_boxes = [d["box"] for d in detections if "person" in d.get("label", "").lower()]
        
        all_relevant_boxes = trigger_boxes + person_boxes if trigger_boxes else [d["box"] for d in detections]

        # Compute merged bounding box [xmin, ymin, xmax, ymax]
        xmins = [b[0] for b in all_relevant_boxes]
        ymins = [b[1] for b in all_relevant_boxes]
        xmaxs = [b[2] for b in all_relevant_boxes]
        ymaxs = [b[3] for b in all_relevant_boxes]

        xmin = min(xmins)
        ymin = min(ymins)
        xmax = max(xmaxs)
        ymax = max(ymaxs)

        bw = xmax - xmin
        bh = ymax - ymin

        # Add generous context buffer padding to ensure VLM sees person & scene context
        pad_x = int(bw * self.buffer_pct)
        pad_y = int(bh * self.buffer_pct)

        # Enforce minimum crop size (at least 40% of frame height & width)
        min_crop_w = int(img_w * 0.40)
        min_crop_h = int(img_h * 0.40)

        crop_xmin = max(0, xmin - pad_x)
        crop_ymin = max(0, ymin - pad_y)
        crop_xmax = min(img_w, xmax + pad_x)
        crop_ymax = min(img_h, ymax + pad_y)

        # Expand if smaller than minimum crop size
        curr_w = crop_xmax - crop_xmin
        curr_h = crop_ymax - crop_ymin

        if curr_w < min_crop_w:
            diff_w = (min_crop_w - curr_w) // 2
            crop_xmin = max(0, crop_xmin - diff_w)
            crop_xmax = min(img_w, crop_xmax + diff_w)

        if curr_h < min_crop_h:
            diff_h = (min_crop_h - curr_h) // 2
            crop_ymin = max(0, crop_ymin - diff_h)
            crop_ymax = min(img_h, crop_ymax + diff_h)

        # Extract raw patch
        raw_patch = frame[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        if raw_patch.size == 0:
            return None, None, {}

        # Super-Resolution / Upscaling
        target_w = int(raw_patch.shape[1] * self.upscale_factor)
        target_h = int(raw_patch.shape[0] * self.upscale_factor)
        
        # High quality Lanczos4 interpolation for sharpening low-res CCTV
        enhanced = cv2.resize(raw_patch, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        # Contrast & Sharpening
        if SHARPEN_IMAGE:
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
            enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b_ch))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Convert to PIL RGB for VLM/CLIP consumption
        rgb_patch = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(rgb_patch)

        crop_meta = {
            "crop_box": [crop_xmin, crop_ymin, crop_xmax, crop_ymax],
            "raw_size": (raw_patch.shape[1], raw_patch.shape[0]),
            "enhanced_size": (target_w, target_h),
            "upscale_factor": self.upscale_factor
        }

        return crop_pil, enhanced, crop_meta

if __name__ == "__main__":
    enhancer = TargetCropEnhancer()
    dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    dummy_dets = [{"box": [100, 100, 200, 200], "is_trigger": True}]
    pil_crop, bgr_crop, meta = enhancer.extract_and_enhance_crop(dummy_frame, dummy_dets)
    print(f"Crop Enhancer Test Passed: {meta}")
