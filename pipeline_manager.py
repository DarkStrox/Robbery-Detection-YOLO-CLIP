import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional

from stage1_yolo_scanner import YOLOEdgeScanner
from stage2_crop_enhancer import TargetCropEnhancer
from stage3_reasoning_engine import ReasoningEngine
from config import VLM_THREAT_KEYWORDS

class RobberyDetectionPipeline:
    """
    3-Stage Robbery Detection Pipeline with VLM False-Positive Verification:
    1. YOLO Stage: Continuous edge detection at 30+ FPS (Candidate Proposal)
    2. Crop & Enhancer Stage: ROI extraction + Super-Resolution (Context Crop)
    3. Reasoning Engine: Florence-2 VLM Auditor & CLIP Verification (False Alarm Suppression)
    """
    def __init__(self):
        print("[Pipeline Manager] Initializing 3-Stage Pipeline with VLM Auditor...")
        self.stage1_scanner = YOLOEdgeScanner()
        self.stage2_enhancer = TargetCropEnhancer()
        self.stage3_reasoner = ReasoningEngine()
        print("[Pipeline Manager] All 3 stages initialized successfully!")

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Processes a single video frame through the 3-stage pipeline.
        Suppresses false alarms if Florence-2 VLM does not confirm an actual weapon/threat.
        """
        if frame is None:
            return {"triggered": False, "verified_threat": False, "annotated_frame": None, "detections": []}

        # Stage 1: Fast Edge Scanning
        yolo_candidate, annotated_frame, detections = self.stage1_scanner.scan_frame(frame)

        if not yolo_candidate or not detections:
            return {
                "triggered": False,
                "verified_threat": False,
                "alert_status": "🟢 Normal Scene",
                "annotated_frame": annotated_frame,
                "detections": detections,
                "crop_pil": None,
                "crop_bgr": None,
                "reasoning_sentence": None,
                "top_category": None,
                "top_score": 0.0,
                "clip_scores": {}
            }

        # Stage 2: Target Crop & Super-Resolution Enhancement
        crop_pil, crop_bgr, crop_meta = self.stage2_enhancer.extract_and_enhance_crop(frame, detections)

        # Stage 3: Reasoning Engine (CLIP + Florence-2 VLM Verification)
        reasoning_results = {}
        if crop_pil is not None:
            reasoning_results = self.stage3_reasoner.analyze_threat_crop(crop_pil)

        reasoning_sentence = reasoning_results.get("reasoning_sentence", "").lower()
        top_cat = reasoning_results.get("top_category", "")
        top_score = reasoning_results.get("top_score", 0.0)

        # Stage 3 VLM Verification Guard
        # Check if Florence-2 sentence or CLIP confirms an actual weapon/threat
        has_vlm_keyword = any(kw in reasoning_sentence for kw in VLM_THREAT_KEYWORDS)
        has_high_clip_threat = ("robbery" in top_cat.lower() or "weapon" in top_cat.lower() or "crime" in top_cat.lower()) and top_score > 55.0

        verified_threat = has_vlm_keyword or has_high_clip_threat

        if verified_threat:
            alert_status = "🚨 REAL THREAT CONFIRMED (YOLO + Florence-2 VLM Verifier)"
        else:
            alert_status = "🟢 FALSE ALARM SUPPRESSED (YOLO candidate rejected by Florence-2 VLM)"

        return {
            "triggered": True,
            "verified_threat": verified_threat,
            "alert_status": alert_status,
            "annotated_frame": annotated_frame,
            "detections": detections,
            "crop_pil": crop_pil,
            "crop_bgr": crop_bgr,
            "crop_meta": crop_meta,
            "top_category": reasoning_results.get("top_category", "Potential Candidate"),
            "top_score": top_score,
            "clip_scores": reasoning_results.get("clip_scores", {}),
            "reasoning_sentence": reasoning_results.get("reasoning_sentence", "Scene analyzed by VLM."),
            "vlm_latency_ms": reasoning_results.get("vlm_latency_ms", 0.0)
        }

if __name__ == "__main__":
    print("Testing Robbery Detection Pipeline...")
    pipeline = RobberyDetectionPipeline()
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    res = pipeline.process_frame(dummy)
    print(f"Pipeline Test Finished: Triggered={res['triggered']}, Verified={res['verified_threat']}")
