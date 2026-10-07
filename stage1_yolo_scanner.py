import os
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Tuple, List, Dict, Any

from config import YOLO_GUNS_WEIGHTS, TRIGGER_CLASSES, YOLO_CONF_THRESHOLD

class YOLOEdgeScanner:
    def __init__(self, model_path: str = YOLO_GUNS_WEIGHTS):
        self.model_path = model_path
        print(f"[Stage 1] Loading YOLOv8 Edge Model from: {model_path}")
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            print(f"[Stage 1 WARNING] Custom weights file not found at {model_path}. Falling back to standard yolov8n.pt")
            self.model = YOLO("yolov8n.pt")
            
    def scan_frame(self, frame: np.ndarray, conf_threshold: float = YOLO_CONF_THRESHOLD) -> Tuple[bool, np.ndarray, List[Dict[str, Any]]]:
        """
        Runs fast YOLO detection on a video frame.
        Returns:
            triggered (bool): True if threat target object detected above threshold
            annotated_frame (np.ndarray): Frame with bounding box visual overlays
            detections (list): Structured detection dictionary items
        """
        if frame is None:
            return False, frame, []

        results = self.model(frame, verbose=False)
        annotated_frame = frame.copy()
        detections = []
        triggered = False

        for result in results:
            classes = result.names
            if result.boxes is None or len(result.boxes) == 0:
                continue

            cls_ids = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            xyxy_boxes = result.boxes.xyxy.cpu().numpy()

            for i in range(len(confs)):
                conf = float(confs[i])
                if conf < conf_threshold:
                    continue

                cls_id = int(cls_ids[i])
                label = classes.get(cls_id, f"object_{cls_id}")
                box = xyxy_boxes[i].astype(int).tolist() # [xmin, ymin, xmax, ymax]

                # Check if this object triggers Stage 2/3
                is_trigger = any(tc in label.lower() for tc in TRIGGER_CLASSES)
                if is_trigger or ("person" in label.lower() and conf > 0.60):
                    triggered = True

                det_item = {
                    "label": label,
                    "confidence": round(conf, 3),
                    "box": box, # [xmin, ymin, xmax, ymax]
                    "is_trigger": is_trigger
                }
                detections.append(det_item)

                # Draw bounding box
                color = (0, 0, 255) if is_trigger else (255, 165, 0)
                xmin, ymin, xmax, ymax = box
                cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), color, 2)
                text = f"{label.upper()} {conf:.2f}"
                cv2.putText(annotated_frame, text, (xmin, max(15, ymin - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        return triggered, annotated_frame, detections

if __name__ == "__main__":
    scanner = YOLOEdgeScanner()
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    trig, ann, dets = scanner.scan_frame(dummy_img)
    print(f"Test Scan Finished: Triggered={trig}, Detections={len(dets)}")
