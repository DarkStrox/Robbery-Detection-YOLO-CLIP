import os
import sys
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Project paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = r"E:\Aitronix Projects\Robber Detection\Yolo Guns Repo\Weapons-and-Knives-Detector-with-YOLOv8\runs\detect\Normal_Compressed\weights\best.pt"
DEFAULT_VIDEO = r"E:\Aitronix Projects\Robber Detection\input_videos\combined_output.mp4"

# Fallback video search if default not found
if not os.path.exists(DEFAULT_VIDEO):
    temp_vid = os.path.join(PROJECT_DIR, "output", "temp_input_video.mp4")
    if os.path.exists(temp_vid):
        DEFAULT_VIDEO = temp_vid

def on_trackbar_conf(val):
    pass

def main():
    print("=========================================================================")
    print("      FAST YOLO ALONE VIDEO TESTER (Ultra-Fast Loop Mode)")
    print("=========================================================================")
    
    # 1. Select Device (GPU if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Execution Device: {device.upper()}")

    # 2. Load YOLO Model
    weights_path = DEFAULT_WEIGHTS
    if not os.path.exists(weights_path):
        print(f"⚠️ Custom weights not found at: {weights_path}")
        weights_path = "yolov8n.pt"
        print(f"Falling back to standard: {weights_path}")

    print(f"📦 Loading YOLO Model: {weights_path}...")
    model = YOLO(weights_path)
    model.to(device)
    print("✅ Model loaded successfully!")

    # 3. Video Capture setup
    video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        return

    print(f"🎥 Video Stream: {video_path}")
    print("\n⌨️ KEYBOARD CONTROLS:")
    print("   [Space] or [p] : Pause / Play")
    print("   [r]             : Restart Video")
    print("   [+] / [-]       : Increase / Decrease Confidence Threshold")
    print("   [q] or [ESC]   : Quit")
    print("=========================================================================\n")

    # Create OpenCV window & Trackbar
    window_name = "Fast YOLO Alone Tester (Press Q to Quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 768)
    
    # Trackbar for interactive confidence adjusting (Default 50%)
    cv2.createTrackbar("Confidence %", window_name, 50, 95, on_trackbar_conf)

    cap = cv2.VideoCapture(video_path)
    
    paused = False
    prev_time = time.time()
    fps = 0.0

    while True:
        if not paused:
            ret, frame = cap.read()
            # Infinite video loop
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Read current confidence threshold from trackbar
            conf_val = cv2.getTrackbarPos("Confidence %", window_name)
            conf_thresh = max(0.05, conf_val / 100.0)

            # Fast YOLO inference (half precision on CUDA for maximum FPS)
            results = model.predict(
                source=frame,
                conf=conf_thresh,
                device=device,
                half=(device == "cuda"),
                verbose=False
            )

            annotated_frame = frame.copy()
            det_count = 0

            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy()
                names = result.names

                for i in range(len(confs)):
                    conf = float(confs[i])
                    cls_id = int(clss[i])
                    label_name = names.get(cls_id, f"Class {cls_id}")
                    xmin, ymin, xmax, ymax = boxes[i].astype(int)

                    det_count += 1
                    
                    # Highlight weapon/gun detections in bright red, others in cyan
                    is_weapon = any(w in label_name.lower() for w in ["gun", "knife", "weapon", "pistol"])
                    box_color = (0, 0, 255) if is_weapon else (255, 255, 0)

                    cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), box_color, 2)
                    text = f"{label_name.upper()} {conf*100:.0f}%"
                    
                    # Text background box
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_frame, (xmin, max(0, ymin - 22)), (xmin + tw + 6, max(22, ymin)), box_color, -1)
                    cv2.putText(annotated_frame, text, (xmin + 3, max(16, ymin - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            # Calculate FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(0.001, curr_time - prev_time))
            prev_time = curr_time

            # Status Overlay Header
            cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 40), (20, 20, 20), -1)
            status_text = f"FPS: {fps:.1f} | Conf Threshold: {conf_thresh:.2f} ({conf_val}%) | Detections: {det_count} | Device: {device.upper()}"
            cv2.putText(annotated_frame, status_text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

            current_display_frame = annotated_frame

        cv2.imshow(window_name, current_display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27 = ESC
            break
        elif key == ord(' ') or key == ord('p'):
            paused = not paused
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        elif key == ord('+') or key == ord('='):
            cur = cv2.getTrackbarPos("Confidence %", window_name)
            cv2.setTrackbarPos("Confidence %", window_name, min(95, cur + 5))
        elif key == ord('-') or key == ord('_'):
            cur = cv2.getTrackbarPos("Confidence %", window_name)
            cv2.setTrackbarPos("Confidence %", window_name, max(5, cur - 5))

    cap.release()
    cv2.destroyAllWindows()
    print("Stream closed.")

if __name__ == "__main__":
    main()
