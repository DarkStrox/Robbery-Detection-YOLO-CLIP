# Robbery Detection Multi-Stage Pipeline (YOLO + CLIP)

An advanced surveillance anomaly detection system combining high-speed YOLO edge detection with deep OpenAI CLIP zero-shot reasoning.

## Architecture
1. **Stage 1 (YOLO Edge Scanner)**: Rapid frame scanning detecting persons, potential weapons, and suspicious interactions at high FPS.
2. **Stage 2 (Target Crop Enhancer)**: Isolates and crops Regions of Interest (ROI) with normalization and enhancement.
3. **Stage 3 (Reasoning Engine)**: Evaluates cropped targets using CLIP zero-shot vision-language embeddings to eliminate false alarms and confirm threat likelihood.

## Requirements
```bash
pip install -r requirements.txt
```

## Running
```bash
python app_ui.py
```
Or run the fast video test script:
```bash
python test_yolo_video_fast.py
```
