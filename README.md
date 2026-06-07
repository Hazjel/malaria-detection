# Malaria Parasite Detection Using YOLO with CBAM Attention

Comparative study of object detection models for malaria parasite detection in thin blood smear microscopic images. Implements YOLOv8n, YOLOv9s, and a novel **YOLOv8n+CBAM** variant with Convolutional Block Attention Module integrated into the backbone and neck.

> Final project — Computer Vision Course, 2025

---

## Results

| Model | mAP50 | mAP50-95 | Precision | Recall | F1 | FPS | Params |
|---|---|---|---|---|---|---|---|
| YOLOv8n | 0.9613 | 0.9365 | 0.9515 | 0.8934 | 0.9215 | 88.2 | 3.0M |
| YOLOv9s | 0.9617 | 0.9396 | 0.9585 | 0.8910 | 0.9235 | 57.5 | 7.2M |
| **YOLOv8n+CBAM** | **0.9600** | **0.9343** | **0.9381** | **0.8825** | **0.9095** | **93.96** | **3.0M** |

YOLOv8n+CBAM achieves competitive accuracy with the fastest inference speed at 93.96 FPS while maintaining the same parameter count as YOLOv8n.

---

## Dataset

- **Sources:** NIH Malaria Cell Images + MP-IDB (Malaria Parasite Image Database)
- **Classes:** `parasitized`, `uninfected`
- **Split:** 21,676 train / 5,606 val / 2,803 test
- **Preprocessing:** CLAHE enhancement, YOLO-format annotation conversion

Dataset not included in this repository due to size. Download from:
- [NIH Malaria Dataset on HuggingFace](https://huggingface.co/datasets/electricsheepafrica/malaria-parasite-detection-yolo)
- [MP-IDB on Roboflow](https://universe.roboflow.com/dsi-malaria/mp-idb)

---

## Project Structure

```
malaria-detection/
├── notebooks/
│   ├── 01_data_preparation.ipynb      # Dataset processing & CLAHE
│   ├── 02_training_yolov8n.ipynb      # YOLOv8n baseline training
│   ├── 03_training_yolov9.ipynb       # YOLOv9s training
│   ├── 04_training_yolov8n_cbam.ipynb # YOLOv8n+CBAM training
│   └── 05_evaluation_comparison.ipynb # Comparative evaluation
├── results/
│   ├── figures/                       # Plots & visualizations
│   └── tables/                        # CSV metrics
├── runs/
│   ├── yolov8n/train/                 # YOLOv8n training outputs
│   ├── yolov9/train/                  # YOLOv9 training outputs
│   └── yolov8n-cbam/train/            # YOLOv8n+CBAM training outputs
└── malaria.yaml                       # Dataset config
```

---

## Methodology

**CBAM Integration:** CBAM (channel + spatial attention) inserted at C2f layers in YOLOv8n backbone (layers 2, 4, 6, 8) and neck (layers 12, 15, 18, 21). This adds no extra parameters beyond the attention modules and improves feature selection without sacrificing speed.

**Training:** 50–80 epochs, AdamW optimizer, image size 640×640, batch size 16.

---

## Requirements

```bash
pip install ultralytics torch torchvision opencv-python matplotlib
```

---

## Usage

```python
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/yolov8n-cbam/train/weights/best.pt")

# Inference
results = model.predict("path/to/image.jpg", conf=0.25)
```
