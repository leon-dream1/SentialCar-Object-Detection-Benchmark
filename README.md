# Comparative Benchmark: YOLO vs. RF-DETR for Automotive Surveillance

An end-to-end Computer Vision research and development pipeline for detecting human presence and activity (`Person_Active` vs. `Person_Not_Active`) in automotive surveillance systems. This project conducts a comparative study between modern CNN-based real-time detectors (**YOLOv10**, **YOLOv12**, **YOLOv26**) and Vision Transformer-based architectures (**RF-DETR Nano** with DINOv2 backbone).

---

## 📌 Key Features & Highlights

* **Leakage-Free Splitting:** Implements a sequence-based split algorithm to ensure frames from the same video sequence/burst are kept within the same split (Train/Val/Test), preventing data leakage.
* **Automotive Domain Augmentations:** Custom Albumentations pipeline addressing low-light noise, brightness jitter, motion blur, and spatial scaling distortion.
* **Multi-Architecture Benchmark:** Trains and evaluates real-time CNN detectors alongside Transformer-based object detectors.
* **In-Depth Qualitative & Latency Analysis:** Comprehensive evaluation covering Precision, Recall, F1-score, mAP (@50, @50:95), hardware inference latency (ms/FPS), False Positives/Negatives, and localization errors.

---

## 📊 Dataset Audit Summary

The dataset was fully audited for bounding box boundary integrity, matching pairs, and class distributions:

| Metric | Count / Percentage |
| :--- | :--- |
| **Total Images** | 18,308 |
| **Labeled Images** | 14,180 |
| **Background / Negative Samples** | 4,128 |
| **Total Bounding Boxes** | 22,623 |
| **Person_Active (Class 0)** | 14,834 (65.57%) |
| **Person_Not_Active (Class 1)** | 7,789 (34.43%) |
| **Corrupt Files / Out-of-Bound BBoxes** | 0 |

### Sequence-Based Split (80 / 10 / 10)
- **Train Set:** 14,646 images (80%)
- **Validation Set:** 1,831 images (10%)
- **Test Set:** 1,831 images (10%)

---

## 📂 Notebooks & Workflow

The analysis and experimental pipeline is structured across 4 sequential Jupyter Notebooks:

### 1. `01_eda_split_and_augmentation.ipynb` — Data Integrity, EDA & Pipeline Setup
* **Integrity Check:** Verifies file count consistency, image-label pairing, and checks for corrupt or empty label files.
* **Exploratory Data Analysis (EDA):** Analyzes class balances and bounding box geometric properties (width, height, area ratio) to quantify object scale distribution.
* **Sequence Splitting:** Prevents data leakage across consecutive video frames using sequence-aware group splitting.
* **Domain Augmentation:** Implements automotive-specific Albumentations (motion blur, low-light noise, spatial transforms) with post-augmentation alignment verification.

### 2. `02_yolo_benchmarking.ipynb` — CNN Detector Training
* **Environment Setup:** Installs and configures `ultralytics`.
* **Data Config:** Generates `data.yaml` dynamically pointing to sequence splits.
* **Multi-Model Training:** Trains **YOLOv10**, **YOLOv12**, and **YOLOv26** for $\ge$ 50 epochs each under consistent hyperparameter settings.
* **Logging & Artifacts:** Tracks loss convergence and mAP (@50, @50:95) progression while saving best model weights.

### 3. `03_rfdetr_transformer_training.ipynb` — Transformer Detector Fine-Tuning
* **Environment Setup:** Configures `rfdetr[train,loggers]` dependencies.
* **Architecture Selection:** Uses **RF-DETR Nano** featuring a DINOv2 Vision Transformer (ViT) backbone with deformable attention to effectively handle fisheye distortion and varying scale positions.
* **Fine-Tuning:** Trains the model using gradient accumulation and EMA weight checkpoints over custom CONFIG parameters.

### 4. `04_comparative_evaluation_and_inference.ipynb` — Comparative Analysis & Benchmarking
* **Held-out Test Evaluation:** Loads best checkpoints for YOLOv10, YOLOv12, YOLOv26, and RF-DETR Nano to report Precision, Recall, F1-Score, and mAP metrics (overall and per-class).
* **Hardware Latency Benchmark:** Benchmarks raw inference latency (ms) and real-time frame rates (FPS).
* **Qualitative Error Analysis:** Isolates and visualizes False Negatives, False Positives, and bounding box localization errors.
* **Trade-off Analysis:** Provides a data-driven recommendation comparing CNN speed vs. ViT global context performance.

---

## 🚀 Getting Started

### Prerequisites
Install dependencies for both Ultralytics YOLO and RF-DETR:

```bash
pip install -r requirements.txt
pip install "rfdetr[train,loggers]"
