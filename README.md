


# 🧠 NeuroAI Pro: Volumetric Medical Image Segmentation

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-BraTS-red)

**NeuroAI Pro** is a state-of-the-art, web-based platform for **3D Brain Tumor Segmentation**. It leverages the **nnFormer** (Interleaved Transformer) architecture to process volumetric MRI scans (NIfTI format) and identify tumor regions with high precision.

The system is designed to handle massive 3D medical data efficiently using **Persistent Storage** and **In-Memory Caching** for real-time analysis.

---

## 📂 Dataset Information

The model is trained on the **BraTS (Brain Tumor Segmentation Challenge)** dataset, which is the gold standard for glioma segmentation.

---

### 🧾 Input Modalities (4 Channels)

The AI processes four distinct MRI sequences simultaneously to distinguish tumor sub-regions:

1. **T1-weighted (T1):** Anatomical structure  
2. **T1-weighted contrast-enhanced (T1ce):** Highlights the active tumor core  
3. **T2-weighted (T2):** Shows edema and inflammation  
4. **FLAIR:** Suppresses fluids to clearly show peritumoral edema  

---

### 🎯 Output Classes (3 Labels)

The model segments the tumor into three clinically relevant regions:

- 🔴 **Necrotic Tumor Core (Label 1):** Dead tissue inside the tumor  
- 🔵 **Peritumoral Edema (Label 2):** Swelling around the tumor  
- 🟡 **Enhancing Tumor (Label 3):** Active, growing tumor tissue  

---

## 🚀 Key Features

### 1️⃣ Advanced AI Engine (nnFormer)

- **Hybrid Architecture:** Combines Convolutional layers for local spatial details with Transformers for long-range global context  
- **Volumetric Processing:** Processes data as 3D patches (128 × 128 × 128), not just 2D slices  
- **Patch-Based Inference:** Sliding window approach with Gaussian overlap stitching  

---

### 2️⃣ Interactive Visualization Dashboard

- **3D Surface Rendering:** Interactive brain & tumor mesh visualization  
- **Instant Slice Explorer:** Real-time Axial, Sagittal, and Coronal slicing  
- **RAM Caching:** Eliminates disk I/O lag during slicing  
- **Biomarker Analytics:**  
  - Tumor volume (cm³)  
  - Tissue ratios  
  - Intensity histograms  

---

### 3️⃣ Clinical Validation Tools

Upload expert ground-truth masks to compute:

- Dice Score  
- IoU (Intersection over Union)  
- Sensitivity & Specificity  
- Hausdorff Distance  

---

## 🧠 Model Architecture: nnFormer

This project implements the **nnFormer (Not-another TransFormer)** architecture for volumetric medical segmentation.

---

###  The "Russian Doll" Strategy

To handle massive 3D MRI volumes without memory overflow:

1. **Volume → Patch:** Sliding window crops of 128 × 128 × 128  
2. **Patch → Window:** Subdivide into 8 × 8 × 8 grids for attention  

---

### 🔬 Key Mechanisms

#### 1️⃣ LV-MSA (Local Volume Multi-head Self-Attention)

Used in early encoder stages to capture fine details.

```
Attention(Q, K, V) = softmax((QKᵀ / √d_k) + B)V
```

---

#### 2️⃣ GV-MSA (Global Volume Multi-head Self-Attention)

Used in the bottleneck to capture full-volume global context.

---

#### 3️⃣ Skip Attention

Instead of concatenation (like U-Net), decoder features use:

- Queries → Decoder features  
- Keys/Values → Encoder features  

This allows selective spatial retrieval.

---

##  Tech Stack

- **Core AI:** PyTorch, MONAI, Nibabel  
- **Web Framework:** Dash (Plotly), Flask  
- **Visualization:** Plotly Graph Objects (3D Mesh & 2D Heatmaps)  
- **Infrastructure:** In-memory caching + persistent filesystem  

---

## 💻 Local Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/AbhiT212/NeuroAI
cd NeuroAI
```

---

### 2️⃣ Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If using GPU, install CUDA-compatible PyTorch.

---

### 4️⃣ Configuration Check

In `app.py`, update:

```python
TEMP_DIR = "temp_sessions"
```

For local usage (instead of `/mnt/data`).

---

### 5️⃣ Run the Application

```bash
python app.py
```

Access dashboard at:

```
http://localhost:8050
```

---

##  Project Structure

```
neuroai-pro/
│
├── app.py
├── model.py
├── config.py
├── requirements.txt
├── assets/
├── checkpoints/
│   └── best.pth
└── README.md
```

---

##  References

1. Zhou, H. Y., et al.  
   **"nnFormer: Volumetric Medical Image Segmentation via a 3D Transformer."**  
   IEEE Transactions on Image Processing, 2021  

2. Brain Tumor Segmentation Challenge (BraTS 2021/2023)

---

##  Developed By

**Abhi** | 2026  
NeuroAI Pro 

