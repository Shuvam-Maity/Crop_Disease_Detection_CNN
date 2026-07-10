# 🌿 Crop Disease Detection using Custom CNNs

[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Models-yellow)](https://huggingface.co/Shuvam-Maity/Crop_Disease_Detection_CNN)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)]()

> Nine crop-specific CNN models detecting 33 plant diseases across 40,000+ leaf images — trained from scratch with no transfer learning. MSc Data Science Dissertation, St. Xavier's College (Autonomous), Kolkata.

---

## 📌 Overview

Plant diseases cause **20–40% of global crop yield loss** every year, costing up to **$220 billion** in economic damage. Early and accurate diagnosis is critical — yet most farmers lack access to expert knowledge.

This project builds **nine specialist CNN models**, one per crop, trained entirely from scratch on the **PlantVillage dataset**. Unlike most prior work that relies on ImageNet pre-trained weights (ResNet, VGG, etc.), these models learn agricultural features purely from leaf morphology — no transfer learning involved.

---

## 🌱 Crops & Performance

| Crop | Botanical Family | Classes | Dataset Size | Test Accuracy |
|------|-----------------|---------|--------------|---------------|
| Tomato | Solanaceae | 10 | 18,166 | 89.80% |
| Grape | Vitaceae | 4 | 4,062 | 98.69% |
| Corn | Poaceae | 4 | 3,852 | 96.03% |
| Apple | Rosaceae | 4 | 3,171 | 97.27% |
| Peach | Rosaceae | 2 | 2,657 | 99.50% |
| Bell Pepper | Solanaceae | 2 | 2,475 | 100% |
| Potato | Solanaceae | 3 | 2,152 | 96.30% |
| Cherry | Rosaceae | 2 | 1,906 | 99.65% |
| Strawberry | Rosaceae | 2 | 1,565 | 99.15% |
| **Total** | **4 Families** | **33** | **40,006** | |

---

## 🏗️ Architecture & Methodology

### Specialist Model Paradigm
Instead of a single unified classifier, one dedicated CNN is trained per crop. This reduces inter-crop feature noise and allows each model to focus on crop-specific pathological patterns.

### Custom CNN Architecture (Keras Sequential API)
```
Input (224x224x3)
    │
    ├── Stochastic Augmentation Layer
    │     RandomFlip → RandomRotation → RandomZoom → RandomContrast
    │
    ├── Conv2D (ReLU) + MaxPooling2D   ← Low-level edges & textures
    ├── Conv2D (ReLU) + MaxPooling2D   ← Mid-level patterns
    ├── Conv2D (ReLU) + MaxPooling2D   ← High-level disease features
    │
    ├── Flatten / Global Average Pooling (crop-dependent)
    ├── Dense (ReLU)
    ├── Dropout (0.5)
    └── Dense (Softmax) → Class Prediction
```

### Key Design Decisions
- **Flatten vs GAP** — Flatten used for Corn & Grape (spatial symptom density matters); GAP used for Peach & Tomato (efficiency + overfitting resistance)
- **Cost-Sensitive Learning** — Class weights computed as `W_j = N / (n_classes × n_j)` to handle class imbalance
- **Data Partitioning** — 70% train / 15% validation / 15% test (strict hold-out)

### Training Stack
- **Loss** — Sparse Categorical Crossentropy
- **Optimizer** — Adam (lr=0.001)
- **Callbacks** — EarlyStopping + ModelCheckpoint (val_loss monitored)
- **Platform** — Google Colab + Kaggle Kernels (NVIDIA Tesla T4/P100)

---

## 📁 Repository Structure

```
Crop_Disease_Detection_CNN/
│
├── assets/
│   ├── Confusion Matrix/
│   │   ├── Apple.jpg
│   │   ├── Bell_Pepper.jpg
│   │   ├── Cherry.jpg
│   │   ├── Corn.jpg
│   │   ├── Grape.jpg
│   │   ├── Peach.jpg
│   │   ├── Potato.jpg
│   │   └── Strawberry.jpg
│   │
│   └── Loss Curves/
│       ├── Apple.jpg
│       ├── Bell_Pepper.jpg
│       ├── Cherry.jpg
│       ├── Corn.jpg
│       ├── Grape.jpg
│       ├── Peach.jpg
│       ├── Potato.jpg
│       ├── Strawberry.jpg
│       └── Tomato.jpg
│
├── models/
│   └── README.md            ← Links to HuggingFace Hub
│
├── notebooks/               ← One Jupyter notebook per crop
│   ├── Apple_Best.ipynb
│   ├── Bell_Pepper_Best.ipynb
│   ├── Cherry_Best.ipynb
│   ├── Corn_Best.ipynb
│   ├── Grape_Best.ipynb
│   ├── Peach_Best.ipynb
│   ├── Potato_Best.ipynb
│   ├── Strawberry_Best.ipynb
│   └── Tomato_Best.ipynb
│
├── reports/
│   └── Crop_Disease_Detection_CN...pdf
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📊 Results

### Loss Curves
Training and validation curves for all nine models are available in [`assets/Loss Curves/`](./assets/Loss%20Curves/).

### Confusion Matrices
Class-level error analysis for all crops is in [`assets/Confusion Matrix/`](./assets/Confusion%20Matrix/).

**Key observations:**

- **Peach, Cherry, Strawberry, Bell Pepper** — Sharp convergence within 10 epochs, stabilizing above 99%
- **Tomato** — Gradual convergence over 50 epochs due to 10-class complexity; slight validation loss oscillation controlled via ModelCheckpoint
- **Corn & Grape** — Moderate convergence with healthy train/val gap, confirming Flatten's suitability for spatial symptom detection

**Notable findings:**
- Custom CNNs trained from scratch match transfer learning baselines when data volume exceeds 40,000 images
- GAP reduces overfitting and model size; Flatten better captures spatial lesion patterns
- Cost-sensitive weighting significantly improves Recall on minority disease classes

---

## 🤗 Trained Models

All `.keras` model files are hosted on HuggingFace Hub:

🔗 [Shuvam-Maity/Crop_Disease_Detection_CNN](https://huggingface.co/Shuvam-Maity/Crop_Disease_Detection_CNN)

### Load a model programmatically
```python
from huggingface_hub import hf_hub_download
from tensorflow import keras

path = hf_hub_download(
    repo_id="Shuvam-Maity/Crop_Disease_Detection_CNN",
    filename="tomato_model.keras"
)
model = keras.models.load_model(path)
```

---

## ⚙️ Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/Shuvam-Maity/Crop_Disease_Detection_CNN.git
cd Crop_Disease_Detection_CNN
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run a notebook
Open any notebook inside `notebooks/` on Google Colab or Kaggle and run all cells.
The PlantVillage dataset needs to be mounted or downloaded separately.

### 4. Run inference
```python
from huggingface_hub import hf_hub_download
from tensorflow import keras
import numpy as np
from PIL import Image

# Load model
path = hf_hub_download(
    repo_id="Shuvam-Maity/Crop_Disease_Detection_CNN",
    filename="tomato_model.keras"
)
model = keras.models.load_model(path)

# Preprocess image
img = Image.open("leaf.jpg").resize((224, 224))
img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

# Predict
pred = model.predict(img_array)
print("Predicted class index:", np.argmax(pred))
```

---

## 🔬 Dataset

**PlantVillage Dataset** (Mohanty et al., 2016)
- 40,006 images across 9 crops and 33 classes
- RGB images on neutral (black/gray) backgrounds
- Curated subset covering fungi, bacteria, virus, and healthy classes

> Dataset not included in this repo. Available via [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) or the original PlantVillage repository.

---

## 📄 Dissertation

The full MSc dissertation report is available in the [`reports/`](./reports/) folder.

> Maity, S. (2026). *Disease Detection in Crops Using Deep Learning*. MSc Dissertation, Department of Data Science, St. Xavier's College (Autonomous), Kolkata.

---

## 👤 Author

**Shuvam Maity**
MSc Data Science — St. Xavier's College (Autonomous), Kolkata
🔗 [HuggingFace](https://huggingface.co/Shuvam-Maity) · [GitHub](https://github.com/Shuvam-Maity)

---

## 📜 License

This project is licensed under the MIT License.
