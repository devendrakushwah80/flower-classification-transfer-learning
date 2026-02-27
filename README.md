# Flower Classification using Transfer Learning (MobileNetV2)

## 📌 Project Overview

This project implements a multi-class image classification model for flower species using **Transfer Learning with MobileNetV2**.

The model classifies images into 7 flower categories:

- Bellflower  
- Daisy  
- Dandelion  
- Lotus  
- Rose  
- Sunflower  
- Tulip  

The dataset contains **11,200 images** (1,600 per class).

---

## 🚀 Features

- Transfer Learning using pretrained **MobileNetV2 (ImageNet weights)**
- Custom safe image loader (handles corrupted images)
- Data augmentation (flip, rotation, zoom)
- EarlyStopping and ModelCheckpoint
- Confusion Matrix visualization
- Classification Report (Precision, Recall, F1-score)
- Training & validation accuracy/loss curves
- Model saving and inference testing

---

## 📂 Dataset Structure
flowers/
│
├── bellflower/
├── daisy/
├── dandelion/
├── lotus/
├── rose/
├── sunflower/
└── tulip/


- Total images: 11,200  
- Classes: 7  

---

## 🧠 Model Architecture

### Base Model
- MobileNetV2 (pretrained on ImageNet)
- `include_top=False`
- Frozen convolutional base

### Custom Classification Head
- GlobalAveragePooling2D
- Dropout (0.3)
- Dense layer (7 units, Softmax)

### Training Configuration
- Loss: Sparse Categorical Crossentropy  
- Optimizer: Adam  
- Batch Size: 32  
- Image Size: 224 × 224  

---

## 📊 Training Results

Final Performance:

- Training Accuracy: ~91%
- Validation Accuracy: ~92%
- Stable loss curves
- No significant overfitting observed

Confusion matrix and classification report show strong per-class performance.

---

## 📈 Evaluation Metrics

The project includes:

- Confusion Matrix
- Precision
- Recall
- F1-score
- Training & Validation Accuracy Curve
- Training & Validation Loss Curve

---

## ⚙️ Installation

### 1️⃣ Clone Repository
git clone https://github.com/yourusername/flower-classification-transfer-learning.git

cd flower-classification-transfer-learning


### 2️⃣ Install Dependencies


pip install -r requirements.txt


### 3️⃣ Add Dataset

Place the dataset inside the project directory:


flower-classification-transfer-learning/
│
├── flowers/
└── Flower_CNN.ipynb


---

## ▶️ Run Training

Open the notebook:


jupyter notebook Flower_CNN.ipynb


Or run training script (if converted to .py):


python train.py


---

## 🔍 Model Inference

After training:

```python
import tensorflow as tf
model = tf.keras.models.load_model("flower_classifier.keras")
The model expects images resized to 224x224.
```
🛠 Tech Stack

Python

TensorFlow / Keras

NumPy

Matplotlib

Seaborn

Scikit-learn

Pillow

🔮 Future Improvements

Fine-tune last convolutional layers

Compare with EfficientNet

Deploy using Streamlit

Convert to ONNX / TensorRT

Add model explainability (Grad-CAM)

👨‍💻 Author

Devendra Kushwah
