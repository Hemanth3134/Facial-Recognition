# Emotion Detection from Facial Images

This repository provides tools and models for detecting human emotions from facial images using classical machine learning (SVM) and deep learning (ONNX) techniques.

## 📁 Repository Contents

| File                          | Description |
|------------------------------|-------------|
| `emotion_data.npy`           | Preprocessed image data or features for training/testing. |
| `emotion_labels.npy`         | Emotion labels corresponding to the data samples. |
| `emotion_svm_model.pkl`      | SVM model trained on emotion data (possibly raw or preprocessed images). |
| `emotion_svm_hog.pkl`        | SVM model trained using HOG (Histogram of Oriented Gradients) features. |
| `emotion-ferplus-8.onnx`     | Pretrained deep learning model based on the FER+ dataset (ONNX format). |
| `Image_detection.ipynb`      | Jupyter Notebook for loading models, making predictions, and visualizing results. |

---

## 🧠 Models

### 1. SVM Models

- `emotion_svm_model.pkl`: May use raw or flattened image data.
- `emotion_svm_hog.pkl`: Uses extracted HOG features for training.

### 2. Deep Learning Model

- `emotion-ferplus-8.onnx`: A model trained on the FER+ dataset to detect 8 emotions:
  - Neutral
  - Happiness
  - Surprise
  - Sadness
  - Anger
  - Disgust
  - Fear
  - Contempt

---

## 🚀 How to Use

1. **Clone this repository**

   ```bash
   git clone https://github.com/your-username/emotion-detection.git
   cd emotion-detection

2. **Install dependencies**

Make sure you have the following Python packages installed:
```bash
pip install numpy opencv-python scikit-learn onnxruntime matplotlib
```
3. **Install dependencies**

  ```bash
jupyter notebook Image_detection.ipynb
 ```
This notebook includes:

Model loading (SVM / ONNX)

Preprocessing and inference steps

Visualization of detected emotions

