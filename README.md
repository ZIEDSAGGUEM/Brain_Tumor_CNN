# 🧠 Brain Tumor Classification & Localization App

This project is a deep learning-powered brain tumor detection app that classifies MRI images into tumor types and highlights the tumor region using Grad-CAM heatmaps. It also generates a PDF report including patient details, prediction results, and annotated images, and logs each diagnosis to a CSV file.

---

## 📌 Features

* ✅ Classify MRI images into 4 categories:

  * **Glioma Tumor**
  * **Meningioma Tumor**
  * **Pituitary Tumor**
  * **No Tumor**

* 🔥 Grad-CAM heatmap for visual explanation

* 📄 Auto-generate PDF report with:

  * Patient name & age
  * Prediction and confidence
  * Original MRI + heatmap overlay
  * Doctor signature field

* 💃 Log each result in a CSV file (`patient_predictions_log.csv`)

---

## 🧠 Model

* CNN model built using Keras **Functional API**
* Trained on a dataset of brain MRI images
* Achieves **>90% accuracy** on validation data

---

## 📂 Project Structure

```
Brain_Tumor/
├── app.py                       # Streamlit app with Grad-CAM & report generation
├── model_train.py              # CNN training script
├── brain_tumor_cnn_model.h5    # Saved trained model
├── patient_predictions_log.csv # Logged reports
├── Brain_Tumor_Dataset/        # Dataset (4 folders for each class)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Setup Instructions

### Clone the repository:

```bash
git clone https://github.com/yourusername/brain-tumor-classification-app.git
cd brain-tumor-classification-app
```

### Create and activate a virtual environment:

```bash
python -m venv brainenv
source brainenv/bin/activate  # On Windows: brainenv\Scripts\activate
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Prepare the dataset:

Place your MRI images into the following folder structure:

```
Brain_Tumor_Dataset/
├── glioma_tumor/
├── meningioma_tumor/
├── pituitary_tumor/
└── no_tumor/
```

### Train the model (optional if `brain_tumor_cnn_model.h5` already exists):

```bash
python model_train.py
```

### Run the app:

```bash
streamlit run app.py
```

---

## 📦 Dependencies

Install with:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
tensorflow
streamlit
Pillow
matplotlib
numpy
opencv-python
fpdf
```

---
