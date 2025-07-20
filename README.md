# Diabetic_Retinopathy
# Diabetic Retinopathy Detection using CNN Architectures

This project aims to detect and classify **Diabetic Retinopathy (DR)** — a leading cause of vision loss — into five severity levels** from retinal fundus images using deep learning models. Leveraging powerful convolutional neural network (CNN) architectures like **ResNet50**, **InceptionV3**, and **DenseNet121**, this project explores performance comparisons and model optimization for medical image analysis.

---

##  Problem Statement

Diabetic Retinopathy is a diabetes-related eye disease that can cause permanent vision loss if not detected early. Manual screening is time-consuming and prone to errors. This project automates the classification of DR severity using CNN-based image classifiers.

---

##  Dataset

- Sourced from [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) / EyePACS (Kaggle)
- Contains high-resolution retinal fundus images labeled with severity levels:
  - **0**: No DR  
  - **1**: Mild  
  - **2**: Moderate  
  - **3**: Severe  
  - **4**: Proliferative DR

---

##  Model Architectures Used

| Model        | Accuracy |
|--------------|----------|
| DenseNet121  | 70%      |
| ResNet50     | 38%      |
| InceptionV3  | 55%      |

DenseNet121 outperformed other architectures in terms of **classification accuracy** and **generalization** across class-imbalanced data.

---

##  Workflow

1. **Data Preprocessing**
   - Resized and normalized images (224x224)
   - Handled class imbalance using **data augmentation**
   - Converted images into RGB format with pixel scaling

2. **Model Training**
   - Used `TensorFlow` and `Keras` for model building
   - Applied **categorical crossentropy loss** and **Adam optimizer**
   - Trained models with **early stopping** and **learning rate decay**

3. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix & Class Distribution Plots
   - Grad-CAM visualization for interpretability


