# 🫀 Myocardial Complications Prediction

This project implements a machine learning pipeline to **predict possible complications** in myocardial infarction patients using clinical and demographic data.

---

## 📂 **Project Structure**

myocardial_project/
├── train_model.py
├── predict_new_patient.py
├── myocardial_data.csv
├── trained_model.pkl
├── scaler.pkl
├── feature_names.csv
└── README.md


---

## 🚀 **Description**

### 1. **train_model.py**
- Loads and preprocesses the dataset
- Handles missing values (median for numeric, mode for categorical)
- Encodes categorical variables (One-Hot)
- Uses **SMOTE** for class balancing
- Trains a **Random Forest Classifier** with class weights balanced
- Evaluates the model (Classification Report, ROC-AUC, Confusion Matrix, Feature Importances)
- Saves:
  - Trained model (`trained_model.pkl`)
  - Scaler (`scaler.pkl`)
  - Feature names (`feature_names.csv`)

---

### 2. **predict_new_patient.py**
- Loads the trained model, scaler, and feature names
- Generates predictions for **new patients** based on their clinical data
- Outputs:
  - Predicted complication class
  - Probability for each possible complication

---

## ⚙️ **Usage**

1. **Install dependencies**

```bash
pip install -r requirements.txt
```
---

###Train the model:
python train_model.py

###Predict for a new patient:
python predict_new_patient.py

---

## **Example Output

🔮 Predicted complication class: 0
📊 Prediction probabilities: [[0.47 0.39 0.   0.08 0.   0.   0.02 0.04]]



