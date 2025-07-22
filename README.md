# 🫀 Myocardial Complications Prediction

This project implements a machine learning pipeline to **predict possible complications** in myocardial infarction patients using clinical and demographic data.

---

## 📂 **Project Structure**

```
myocardial_project/ 
├── notebook/ 
│   └── notebook.py
├── myocardial_data.csv
├── train_model.py
├── predict.py
├── scaler.pkl
├── trained_model.pkl
├── feature_names.csv
├── README.md
└── requirements.txt

└── README.md              
```

---

🚀 What does this project do?
Basic idea:
This project uses a computer program (machine learning) to learn from past patient data who had a heart attack, to predict which patients are more likely to develop complications afterward.

How does it work?
1. Training the model (train_model.py):
Load Data: Reads the patient data, which includes clinical info (e.g., blood pressure, cholesterol) and demographic info (e.g., age, gender).

Clean Data: If some values are missing, it fills them with the most common value (for categories) or the median value (for numbers).

Convert Categories: Turns categorical data (like gender or yes/no answers) into numbers that the computer can understand.

Balance Classes: If some complication types are rare, it uses a technique called SMOTE to create synthetic examples so the model doesn’t ignore rare cases.

Train Model: Learns patterns using a Random Forest Classifier, which is an algorithm that combines many decision trees to make predictions.

Evaluate Model: Checks how well the model predicts using various metrics (like accuracy, ROC-AUC, confusion matrix).

Save Files: Saves the trained model (trained_model.pkl), a scaler used for data normalization (scaler.pkl), and the list of feature names (feature_names.csv) for future use.


2. Predicting complications (predict.py):
Loads the saved model, scaler, and feature names.

Takes new patient data as input.

Processes the data in the same way as during training (scaling, encoding).

Predicts the most likely complication class for the new patient.

Outputs the predicted class along with probabilities for all possible complication types.


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

### 2. **predict.py**

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
python predict.py

---





