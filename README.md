# Myocardial Infarction Complication Prediction

## Overview

This project is a machine learning application for predicting the risk of complications after myocardial infarction.

The goal is to build a clinical decision-support prototype that uses patient demographic and clinical data to estimate whether a patient belongs to one of the following categories:

* **0 → No complication**
* **1 → Moderate complication**
* **2 → Severe complication**

The project includes:

* Data exploration
* Preprocessing
* Model training
* Model evaluation
* Explainability using SHAP
* A Streamlit web application

---

## Why this project matters

Post-myocardial infarction complications can be serious and may require early clinical attention.

This project demonstrates how machine learning can be used to:

* Analyze clinical patient data
* Detect patterns related to complications
* Support risk classification
* Provide interpretable predictions
* Present results through a web interface

---

## Project Features

* Exploratory Data Analysis (EDA)
* Missing value handling
* One-Hot Encoding
* Feature scaling
* Class imbalance handling with SMOTE
* Class merging for better clinical grouping
* Multi-class classification
* Model evaluation with several metrics
* Confusion matrix visualization
* Feature importance analysis
* SHAP explainability
* Patient-specific prediction
* Streamlit web app

---

## Project Structure

```bash
Myocardial_Project/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── myocardial_data.csv
│
├── models/
│   ├── trained_model.pkl
│   ├── scaler.pkl
│   └── feature_names.csv
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── shap_summary.png
│   ├── shap_patient_explanation.png
│   └── eda_outputs/
│
└── src/
    ├── eda.py
    ├── train_and_save_model.py
    └── predict.py
```

## Dataset Description

* **1700 patient records**
* **124 clinical and demographic features**

Examples of features:

* Age
* Sex
* Previous infarction history
* Heart failure indicators
* Myocardial rupture indicators
* Clinical complications

Target variable:

```text
LET_IS
```

Originally, the target variable had 8 classes:

```text
0, 1, 2, 3, 4, 5, 6, 7
```

Original class distribution:

```text
0 → 1429
1 → 110
3 → 54
7 → 27
6 → 27
4 → 23
2 → 18
5 → 12
```

The dataset was highly imbalanced.

---

## Class Merging Strategy

To improve predictive performance, the original 8 classes were merged into 3 clinically meaningful categories:

```text
0 → No complication
1 → Moderate complication
2 → Severe complication
```

This significantly improved the model’s ability to learn from minority classes and improved macro-average F1-score.

---

## Data Preprocessing

### 1. Missing Value Handling

Numerical missing values are replaced using the median:

```python
df[col] = df[col].fillna(df[col].median())
```

Categorical missing values are replaced using the mode:

```python
df[col] = df[col].fillna(df[col].mode()[0])
```

### 2. Feature Selection

```python
target = "LET_IS"
X = df.drop(columns=[target, "ID"])
```

### 3. One-Hot Encoding

```python
X = pd.get_dummies(X)
```

### 4. Feature Scaling

```python
StandardScaler()
```

---

## Class Imbalance Handling

SMOTE was applied:

```python
SMOTE(random_state=42)
```

---

## Model Evaluation

Metrics used:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Classification Report
* Confusion Matrix
* Cross-validation

### Final Results

```text
Accuracy:  0.8912
Precision: 0.8808
Recall:    0.8912
F1-score:  0.8760
ROC-AUC:   0.9069
```

### Classification Report

```text
              precision    recall  f1-score   support

0              0.91      0.98      0.94       286
1              0.65      0.50      0.56        22
2              0.79      0.34      0.48        32
```

---

## Explainable AI with SHAP

Generated files:

```text
outputs/shap_summary.png
outputs/shap_patient_explanation.png
```

---

## Streamlit Web Application

Run locally:

```bash
python3 -m streamlit run app.py
```

---

## How to Run

```bash
git clone https://github.com/AfroditiTzama/Myocardial_Project.git
cd Myocardial_Project
pip install -r requirements.txt
python src/eda.py
python src/train_and_save_model.py
python src/predict.py
python3 -m streamlit run app.py
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn
* XGBoost
* SHAP
* Streamlit
* Matplotlib
* Seaborn

---

## Limitations

* Dataset imbalance
* Limited minority class samples
* No external validation dataset
* Not clinically validated
* Educational use only

---

## Future Improvements

* Hyperparameter tuning
* Model comparison
* Probability calibration
* Better Streamlit UI
* Online deployment

---

## Disclaimer

This project is for educational and portfolio purposes only.
