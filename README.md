# Myocardial Infarction Complication Prediction

## Overview

This project is a machine learning application for predicting the risk of complications after myocardial infarction.

The goal is to build a clinical decision-support prototype that uses patient demographic and clinical data to estimate whether a patient belongs to one of the following categories:

- **0 → No complication**
- **1 → Moderate complication**
- **2 → Severe complication**

The project includes data exploration, preprocessing, model training, evaluation, explainability using SHAP, and a Streamlit web application.

---

## Why this project matters

Post-myocardial infarction complications can be serious and may require early clinical attention.

This project demonstrates how machine learning can be used to:

- analyze clinical patient data
- detect patterns related to complications
- support risk classification
- provide interpretable predictions
- present results through a simple web interface

---

## Project Features

- Exploratory Data Analysis
- Missing value handling
- One-Hot Encoding
- Feature scaling
- Class imbalance handling with SMOTE
- Class merging for better clinical grouping
- Multi-class classification
- Model evaluation with several metrics
- Confusion matrix visualization
- Feature importance analysis
- SHAP explainability
- Patient-specific prediction
- Streamlit web app

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

Dataset Description

The dataset contains:

1700 patient records
124 clinical and demographic features

Examples of features include:

Age
Sex
Previous infarction history
Heart failure indicators
Myocardial rupture indicators
Clinical complications

The target variable is:

LET_IS

Originally, the target variable had 8 classes:

0, 1, 2, 3, 4, 5, 6, 7

Original class distribution:

0 → 1429
1 → 110
3 → 54
7 → 27
6 → 27
4 → 23
2 → 18
5 → 12

The dataset was highly imbalanced.

Class Merging Strategy

To improve predictive performance, the original 8 classes were merged into 3 clinically meaningful categories:

0 → No complication
1 → Moderate complication
2 → Severe complication

This significantly improved the model’s ability to learn from minority classes and improved macro-average F1-score.

Data Preprocessing

The preprocessing pipeline includes:

1. Missing Value Handling

Numerical missing values are replaced using the median:

df[col] = df[col].fillna(df[col].median())

Categorical missing values are replaced using the mode:

df[col] = df[col].fillna(df[col].mode()[0])
2. Feature Selection

The following columns are removed:

target = "LET_IS"
X = df.drop(columns=[target, "ID"])

The ID column is removed because it has no predictive value.

3. One-Hot Encoding

Categorical variables are converted into numerical format:

X = pd.get_dummies(X)
4. Feature Scaling

Numerical features are standardized:

StandardScaler()
Class Imbalance Handling

The dataset was imbalanced.

To address this, SMOTE (Synthetic Minority Over-sampling Technique) was applied:

SMOTE(random_state=42)

This creates synthetic examples of minority classes during training.

Model

The model is trained to predict complication risk.

Files generated after training:

models/trained_model.pkl
models/scaler.pkl
models/feature_names.csv

These are used later by:

predict.py
app.py
Model Evaluation

The model is evaluated using:

Accuracy
Precision
Recall
F1-score
ROC-AUC
Classification Report
Confusion Matrix
Cross-validation
Final Results
Accuracy:  0.8912
Precision: 0.8808
Recall:    0.8912
F1-score:  0.8760
ROC-AUC:   0.9069
Classification Report
              precision    recall  f1-score   support

0              0.91      0.98      0.94       286
1              0.65      0.50      0.56        22
2              0.79      0.34      0.48        32

accuracy                           0.89       340
macro avg       0.78      0.61      0.66       340
weighted avg    0.88      0.89      0.88       340

The model performs strongly overall, although severe complications remain harder to predict due to fewer examples.

Exploratory Data Analysis (EDA)

The project performs EDA before training.

Generated plots include:

Missing Values Heatmap
Correlation Heatmap
Target Distribution
Histograms
Boxplots

Run:

python src/eda.py

Saved in:

outputs/eda_outputs/
Feature Importance

The project visualizes the most important clinical features.

Example output:

outputs/feature_importance.png

This helps identify which variables influence predictions the most.

Explainable AI with SHAP

The project uses SHAP (SHapley Additive Explanations).

SHAP provides:

Global explanation of model behavior
Patient-specific explanation

Generated files:

outputs/shap_summary.png
outputs/shap_patient_explanation.png

This improves interpretability and trust in the model.

Prediction Example

Example patient:

new_patient["AGE"] = 75
new_patient["SEX"] = 1

Output example:

Predicted complication class: 2
Clinical meaning: Severe complication

Prediction probabilities:
Class 0: 0.18
Class 1: 0.10
Class 2: 0.72

Run:

python src/predict.py
Streamlit Web Application

This project includes an interactive Streamlit web app.

Run:

python3 -m streamlit run app.py

The app allows the user to:

enter patient information
get risk prediction
view class probabilities
view top SHAP factors

This provides a user-friendly interface for demonstrating the model.

How to Run the Project
Clone repository
git clone https://github.com/AfroditiTzama/Myocardial_Project.git
cd Myocardial_Project
Install dependencies
pip install -r requirements.txt

or:

python3 -m pip install -r requirements.txt
Run EDA
python src/eda.py
Train model
python src/train_and_save_model.py
Run prediction
python src/predict.py
Run Streamlit app
python3 -m streamlit run app.py
Technologies Used
Python
Pandas
NumPy
Scikit-learn
Imbalanced-learn
XGBoost
SHAP
Streamlit
Matplotlib
Seaborn
Pickle
Limitations

This project has limitations:

Dataset imbalance
Limited minority class samples
No external validation dataset
Not clinically validated
Educational use only
Future Improvements

Possible future work:

Hyperparameter tuning
Comparison with more models
Probability calibration
Better Streamlit UI
More patient input features
Online deployment
External validation dataset
Disclaimer

