# ===========================================
# 📓 1. Introduction
# ===========================================

"""
In this notebook, we build a model to predict complications after myocardial infarction 
using a Random Forest classifier.
"""

# ===========================================
# 📓 2. Loading Data
# ===========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from imblearn.over_sampling import SMOTE
from collections import Counter

# Load dataset
df = pd.read_csv('../myocardial_data.csv')
print("Dataset shape:", df.shape)
df.head()

# ===========================================
# 📓 3. Exploratory Data Analysis (EDA)
# ===========================================

# Missing values
df.isna().sum()

# Target distribution
sns.countplot(x='LET_IS', data=df)
plt.title('Target Distribution (LET_IS)')
plt.show()

# ===========================================
# 📓 4. Preprocessing
# ===========================================

# Handle missing values
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# One-hot encoding
X = df.drop(columns=['LET_IS', 'ID'])
y = df['LET_IS']

X = pd.get_dummies(X)

# ===========================================
# 📓 5. Train-Test Split & SMOTE
# ===========================================

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SMOTE for class balancing
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Original training set shape:", Counter(y_train))
print("Resampled training set shape:", Counter(y_train_res))

# ===========================================
# 📓 6. Model Training
# ===========================================

model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train_res, y_train_res)

# ===========================================
# 📓 7. Evaluation
# ===========================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("ROC-AUC (multiclass ovr):", roc_auc_score(y_test, y_prob, multi_class='ovr'))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ===========================================
# 📓 8. Feature Importance
# ===========================================

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

plt.figure(figsize=(12,6))
plt.title("Top 20 Feature Importances")
sns.barplot(x=importances[indices][:20], y=feature_names[indices][:20], palette='viridis')
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# ===========================================
# 📓 9. Prediction Demo for New Patient
# ===========================================

# New patient example
new_patient = {col: 0 for col in feature_names}
new_patient['AGE'] = 75
new_patient['SEX'] = 1

# Convert to DataFrame
new_patient_df = pd.DataFrame([new_patient])

# Scale
new_patient_scaled = scaler.transform(new_patient_df)

# Predict
prediction = model.predict(new_patient_scaled)
prediction_proba = model.predict_proba(new_patient_scaled)

print("🔮 Predicted complication class:", prediction[0])
print("📊 Prediction probabilities:", prediction_proba)

# ===========================================
# ✅ END OF NOTEBOOK
# ===========================================
