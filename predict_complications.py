# train_and_save_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from imblearn.over_sampling import SMOTE
from collections import Counter

# 1. Φόρτωση δεδομένων
df = pd.read_csv('myocardial_data.csv')
print("Dataset shape:", df.shape)
print("Missing values per column:\n", df.isna().sum())

# 2. Διαχείριση missing values
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 3. Στόχος
target = 'LET_IS'
print(f"Target '{target}' value counts:\n", df[target].value_counts())

# 4. Προεπεξεργασία
X = df.drop(columns=[target, 'ID'])
y = df[target]

X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

# Τυποποίηση
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Original training set shape:", Counter(y_train))
print("Resampled training set shape:", Counter(y_train_res))

# 6. Μοντέλο
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train_res, y_train_res)

# 7. Αξιολόγηση
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

# Feature Importance
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

# Αποθήκευση μοντέλου, scaler, feature names
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

pd.Series(feature_names).to_csv('feature_names.csv', index=False)

print("✅ Model, scaler, and feature names saved successfully.")
