# predict.py

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Φόρτωση μοντέλου, scaler και feature names
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

feature_names = pd.read_csv(FEATURE_NAMES_PATH).squeeze().tolist()

# 2. Δημιουργία νέου ασθενή
new_patient = {col: 0 for col in feature_names}

new_patient["AGE"] = 75

if "SEX" in feature_names:
    new_patient["SEX"] = 1
elif "SEX_1" in feature_names:
    new_patient["SEX_1"] = 1

# 3. Μετατροπή σε DataFrame
new_patient_df = pd.DataFrame([new_patient])
new_patient_df = new_patient_df[feature_names]

# 4. Scaling
new_patient_scaled = scaler.transform(new_patient_df)

# 5. Πρόβλεψη
prediction = model.predict(new_patient_scaled)
prediction_proba = model.predict_proba(new_patient_scaled)

class_mapping = {
    0: "No complication",
    1: "Moderate complication",
    2: "Severe complication"
}

print("\n===== Prediction Result =====")
print("Predicted complication class:", prediction[0])
print("Clinical meaning:", class_mapping.get(prediction[0], "Unknown class"))

print("\nPrediction probabilities:")
for cls, prob in zip(model.classes_, prediction_proba[0]):
    print(f"Class {cls} ({class_mapping.get(cls, 'Unknown')}): {prob:.4f}")

# 6. Feature Importance γράφημα
print("\nΔημιουργία γραφήματος Feature Importance...")

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 20

plt.figure(figsize=(12, 8))
plt.title("Top 20 Most Important Clinical Features")

sns.barplot(
    x=importances[indices][:top_n],
    y=[feature_names[i] for i in indices[:top_n]],
    palette="magma"
)

plt.xlabel("Importance Score")
plt.ylabel("Clinical Feature")
plt.tight_layout()

feature_importance_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
plt.savefig(feature_importance_path, dpi=300, bbox_inches="tight")
print(f"Το γράφημα αποθηκεύτηκε εδώ: {feature_importance_path}")

plt.show()

# 7. SHAP Explainability
print("\nGenerating SHAP explanation...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(new_patient_scaled)

predicted_class_index = list(model.classes_).index(prediction[0])

if isinstance(shap_values, list):
    selected_shap_values = shap_values[predicted_class_index]
else:
    selected_shap_values = shap_values[:, :, predicted_class_index]

shap.summary_plot(
    selected_shap_values,
    new_patient_scaled,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()
shap_summary_path = os.path.join(OUTPUT_DIR, "shap_summary.png")
plt.savefig(shap_summary_path, dpi=300, bbox_inches="tight")
print(f"Το SHAP summary plot αποθηκεύτηκε εδώ: {shap_summary_path}")

plt.show()

# 8. SHAP bar plot για τον συγκεκριμένο ασθενή
patient_shap_values = selected_shap_values[0]

shap_df = pd.DataFrame({
    "Feature": feature_names,
    "SHAP Value": patient_shap_values
})

shap_df["Absolute SHAP Value"] = shap_df["SHAP Value"].abs()
shap_df = shap_df.sort_values("Absolute SHAP Value", ascending=False).head(20)

plt.figure(figsize=(12, 8))
sns.barplot(
    data=shap_df,
    x="SHAP Value",
    y="Feature",
    palette="coolwarm"
)

plt.title("Top 20 SHAP Values for This Patient")
plt.xlabel("SHAP Value")
plt.ylabel("Clinical Feature")
plt.tight_layout()

shap_bar_path = os.path.join(OUTPUT_DIR, "shap_patient_explanation.png")
plt.savefig(shap_bar_path, dpi=300, bbox_inches="tight")
print(f"Το SHAP patient explanation αποθηκεύτηκε εδώ: {shap_bar_path}")

plt.show()