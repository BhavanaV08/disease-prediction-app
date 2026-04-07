import pickle
import pandas as pd
import numpy as np

# ===================== LOAD MODEL FILES =====================
model = pickle.load(open("disease_model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
mlb = pickle.load(open("mlb.pkl", "rb"))

# ===================== LOAD CSV FILES =====================
desc_df = pd.read_csv("symptom_Description.csv")
prec_df = pd.read_csv("symptom_precaution.csv")
severity_df = pd.read_csv("Symptom-severity.csv")

# Normalize column names
desc_df.columns = ["Disease", "Description"]
severity_df.columns = ["Symptom", "weight"]

# ===================== CREATE LOOKUP DICTS =====================
description_dict = dict(zip(desc_df["Disease"], desc_df["Description"]))

severity_dict = dict(
    zip(severity_df["Symptom"].str.strip(), severity_df["weight"])
)

precaution_dict = {}
for _, row in prec_df.iterrows():
    precaution_dict[row["Disease"]] = [
        str(row[col]) for col in prec_df.columns[1:]
        if pd.notna(row[col])
    ]

# ===================== SYMPTOM INPUT =====================
print("\nAnswer symptoms with y/n:\n")

user_symptoms = []
symptom_count = 0
severity_score = 0

for symptom in mlb.classes_:
    ans = input(f"Do you have {symptom.replace('_',' ')}? (y/n): ").lower()
    if ans == "y":
        user_symptoms.append(symptom)
        symptom_count += 1
        severity_score += severity_dict.get(symptom, 0)

print("\nSymptoms provided      :", symptom_count)

# ===================== PREDICTION =====================
input_vector = mlb.transform([user_symptoms])
proba = model.predict_proba(input_vector)[0]

pred_index = np.argmax(proba)
predicted_disease = label_encoder.inverse_transform([pred_index])[0]
confidence = proba[pred_index] * 100

# ===================== OUTPUT =====================
print("\n🩺 Predicted Disease :", predicted_disease)
print(f"📊 Confidence       : {confidence:.2f}%")

# ===================== DESCRIPTION =====================
print("\n📖 Disease Description:")
print(description_dict.get(predicted_disease, "Description not available"))

# ===================== PRECAUTIONS =====================
print("\n🛡️ Precautions:")
precautions = precaution_dict.get(predicted_disease, [])

if precautions:
    for i, p in enumerate(precautions, 1):
        print(f"{i}. {p}")
else:
    print("Precautions not available")

# ===================== SEVERITY RESULT =====================
print("\n⚠️ Severity Analysis:")
if severity_score <= 20:
    print("Low severity – Home care is usually sufficient.")
elif severity_score <= 40:
    print("Medium severity – Consult a doctor if symptoms persist.")
else:
    print("High severity – Seek medical attention immediately.")

print("\n✅ Prediction completed successfully.")
