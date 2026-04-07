import streamlit as st
import joblib
import numpy as np

# -------------------------------
# Load model and encoder
# -------------------------------
model = joblib.load("disease_model.pkl")
mlb = joblib.load("symptom_encoder.pkl")

# -------------------------------
# Page settings
# -------------------------------
st.set_page_config(page_title="Disease Prediction System")

st.title("🩺 Symptom-Based Disease Prediction")
st.write("Select your symptoms and click Predict")

# -------------------------------
# Symptom Selection
# -------------------------------
all_symptoms = sorted(mlb.classes_)

selected_symptoms = st.multiselect(
    "Search and select symptoms:",
    all_symptoms
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Disease"):

    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        # Convert to model format
        user_input = mlb.transform([selected_symptoms])

        # Predict probabilities
        probs = model.predict_proba(user_input)[0]

        # Get top 3 predictions
        top3_indices = np.argsort(probs)[-3:][::-1]

        st.subheader("Top Predictions:")

        for idx in top3_indices:
            disease = model.classes_[idx]
            probability = probs[idx] * 100
            st.write(f"**{disease}** — {probability:.2f}%")
