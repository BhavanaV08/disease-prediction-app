import streamlit as st
import pandas as pd
import pickle
import numpy as np
import base64
import os
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Disease Predictor", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------
# BACKGROUND IMAGE
# -------------------------------
def set_bg(image_name):
    image_path = os.path.join(BASE_DIR, image_name)
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

set_bg("background.png")

# -------------------------------
# CSS
# -------------------------------
st.markdown("""
<style>
.card {
    background: rgba(0,0,0,0.92);
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 20px;
    text-align: center;
}

h1, h2, h3 {
    text-align: center;
    color: white;
}

p {
    color: white;
    font-size:16px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODELS
# -------------------------------
rf_model = pickle.load(open(os.path.join(BASE_DIR,"rf_model.pkl"),"rb"))
lr_model = pickle.load(open(os.path.join(BASE_DIR,"lr_model.pkl"),"rb"))
nb_model = pickle.load(open(os.path.join(BASE_DIR,"nb_model.pkl"),"rb"))
le = pickle.load(open(os.path.join(BASE_DIR,"label_encoder.pkl"),"rb"))
symptoms = pickle.load(open(os.path.join(BASE_DIR,"symptoms.pkl"),"rb"))

# -------------------------------
# LOAD DATA
# -------------------------------
description = pd.read_csv(os.path.join(BASE_DIR,"symptom_Description.csv"))
precaution = pd.read_csv(os.path.join(BASE_DIR,"symptom_precaution.csv"))
severity = pd.read_csv(os.path.join(BASE_DIR,"Symptom-severity.csv"))

# ✅ CLEAN SEVERITY FILE PROPERLY
severity.columns = severity.columns.str.strip().str.lower()
severity['symptom'] = severity['symptom'].str.strip().str.lower()
severity['weight'] = pd.to_numeric(severity['weight'], errors='coerce')

# -------------------------------
# TITLE
# -------------------------------
st.markdown("""
<div class="card">
    <h1>🩺 Disease Prediction System</h1>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# INPUT
# -------------------------------
selected = st.multiselect("Select Symptoms", symptoms)

input_data = [1 if s in selected else 0 for s in symptoms]
input_df = pd.DataFrame([input_data], columns=symptoms)

# -------------------------------
# PREDICT
# -------------------------------
if st.button("Predict"):

    if len(selected) < 3:
        st.markdown("""
        <div class="card">
            <h3>⚠ Select at least 3 symptoms</h3>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # -------------------------------
    # GET PROBABILITIES
    # -------------------------------
    rf_probs = rf_model.predict_proba(input_df)[0]
    lr_probs = lr_model.predict_proba(input_df)[0]
    nb_probs = nb_model.predict_proba(input_df)[0]

    rf_probs /= rf_probs.sum()
    lr_probs /= lr_probs.sum()
    nb_probs /= nb_probs.sum()

    rf_idx = rf_probs.argsort()[-3:][::-1]
    lr_idx = lr_probs.argsort()[-3:][::-1]
    nb_idx = nb_probs.argsort()[-3:][::-1]

    rf_res = {le.inverse_transform([i])[0]: rf_probs[i]*100 for i in rf_idx}
    lr_res = {le.inverse_transform([i])[0]: lr_probs[i]*100 for i in lr_idx}
    nb_res = {le.inverse_transform([i])[0]: nb_probs[i]*100 for i in nb_idx}

    # -------------------------------
    # ENSEMBLE (AVERAGE OF 3)
    # -------------------------------
    combined = {}

    all_diseases = set(list(rf_res.keys()) +
                       list(lr_res.keys()) +
                       list(nb_res.keys()))

    for disease in all_diseases:
        rf_val = rf_res.get(disease, 0)
        lr_val = lr_res.get(disease, 0)
        nb_val = nb_res.get(disease, 0)
        combined[disease] = (rf_val + lr_val + nb_val) / 3

    final = max(combined, key=combined.get)
    confidence = round(combined[final], 2)

    # -------------------------------
    # FINAL PREDICTION
    # -------------------------------
    st.markdown(f"""
    <div class="card">
        <h2><b>🧠 Final Prediction</b></h2>
        <h1>{final}</h1>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------
    # CONFIDENCE
    # -------------------------------
    st.markdown(
        f"""
<div class="card">
<h3><b>📊 Confidence</b></h3>
<p style="font-size:20px;margin-bottom:15px;">{confidence}%</p>
<div style="background:#333;border-radius:10px;height:18px;width:100%;margin-top:10px;">
<div style="width:{confidence}%;background:#00e5ff;height:100%;border-radius:10px;"></div>
</div>
</div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------------
    # GRAPH
    # -------------------------------
    diseases = list(all_diseases)

    rf_vals = [rf_res.get(d,0) for d in diseases]
    lr_vals = [lr_res.get(d,0) for d in diseases]
    nb_vals = [nb_res.get(d,0) for d in diseases]

    x = np.arange(len(diseases))
    width = 0.25

    fig, ax = plt.subplots()
    ax.bar(x - width, rf_vals, width, label='Random Forest')
    ax.bar(x, lr_vals, width, label='Logistic Regression')
    ax.bar(x + width, nb_vals, width, label='Naive Bayes')

    ax.set_xticks(x)
    ax.set_xticklabels(diseases, rotation=30)
    ax.set_ylabel("Probability (%)")
    ax.legend()

    st.pyplot(fig)

    # -------------------------------
    # ✅ FIXED SEVERITY LOGIC
    # -------------------------------
    total_weight = 0

    for s in selected:
        symptom_key = s.strip().lower()
        row = severity[severity['symptom'] == symptom_key]

        if not row.empty:
            total_weight += row.iloc[0]['weight']

    max_possible = len(selected) * 7  # 7 is max weight in your file

    if max_possible > 0:
        severity_percent = round((total_weight / max_possible) * 100, 2)
    else:
        severity_percent = 0

    if severity_percent < 35:
        severity_text = f"🟢 Low Severity ({severity_percent}%)"
    elif severity_percent < 65:
        severity_text = f"🟡 Moderate Severity ({severity_percent}%)"
    else:
        severity_text = f"🔴 High Severity ({severity_percent}%) – Seek medical help!"

    st.markdown(f"""
    <div class="card">
        <h3><b>⚠ Severity Level</b></h3>
        <p>{severity_text}</p>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------
    # DESCRIPTION
    # -------------------------------
    desc = description[description['Disease']==final]
    if not desc.empty:
        st.markdown(f"""
        <div class="card">
            <h3><b>📄 Description</b></h3>
            <p>{desc.iloc[0]['Description']}</p>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------
    # PRECAUTIONS
    # -------------------------------
    prec = precaution[precaution['Disease']==final]
    if not prec.empty:
        precautions_html = ""
        for i in range(1,5):
            val = prec.iloc[0][f'Precaution_{i}']
            if pd.notna(val):
                precautions_html += f"<p>✔ {val}</p>"

        st.markdown(f"""
        <div class="card">
            <h3><b>🛡 Precautions</b></h3>
            {precautions_html}
        </div>
        """, unsafe_allow_html=True)
