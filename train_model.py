# =========================================
# 1. Imports
# =========================================
import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# 2. Fix Randomness
# =========================================
random.seed(42)
np.random.seed(42)

# =========================================
# 3. Load Dataset
# =========================================
df = pd.read_csv("dataset.csv")

symptom_cols = [col for col in df.columns if col.startswith("Symptom")]

df['symptom_list'] = df[symptom_cols].values.tolist()
df['symptom_list'] = df['symptom_list'].apply(
    lambda x: [sym.strip().lower() for sym in x if pd.notna(sym)]
)

# =========================================
# 4. Train-Test Split (Stratified)
# =========================================
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["Disease"],
    random_state=42
)

# =========================================
# 5. Data Augmentation
# =========================================
all_symptoms = list(set(df['symptom_list'].explode()))

def drop_symptoms(symptoms, drop_rate=0.5):
    return [s for s in symptoms if random.random() > drop_rate]

def add_noise(symptoms, noise_rate=0.3):
    n = max(1, int(len(symptoms) * noise_rate))
    noise = random.sample(all_symptoms, min(n, len(all_symptoms)))
    return list(set(symptoms + noise))

common_symptoms = ["fever", "fatigue", "headache"]

def inject_common(symptoms):
    if random.random() < 0.5:
        symptoms.append(random.choice(common_symptoms))
    return list(set(symptoms))

def augment(symptoms):
    symptoms = drop_symptoms(symptoms, 0.5)
    symptoms = add_noise(symptoms, 0.3)
    symptoms = inject_common(symptoms)
    return list(set(symptoms))

train_df["symptom_list"] = train_df["symptom_list"].apply(augment)
test_df["symptom_list"]  = test_df["symptom_list"].apply(augment)

# =========================================
# 6. MultiLabel Binarizer
# =========================================
mlb = MultiLabelBinarizer()
mlb.fit(train_df["symptom_list"])

X_train = mlb.transform(train_df["symptom_list"])
X_test  = mlb.transform(test_df["symptom_list"])

y_train = train_df["Disease"]
y_test  = test_df["Disease"]

print("Total Unique Symptoms:", len(mlb.classes_))

# =========================================
# 7. Train Models
# =========================================

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42
)
rf.fit(X_train, y_train)

# Logistic Regression
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)

# =========================================
# 8. Evaluation Function
# =========================================
def evaluate_model(model, name):
    pred = model.predict(X_test)

    print(f"\n========== {name} ==========")
    print("Accuracy:", accuracy_score(y_test, pred))
    print("F1 Score (Weighted):", f1_score(y_test, pred, average='weighted'))
    print("Train Accuracy:", model.score(X_train, y_train))
    print("Test Accuracy :", model.score(X_test, y_test))

# Evaluate all models
evaluate_model(rf, "Random Forest")
evaluate_model(lr, "Logistic Regression")
evaluate_model(nb, "Naive Bayes")

# =========================================
# 9. Cross Validation (RF)
# =========================================
rf_cv = cross_val_score(rf, X_train, y_train, cv=5)
print("\nRandom Forest Cross-Val Accuracy:", rf_cv.mean())

# =========================================
# 10. Feature Importance (RF)
# =========================================
importances = rf.feature_importances_

feature_importance = sorted(
    zip(mlb.classes_, importances),
    key=lambda x: -x[1]
)

print("\nTop 15 Important Symptoms:")
for sym, score in feature_importance[:15]:
    print(f"{sym} -> {score:.4f}")
