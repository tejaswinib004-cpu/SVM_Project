import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# -------------------------------
# Load and preprocess data
# -------------------------------
df = pd.read_csv("medical_Diagnosis_dataset.csv")

# Drop unwanted features
df = df.drop(["feature_6", "feature_7", "feature_8", "feature_9", "feature_10"], axis=1)

# One-hot encoding
df = pd.get_dummies(df, columns=["smoking_status"])

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
model.fit(X_train, y_train)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🏥 Medical Diagnosis Prediction")

st.write("Enter patient details:")

# Inputs
glucose = st.number_input("Glucose Level")
bp = st.number_input("Blood Pressure")
chol = st.number_input("Cholesterol")
bmi = st.number_input("BMI")
age = st.number_input("Age")

smoking = st.selectbox("Smoking Status", ["Former", "Non-smoker", "Smoker"])

# Convert smoking to one-hot
smoking_former = 1 if smoking == "Former" else 0
smoking_non = 1 if smoking == "Non-smoker" else 0
smoking_smoker = 1 if smoking == "Smoker" else 0

# Prediction
if st.button("Predict"):
    input_data = np.array([[
        glucose, bp, chol, bmi, age,
        smoking_former, smoking_non, smoking_smoker
    ]])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Disease")
    else:
        st.success("✅ Low Risk (Healthy)")