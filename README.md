# 🏥 Medical Diagnosis Prediction using SVM & Streamlit

This project predicts whether a person is at risk of a disease using machine learning.  
A Support Vector Machine (SVM) model is trained on a medical dataset and deployed using a Streamlit web application.

---

## 📊 Dataset

The dataset contains **1000 records** with medical features such as:

- glucose_level  
- blood_pressure  
- cholesterol  
- bmi  
- age  
- smoking_status (categorical)  
- target (0 = No disease, 1 = Disease)

---

## 🔍 Data Preprocessing

The following preprocessing steps were performed:

- Checked for missing values (none found)
- Converted **smoking_status** into numerical form using one-hot encoding
- Scaled and adjusted the **age** feature to a meaningful range
- Handled outliers using the **IQR method**
- Removed irrelevant features:
  - feature_6  
  - feature_7  
  - feature_8  
  - feature_9  
  - feature_10  

👉 This improved model clarity and interpretability

---

## 🧠 Model Used

- **Support Vector Machine (SVM)**
  - Kernel: RBF
  - Class weight: Balanced

The model was trained using an 80-20 train-test split.

---

## 🌐 Streamlit Web Application

A simple web app was created using Streamlit.

### Features:
- User inputs patient details
- Model predicts disease risk instantly
- Clean and interactive interface

### Input Fields:
- Glucose Level  
- Blood Pressure  
- Cholesterol  
- BMI  
- Age  
- Smoking Status  

---

# ⚙️ Technologies Used
Python, 
Pandas, 
NumPy, 
Scikit-learn, 
Streamlit.

---

## ⚙️ How the App Works

1. Loads dataset
2. Applies same preprocessing (removes unwanted features + encoding)
3. Trains SVM model
4. Takes user input from UI
5. Converts input into model format
6. Predicts output:
   - ⚠️ High Risk  
   - ✅ Low Risk  

---

## 📈 Result

- Successfully built a classification model
- Deployed using Streamlit for real-time prediction

---

## 📌 Conclusion

This project demonstrates a complete machine learning pipeline:
- Data preprocessing  
- Feature selection  
- Model training  
- Deployment using Streamlit  

It highlights the importance of removing irrelevant features and preparing clean data for better predictions.
