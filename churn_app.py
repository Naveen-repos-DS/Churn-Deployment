import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================
# Load model & columns
# =========================
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))  # saved X.columns

st.set_page_config(page_title="Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict churn")

# =========================
# USER INPUTS
# =========================

st.subheader("Customer Information")

# Numeric inputs
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
monthly = st.number_input("Monthly Charges", min_value=0.0, value=50.0)

# Derived feature (optional)
total = tenure * monthly

# Categorical inputs
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

payment = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)

security = st.selectbox("Online Security", ["Yes", "No"])
support = st.selectbox("Tech Support", ["Yes", "No"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

# =========================
# PREPROCESS FUNCTION
# =========================

def preprocess():
    input_dict = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment,
        "OnlineSecurity": security,
        "TechSupport": support,
        "PaperlessBilling": paperless
    }

    input_df = pd.DataFrame([input_dict])

    # One-hot encode
    input_df = pd.get_dummies(input_df)

    # Align with training columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    return input_df


# =========================
# PREDICTION
# =========================

if st.button("Predict Churn"):

    input_df = preprocess()

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to CHURN\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Customer is likely to STAY\n\nProbability: {probability:.2f}")

    # Extra insight
    st.subheader("Confidence Level")
    st.progress(int(probability * 100))
