import streamlit as st
import pandas as pd
import joblib
from scripts.preprocess_data import preprocess_data
from scripts.detect_anomalies import detect_anomalies

st.title("🚨 Fraud or Equipment Malfunction Detection Dashboard")

uploaded_file = st.file_uploader("Upload CSV Data", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    model = joblib.load("models/trained_model.pkl")
    X = preprocess_data(df)
    results = detect_anomalies(model, X, df)
    st.subheader("🔍 Detected Anomalies")
    st.dataframe(results)
