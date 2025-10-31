import joblib
import pandas as pd

def detect_anomalies(model, X, data):
    """Predict anomalies and tag them."""
    preds = model.predict(X)
    data['anomaly'] = preds
    data['label'] = data['anomaly'].map({1: 'Normal', -1: 'Suspicious'})
    suspicious = data[data['label'] == 'Suspicious']
    print(f"⚠️ Detected {suspicious.shape[0]} anomalies.")
    return suspicious

if __name__ == "__main__":
    from preprocess_data import preprocess_data
    from load_data import load_data
    model = joblib.load("../models/trained_model.pkl")
    df = load_data("../data/system_data.csv")
    X = preprocess_data(df)
    anomalies = detect_anomalies(model, X, df)
    anomalies.to_csv("../reports/anomaly_results.csv", index=False)
