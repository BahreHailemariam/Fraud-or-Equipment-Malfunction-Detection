from sklearn.ensemble import IsolationForest
import joblib

def train_model(X):
    """Train Isolation Forest model for anomaly detection."""
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)
    joblib.dump(model, "../models/trained_model.pkl")
    print("✅ Model trained and saved to models/trained_model.pkl.")
    return model

if __name__ == "__main__":
    import pandas as pd
    from preprocess_data import preprocess_data
    from load_data import load_data

    df = load_data("../data/system_data.csv")
    X = preprocess_data(df)
    train_model(X)
