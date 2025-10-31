import pandas as pd

def create_features(df):
    """Generate time-based and statistical features."""
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['sensor_to_transaction_ratio'] = df['sensor_reading'] / (df['transaction_amount'] + 1e-5)
    print("✅ Feature engineering completed.")
    return df

if __name__ == "__main__":
    from load_data import load_data
    df = load_data("../data/system_data.csv")
    engineered = create_features(df)
