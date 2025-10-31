import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """Clean missing values and normalize numeric columns."""
    df = df.fillna(df.mean(numeric_only=True))
    numeric_features = df.select_dtypes(include=['float64', 'int64'])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_features)
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_features.columns)
    print("✅ Data preprocessed successfully.")
    return scaled_df

if __name__ == "__main__":
    from load_data import load_data
    df = load_data("../data/system_data.csv")
    preprocessed = preprocess_data(df)
