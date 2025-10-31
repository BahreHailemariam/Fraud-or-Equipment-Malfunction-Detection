import pandas as pd

def load_data(file_path):
    """Load system or transactional dataset."""
    df = pd.read_csv(file_path)
    print(f"✅ Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

if __name__ == "__main__":
    data = load_data("../data/system_data.csv")
