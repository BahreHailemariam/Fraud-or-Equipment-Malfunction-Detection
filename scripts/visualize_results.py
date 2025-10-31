import matplotlib.pyplot as plt

def visualize_results(data):
    """Plot sensor readings and highlight anomalies."""
    plt.figure(figsize=(10, 6))
    plt.plot(data['timestamp'], data['sensor_reading'], label='Sensor Reading')
    anomalies = data[data['label'] == 'Suspicious']
    plt.scatter(anomalies['timestamp'], anomalies['sensor_reading'], color='red', label='Anomaly')
    plt.xlabel('Timestamp')
    plt.ylabel('Sensor Reading')
    plt.title('Sensor Data with Anomalies')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("../reports/anomaly_results.csv")
    visualize_results(df)
