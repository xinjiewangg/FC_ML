import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# List of file paths for the new datasets
file_paths = [
    "data/2_Quantification/Natural_MPs_045_500625/MPs_01mlMPs_9k_test.csv",
    "data/2_Quantification/Natural_MPs_045_500625/MPs_9k_test.csv",
    "data/2_Quantification/Natural_MPs_045_500625/01mlMPs_9k_test.csv",

    "data/2_Quantification/Natural_MPs_045_500625/Natural water_5_3-2023-05-05-17_e.csv",
    "data/2_Quantification/Natural_MPs_045_500625/Natural water_5_2-2023-05-05-17_e.csv",
    "data/2_Quantification/Natural_MPs_045_500625/Natural water_5_1-2023-05-05-17_e.csv",

    "data/2_Quantification/Natural_MPs_045_500625/Natural water_2.5_3-2023-05-05-_e.csv",
    "data/2_Quantification/Natural_MPs_045_500625/Natural water_2.5_2-2023-05-05-_e.csv",
    "data/2_Quantification/Natural_MPs_045_500625/Natural water_2.5_1-2023-05-05-_e.csv",

    "data/2_Quantification/Natural_MPs_045_500625/Natural water_1_3-2023-05-05-16_e.csv",
    "data/2_Quantification/Natural_MPs_045_500625/Natural water_1_2-2023-05-05-16_e.csv",
    "data/2_Quantification/Natural_MPs_045_500625/Natural water_1_1-2023-05-05-16_e.csv",

    "data/2_Quantification/Natural_MPs_045_500625/Natural water_0.1_2-2023-05-05-_e.csv",
    "data/2_Quantification/Natural_MPs_045_500625/Natural water_0.1_1-2023-05-05-_e.csv",
    #"data/2_Quantification/Natural_MPs_045_500625/50Natural water_0.1_1-2023-05-0_e.csv"

]

# Load the trained model
rf_classifier = joblib.load('trained_rf_classifier_NW045_01.pkl')
scaler = joblib.load('scaler_NW045_01.pkl')


results = []

for file_path in file_paths:
    
    new_data = pd.read_csv(file_path)
    
    print(f"Head of the dataset: {file_path}")
    print(new_data.head())
    
    features_new = new_data.iloc[:, :]

    # Preprocess the new data (scaling)
    features_new_scaled = scaler.transform(features_new)

    print("new_data_Standard")
    print(features_new_scaled[:5])

    # Predict 
    predicted_labels = rf_classifier.predict(features_new_scaled)

    # Count 
    class_0_count = sum(predicted_labels == 0)
    class_1_count = sum(predicted_labels == 1)

    total_events = len(new_data)

    results.append({
        "File Path": file_path,
        "Total Events": total_events,
        "Class 0 Count": class_0_count,
        "Class 1 Count": class_1_count
    })

# Print or process the results
for result in results:
    print(f"Results for dataset: {result['File Path']}")
    print(f"Total number of events in the dataset: {result['Total Events']}")
    print(f"Number of events predicted as Class 0: {result['Class 0 Count']}")
    print(f"Number of events predicted as Class 1: {result['Class 1 Count']}")
