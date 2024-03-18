import pandas as pd
import random
import os

# Define the number of rows to randomly extract from each sample
num_rows_to_extract = 3000

# Set a random seed for reproducibility
random.seed(42) #42

# 
folder_path = "data/minerals"
sample_file_names = [
    "FeOx_1-2023-04-19-111632-12 A_e.csv",
    "FeOx_2-2023-04-19-111710-13 A_e.csv",
    "FeOx_3-2023-04-19-111751-14 A_e.csv",
]

samples = []
feature_names = None

for file_name in sample_file_names:
    file_path = os.path.join(folder_path, file_name)
    sample = pd.read_csv(file_path, header=0)
    
    if feature_names is None:
        feature_names = sample.columns.tolist()  
    
    # Randomly extractrows (excluding the first row)
    extracted_rows = sample.iloc[1:].sample(n=num_rows_to_extract, random_state=42)
    samples.append(extracted_rows)

combined_sample = pd.concat(samples, ignore_index=True)
combined_sample.columns = feature_names
combined_sample['class'] = 3 # set class names 

# Save the combined sample
output_file_path = os.path.join(folder_path, "FeOx_9k.csv")
combined_sample.to_csv(output_file_path, index=False)

print(combined_sample.head())
print(combined_sample.shape)