import pandas as pd
import os

# 
folder_path = "data/minerals"
file_names = ["CK_9k.csv", "CM_9k.csv", "Q12_9k.csv", "FeOx_9k.csv"]

dfs = []
for file_name in file_names:

    file_path = os.path.join(folder_path, file_name)
    df = pd.read_csv(file_path)
    dfs.append(df)

# Combine 
combined_df = pd.concat(dfs, ignore_index=True)

# Save 
output_file_path = os.path.join(folder_path, "minerals_36k.csv")
combined_df.to_csv(output_file_path, index=False)

print(combined_df.head())
print(combined_df.shape)
