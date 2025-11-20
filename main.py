import kagglehub
import pandas as pd
import os

# Download latest version
path = kagglehub.dataset_download("abdullahorzan/moodify-dataset")

print("Path to dataset files:", path)

# Find the CSV file in the dataset directory
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

if csv_files:
    # Load the first CSV file found
    csv_file_path = os.path.join(path, csv_files[0])
    df = pd.read_csv(csv_file_path)
    print(f"Loaded dataset: {csv_files[0]}")
    print(df.head())
else:
    print("No CSV file found in the downloaded dataset.")
