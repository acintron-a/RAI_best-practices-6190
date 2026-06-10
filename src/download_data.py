# download_data.py
import os
from fairlearn.datasets import fetch_diabetes_hospital

# Resolve absolute path for the project data directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(script_dir), "data")

# Ensure the data directory exists
os.makedirs(data_dir, exist_ok=True)


print("Fetching dataset from Fairlearn...")
# This fetches the data structure without loading it entirely into a massive Pandas DataFrame view
bunch = fetch_diabetes_hospital(as_frame=True)
df = bunch.frame

print("Saving raw data to local external storage (data/diabetes_raw.csv)...")
df.to_csv(os.path.join(data_dir, "diabetes_raw.csv"), index=False)
print("Download complete!")
