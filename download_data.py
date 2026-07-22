import pandas as pd
import urllib.request
import os

# Create the data/raw folder if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

# Download the dataset
url = "https://github.com/visharadsingh/telco_customer_churn_analysis/raw/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"
urllib.request.urlretrieve(url, "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Load and verify it
df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print("Dataset downloaded successfully!")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn info:")
print(df.info())