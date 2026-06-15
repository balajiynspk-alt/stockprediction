import os
import time
import sys
from kaggle.api.kaggle_api_extended import KaggleApi

# Use the verified API token
os.environ["KAGGLE_API_TOKEN"] = "KGAT_b23502b3f20a0303ba5c36660bacfc4e"

def run():
    api = KaggleApi()
    api.authenticate()
    
    dataset_id = "ybalaji098/nse-us-stock-market-historical-data"
    print(f"Monitoring status for: {dataset_id}")
    print("Press Ctrl+C to stop checking manually.\n")
    
    attempts = 0
    while attempts < 40:  # Check for ~10 minutes max
        try:
            # Get list of user's datasets
            datasets = [d.ref for d in api.dataset_list(mine=True)]
            if dataset_id in datasets:
                print("\n============================================================")
                print("SUCCESS! Your dataset is fully processed and available!")
                print(f"Link: https://www.kaggle.com/datasets/{dataset_id}")
                print("============================================================")
                sys.exit(0)
            else:
                print(f"[{attempts+1}] Kaggle is still extracting and processing files...")
        except Exception as e:
            print(f"[{attempts+1}] Waiting for Kaggle server to register dataset...")
            
        attempts += 1
        time.sleep(15)
        
    print("\nTimeout: Kaggle is taking longer than expected to process. Please check the website directly in a few minutes.")

if __name__ == "__main__":
    run()
