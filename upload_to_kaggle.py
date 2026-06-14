import os
import sys

try:
    import kagglehub
except ImportError:
    print("kagglehub package is not installed. Please run: pip install kagglehub")
    sys.exit(1)

# =====================================================================
# CONFIGURATION
# =====================================================================
# Replace this with your actual Kaggle username and your dataset slug:
KAGGLE_USERNAME = "balajiynspk"  # Update with your Kaggle username
DATASET_SLUG = "nse-us-stock-market-historical-data"

# Optional: If you haven't placed your 'kaggle.json' in C:\Users\balaj\.kaggle\kaggle.json,
# you can uncomment and define them as environment variables here:
# os.environ["KAGGLE_USERNAME"] = "your_kaggle_username"
# os.environ["KAGGLE_KEY"] = "your_kaggle_api_key"
# =====================================================================

handle = f"{KAGGLE_USERNAME}/{DATASET_SLUG}"
local_dataset_dir = "data"

def upload():
    # Verify that the data directory exists and contains files
    if not os.path.exists(local_dataset_dir):
        print(f"Error: Local directory '{local_dataset_dir}' does not exist.")
        sys.exit(1)
        
    csv_files = [f for f in os.listdir(local_dataset_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"Error: No CSV files found in '{local_dataset_dir}' to upload.")
        sys.exit(1)

    print("=" * 60)
    print(f"Uploading {len(csv_files)} CSV files from '{local_dataset_dir}/' to Kaggle...")
    print(f"Target Dataset Handle: {handle}")
    print("=" * 60)
    
    try:
        # Uploading dataset using kagglehub
        kagglehub.dataset_upload(
            handle=handle,
            local_directory=local_dataset_dir,
            ignore_patterns=["*.tmp", "batch_training_summary.csv", "datapackage.json"]
        )
        print("\n🎉 Success! Dataset upload to Kaggle completed.")
        print(f"View your dataset at: https://www.kaggle.com/datasets/{handle}")
    except Exception as e:
        print(f"\n❌ Error uploading dataset: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure your 'kaggle.json' is located at: C:\\Users\\balaj\\.kaggle\\kaggle.json")
        print("2. Ensure you have created the dataset on the Kaggle website first if you are updating it,")
        print("   or check your internet connection and Kaggle API quota limits.")

if __name__ == "__main__":
    upload()
