import os
import sys
import json
import zipfile
import shutil
import subprocess

# =====================================================================
# CONFIGURATION
# =====================================================================
KAGGLE_USERNAME = "ybalaji098"
DATASET_SLUG = "nse-us-stock-market-historical-data"
local_dataset_dir = "data"
upload_temp_dir = "upload_temp"
# =====================================================================

handle = f"{KAGGLE_USERNAME}/{DATASET_SLUG}"

def upload():
    # Verify that the data directory exists and contains files
    if not os.path.exists(local_dataset_dir):
        print(f"Error: Local directory '{local_dataset_dir}' does not exist.")
        sys.exit(1)
        
    csv_files = [f for f in os.listdir(local_dataset_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"Error: No CSV files found in '{local_dataset_dir}' to upload.")
        sys.exit(1)

    # 1. Load credentials from C:\Users\balaj\.kaggle\kaggle.json or environment
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
    api_token = None
    
    if os.path.exists(kaggle_json_path):
        try:
            with open(kaggle_json_path, 'r') as f:
                creds = json.load(f)
                api_token = creds.get("key")
        except Exception as e:
            print(f"Warning: Could not parse {kaggle_json_path}: {e}")
            
    if not api_token:
        # Check if environment variable is set
        api_token = os.environ.get("KAGGLE_API_TOKEN")

    if not api_token:
        print("Error: Kaggle API token not found.")
        print("Please ensure kaggle.json exists in C:\\Users\\balaj\\.kaggle\\kaggle.json or KAGGLE_API_TOKEN env var is set.")
        sys.exit(1)

    # 2. Create temporary upload directory
    os.makedirs(upload_temp_dir, exist_ok=True)

    # 3. Create zip archive of CSV files
    zip_path = os.path.join(upload_temp_dir, "stock_historical_data.zip")
    print("=" * 60)
    print(f"Creating zip archive of {len(csv_files)} CSV files: {zip_path}...")
    print("=" * 60)
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for i, file in enumerate(csv_files):
                file_path = os.path.join(local_dataset_dir, file)
                zipf.write(file_path, arcname=file)
                if (i + 1) % 200 == 0 or (i + 1) == len(csv_files):
                    print(f"Zipped {i + 1}/{len(csv_files)} files...")
        print(f"Zip archive created. Size: {os.path.getsize(zip_path) / (1024 * 1024):.2f} MB\n")
    except Exception as e:
        print(f"Error creating zip archive: {e}")
        shutil.rmtree(upload_temp_dir, ignore_errors=True)
        sys.exit(1)

    # 4. Set environment variable for Kaggle CLI
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = api_token

    # 5. Ensure dataset-metadata.json exists in the upload_temp directory
    metadata_path = os.path.join(upload_temp_dir, "dataset-metadata.json")
    metadata = {
        "title": "NSE and US Stock Market Historical Data",
        "id": handle,
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # 6. Check if dataset already exists
    print("Checking if dataset already exists on Kaggle...")
    status_cmd = subprocess.run(
        ["kaggle", "datasets", "status", handle],
        env=env,
        capture_output=True,
        text=True
    )
    
    dataset_exists = (status_cmd.returncode == 0)

    if dataset_exists:
        print("Dataset exists on Kaggle. Creating a new version...")
        upload_cmd = ["kaggle", "datasets", "version", "-p", upload_temp_dir, "-m", "Updated dataset version"]
    else:
        print("Dataset does not exist on Kaggle. Creating a new dataset...")
        upload_cmd = ["kaggle", "datasets", "create", "-p", upload_temp_dir]

    # 7. Perform the upload
    print(f"Running command: {' '.join(upload_cmd)}")
    success = False
    try:
        # Run and stream the output to the console
        process = subprocess.Popen(
            upload_cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream stdout/stderr in real-time
        for line in process.stdout:
            print(line, end="")
            
        process.wait()
        
        if process.returncode == 0:
            print("\nSuccess! Dataset upload to Kaggle completed.")
            print(f"View your dataset at: https://www.kaggle.com/datasets/{handle}")
            success = True
        else:
            print(f"\nError: Upload command exited with code {process.returncode}")
            
    except Exception as e:
        print(f"\nError running upload command: {e}")
    finally:
        # Clean up temporary directory
        print("\nCleaning up temporary upload directory...")
        shutil.rmtree(upload_temp_dir, ignore_errors=True)
        print("Cleanup completed.")
        
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    upload()
