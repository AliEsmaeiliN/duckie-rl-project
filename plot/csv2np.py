import os
import re
import argparse
import pandas as pd
import numpy as np

def convert_single_csv(csv_path, dest_folder, output_name):
    """
    Converts a single specified CSV file into a .npy file inside artifacts/<dest_folder>/
    """
    source_root_dir = "artifacts"
    base_target_dir = os.path.join(source_root_dir, dest_folder)
    
    # Verify input file exists
    if not os.path.exists(csv_path):
        print(f"Error: The input CSV file '{csv_path}' does not exist.")
        return False

    # Ensure output filename ends with .npy
    if not output_name.endswith(".npy"):
        output_name += ".npy"

    try:
        # Load and clean columns
        df = pd.read_csv(csv_path)
        df.columns = [col.strip() for col in df.columns]

        if "step" not in df.columns or "avg_reward" not in df.columns:
            print(f"Error: Missing required columns 'step' or 'avg_reward' in {csv_path}.")
            print(f"Found columns: {list(df.columns)}")
            return False

        # Sort chronologically by training steps
        df = df.sort_values(by="step")
        rewards_array = df["avg_reward"].to_numpy(dtype=np.float32)

        # Build output directory and save path
        os.makedirs(base_target_dir, exist_ok=True)
        output_file_path = os.path.join(base_target_dir, output_name)
        
        np.save(output_file_path, rewards_array)
        
        print(f"Successfully processed: {csv_path}")
        print(f"\t-> Saved to: {output_file_path} ({len(rewards_array)} entries)")
        return True

    except Exception as e:
        print(f"Failure processing {csv_path}: {e}")
        return False


def convert_all_artifacts():
    """
    Automatically parses all CSV files in the 'artifacts' directory and converts them.
    Maps naming pattern e.g., 'sac_r1_s1_p.csv' -> folder: 'SAC_r1_p', filename: 'seed_1.npy'
    """
    source_root_dir = "artifacts"
    
    if not os.path.exists(source_root_dir):
        print(f"Error: '{source_root_dir}' directory not found.")
        return

    # Regex to match: (algo)_(run)_s(seed)_(type).csv
    # e.g., sac_r1_s1_p.csv -> algo=sac, run=r1, seed=1, type=p
    pattern = re.compile(r"^([a-zA-Z0-9]+)_(r\d+)_s(\d+)_([a-zA-Z0-9]+)\.csv$")
    
    files = os.listdir(source_root_dir)
    csv_files = [f for f in files if f.endswith(".csv")]
    
    if not csv_files:
        print("No CSV files found in artifacts to process.")
        return

    print(f"Found {len(csv_files)} CSV files. Starting auto-conversion...\n")
    success_count = 0

    for file_name in csv_files:
        match = pattern.match(file_name)
        if not match:
            print(f"Skipping {file_name}: Does not match expected pattern (e.g., sac_r1_s1_p.csv)")
            continue
            
        algo, run, seed, file_type = match.groups()
        
        # Format destination folder (e.g., sac -> SAC, combined with _r1_p)
        dest_folder = f"{algo.upper()}_{run}_{file_type}"
        # Format output filename (e.g., seed_1)
        output_name = f"seed_{seed}"
        
        full_csv_path = os.path.join(source_root_dir, file_name)
        
        if convert_single_csv(full_csv_path, dest_folder, output_name):
            success_count += 1

    print(f"\nAuto-conversion complete. Successfully processed {success_count}/{len(csv_files)} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert evaluation CSV logs into a customized NumPy format."
    )
    
    parser.add_argument("--csv", type=str, help="Path to a single source CSV file.")
    parser.add_argument("--folder", type=str, help="Destination folder name inside 'artifacts/'.")
    parser.add_argument("--name", type=str, help="Output file name.")
    
    args = parser.parse_args()
    
    if args.csv and args.folder and args.name:
        convert_single_csv(csv_path=args.csv, dest_folder=args.folder, output_name=args.name)
    else:
        print("No manual arguments provided. Defaulting to automatic batch conversion...")
        convert_all_artifacts()