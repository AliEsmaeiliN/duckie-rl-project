import os
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
        return

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
            return

        # Sort chronologically by training steps
        df = df.sort_values(by="step")
        rewards_array = df["avg_reward"].to_numpy(dtype=np.float32)

        # Build output directory and save path
        os.makedirs(base_target_dir, exist_ok=True)
        output_file_path = os.path.join(base_target_dir, output_name)
        
        np.save(output_file_path, rewards_array)
        
        print(f"Successfully processed: {csv_path}")
        print(f"\t-> Saved to: {output_file_path} ({len(rewards_array)} entries)")

    except Exception as e:
        print(f"Failure processing {csv_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a specific evaluation CSV log into a customized NumPy format."
    )
    
    # Named/Flag arguments for clear inputs
    parser.add_argument(
        "--csv", 
        type=str, 
        required=True, 
        help="Path to the source CSV file (e.g., 'artifacts/sac_s0.csv')."
    )
    parser.add_argument(
        "--folder", 
        type=str, 
        required=True, 
        help="The destination folder name inside 'artifacts/' (e.g., 'SAC_R1')."
    )
    parser.add_argument(
        "--name", 
        type=str, 
        required=True, 
        help="The output file name (e.g., 'seed_0.npy' or 'seed_0')."
    )
    
    args = parser.parse_args()
    convert_single_csv(
        csv_path=args.csv, 
        dest_folder=args.folder, 
        output_name=args.name
    )