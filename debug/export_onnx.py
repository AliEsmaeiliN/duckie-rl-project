#!/usr/bin/env python3
import os
import sys
import torch

from models import SACActor, TD3Actor

def export_model_file(base_dir, model_filename):
    """Handles the extraction and ONNX conversion for a single model file."""
    checkpoint_path = os.path.join(base_dir, model_filename)
    
    if model_filename.endswith(".cleanrl_model"):
        onnx_filename = model_filename.replace(".cleanrl_model", ".onnx")
    else:
        base_name = os.path.splitext(model_filename)[0]
        model_filename = base_name + ".cleanrl_model"
        checkpoint_path = os.path.join(base_dir, model_filename)
        onnx_filename = base_name + ".onnx"

    output_onnx_path = os.path.join(base_dir, onnx_filename)

    if not os.path.exists(checkpoint_path):
        print(f"Error: Could not find checkpoint file at {checkpoint_path}")
        return False

    prefix = model_filename[:3].lower()
    if prefix == "sac":
        algo_type = "sac"
    elif prefix == "td3":
        algo_type = "td3"
    else:
        print(f"Error: Unknown algorithm prefix '{model_filename[:3]}'. Filename must start with 'sac' or 'td3'.")
        return False

    grayscale = True
    frame_stack = 4
    channels = 4 if grayscale else 12
    device = torch.device("cpu") # CPU trace keeps the file clean of runtime device flags

    print("\n" + "-"*50)
    print(f"Detected Algorithm : {algo_type.upper()}")
    print(f"Reading Weights    : {checkpoint_path}")
    print(f"Target Output      : {output_onnx_path}")
    print("-"*50)

    print("Instantiating Model Architecture...")
    if algo_type == "sac":
        actor = SACActor(grayscale=grayscale).to(device)
    else:
        actor = TD3Actor(grayscale=grayscale).to(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        actor.load_state_dict(checkpoint['actor_state_dict'])
        actor.eval()

        dummy_input = torch.randint(
            low=0, 
            high=255, 
            size=(1, channels, 42, 42), 
            dtype=torch.uint8
        ).to(device)
        
        input_names = ["input_observations"]
        output_names = ["actions"]

        print("Tracing computational network graph and exporting...")
        torch.onnx.export(
            actor,
            dummy_input,
            output_onnx_path,
            export_params=True,        # Embed the loaded parameter weights straight inside the graph binary
            opset_version=11,          # Highly stable ONNX runtime configuration
            do_constant_folding=True,  # Optimizes network parameters by collapsing constants
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=None          # Hardlocks batch size to 1 for the absolute lowest inference latency
        )
        print(f"Success! Model successfully exported to: {output_onnx_path}\n")
        return True
    except Exception as e:
        print(f"Error processing {model_filename}: {e}")
        return False


def main():
    base_dir = os.path.expanduser("~/workspace/rl_models/")

    if "--all" in sys.argv:
        if not os.path.exists(base_dir):
            print(f"Error: Base directory {base_dir} does not exist.")
            return

        model_files = [f for f in os.listdir(base_dir) if f.endswith(".cleanrl_model")]
        
        if not model_files:
            print(f"No '.cleanrl_model' files found in {base_dir}")
            return

        print(f"Found {len(model_files)} model(s) in {base_dir}. Batch exporting started...")
        for model_file in model_files:
            export_model_file(base_dir, model_file)
        
        print("Batch export completed completed.")
        return

    if len(sys.argv) > 1:
        model_input = sys.argv[1]
    else:
        print("\n=== Duckiebot RL-to-ONNX Exporter ===")
        model_input = input("Input model filename (e.g., sac_v10.cleanrl_model): ").strip()

    if not model_input:
        print("Error: No model file name specified. Exiting.")
        return

    if model_input.endswith(".cleanrl_model"):
        model_filename = model_input
    else:
        model_filename = model_input + ".cleanrl_model"

    export_model_file(base_dir, model_filename)

if __name__ == "__main__":
    main()