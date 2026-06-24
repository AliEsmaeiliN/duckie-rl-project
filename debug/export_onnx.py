import os
import torch
import numpy as np

# Ensure your python path can find the local packages
import sys
sys.path.append(os.path.abspath("packages/rl_package/src"))

from models import SACActor

def export_sac_to_onnx(checkpoint_path, output_onnx_path, grayscale=True, frame_stack=4):
    device = torch.device("cpu") # Exporting on CPU is safer and cleaner
    channels = 1 if grayscale else 3
    
    print(f"Instantiating SACActor (Grayscale={grayscale}, Stack={frame_stack})...")
    actor = SACActor(grayscale=grayscale).to(device)
    
    print(f"Loading weights from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    actor.eval()
    
    # Define the precise input dimensions that the model expects: (Batch, Channels * Stack, Height, Width)
    dummy_input = torch.zeros(1, channels * frame_stack, 84, 84, dtype=torch.float32)
    
    # We name the input and output nodes so we can target them exactly in TensorRT later
    input_names = ["input_observations"]
    output_names = ["actions"]
    
    print(f"Exporting computational graph to ONNX at {output_onnx_path}...")
    torch.onnx.export(
        actor,
        dummy_input,
        output_onnx_path,
        export_params=True,        # Store the trained parameter weights inside the file
        opset_version=11,          # Standard, highly compatible ONNX operator set version
        do_constant_folding=True,  # Optimizes constants by pre-computing them
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None          # We hardlock batch size to 1 for minimum latency inference
    )
    print("ONNX Export completed successfully!")

if __name__ == "__main__":
    # Update these paths to match your local repository setup
    CHECKPOINT_PATH = "~/workspace/rl_models/sac_vu2b.cleanrl_model"
    OUTPUT_ONNX_PATH = "~/workspace/rl_models/sac_vu2b.onnx"
    
    export_sac_to_onnx(CHECKPOINT_PATH, OUTPUT_ONNX_PATH)