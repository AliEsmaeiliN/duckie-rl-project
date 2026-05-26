import os
import torch
import gymnasium as gym
import numpy as np

from rl.sac_continuous_action import Actor as SimSACActor
from rl.td3_continuous_action import Actor as SimTD3Actor
from rl.cnn_architectures import ImpalaCNN
from models import SACActor as RealSACActor
from models import TD3Actor as RealTD3Actor


class DummyEnv:
    def __init__(self):
        self.single_observation_space = gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)


def diagnose_saved_model(checkpoint_path: str, algo_choice: str):
    print("\n" + "=" * 70)
    print(f"  DIAGNOSING BUNDLED CHECKPOINT: {checkpoint_path}")
    print("=" * 70)

    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Could not find checkpoint file at '{checkpoint_path}'")
        print("=" * 70 + "\n")
        return

    try:
        # Load the dictionary bundle safely
        checkpoint_bundle = torch.load(checkpoint_path, map_location="cpu")
        
        # Metadata Extraction
        print("\n Experiment Metadata:")
        print(f"   - Environment ID: {checkpoint_bundle.get('env_id', 'Unknown')}")
        print(f"   - Global Step   : {checkpoint_bundle.get('global_step', 'Unknown')}")
        print(f"   - Run Notes     : {checkpoint_bundle.get('run_notes', 'None')}")
        print(f"   - Env Params    : {checkpoint_bundle.get('env_params', 'None')}")
        print("-" * 70)

        # Extract the actual actor state dictionary
        if "actor_state_dict" not in checkpoint_bundle:
            print("❌ Critical Error: 'actor_state_dict' key missing from the checkpoint file bundle.")
            print("=" * 70 + "\n")
            return
            
        actor_state_dict = checkpoint_bundle["actor_state_dict"]

        print("Inner Actor Weights:")
        saved_scale = actor_state_dict.get("action_scale", None)
        saved_bias = actor_state_dict.get("action_bias", None)
        
        print(f"   - Saved 'action_scale': {saved_scale.tolist() if saved_scale is not None else '❌ NOT FOUND IN ACTOR'}")
        print(f"   - Saved 'action_bias' : {saved_bias.tolist() if saved_bias is not None else '❌ NOT FOUND IN ACTOR'}")
        print("-" * 70)

        dummy_env = DummyEnv()

        if algo_choice == "sac":
            print(" Simulation 'SACActor' Class Integration:")
            sim_actor = SimSACActor(env=dummy_env, grayscale=True)
            print(f"   - Before Load -> scale: {sim_actor.action_scale.tolist()}, bias: {sim_actor.action_bias.tolist()}")
            sim_actor.load_state_dict(actor_state_dict)
            print(f"   - After Load  -> scale: {sim_actor.action_scale.tolist()}, bias: {sim_actor.action_bias.tolist()}")
            print("-" * 70)

            print(" Deployment 'SACActor' Class Integration:")
            bot_actor = RealSACActor(grayscale=True)
            print(f"   - Before Load -> scale: {bot_actor.action_scale.tolist()}, bias: {bot_actor.action_bias.tolist()}")
            bot_actor.load_state_dict(actor_state_dict, strict=True)
            print(f"   - After Load  -> scale: {bot_actor.action_scale.tolist()}, bias: {bot_actor.action_bias.tolist()}")
            print("=" * 70 + "\n")

        elif algo_choice == "td3":
            print(" Simulation 'TD3Actor' Class Integration:")
            sim_actor = SimTD3Actor(env=dummy_env)
            print(f"   - Before Load -> scale: {sim_actor.action_scale.tolist()}, bias: {sim_actor.action_bias.tolist()}")
            sim_actor.load_state_dict(actor_state_dict)
            print(f"   - After Load  -> scale: {sim_actor.action_scale.tolist()}, bias: {sim_actor.action_bias.tolist()}")
            print("-" * 70)

            print(" Deployment 'TD3Actor' Class Integration:")
            bot_actor = RealTD3Actor(grayscale=True)
            print(f"   - Before Load -> scale: {bot_actor.action_scale.tolist()}, bias: {bot_actor.action_bias.tolist()}")
            bot_actor.load_state_dict(actor_state_dict, strict=True)
            print(f"   - After Load  -> scale: {bot_actor.action_scale.tolist()}, bias: {bot_actor.action_bias.tolist()}")
            print("=" * 70 + "\n")

    except Exception as e:
        print(f"  Analysis failed: {e}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    user_path = input("Enter model checkpoint path: ").strip()
    if not user_path:
        user_path = "/home/knoji/workspace/rl_models/sac_v16.cleanrl_model"
        
    choice = input("Enter architecture type (sac/td3): ").strip().lower()
    if choice not in ["sac", "td3"]:
        choice = "sac"
        
    diagnose_saved_model(user_path, choice)