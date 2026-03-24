import wandb
import torch
#wandb api: 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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
api = wandb.Api()
model_address = input("Paste the wandb artifact address: ")
if model_address == "":
    model_address = "ali-esm-unipd/DT_RL_TD3/baselineV1__td3_continuous_action__1__1774283945_Final:v0"
artifact = api.artifact(model_address)
artifact_dir = artifact.download()

model_path = f"{artifact_dir}/sac_Final.cleanrl_model"
print(f"Loading model from {model_path}")
checkpoint = torch.load(model_path, map_location=device)

wandb_metadata = artifact.metadata
        
    # Build sim_params from WandB metadata
sim_params = {
    "domain_rand": wandb_metadata.get("domain_rand", False),
    "distortion": wandb_metadata.get("distortion", True),
    "dynamics_rand": wandb_metadata.get("dynamics_rand", False),
    "camera_rand": wandb_metadata.get("camera_rand", False),
}

print(f"--- Metadata Extracted ---")
print(f"Randomizations: {sim_params}")