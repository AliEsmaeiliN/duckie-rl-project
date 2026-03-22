import wandb
import torch
#wandb api: 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

api = wandb.Api()
model_address = input("Paste the wandb artifact address: ")
artifact = api.artifact(model_address)
artifact_dir = artifact.download()

model_path = f"{artifact_dir}/sac_step_latest.cleanrl_model"