import wandb
import torch

api = wandb.Api()

artifact = api.artifact("ali-esm-unipd/DT_RL/run-7hwtyxw0-history:v0")
artifact_dir = artifact.download()

model_path = f"{artifact_dir}/sac_step_latest.cleanrl_model"