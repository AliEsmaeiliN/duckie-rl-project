import matplotlib.pyplot as plt
import os
from torch import save
import wandb


def plot_model_input(s_obs, global_step):
    # Take the first environment's observation from the batch
    # s_obs shape is (Batch, 12, 120, 160)
    sample_obs = s_obs[0].cpu().numpy() 

    # Extract the first 3 channels (the most recent RGB frame)
    first_frame = sample_obs[0:3, :, :].transpose(1, 2, 0)

    plt.imshow(first_frame)
    plt.title(f"Input to Model - Step {global_step}")
    plt.show() 

def save_model(actor, qf1, qf2, step, run_name, suffix=""):
    
    model_dir = f"runs/{run_name}/models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    label = suffix if suffix else "latest"
    model_path = f"{model_dir}/sac_step_{label}.cleanrl_model"

    save({
        'actor_state_dict': actor.state_dict(),
        'qf1_state_dict': qf1.state_dict(),
        'qf2_state_dict': qf2.state_dict(),
        'global_step': step,
    }, model_path)

    if wandb.run is not None:
        # Use the suffix or the global step to make the artifact version clear
        label = suffix if suffix else f"step_{step}"
        artifact = wandb.Artifact(name=f"{run_name}_model", type="model")
        artifact.add_file(model_path)      
        artifact.metadata = {"global_step": step, "suffix": suffix}
        
        wandb.log_artifact(artifact)
    
    print(f"Saved: {model_path} at Step:{step}")