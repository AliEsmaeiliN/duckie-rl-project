import matplotlib.pyplot as plt


def plot_model_input(s_obs, global_step):
    # Take the first environment's observation from the batch
    # s_obs shape is (Batch, 12, 120, 160)
    sample_obs = s_obs[0].cpu().numpy() 

    # Extract the first 3 channels (the most recent RGB frame)
    first_frame = sample_obs[0:3, :, :].transpose(1, 2, 0)

    plt.imshow(first_frame)
    plt.title(f"Input to Model - Step {global_step}")
    plt.show() 