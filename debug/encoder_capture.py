import torch
import torch.nn as nn
import socket
import pickle
import struct
import matplotlib.pyplot as plt
import numpy as np
from models import SACActor, TD3Actor
from rl_env_debug import DuckieOvalEnv
from laptop_viewer import start_laptop_receiver


class FeatureVisualizer:
    def __init__(self, model_path, algo_type="sac", grayscale=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.algo_type = algo_type.lower()
        self.grayscale = grayscale
        
        if self.algo_type == "sac":
            self.actor = SACActor(grayscale=self.grayscale).to(self.device)
        elif self.algo_type == "td3":
            self.actor = TD3Actor(grayscale=self.grayscale).to(self.device)
        else:
            raise ValueError(f"Unknown algo type: {self.algo_type}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor.eval()

        self.c = 1 if grayscale else 3

        self.outputs = {}

    def _get_hook(self, name):
        """Internal function to capture layer output."""
        def hook(model, input, output):
            self.outputs[name] = output.detach()
        return hook
    
    def _env_step(self, domain_rand=False, camera_rand=False, distortion=False, gray_scale=True):
        """Create the Test Env"""
        run_name = "feature_debug"
        env = DuckieOvalEnv.create_wrapped(
            run_name=run_name,
            capture_video=False,
            motion_blur=False,
            grayscale=gray_scale,
            frame_stack=4,
            domain_rand=domain_rand,
            distortion=distortion,
            camera_rand=camera_rand
        )

        obs, _ = env.reset(seed=42)
        print("the feature extraction environment created!")

        constant_action = np.array([0.4, 0.4], dtype=np.float32)
        steps = 10
        for _ in range(steps):
            obs, _, _, _, _ = env.step(constant_action)
        print(f"taking {steps} with conastant action of {constant_action} in the environment")

        if self.grayscale:
            plt.imshow(obs[-1], cmap='gray')
        else:
            frame = obs[-3:].transpose(1, 2, 0)
            plt.imshow(frame)
        plt.title("Environment Output (Final Processed Frame)")
        plt.axis('off')
        plt.show()

        stack_obs = torch.Tensor(obs).unsqueeze(0).to(self.device)

        return stack_obs

    def visualize_layers(self, input_stack):
        """
        Registers hooks, runs a forward pass, and plots feature maps.
        input_stack: Tensor of shape (1, C*Stack, 84, 84)
        """
    
        # main[0] is the first Conv2d
        # main[2] is the second Conv2d
        layers = ['Conv1_Raw', 'Conv1_Act', 'Conv2_Raw', 'Conv2_Act', 'Latent']
        
        h0 = self.actor.encoder.main[0].register_forward_hook(self._get_hook(layers[0]))
        h1 = self.actor.encoder.main[1].register_forward_hook(self._get_hook(layers[1]))
        h2 = self.actor.encoder.main[2].register_forward_hook(self._get_hook(layers[2]))
        h3 = self.actor.encoder.main[3].register_forward_hook(self._get_hook(layers[3]))
        h_emb = self.actor.encoder.main[5].register_forward_hook(self._get_hook(layers[4]))

        if input_stack is None:
            input_stack = self._env_step()

        with torch.no_grad():
            _ = self.actor(input_stack.to(self.device))

        for layer in layers:
            if layer == 'Latent':
                break
            fm = self.outputs[layer][0].cpu().numpy()
            grid = int(np.ceil(np.sqrt(fm.shape[0])))
            fig, axes = plt.subplots(grid, grid, figsize=(8, 8))
            fig.suptitle(f"Spatial Features: {layer}")
            for i, ax in enumerate(axes.flat):
                if i < fm.shape[0]:
                    ax.imshow(fm[i], cmap='viridis')
                ax.axis('off')
            plt.show() 

        embedding = self.outputs['Latent'][0].cpu().numpy()
        plt.figure(figsize=(15, 2))
        plt.imshow(embedding.reshape(1, -1), aspect='auto', cmap='magma')
        plt.title(f"3. Final Latent Features (Size: {embedding.shape[0]})")
        plt.axis('off')
        plt.show()

        #self.calculate_feature_impact()

        # Cleanup
        for h in [h0, h1, h2, h3, h_emb]: h.remove()

    def visualize_robot_stream(self):
        print("Waiting for the first frame from the Duckiebot...")
        for msg in start_laptop_receiver(yield_data=True):
        
            img_stack = np.array(msg["image"], dtype=np.uint8)

            if self.grayscale:
                plt.imshow(img_stack[-1], cmap='gray')
            else:
                frame = img_stack[-3:].transpose(1, 2, 0)
                plt.imshow(frame)
            plt.title("Robot Output (Final Processed Frame)")
            plt.axis('off')
            plt.show()
            
            input_tensor = torch.Tensor(img_stack).unsqueeze(0).to(visualizer.device)
            
            print("Snapshot received! Generating visualizations...")
            
            visualizer.visualize_layers(input_tensor)
            
            break 

        print("Analysis finished. Script exiting.")

    def calculate_feature_impact(self):
        """Analyze the weights of the final linear layer to find important features."""
        linear_layer = self.actor.encoder.main[5]
        weights = linear_layer.weight.data.abs().cpu().numpy()
        
        impact = np.mean(weights, axis=1) 
        
        plt.figure(figsize=(12, 4))
        plt.bar(range(len(impact)), impact, color='teal')
        plt.title("Feature Impact: Importance of each Latent Neuron in the Final Layer")
        plt.xlabel("Latent Feature Index")
        plt.ylabel("Mean Absolute Weight (Impact)")
        plt.show()
        

# shape = (1, 4, 84, 84) for grayscale or (1, 12, 84, 84) for RGB
if __name__ == "__main__":
    
    visualizer = FeatureVisualizer("artifacts/sac_V5.cleanrl_model", algo_type="sac")
    
    MODE = 'sim' 

    if MODE == 'sim':
        visualizer.visualize_layers(input_stack=None)
    else:
        visualizer.visualize_robot_stream()