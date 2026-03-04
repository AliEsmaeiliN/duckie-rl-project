import torch
import gymnasium as gym

# Check GPU status
print(f"PyTorch version: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using: {torch.cuda.get_device_name(0)}")

# Check Gymnasium
try:
    env = gym.make("CartPole-v1")
    obs, info = env.reset()
    print("Gymnasium: CartPole-v1 loaded successfully!")
    env.close()
except Exception as e:
    print(f"Gymnasium Error: {e}")