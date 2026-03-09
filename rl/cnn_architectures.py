import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Initialization Helpers ---

def weight_init(m):
    """Orthogonal initialization for stable gradients in RL."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data') and m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data') and m.bias is not None:
            m.bias.data.fill_(0.0)

def impala_init(module, weight_init, bias_init, gain=1):
    """Custom initialization utility for Impala-style architectures."""
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

# --- Utility Layers ---


class LayerNormalization(nn.Module):
    """Applies Layer Normalization to the input tensor."""
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return F.layer_norm(input, input.size())
    def extra_repr(self) -> str:
        return "Layer Normalization"

# --- Encoder Architectures ---

class ImpalaCNNLN(nn.Module):
    """A more efficient, high-stride CNN for faster feature extraction."""
    def __init__(self, in_channels=9, feature_dim=50, default_init=True):

        super().__init__()

        if default_init:
            init_ = lambda m: impala_init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        else:
            init_ = lambda m:m # no init, use pytorch default

        self.main = nn.Sequential(
            init_(nn.Conv2d(in_channels, 16, 8, stride=4)), LayerNormalization(), nn.LeakyReLU(),
            init_(nn.Conv2d(16, 32, 4, stride=2)), LayerNormalization(), nn.LeakyReLU(), nn.Flatten(),
            init_(nn.Linear(32 * 81, feature_dim)))
        
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Num params of encoder: {num_params}")
        
    def forward(self, obs):
        x = self.main(torch.unsqueeze(obs, dim=0)/255 - 0.5)
        return torch.squeeze(F.layer_norm(x, x.size()))


class ImpalaCNN(nn.Module):
    """A more efficient, high-stride CNN for faster feature extraction."""
    def __init__(self, in_channels=9, feature_dim=50, default_init=True):

        super().__init__()

        if default_init:
            init_ = lambda m: impala_init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        else:
            init_ = lambda m:m # no init, use pytorch default

        self.main = nn.Sequential(
            init_(nn.Conv2d(in_channels, 16, 8, stride=4)), nn.LeakyReLU(),
            init_(nn.Conv2d(16, 32, 4, stride=2)), nn.LeakyReLU(), nn.Flatten(),
            init_(nn.Linear(32 * 81, feature_dim)))
        
        num_params = sum(p.numel() for p in self.parameters())
        print(f"Num params of encoder: {num_params}")
        
    def forward(self, obs):
        x = self.main(torch.unsqueeze(obs, dim=0)/255 - 0.5)
        return torch.squeeze(F.layer_norm(x, x.size()))


class DrQEncoderV2(nn.Module):  
    def __init__(self, obs_shape=9, feature_dim=50, pretrained=False):
        super().__init__()

        #assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape, 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(),
                                     nn.Flatten())
        self.linear = nn.Linear(self.repr_dim, feature_dim)
        self.apply(weight_init)

        #if pretrained:
        #    pretrained_agent = torch.load(pretrained)
        #    self.load_state_dict(pretrained_agent.encoder.state_dict())

        num_params = sum(p.numel() for p in self.parameters())
        print(f"Num params of encoder: {num_params}")

    def forward(self, obs):
        obs = torch.unsqueeze(obs, dim=0) / 255.0 - 0.5
        h = torch.squeeze(self.convnet(obs), dim=0)

        h = self.linear(h)

        #h = h.view(h.shape[0], -1)
        return F.layer_norm(h, h.size())


class DQNEncoder(nn.Module):  
    """Classic Nature CNN architecture modified for continuous control and Duckitown 160*120*3 with frame stacking (4) ."""
    def __init__(self, obs_shape=(12, 120, 160), feature_dim=256):
        super().__init__()

        # 64 * 11 * 16 = 11264
        self.repr_dim = 11264

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 8, stride=4),
                                     nn.ReLU(), nn.Conv2d(32, 64, 4, stride=2),
                                     nn.ReLU(), nn.Conv2d(64, 64, 3, stride=1),
                                     nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            self.repr_dim = self.convnet(dummy_input).shape[1]

        self.linear = nn.Sequential(nn.Linear(self.repr_dim, 512), 
                                    nn.ReLU(),
                                    nn.Linear(512, feature_dim)
        )
        self.apply(weight_init)

        num_params = sum(p.numel() for p in self.parameters())
        print(f"Num params of encoder: {num_params}")

    def forward(self, obs):
        #data are already normalized so we ignore these for now 
        #obs = torch.unsqueeze(obs, dim=0) / 255.0 - 0.5
        #h = torch.squeeze(self.convnet(obs), dim=0)

        h = self.convnet(obs)
        h = self.linear(h)

        return F.layer_norm(h, (h.size(-1),))


class StreamRLEncoder(nn.Module):  
    def __init__(self, obs_shape=9, feature_dim=50):
        super().__init__()

        self.repr_dim = 256

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape, 32, 8, stride=5),
                                     nn.LeakyReLU(), nn.Conv2d(32, 64, 4, stride=3),
                                     nn.LeakyReLU(), nn.Conv2d(64, 64, 3, stride=2),
                                     nn.LeakyReLU(), nn.Flatten(),
                                     nn.Linear(self.repr_dim, 256), nn.LeakyReLU())
        self.linear = nn.Linear(256, feature_dim)
        self.apply(weight_init)

        num_params = sum(p.numel() for p in self.parameters())
        print(f"Num params of encoder: {num_params}")

    def forward(self, obs):
        obs = torch.unsqueeze(obs, dim=0) / 255.0 - 0.5
        h = torch.squeeze(self.convnet(obs), dim=0)

        h = self.linear(h)

        return F.layer_norm(h, h.size())