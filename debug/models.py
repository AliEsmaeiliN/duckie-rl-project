import torch
import torch.nn as nn
import torch.nn.functional as F

class ImpalaCNN(nn.Module):
    def __init__(self, in_channels=12, feature_dim=256):

        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 16, 8, stride=4), nn.LeakyReLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.LeakyReLU(), 
            nn.Flatten(),
            nn.Linear(32 * 81, feature_dim)
        )
        
    def forward(self, obs):
        x = obs.float() / 255.0 - 0.5
        h = self.main(x)
        return F.layer_norm(h, (h.size(-1),))
    
class SACActor(nn.Module):
    def __init__(self, grayscale=True, action_dim=2):
        super().__init__()

        self.channels = 4 if grayscale else 12
        self.encoder = ImpalaCNN(in_channels=self.channels,feature_dim=256)
        
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)

        # Action scaling (standard Duckietown is [-1, 1])
        self.register_buffer("action_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x):
        x = self.encoder(x)
        return self.fc_mean(x), self.fc_logstd(x)

    def get_action(self, x):
        """Only returns the mean action"""
        mean, _ = self.forward(x)
        v = torch.sigmoid(mean[:, 0:1])
        omega = torch.tanh(mean[:, 1:2])
        
        action = torch.cat([v, omega], dim=-1)
        return None, None, action * self.action_scale + self.action_bias
    
class TD3Actor(nn.Module):
    def __init__(self, grayscale=True, action_dim=2):
        super().__init__()
        self.channels = 4 if grayscale else 12

        self.encoder = ImpalaCNN(in_channels=self.channels, feature_dim=256)
        
        self.fc_mu = nn.Linear(256, action_dim)
        self.register_buffer("action_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        v = torch.tanh(mu[:, 0:1]).clamp(min=0.1)
        omega = torch.tanh(mu[:, 1:2])
        action = torch.cat([v, omega], dim=-1)
        return action * self.action_scale + self.action_bias