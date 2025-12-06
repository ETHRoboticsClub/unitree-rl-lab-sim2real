import torch
import torch.nn as nn

def _diag_gauss(mean, log_std):
    std = torch.exp(log_std).expand_as(mean)
    return torch.distributions.Normal(mean, std)

class Go2StandPolicy(nn.Module):
    """
    Policy class for Unitree Go2 Stand-Still task.
    Implements an Actor-Critic architecture using MLPs.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden=(512, 512, 256, 128)):
        super().__init__()
        
        # Actor Network
        actor_layers = []
        last_dim = obs_dim
        for h in hidden:
            actor_layers.append(nn.Linear(last_dim, h))
            actor_layers.append(nn.ReLU())
        self.actor = nn.Sequential(*actor_layers, nn.Linear(last_dim, action_dim))
        
        # Learnable log standard deviation for action distribution
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Value Function Network (Critic)
        # Using a simpler network for value function as seen in other implementations, 
        # or we could match actor complexity. Here we use a standard small critic.
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256), 
            nn.ReLU(), 
            nn.Linear(256, 1)
        )

    def act(self, obs):
        """
        Compute action and its log probability for a given observation.
        """
        mean = self.actor(obs)
        dist = _diag_gauss(mean, self.log_std)
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1)

    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions to get log probability, entropy, and value estimate.
        """
        mean = self.actor(obs)
        dist = _diag_gauss(mean, self.log_std)
        
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)
        
        return log_prob, entropy, value, dist

    def value(self, obs):
        """
        Estimate value for a given observation.
        """
        return self.critic(obs).squeeze(-1)
