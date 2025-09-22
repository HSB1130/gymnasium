from tqdm import tqdm
import numpy as np
import gymnasium as gym
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.normal import Normal


env = gym.make(
    id='BipedalWalker-v3',
    render_mode=None
)


class RolloutBuffer:
    def __init__(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def save(self, state, action_log_prob, next_state, reward, done):
        trajactory = (state, action_log_prob, next_state, reward, done)
        self.buffer.append(trajactory)

    def sample(self):
        states, action_log_probs, next_states, rewards, dones = map(torch.stack, zip(*self.buffer))
        rewards = rewards.unsqueeze(dim=1)
        dones = dones.unsqueeze(dim=1)
        self.buffer.clear()

        return states, action_log_probs, next_states, rewards, dones



class PolicyNet(nn.Module):
    def __init__(self, observation_size, action_size, hiddne_size=128):
        super().__init__()

        self.FC = nn.Sequential(
            nn.Linear(observation_size, hiddne_size),
            nn.Tanh(),
            nn.Linear(hiddne_size, hiddne_size),
            nn.Tanh(),
        )
        self.avg_header = nn.Linear(hiddne_size, action_size)
        self.log_std_header = nn.Linear(hiddne_size, action_size)

    def forward(self, state):
        h = self.FC(state)
        avg = self.avg_header(h)
        log_std = self.log_std_header(h)
        std = nn.functional.softplus(log_std) + 1e-4
        return avg, std


class ValueNet(nn.Module):
    def __init__(self, observation_size, hiddne_size=128):
        super().__init__()

        self.FC = nn.Sequential(
            nn.Linear(observation_size, hiddne_size),
            nn.Tanh(),
            nn.Linear(hiddne_size, hiddne_size),
            nn.Tanh(),
            nn.Linear(hiddne_size, 1)
        )

    def forward(self, state):
       return self.FC(state)


class PpoAgent:
    def __init__(self, observation_size, action_size, n_steps, n_epochs, lr, gae_lmda, clip_ration, vf_coef, entropy_coef):
        self.observation_size = observation_size
        self.action_size = action_size

        self.n_steps = n_steps
        self.n_epochs = n_epochs

        self.gamma = 0.99
        self.lr = lr
        self.gae_lmda = gae_lmda
        self.clip_ratio = clip_ration
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef

        self.buffer = RolloutBuffer()

        self.policy_net = PolicyNet(self.observation_size, self.action_size)
        self.policy_optimizer = optim.Adam(params=self.policy_net.parameters(), lr=self.lr)

        self.value_net = ValueNet(self.observation_size)
        self.value_optimizer = optim.Adam(params=self.value_net.parameters(), lr=self.lr)

    def add_buffer(self, state, action_log_prob, next_state, reward, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.long)

        trajactory = (state, action_log_prob, next_state, reward, done)
        self.buffer.save(trajactory)

    def get_action_determistic(self, state):
        state = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            avg, std = self.policy_net(state)
            action_tanh = torch.tanh(avg)

        return action_tanh.detach().numpy()

    def get_action_log_prob(self, state):
        eps = 1e-6
        state = torch.tensor(state, dtype=torch.float32)

        avg, std = self.policy_net(state)

        distribution = Normal(avg, std)
        z = distribution.rsample()
        action_tanh = torch.tanh(z)

        log_prob = distribution.log_prob(z).sum(-1)
        log_prob -= torch.sum(torch.log(1 - action_tanh.pow(2) + eps), dim=-1)

        return action_tanh.detach().numpy(), log_prob

    def update_policy_value_net(self):
        pass



if __name__=='__main__':
    pass