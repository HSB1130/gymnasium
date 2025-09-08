import random
from collections import deque
import numpy as np
import gymnasium as gym
import torch
from torch import nn, optim

env = gym.make(
    'LunarLander-v3',
    # render_mode='human'
    render_mode=None
)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.buffer)

    def add_buffer(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))


class QNet(nn.Module):
    def __init__(self, observation_size, action_size, hidden_size=4):
        super().__init__()
        self.FC = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state):
        Q = self.FC(state)
        return Q


class DqnAgent:
    def __init__(self, observation_size, action_size):
        self.gamma = 0.9
        self.action_size = action_size
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.lr = 2e-5


        self.qnet = QNet(observation_size, action_size)
        self.optimizer = optim.Adam(params=self.qnet.parameters(), lr=self.lr)

    def get_action_from_qnet(self, state):
        state = torch.tensor(state, dtype=torch.float32)

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            with torch.no_grad():
                q_values = self.qnet(state)
                return torch.argmax(q_values).item()

    def update_qnet(self, state, action, next_state, reward, done):


def render_agent(agent: DqnAgent, num_episodes):
    reward_history = []

    for episode in range(1, num_episodes):
        state, info = env.reset()
        total_reward = 0

        while True:
            with torch.no_grad():
                tensor_state = torch.tensor(state, dtype=torch.float32)
                q_values = agent.qnet(tensor_state)
                action = torch.argmax(q_values).item()

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if done:
                break
            state = next_state



if __name__=='__main__':
    agent = DqnAgent(
        observation_size=env.observation_space.shape[0],
        action_size=env.action_space.n
    )