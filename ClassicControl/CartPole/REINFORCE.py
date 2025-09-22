import numpy as np
import gymnasium as gym
import torch
from torch import nn, optim
from torch.distributions import Categorical

env = gym.make(
    id="CartPole-v1",
   render_mode=None
)


class PolicyNet(nn.Module):
    def __init__(self, observation_size, action_size):
        super().__init__()
        self.FC = nn.Sequential(
            nn.Linear(observation_size, 32),
            nn.ReLU(),
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Linear(24, action_size),
        )

    def forward(self, state):
        y = self.FC(state)
        return y


class Agent:
    def __init__(self, observation_size, action_size):
        self.gamma = 0.99
        self.lr = 9.8e-4

        self.policy_net = PolicyNet(observation_size, action_size)
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=self.lr)
        self.memory = []

    def get_action_log_prob(self, state):
        state = torch.as_tensor(state, dtype=torch.float32)
        logits = self.policy_net(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob

    def add_memory(self, reward, log_prob):
        self.memory.append((float(reward), log_prob))

    def update_PolicyNet(self):
        G, loss = 0.0, 0.0
        for reward, log_prob in reversed(self.memory):
            G = reward + self.gamma*G
            loss += -G*log_prob

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory.clear()


def train_agent(agent:Agent, num_spiodes):
    reward_history = []

    for episode in range(1, num_spiodes+1):
        state, info = env.reset()
        total_reward = 0

        while True:
            action, log_prob = agent.get_action_log_prob(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.add_memory(reward, log_prob)
            total_reward += reward

            if done:
                break

            state = next_state

        agent.update_PolicyNet()
        reward_history.append(total_reward)

        if episode%100==0:
            recent_reward = np.mean(reward_history[-100:])
            print(f'Episode:{episode}  Recent Reward:{recent_reward}')


def render_agent(agent: Agent, num_episodes=1):
    _ = str(input('Press ENTER to start rendering : '))

    for episode in range(num_episodes):
        tmp_env = gym.make(
            id="CartPole-v1",
            render_mode='human'
        )

        state, info = tmp_env.reset()
        total_reward = 0

        while True:
            action, log_prob = agent.get_action_log_prob(state)
            next_state, reward, terminated, truncated, info = tmp_env.step(action)

            done = terminated or truncated
            total_reward += reward

            if done:
                tmp_env.close()
                break

            state = next_state

        print(f'Total Reward : {total_reward}')


if __name__=='__main__':
    agent = Agent(
        observation_size=env.observation_space.shape[0],
        action_size=env.action_space.n
    )

    train_agent(agent, num_spiodes=2000)
    render_agent(agent, num_episodes=10)