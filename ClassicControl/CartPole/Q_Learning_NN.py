import time
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import torch
from torch import nn

env = gym.make(
    id="CartPole-v1",
    render_mode=None
)


class QNet(nn.Module):
    def __init__(self, observation_size, action_size):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, state):
        y = self.layer(state)
        return y


class QLearningAgent:
    def __init__(self, observation_size=4, action_size=2):
        self.gamma = 0.999
        self.epsilon = 1.0  # 시작을 1.0으로 변경
        self.epsilon_min = 0.01  # 최소 epsilon 값
        self.epsilon_decay = 0.9995  # decay rate
        self.action_size = action_size

        self.QNet = QNet(observation_size, action_size)
        self.optimizer = torch.optim.Adam(params=self.QNet.parameters(), lr=0.001)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_decay*self.epsilon, self.epsilon_min)

    def get_action_from_QNet(self, state):
        state = torch.tensor(state, dtype=torch.float32)

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            with torch.no_grad():
                q_values = self.QNet(state)
                max_q_value = torch.max(q_values)
                max_action_candidates = [idx for idx, q_value in enumerate(q_values) if q_value==max_q_value]
                return int(np.random.choice(max_action_candidates))

    def update_QNet(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)

        pred_q = self.QNet(state)[action]

        if done:
            next_max_q = torch.zeros((), dtype=torch.float32)
        else:
            with torch.no_grad():
                next_q_values = self.QNet(next_state)
                next_max_q = torch.max(next_q_values)

        target_q = reward + self.gamma*next_max_q
        target_q = target_q.detach()

        loss = nn.functional.mse_loss(target_q, pred_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def train_agent(agent:QLearningAgent, num_epidoes):
    reward_per_episode = []
    max_avg_reward = 0

    for episode in range(num_epidoes):
        state, info = env.reset()

        total_loss = 0
        step_cnt = 0
        total_reward = 0

        while True:
            action = agent.get_action_from_QNet(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            loss = agent.update_QNet(state, action, reward, next_state, done)

            total_loss += loss
            total_reward += reward
            step_cnt += 1

            if done:
                break

            state = next_state

        reward_per_episode.append(total_reward)

        agent.update_epsilon()

        if episode%100 == 0:
            avg_loss = total_loss / step_cnt if step_cnt > 0 else 0
            recent_rewards = np.mean(reward_per_episode[-100:])
            print(f"Episode {episode}: Avg Loss = {avg_loss:.5f}, Recent Avg Reward = {recent_rewards:.2f}")


def render_agent(agent: QLearningAgent, num_episodes=1):
    _ = str(input('Press ENTER to start rendering : '))

    for episode in range(num_episodes):
        tmp_env = gym.make(
            id="CartPole-v1",
            render_mode='human'
        )

        state, info = tmp_env.reset()
        total_reward = 0

        while True:
            with torch.no_grad():
                q_values = agent.QNet(torch.tensor(state, dtype=torch.float32))
                max_q_value = torch.max(q_values)
                action_candidates = [idx for idx, q_value in enumerate(q_values) if q_value==max_q_value]
                action = int(np.random.choice(action_candidates))

            next_state, reward, terminated, truncated, info = tmp_env.step(action)

            done = terminated or truncated
            total_reward += reward

            if done:
                tmp_env.close()
                break

            state = next_state

        print(f'Total Reward : {total_reward}')


if __name__=='__main__':
    agent = QLearningAgent()
    train_agent(agent, num_epidoes=5000)
    render_agent(agent, num_episodes=10)

