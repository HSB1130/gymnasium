import time
import random
import numpy as np
import gymnasium as gym
from collections import deque
import torch
from torch import nn, optim


env = gym.make(
    id="CartPole-v1",
   render_mode=None
)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, next_state, reward, done):
        self.buffer.append((state, action, next_state, reward, done))

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = np.stack([x[0] for x in data])
        action = np.array([x[1] for x in data])
        next_state = np.stack([x[2] for x in data])
        reward = np.array([x[3] for x in data])
        done = np.array([x[4] for x in data])

        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.long)

        return state, action, next_state, reward, done


class QNet(nn.Module):
    def __init__(self, observation_size=4, action_size=2):
        super().__init__()
        self.FC = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

        for layer in [self.FC]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')

    def forward(self, state):
        Q = self.FC(state)
        return Q


class DqnAgent:
    def __init__(self, observation_size=4, action_size=2):
        self.gamma = 0.99
        self.epsilon = 0.99
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.observation_size = observation_size
        self.action_size = action_size
        self.replay_buffer = ReplayBuffer(buffer_size=5000, batch_size=32)

        self.QNet = QNet(observation_size, action_size)
        self.QNet_target = QNet(observation_size, action_size).eval()
        self.QNet_target.load_state_dict(self.QNet.state_dict())
        self.optimizer = optim.Adam(params=self.QNet.parameters(), lr=0.001)

    def sync_QNet(self):
        self.QNet_target.load_state_dict(self.QNet.state_dict())

    def get_action_from_QNet(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(self.action_size))
        else:
            with torch.no_grad():
                action_q_values = self.QNet(state)
                return torch.argmax(action_q_values).item()

    def update_QNet(self, state, action, next_state, reward, done):
        self.replay_buffer.add(state, action, next_state, reward, done)
        if len(self.replay_buffer) < self.replay_buffer.batch_size:
            return

        state, action, next_state, reward, done = self.replay_buffer.get_batch()

        q_values = self.QNet(state)
        q_pred = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        next_q = self.QNet_target(next_state)
        next_max_q, _ = torch.max(next_q, dim=1)

        target_q = reward + (1-done)*self.gamma*next_max_q
        target_q = target_q.detach()

        loss = nn.functional.mse_loss(q_pred, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def train_agent(agent:DqnAgent, num_episodes):
    reward_per_episode = []

    for episode in range(num_episodes):
        state, info = env.reset()

        total_reward = 0
        total_loss = 0
        step_cnt = 0

        while True:
            action = agent.get_action_from_QNet(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            loss = agent.update_QNet(state, action, next_state, reward, done)

            if loss is not None:
                total_loss += loss
                step_cnt += 1

            total_reward += reward

            if done:
                break

            state = next_state

        reward_per_episode.append(total_reward)

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        if (episode+1)%10 == 0:
            agent.sync_QNet()

        if (episode+1)%100 == 0:
            recent_avg_reward = np.mean(reward_per_episode[-100:])
            print(f'Episode:{episode+1} Avg_Loss:{total_loss/step_cnt:.4f}  Recent_Avg_Reward:{recent_avg_reward:.4f}')


def render_agent(agent: DqnAgent, num_episodes=1):
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
                action = torch.argmax(q_values).item()

            next_state, reward, terminated, truncated, info = tmp_env.step(action)

            done = terminated or truncated
            total_reward += reward

            if done:
                tmp_env.close()
                break

            state = next_state

        print(f'Total Reward : {total_reward}')


if __name__=='__main__':
    agent = DqnAgent(
        observation_size=env.observation_space.shape[0],
        action_size=env.action_space.n
    )

    train_agent(agent, num_episodes=2000)
    render_agent(agent, num_episodes=10)


