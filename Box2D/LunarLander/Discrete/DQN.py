import random
from datetime import datetime
from collections import deque
import numpy as np
import gymnasium as gym
import torch
from torch import nn, optim

env = gym.make(
    'LunarLander-v3',
    render_mode=None
)


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.buffer)

    def add_buffer(self, state, action, next_state, reward, done):
        self.buffer.append((state, action, next_state, reward, done))

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        states = np.stack([x[0] for x in data])
        actions = np.array([x[1] for x in data])
        next_states = np.stack([x[2] for x in data])
        rewards = np.array([x[3] for x in data])
        dones = np.array([x[4] for x in data])

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.long)

        return states, actions, next_states, rewards, dones


class QNet(nn.Module):
    def __init__(self, observation_size, action_size, hidden_size=64):
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
    def __init__(
        self,
        observation_size,
        action_size,
        batch_size=128,
        lr=5e-4,
        gamma=0.99,
        buffer_size=10000,
        tau=1e-3
    ):
        self.gamma = gamma
        self.action_size = action_size
        self.epsilon = 0.999
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.lr = lr

        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size)

        self.qnet = QNet(observation_size, action_size)
        self.qnet_target = QNet(observation_size, action_size).eval()
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.optimizer = optim.Adam(params=self.qnet.parameters(), lr=self.lr)

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)

    def add_replay_buffer(self, state, action, next_state, reward, done):
        self.replay_buffer.add_buffer(state, action, next_state, reward, done)

    def get_action_from_qnet(self, state):
        state = torch.tensor(state, dtype=torch.float32)

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            with torch.no_grad():
                q_values = self.qnet(state)
                return torch.argmax(q_values).item()

    def update_qnet(self):
        if len(self.replay_buffer) < 1000:
            return

        states, actions, next_states, rewards, dones = self.replay_buffer.get_batch()

        # q_preds = Q(s, a)
        q_values = self.qnet(states)
        q_preds = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # target = rewards + gamma * max a' Q(s', a')
        with torch.no_grad():
            next_q_values = self.qnet_target(next_states)
            next_max_q_values, _ = torch.max(next_q_values, dim=1)

        targets = rewards + (1-dones)*self.gamma*next_max_q_values
        targets.detach()

        # back-propagation
        loss = nn.functional.mse_loss(q_preds, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train_qnet(agent: DqnAgent, num_episodes, update_qnet_step=5, target_reward=220):
    reward_history = []

    for episode in range(1, num_episodes+1):
        state, info = env.reset()
        done = False

        total_reward = 0
        episode_step = 0

        while not done and episode_step<=1500:
            action = agent.get_action_from_qnet(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_step += 1
            total_reward += reward

            agent.add_replay_buffer(state, action, next_state, reward, done)

            if episode_step%update_qnet_step==0:
                agent.update_qnet()
                agent.sync_qnet()

            state = next_state

        agent.decay_epsilon()
        reward_history.append(total_reward)

        if episode%100 == 0:
            recent_reward = np.mean(reward_history[-20:])
            print(f'Episode : {episode}   Recent Reward : {recent_reward}')

            # early stopping
            if recent_reward>=target_reward:
                print('Target reward achieved!!')
                break


def save_qnet_params(agent: DqnAgent):
    is_save = str(input('Do you want to save this Qnet parameters? [Yes/No] : '))
    if is_save.strip().lower() == 'yes':
        now = datetime.now()
        datetime_str = now.strftime("%Y-%m-%d_%H:%M:%S")
        state_dict_saved_path = f'./model/dqn_qnet_state_dict_{datetime_str}.pt'

        torch.save(agent.qnet.state_dict(), state_dict_saved_path)
        print('Model state dict saved!!')
    else:
        print('Model doesn\'t saved!!')


def render_agent(agent: DqnAgent, num_episodes):
    _ = str(input('Press ENTER to start rendering : '))

    for episode in range(1, num_episodes+1):
        tmp_env = gym.make(
            'LunarLander-v3',
            render_mode='human'
        )

        state, info = tmp_env.reset()
        total_reward = 0

        while True:
            with torch.no_grad():
                tensor_state = torch.tensor(state, dtype=torch.float32)
                q_values = agent.qnet(tensor_state)
                action = torch.argmax(q_values).item()

            next_state, reward, terminated, truncated, info = tmp_env.step(action)
            done = terminated or truncated

            total_reward += reward

            if done:
                tmp_env.close()
                break

            state = next_state

        print(f'Reward : {total_reward}')

    save_qnet_params(agent)


if __name__=='__main__':
    agent = DqnAgent(
        observation_size=env.observation_space.shape[0],
        action_size=env.action_space.n
    )

    train_qnet(agent, num_episodes=10000)
    render_agent(agent, num_episodes=10)
