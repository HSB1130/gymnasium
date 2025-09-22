import time
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import torch
from torch import nn, optim

env = gym.make(id='Taxi-v3')

class QNet(nn.Module):
    def __init__(self, state_size=500, action_size=6):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=state_size, embedding_dim=action_size)

    def forward(self, state):
        return self.embedding_layer(state)


class QLearningAgent:
    def __init__(self, action_size=4):
        self.gamma = 0.9
        self.epsilon = 0.1
        self.action_size = action_size

        self.QNet = QNet()
        self.optimizer = optim.SGD(params=self.QNet.parameters(), lr=0.1)

    def get_action_from_QNet(self, state):
        state = torch.tensor(state, dtype=torch.long)

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            with torch.no_grad():
                action_values = self.QNet(state)
                max_value = torch.max(action_values)
                max_action_candidates = [idx for idx, action_value in enumerate(action_values) if action_value==max_value]
                return int(np.random.choice(max_action_candidates))

    def update_QNet(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.long)
        next_state = torch.tensor(next_state, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)

        if done:
            next_max_q = torch.tensor(0.0, dtype=torch.float32)
        else:
            with torch.no_grad():
                next_action_values = self.QNet(next_state)
                next_max_q = torch.max(next_action_values)

        target_q = reward + self.gamma * next_max_q
        pred_q = self.QNet(state)[action]

        target_q = target_q.detach()
        loss = nn.functional.mse_loss(pred_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def run_episodes(agent: QLearningAgent, num_episodes):
    episode_rewards = []

    for episode in tqdm(range(num_episodes)):
        state, info = env.reset()

        total_loss = 0
        total_reward = 0
        step_count = 0

        while True:
            action = agent.get_action_from_QNet(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            loss = agent.update_QNet(state, action, reward, next_state, done)

            total_loss += loss
            total_reward += reward
            step_count += 1

            if done:
                break

            state = next_state

        episode_rewards.append(total_reward)

        if episode % 100 == 0:
            avg_loss = total_loss / step_count if step_count > 0 else 0
            recent_avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}: Avg Loss = {avg_loss:.4f}, Recent Avg Reward = {recent_avg_reward:.2f}")


def get_optimal_policy(agent: QLearningAgent):
    optimal_policy = {}

    for state in range(env.observation_space.n):
        tensor_state = torch.tensor(state, dtype=torch.long)
        with torch.no_grad():
            q_values = agent.QNet(tensor_state)
            max_action = torch.argmax(q_values).item()
        optimal_policy[state] = max_action
    return optimal_policy


def render_optimal_policy(optimal_policy:dict, num_render=1):
    _ = str(input('Press ENTER to start rendering : '))

    for _ in range(num_render):
        tmp_env = gym.make(
            id='Taxi-v3',
            render_mode='human'
        )

        state, info = tmp_env.reset()
        step_count = 0

        while True:
            action = optimal_policy[state]
            next_state, reward, terminated, truncated, info = tmp_env.step(action)

            print(f'Step      : {step_count}')
            print(f'State     : {state}')
            print(f'Action    : {action}')
            print(f'Next State: {next_state}')
            print(f'Reward    : {reward}', end='\n\n')
            time.sleep(1)

            step_count += 1

            if terminated or truncated:
                break

            state = next_state

        tmp_env.close()


if __name__ == '__main__':
    agent = QLearningAgent()
    run_episodes(agent, num_episodes=10000)

    optimal_policy = get_optimal_policy(agent)
    render_optimal_policy(optimal_policy)
