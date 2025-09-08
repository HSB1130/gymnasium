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


class ValueNet(nn.Module):
    def __init__(self, observation_size):
        super().__init__()

        self.FC = nn.Sequential(
            nn.Linear(observation_size, 32),
            nn.ReLU(),
            nn.Linear(32, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )

    def forward(self, state):
        y = self.FC(state)
        return y


class Agent:
    def __init__(self, observation_size, action_size):
        self.gamma = 0.99

        self.policy_net = PolicyNet(observation_size, action_size)
        self.value_net = ValueNet(observation_size)
        self.optimizer_policy = optim.Adam(params=self.policy_net.parameters(), lr=5e-4)
        self.optimizer_value = optim.Adam(params=self.value_net.parameters(), lr=2e-4)

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        logits = self.policy_net(state)
        distribution = torch.distributions.Categorical(logits=logits)
        chosen_action = distribution.sample()
        chosen_action_log_prob = distribution.log_prob(chosen_action)
        return chosen_action.item(), chosen_action_log_prob

    def update_V_Policy(self, state, action_log_prob, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.long)

        V = self.value_net(state)
        target = reward + (1-done)*self.gamma*self.value_net(next_state)
        target = target.detach()
        loss_value = torch.nn.functional.mse_loss(V, target)

        delta = target - V
        delta = delta.detach()
        loss_policy = -delta*action_log_prob

        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()

        loss_value.backward()
        loss_policy.backward()

        self.optimizer_value.step()
        self.optimizer_policy.step()


def run_episodes(agent:Agent, num_episodes):
    reward_history = []

    for episode in range(1, num_episodes+1):
        state, info = env.reset()
        total_rewards = 0

        while True:
            action, action_log_prob = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_rewards += reward
            agent.update_V_Policy(state, action_log_prob, reward, next_state, done)

            if done:
                break

            state = next_state

        reward_history.append(total_rewards)

        if episode%100 == 0:
            recent_reward = np.mean(reward_history[-100:])
            print(f'Episode:{episode}  Recent_Reward:{recent_reward}')


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
            action, action_log_prob = agent.get_action(state)
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

    run_episodes(agent, num_episodes=3000)
    render_agent(agent, num_episodes=10)
