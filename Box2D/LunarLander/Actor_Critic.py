import numpy as np
from tqdm import tqdm
from datetime import datetime
import gymnasium as gym
import torch
from torch import nn, optim
from torch.distributions.categorical import Categorical

env = gym.make(
    id='LunarLander-v3',
    render_mode=None
)


class PolicyNet(nn.Module):
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
        return self.FC(state)


class ValueNet(nn.Module):
    def __init__(self, observation_size, hidden_size=64):
        super().__init__()
        self.FC = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        return self.FC(state)


class AcAgent:
    def __init__(self, observation_size, action_size, lr=1e-3):
        self.gamma = 0.99
        self.lr_policy = lr
        self.lr_value = lr

        self.policy_net = PolicyNet(observation_size, action_size)
        self.optimizer_policy = optim.Adam(params=self.policy_net.parameters(), lr=self.lr_policy)

        self.value_net = ValueNet(observation_size)
        self.optimizer_value = optim.Adam(params=self.value_net.parameters(), lr=self.lr_value)

    def get_action_from_policy_net(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            logits = self.policy_net(state)
            distribution = Categorical(logits=logits)
            chosen_action = distribution.sample()
        return chosen_action.item()

    def get_action_by_deterministic(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            logits = self.policy_net(state)
            chosen_action = torch.argmax(logits)
        return chosen_action.item()

    def update_policy_value_net(self, state, action, next_state, reward, done):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.long)

        # Value Net : loss = [r+gama*Vw(s')-Vw(s)]^2
        pred_state_value = self.value_net(state)

        with torch.no_grad():
            next_state_value = self.value_net(next_state)
            target_state_value = reward + (1-done)*self.gamma*next_state_value

        target_state_value = target_state_value.detach()
        loss_value = nn.functional.mse_loss(pred_state_value, target_state_value)

        # Policy Net : loss = -[{r+gamma*Vw(s')-Vw(s)} * log pi_theta(a|s)]
        delta_v = target_state_value - pred_state_value
        delta_v = delta_v.detach()

        logits = self.policy_net(state)
        distribution = Categorical(logits=logits)
        action_log_prob = distribution.log_prob(action)

        loss_policy = -delta_v*action_log_prob

        # Update value_net, policy_net
        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()

        loss_value.backward()
        loss_policy.backward()

        self.optimizer_value.step()
        self.optimizer_policy.step()


def train_agent(agent: AcAgent, num_episodes):
    reward_history = []

    for episode in tqdm(range(1, num_episodes+1)):
        state, info = env.reset()
        done = False

        total_reward = 0

        while not done:
            action = agent.get_action_from_policy_net(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.update_policy_value_net(state, action, next_state, reward, done)

            total_reward += reward
            state = next_state

        reward_history.append(total_reward)
        recent_reward = np.mean(reward_history[-50:])

        # Early Stopping
        if recent_reward>=245:
            print('Early Stopped!!')
            print(f'Episode : {episode}   Recent reward : {recent_reward:.4f}')
            break

        if episode%50 == 0:
            print(f'Episode : {episode}   Recent reward : {recent_reward:.4f}')


def save_policy_net_params(agent: AcAgent):
    is_save = str(input('Do you want to save this Policy Net parameters? [Yes/No] : '))
    if is_save.strip().lower() == 'yes':
        now = datetime.now()
        datetime_str = now.strftime("%Y-%m-%d_%H:%M:%S")
        state_dict_saved_path = f'./model/actor_critic_policy_net_state_dict_{datetime_str}.pt'

        torch.save(agent.policy_net.state_dict(), state_dict_saved_path)
        print('Model state dict saved!!')
    else:
        print('Model doesn\'t saved!!')


def render_agent(agent: AcAgent, num_episodes):
    render_env = gym.make(
        id='LunarLander-v3',
        render_mode='human'
    )

    _ = str(input('Press ENTER to start rendering : '))

    for episode in range(1, num_episodes+1):
        state, info = render_env.reset()
        done = False

        total_reward = 0.0

        while not done:
            action =  agent.get_action_by_deterministic(state)
            next_state, reward, terminated, truncated, info = render_env.step(action)
            done = terminated or truncated

            total_reward += reward

            state = next_state

        print(f'Episode : {episode}  Reward : {total_reward:.4f}')

    render_env.close()
    save_policy_net_params(agent)


if __name__=='__main__':
    agent = AcAgent(
        observation_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        lr=1e-3
    )

    train_agent(agent, num_episodes=5000)
    render_agent(agent, num_episodes=10)