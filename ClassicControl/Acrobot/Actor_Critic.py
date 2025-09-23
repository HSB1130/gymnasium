import numpy as np
from tqdm import tqdm
import gymnasium as gym
import torch
from torch import nn, optim
from torch.distributions.categorical import Categorical

env = gym.make(
    id="Acrobot-v1",
    render_mode=None
)

class PolicyNet(nn.Module):
    def __init__(self, observation_size=6, action_size=3, hidden_size=32):
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
    def __init__(self, observation_size=6, hidden_size=32):
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
    def __init__(self, observation_size=6, action_size=3, lr=1e-4):
        self.gamma = 0.99
        self.observation_size = observation_size
        self.action_size = action_size
        self.lr = lr

        self.policy_net = PolicyNet(self.observation_size, self.action_size)
        self.optimizer_policy = optim.Adam(params=self.policy_net.parameters(), lr=self.lr)

        self.value_net = ValueNet(self.observation_size)
        self.optimizer_value = optim.Adam(params=self.value_net.parameters(), lr=self.lr)

    def get_action_from_policy_net(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            logits = self.policy_net(state)
            distribution = Categorical(logits=logits)
            action = distribution.sample()
        return action.item()

    def get_action_deterministic(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            logits = self.policy_net(state)
            choesen_action = torch.argmax(logits)
        return choesen_action.item()

    def update_policy_value_net(self, state, action, next_state, reward, done):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.long)

        # ValueNet : Loss = [{r+gamma*Vw(s')} - Vw(s)]^2
        pred_state_value = self.value_net(state)

        with torch.no_grad():
            next_state_value = self.value_net(next_state)
            target_state_value = reward + (1-done)*self.gamma*next_state_value

        target_state_value = target_state_value.detach()
        loss_value_net = nn.functional.mse_loss(pred_state_value, target_state_value)

        # PolicyNet : Loss = -[{r+gamma*Vw(s')}-Vw(s)] * log(pi_theta(a|s))
        delta = target_state_value - pred_state_value
        delta = delta.detach()

        logits = self.policy_net(state)
        distribution = Categorical(logits=logits)
        action_log_prob = distribution.log_prob(action)

        loss_policy_net = -delta*action_log_prob

        # Update Neural Net
        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()

        loss_value_net.backward()
        loss_policy_net.backward()

        self.optimizer_value.step()
        self.optimizer_policy.step()


def train_agent(agent: AcAgent, num_episodes):
    reward_history = []

    for episode in tqdm(range(1, num_episodes+1)):
        state, info = env.reset()
        done = False

        total_reward = 0.0

        while not done:
            action = agent.get_action_from_policy_net(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.update_policy_value_net(state, action, next_state, reward, done)
            total_reward += reward

            state = next_state

        reward_history.append(total_reward)
        recent_reward = np.mean(reward_history[-50:])

        if recent_reward>=-85:
            print('Early Stop!!')
            print(f'Episode : {episode}   Recent_Reward : {recent_reward:.4f}')
            break

        if episode%50==0:
            recent_reward = np.mean(reward_history[-50:])
            print(f'Episode : {episode}   Recent_Reward : {recent_reward:.4f}')


def render_agent(agent: AcAgent, num_episodes=10):
    render_env = gym.make(
        id="Acrobot-v1",
        render_mode='human'
    )

    _ = str(input('Press ENTER to start rendering : '))

    for episode in range(1, num_episodes+1):
        state, info = render_env.reset()
        done = False

        total_reward = 0

        while not done:
            action = agent.get_action_deterministic(state)
            next_state, reward, terminated, truncated, info = render_env.step(action)
            done = terminated or truncated

            total_reward += reward
            state = next_state

        print(f'Episode {episode} : Reward {total_reward}')
    render_env.close()


if __name__=='__main__':
    agent = AcAgent(
        observation_size=env.observation_space.shape[0],
        action_size=env.action_space.n
    )

    train_agent(agent, num_episodes=3000)
    render_agent(agent, num_episodes=20)
