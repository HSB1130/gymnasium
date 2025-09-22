from tqdm import tqdm
import numpy as np
import gymnasium as gym
import torch
from torch import nn, optim
from torch.distributions.normal import Normal


env = gym.make(
    id='Pendulum-v1',
    render_mode = None
)


class RolloutBuffer:
    def __init__(self):
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def save(self, transition):
        self.buffer.append(transition)

    def sample(self):
        states, action_log_probs, next_states, rewards, dones = map(torch.stack, zip(*self.buffer))
        rewards = rewards.unsqueeze(dim=1)
        dones = dones.unsqueeze(dim=1)
        self.buffer.clear()

        return states, action_log_probs, next_states, rewards, dones

    @property
    def size(self):
        return len(self.buffer)


class PolicyNet(nn.Module):
    def __init__(self, observation_size, action_size, hidden_size=64):
        super().__init__()

        self.FC = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.avg_layer = nn.Linear(hidden_size, action_size)
        self.log_std_layer = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        h = self.FC(state)
        avg = self.avg_layer(h)

        # # Use tanh+exp
        # log_std = torch.tanh(self.log_std_layer(h))
        # std = torch.exp(log_std)

        # Use SoftPlus
        std = nn.functional.softplus(self.log_std_layer(h)) + 1e-4

        return avg, std


class ValueNet(nn.Module):
    def __init__(self, observation_size, hidden_size=64):
        super().__init__()

        self.FC = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state):
        return self.FC(state)


class AcAgent:
    def __init__(self, observation_size, action_size, lr, lmda):
        self.gamma = 0.95
        self.observation_size = observation_size
        self.action_size = action_size
        self.lr = lr

        self.lmda = lmda
        self.buffer = RolloutBuffer()

        self.policy_net = PolicyNet(observation_size, action_size, hidden_size=64)
        self.policy_optimizer = optim.Adam(params=self.policy_net.parameters(), lr=self.lr)

        self.value_net = ValueNet(observation_size, hidden_size=32)
        self.value_optimizer = optim.Adam(params=self.value_net.parameters(), lr=self.lr)

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
        z = distribution.rsample()  # reparameterization trick 사용!
        action_tanh = torch.tanh(z)  # [-1, 1] 범위

        log_prob = distribution.log_prob(z).sum(-1)
        log_prob -= torch.sum(torch.log(1 - action_tanh.pow(2) + eps), dim=-1)

        return action_tanh.detach().numpy(), log_prob

    def add_buffer(self, state, action_log_prob, next_state, reward, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.long)

        trajactory = (state, action_log_prob, next_state, reward, done)
        self.buffer.save(trajactory)

    def update_policy_net(self):
        states, action_log_probs, next_states, rewards, dones = self.buffer.sample()

        # GAE
        with torch.no_grad():
            state_value = self.value_net(states)
            next_state_value = self.value_net(next_states)

            # delta = {Rt + gamma*Vw(St+1)} - Vw(St)
            deltas = rewards + (1-dones)*self.gamma*next_state_value - state_value

            advantages = torch.zeros_like(rewards)
            last_advantage = 0.0

            for t in reversed(range(len(rewards))):
                last_advantage = deltas[t] + (1-dones[t])*self.gamma*self.lmda*last_advantage
                advantages[t] += last_advantage

            returns = advantages + state_value

        # Critic
        returns = returns.detach()
        pred_state_values = self.value_net(states)
        loss_value = nn.functional.mse_loss(pred_state_values, returns)

        # Actor
        advantages = advantages.detach().squeeze()
        loss_policy = -(advantages*action_log_probs).mean()

        # Update
        self.value_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()

        loss_value.backward()
        loss_policy.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)

        self.value_optimizer.step()
        self.policy_optimizer.step()


def train_agent(agent: AcAgent, num_episodes):
    reward_history = []

    for episode in tqdm(range(1, num_episodes+1)):
        state, info = env.reset()
        done = False

        total_reward = 0.0

        while not done:
            action, action_log_prob = agent.get_action_log_prob(state)
            next_state, reward, terminated, truncated, info = env.step(2*action)
            done = terminated or truncated

            agent.add_buffer(state, action_log_prob, next_state, reward, done)

            total_reward += reward
            state = next_state

        agent.update_policy_net()

        reward_history.append(total_reward)
        recent_reward = np.mean(reward_history[-100:])

        if recent_reward >= -350:
            print(f'Early Stopped with Recent Reward : {recent_reward:.4f}')
            break

        if episode%100 == 0:
            recent_reward = np.mean(reward_history[-100:])
            print(f'Episode : {episode}   Recent reward : {recent_reward:.4f}')


def render_agent(agent: AcAgent, num_episodes=20):
    render_env = gym.make(
        id='Pendulum-v1',
        render_mode = 'human'
    )

    _ = str(input('Press ENTER to start rendering : '))

    for episode in range(1, num_episodes+1):
        state, info = render_env.reset()
        done = False

        total_reward = 0

        while not done:
            action = agent.get_action_determistic(state)
            next_state, reward, terminated, truncated, info = render_env.step(2*action)
            done = terminated or truncated

            total_reward += reward

            state = next_state

        print(f'Episode : {episode}  Reward : {total_reward:.4f}')


if __name__=='__main__':
    agent = AcAgent(
        observation_size=env.observation_space.shape[0],
        action_size=env.action_space.shape[0],
        lr=2e-4,
        lmda=0.7
    )

    train_agent(agent, num_episodes=30000)
    render_agent(agent, num_episodes=20)
