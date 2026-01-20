import numpy as np
import gymnasium as gym
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader, TensorDataset


env = gym.make(
    id="LunarLander-v3",
    continuous=True,
    render_mode=None
)


class RolloutBuffer:
    def __init__(self, max_len):
        self.memory = []
        self.max_len = max_len

    def __len__(self):
        return len(self.memory)

    def is_full(self):
        return len(self.memory) >= self.max_len

    def save(self, trajectory):
        self.memory.append(trajectory)

    def sample(self):
        states, actions, raw_actions, raw_action_log_probs, next_states, rewards, dones = map(torch.stack, zip(*self.memory))
        rewards = rewards.unsqueeze(dim=1)
        dones = dones.unsqueeze(dim=1)
        self.memory.clear()
        return states, actions, raw_actions, raw_action_log_probs, next_states, rewards, dones


class PolicyNet(nn.Module):
    def __init__(self, observation_size, action_size, hidden_size=64):
        super().__init__()

        self.FC = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.avg_head = nn.Linear(hidden_size, action_size)
        self.raw_std_head = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        hidden_state = self.FC(state)
        avg = self.avg_head(hidden_state)
        raw_std = self.raw_std_head(hidden_state)
        std = nn.functional.softplus(raw_std) + 1e-4
        return avg, std


class ValueNet(nn.Module):
    def __init__(self, observation_size, hidden_size=64):
        super().__init__()

        self.FC = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        return self.FC(state)


class PpoAgent:
    def __init__(self, observation_size, action_size, gamma, lr, buffer_max_len, gae_lmda, n_epochs, batch_size, clip_ratio, vf_coef, entropy_coef):
        self.gamma = gamma
        self.lr = lr
        self.gae_lmda = gae_lmda

        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef

        self.buffer = RolloutBuffer(buffer_max_len)

        self.policy_net = PolicyNet(observation_size, action_size)
        self.optim_policy_net = optim.Adam(params=self.policy_net.parameters(), lr=self.lr)

        self.value_net = ValueNet(observation_size)
        self.optim_value_net = optim.Adam(params=self.value_net.parameters(), lr=self.lr)

    def add_buffer(self, state, action, raw_action, raw_action_log_prob, next_state, reward, done):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        raw_action = torch.tensor(raw_action, dtype=torch.float32)
        raw_action_log_prob = torch.tensor(raw_action_log_prob, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.long)

        trajectory = (state, action, raw_action, raw_action_log_prob, next_state, reward, done)
        self.buffer.save(trajectory)

    def get_action_deterministic(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            avg, std = self.policy_net(state)
            action = torch.tanh(avg)
        return action.detach().numpy()

    def get_action_log_prob(self, state):
        state = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            avg, std = self.policy_net(state)
            distribution = Normal(loc=avg, scale=std)

            chosen_raw_action = distribution.sample()
            chosen_raw_action_log_prob = distribution.log_prob(chosen_raw_action).sum(dim=-1)
            chosen_action = torch.tanh(chosen_raw_action)

        return chosen_action.detach().numpy(), chosen_raw_action.detach().numpy(), chosen_raw_action_log_prob.item()

    def update_policy_value_net(self):
        states, actions, raw_actions, raw_action_log_probs, next_states, rewards, dones = self.buffer.sample()

        # GAE
        with torch.no_grad():
            pred_state_values = self.value_net(states)
            next_state_values = self.value_net(next_states)

            deltas = rewards + (1-dones)*self.gamma*(next_state_values) - pred_state_values

            advantages = torch.zeros_like(rewards)
            last_advantage = 0.0

            for t in reversed(range(len(states))):
                advantages[t] = (1-dones[t])*self.gae_lmda*self.gamma*last_advantage + deltas[t]
                last_advantage = advantages[t]

            returns = advantages + pred_state_values

            advantages = advantages.squeeze(dim=-1)
            returns =returns.squeeze(dim=-1)

            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        dataset = TensorDataset(states, raw_actions, raw_action_log_probs, returns, advantages)
        dataloader = DataLoader(dataset, self.batch_size, shuffle=True, drop_last=False)

        for epoch in range(1, self.n_epochs+1):
            for batch in dataloader:
                states, raw_actions, raw_action_log_probs, returns, advantages = batch

                # Policy Loss
                avgs, stds = self.policy_net(states)
                distributions = Normal(avgs, stds)
                current_raw_action_log_probs = distributions.log_prob(raw_actions).sum(dim=-1)

                ratio = torch.exp(current_raw_action_log_probs - raw_action_log_probs)
                surrogate_1 = ratio*advantages
                surroaget_2 = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio)*advantages
                loss_policy = -torch.min(surrogate_1, surroaget_2).mean()

                # Value Loss
                pred_state_values = self.value_net(states).squeeze(-1)
                loss_value = nn.functional.mse_loss(returns, pred_state_values)

                # Entropy Loss
                entropy = distributions.entropy().sum(dim=-1).mean()

                # Total Loss
                total_loss = loss_policy + self.vf_coef*loss_value - self.entropy_coef*entropy

                # Update Network
                self.optim_policy_net.zero_grad()
                self.optim_value_net.zero_grad()

                total_loss.backward()

                self.optim_policy_net.step()
                self.optim_value_net.step()


def train_agent(agent:PpoAgent, num_steps):
    episode_reward_history = []
    episode_reward = 0.0
    episode_cnt = 0

    state, info = env.reset()

    for step in tqdm(range(1, num_steps+1)):
        action, raw_action, raw_action_log_prob = agent.get_action_log_prob(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.add_buffer(state, action, raw_action, raw_action_log_prob, next_state, reward, done)

        episode_reward += reward
        state = next_state

        if done:
            episode_reward_history.append(episode_reward)
            episode_reward = 0.0
            episode_cnt += 1

            recent_reward = np.mean(episode_reward_history[-100:])

            if recent_reward>=260:
                print('Early Stopped!!')
                print(f'Episode : {episode_cnt}  Recnet reward : {recent_reward:.4f}')
                break

            if episode_cnt%10==0:
                print(f'Episode : {episode_cnt}  Recnet reward : {recent_reward:.4f}')

            state, info = env.reset()

        if agent.buffer.is_full():
            agent.update_policy_value_net()


def render_agent(agent:PpoAgent, num_episodes=10):
    render_env = gym.make(
        id="LunarLander-v3",
        continuous=True,
        render_mode='human'
    )

    _ = str(input('Press ENTER to start rendering : '))

    for episode in range(1, num_episodes+1):
        state, info = render_env.reset()
        total_reward = 0.0

        while True:
            action = agent.get_action_deterministic(state)
            next_state, reward, terminated, truncated, info = render_env.step(action)
            done = terminated or truncated
            total_reward += reward

            if done:
                break

            state = next_state

        print(f'Episode : {episode}  Total Reward : {total_reward:.4f}')
    render_env.close()


if __name__=='__main__':
    agent = PpoAgent(
        observation_size=env.observation_space.shape[0],
        action_size=env.action_space.shape[0],
        gamma=0.99,
        lr=3e-4,
        buffer_max_len=2048,
        gae_lmda=0.95,
        n_epochs=10,
        batch_size=128,
        clip_ratio=0.2,
        vf_coef=0.5,
        entropy_coef=0.005
    )

    train_agent(agent, num_steps=10000000)
    render_agent(agent, num_episodes=20)

