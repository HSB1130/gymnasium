import numpy as np
import gymnasium as gym
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.distributions.normal import Normal
from torch.utils.data import TensorDataset, DataLoader


env = gym.make(
    id='Swimmer-v5',
    render_mode=None
)


class PolicyNet(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dim):
        super().__init__()

        self.FC = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.avg_header = nn.Linear(hidden_dim, action_dim)
        self.raw_std_header = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        h = self.FC(state)
        avg = self.avg_header(h)
        raw_std = self.raw_std_header(h)
        std = nn.functional.softplus(raw_std) + 1e-4
        return avg, std


class ValueNet(nn.Module):
    def __init__(self, observation_dim, hidden_dim):
        super().__init__()

        self.FC = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.FC(state)


class RolloutBuffer:
    def __init__(self, buffer_size):
        self.buffer = []
        self.buffer_size = buffer_size

    def __len__(self):
        return len(self.buffer)

    def is_full(self):
        return len(self.buffer)>=self.buffer_size

    def save(self, trajectory):
        self.buffer.append(trajectory)

    def sample(self):
        states, actions, raw_actions, raw_action_log_probs, next_states, rewards, dones = map(torch.stack, zip(*self.buffer))
        rewards = rewards.unsqueeze(dim=1)
        dones = dones.unsqueeze(dim=1)
        self.buffer.clear()
        return states, actions, raw_actions, raw_action_log_probs, next_states, rewards, dones


class PpoAgent:
    def __init__(self, observation_dim, action_dim, hidden_dim, gamma, lr, buffer_size, n_epoch, batch_size, gae_lmda, clip_ratio, vf_coef, entropy_coef):
        self.gamma = gamma
        self.lr = lr

        self.buffer_size = buffer_size
        self.n_epoch = n_epoch
        self.batch_size = batch_size

        self.gae_lmda = gae_lmda
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef

        self.buffer= RolloutBuffer(self.buffer_size)

        self.policy_net = PolicyNet(observation_dim, action_dim, hidden_dim)
        self.policy_optimizer = optim.Adam(params=self.policy_net.parameters(), lr=self.lr)

        self.value_net = ValueNet(observation_dim, hidden_dim)
        self.value_optimizer = optim.Adam(params=self.value_net.parameters(), lr=self.lr)

    def save_buffer(self, state, action, raw_action, raw_action_log_prob, next_state, reward, done):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        raw_action = torch.tensor(raw_action, dtype=torch.float32)
        raw_action_log_prob = torch.tensor(raw_action_log_prob, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        trajectory = (state, action, raw_action, raw_action_log_prob, next_state, reward, done)
        self.buffer.save(trajectory)

    def get_action_log_prob(self, state):
        state = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            avg, std = self.policy_net(state)
            distribution = Normal(avg, std)

            chosen_raw_action = distribution.sample()
            chosen_raw_action_log_prob = distribution.log_prob(chosen_raw_action).sum(dim=-1)
            chosen_action = torch.tanh(chosen_raw_action)

        return chosen_action.detach().numpy(), chosen_raw_action.detach().numpy(), chosen_raw_action_log_prob.item()

    def get_action_deterministic(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            avg, std = self.policy_net(state)
            action = torch.tanh(avg)
        return action.detach().numpy()

    def update_policy_value_net(self):
        states, actions, raw_actions, raw_action_log_probs, next_states, rewards, dones = self.buffer.sample()

        # GAE
        with torch.no_grad():
            state_values = self.value_net(states)
            next_state_values = self.value_net(next_states)

            deltas = rewards + (1-dones)*self.gamma*next_state_values - state_values

            advantages = torch.zeros_like(rewards)
            last_advantage = 0.0

            for t in reversed(range(len(rewards))):
                last_advantage = deltas[t] + (1-dones[t])*self.gae_lmda*self.gamma*last_advantage
                advantages[t] = last_advantage

            returns = advantages + state_values

            advantages = advantages.squeeze(dim=-1)
            returns = returns.squeeze(dim=-1)

            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        dataset = TensorDataset(states, raw_actions, raw_action_log_probs, advantages, returns)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        for epoch in range(1, self.n_epoch+1):
            for batch in dataloader:
                states, raw_actions, raw_action_log_probs, advantages, returns = batch

                # Policy Loss
                avgs, stds = self.policy_net(states)
                distributions = Normal(avgs, stds)
                current_raw_action_log_probs = distributions.log_prob(raw_actions).sum(dim=-1)

                ratio = torch.exp(current_raw_action_log_probs - raw_action_log_probs)
                surrogate_1 =  ratio * advantages
                surrogate_2 = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * advantages
                loss_policy = -torch.min(surrogate_1, surrogate_2).mean()

                # Value Loss
                state_values = self.value_net(states).squeeze(-1)
                loss_value = nn.functional.mse_loss(state_values, returns)

                # Enropy
                entropy = distributions.entropy().sum(dim=-1).mean()

                # Total Loss
                total_loss = loss_policy + self.vf_coef*loss_value - self.entropy_coef*entropy

                # Update Network
                self.value_optimizer.zero_grad()
                self.policy_optimizer.zero_grad()

                total_loss.backward()

                self.value_optimizer.step()
                self.policy_optimizer.step()


def train_agent(agent: PpoAgent, num_steps):
    episode_reward_history = []
    episode_reward = 0.0
    episode_cnt = 0

    state, info = env.reset()

    for step in tqdm(range(1, num_steps+1)):
        action, raw_action, raw_action_log_prob = agent.get_action_log_prob(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.save_buffer(state, action, raw_action, raw_action_log_prob, next_state, reward, done)

        episode_reward += reward
        state = next_state

        if done:
            episode_reward_history.append(episode_reward)
            episode_reward = 0.0
            episode_cnt += 1

            recent_reward = np.mean(episode_reward_history[-100:])

            if recent_reward>=100:
                print('Early Stopped!!')
                print(f'Episode : {episode_cnt}  Recent reward : {recent_reward:.4f}')
                break

            if episode_cnt%10==0:
                print(f'Episode : {episode_cnt}  Recent reward : {recent_reward:.4f}')

            state, info = env.reset()

        if agent.buffer.is_full():
            agent.update_policy_value_net()


def render_agent(agent: PpoAgent, num_episodes):
    render_env = gym.make(
        id='Swimmer-v5',
        render_mode='human'
    )

    _ = str(input('Press ENTER to start rendering : '))

    for episode in range(1, num_episodes+1):
        state, info = render_env.reset()

        total_reward = 0.0
        done = False

        while not done:
            action = agent.get_action_deterministic(state)
            next_state, reward, terminated, truncated, info = render_env.step(action)
            done = terminated or truncated

            total_reward += reward
            state = next_state

        print(f'Episode : {episode}  Total Reward : {total_reward:.4f}')
    render_env.close()


if __name__=='__main__':
    agent = PpoAgent(
        observation_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_dim=64,
        gamma=0.999,
        lr=3e-4,
        buffer_size=4096,
        n_epoch=10,
        batch_size=128,
        gae_lmda=0.95,
        clip_ratio=0.2,
        vf_coef=0.5,
        entropy_coef=0.01
    )

    train_agent(
        agent,
        num_steps=100000000
    )

    render_agent(
        agent,
        num_episodes=10
    )
