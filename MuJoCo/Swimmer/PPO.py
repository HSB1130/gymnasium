from tqdm import tqdm
import numpy as np
import gymnasium as gym
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
        self.log_std_header = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        h = self.FC(state)
        avg = self.avg_header(h)
        log_std = self.log_std_header(h)
        std = nn.functional.softplus(log_std)
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
        states, actions, action_log_probs, next_states, rewards, dones = map(torch.stack, zip(*self.buffer))
        rewards = rewards.unsqueeze(dim=1)
        dones = dones.unsqueeze(dim=1)
        self.buffer.clear()
        return states, actions, action_log_probs, next_states, rewards, dones


class PpoAgent:
    def __init__(self, observation_dim, action_dim, hidden_dim, gamma, lr, buffer_size, n_epoch, batch_size, gae_lmda, clip_ratio, vf_coef, entropy_coef):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

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

    def save_buffer(self, state, action, action_log_prob, next_state, reward, done):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        action_log_prob = torch.tensor(action_log_prob, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        trajectory = (state, action, action_log_prob, next_state, reward, done)
        self.buffer.save(trajectory)

    def get_action_log_prob(self, state):
        state = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            avg, std = self.policy_net(state)
            distribution = Normal(avg, std)

            action = distribution.sample()
            action_tanh = torch.tanh(action)

            action_log_prob = distribution.log_prob(action).sum(dim=-1)
            action_log_prob -= torch.log(1 - action_tanh.pow(2) + 1e-6).sum(dim=-1)

        return action_tanh.detach().numpy(), action_log_prob.item()

    def get_action_deterministic(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            avg, std = self.policy_net(state)
            action = torch.tanh(avg)
        return action.detach().numpy()

    def update_policy_value_net(self):
        states, actions, action_log_probs, next_states, rewards, dones = self.buffer.sample()

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

        dataset = TensorDataset(states, actions, action_log_probs, advantages, returns)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        for epoch in range(1, self.n_epoch+1):
            for batch in dataloader:
                states_b, actions_b, old_action_log_probs_b, advantages_b, returns_b = batch

                # Value Loss
                state_values = self.value_net(states_b).squeeze(-1)
                loss_value = nn.functional.mse_loss(state_values, returns_b)

                # Policy Loss
                avgs, stds = self.policy_net(states_b)
                distributions = Normal(avgs, stds)

                eps = 1e-6
                actions_b = actions_b.clamp(-1+eps, 1-eps)
                z = torch.atanh(actions_b)

                chosen_action_log_probs = distributions.log_prob(z).sum(dim=-1)
                chosen_action_log_probs -= torch.log(1-actions_b.pow(2) + eps).sum(dim=-1)

                ratio = torch.exp(chosen_action_log_probs - old_action_log_probs_b)
                surrogate_1 = advantages_b * ratio
                surrogate_2 = advantages_b * torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio)
                loss_policy = -torch.min(surrogate_1, surrogate_2).mean()

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
        action, action_log_prob = agent.get_action_log_prob(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.save_buffer(state, action, action_log_prob, next_state, reward, done)

        episode_reward += reward
        state = next_state

        if done:
            episode_reward_history.append(episode_reward)
            episode_reward = 0.0
            episode_cnt += 1

            recent_reward = np.mean(episode_reward_history[-100:])

            if recent_reward>=80:
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
        buffer_size=2048,
        n_epoch=10,
        batch_size=64,
        gae_lmda=0.95,
        clip_ratio=0.2,
        vf_coef=0.5,
        entropy_coef=0.01
    )

    train_agent(
        agent,
        num_steps=10000000
    )

    render_agent(
        agent,
        num_episodes=10
    )
