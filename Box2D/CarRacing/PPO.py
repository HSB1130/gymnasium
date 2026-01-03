import numpy as np
import gymnasium as gym
from tqdm import tqdm
from datetime import datetime
import torch
from torch import nn, optim
from torch.distributions.categorical import Categorical
from torch.utils.data import TensorDataset, DataLoader


env = gym.make(
    id="CarRacing-v3",
    render_mode=None,
    domain_randomize=False,
    continuous=False,
)


class PolicyNet(nn.Module):
    def __init__(self, observation_shape: tuple[int, int, int], action_size: int) -> None:
        super().__init__()
        self.observation_shape = observation_shape
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        conv_out_dim = self._get_conv_out_dim()
        self.head = nn.Sequential(
            nn.Linear(conv_out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_size),
        )

    def _get_conv_out_dim(self) -> int:
        with torch.no_grad():
            h, w, c = self.observation_shape
            dummy = torch.zeros(1, c, h, w)
            return self.conv(dummy).reshape(1, -1).size(1)

    def _format_obs(self, x: torch.Tensor) -> torch.Tensor:
        # Gymnasium returns HWC uint8; normalize and convert to CHW.
        # Handle both single observation (H, W, C) and batch (N, H, W, C)
        if x.dim() == 3:  # Single observation (H, W, C)
            x = x.permute(2, 0, 1).unsqueeze(0)
        elif x.dim() == 4:  # Batch (N, H, W, C)
            x = x.permute(0, 3, 1, 2)
        return x.float() / 255.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._format_obs(x)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.head(x)


class ValueNet(nn.Module):
    def __init__(self, observation_shape: tuple[int, int, int]) -> None:
        super().__init__()
        self.observation_shape = observation_shape
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        conv_out_dim = self._get_conv_out_dim()
        self.head = nn.Sequential(
            nn.Linear(conv_out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def _get_conv_out_dim(self) -> int:
        with torch.no_grad():
            h, w, c = self.observation_shape
            dummy = torch.zeros(1, c, h, w)
            return self.conv(dummy).reshape(1, -1).size(1)

    def _format_obs(self, x: torch.Tensor) -> torch.Tensor:
        # Gymnasium returns HWC uint8; normalize and convert to CHW.
        # Handle both single observation (H, W, C) and batch (N, H, W, C)
        if x.dim() == 3:  # Single observation (H, W, C)
            x = x.permute(2, 0, 1).unsqueeze(0)
        elif x.dim() == 4:  # Batch (N, H, W, C)
            x = x.permute(0, 3, 1, 2)
        return x.float() / 255.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._format_obs(x)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.head(x).squeeze(-1)


class RolloutBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def is_full(self):
        return len(self.buffer) >= self.buffer_size

    def save(self, state, action, action_log_prob, next_state, reward, done):
        trajectory = (state, action, action_log_prob, next_state, reward, done)
        self.buffer.append(trajectory)

    def sample(self):
        states, actions, action_log_probs, next_states, rewards, dones = map(torch.stack, zip(*self.buffer))
        rewards = rewards.unsqueeze(dim=1)
        dones = dones.unsqueeze(dim=1)
        self.buffer.clear()
        return states, actions, action_log_probs, next_states, rewards, dones


class PpoAgent:
    def __init__(self, observation_shape, action_size, gamma, gae_lmda, n_steps, n_epochs, lr, batch_size, clip_ratio, vf_coef, entropy_coef):
        self.observation_shape = observation_shape
        self.action_size = action_size

        self.gamma = gamma
        self.gae_lmda = gae_lmda

        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size

        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef

        self.policy_net = PolicyNet(self.observation_shape, self.action_size)
        self.optim_policy_net = optim.Adam(params=self.policy_net.parameters(), lr=self.lr)

        self.value_net = ValueNet(self.observation_shape)
        self.optim_value_net = optim.Adam(params=self.value_net.parameters(), lr=self.lr)

        self.rollout_buffer = RolloutBuffer(buffer_size=n_steps)

    def is_buffer_full(self):
        return self.rollout_buffer.is_full()

    def save_buffer(self, state, action, action_log_prob, next_state, reward, done):
        state = torch.tensor(state, dtype=torch.uint8)
        action = torch.tensor(action, dtype=torch.long)
        action_log_prob = torch.tensor(action_log_prob, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.uint8)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.long)
        self.rollout_buffer.save(state, action, action_log_prob, next_state, reward, done)

    def get_action_log_prob(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.uint8)
            logit = self.policy_net(state)
            distribution = Categorical(logits=logit)
            chosen_action = distribution.sample()
            chosen_action_log_prob = distribution.log_prob(chosen_action)
        return chosen_action.detach().item(), chosen_action_log_prob.item()

    def get_action_determistic(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.uint8)
            logit = self.policy_net(state)
            chosen_action = torch.argmax(logit)
        return chosen_action.detach().item()

    def update_policy_value_net(self):
        states, actions, action_log_probs_old, next_states, rewards, dones = self.rollout_buffer.sample()

        # GAE
        with torch.no_grad():
            state_values = self.value_net(states).unsqueeze(-1)
            next_state_values = self.value_net(next_states).unsqueeze(-1)

            # delta  = {Rt+gamma*Vw(St+1)} - Vw(St)
            deltas = rewards + (1-dones)*self.gamma*next_state_values - state_values

            last_advantage = 0.0
            advantages = torch.zeros_like(rewards)

            # A_t = delta_t + lmda*gamma*A_t+1
            for t in reversed(range(len(rewards))):
                last_advantage = deltas[t] + (1-dones[t])*self.gamma*self.gae_lmda*last_advantage
                advantages[t] += last_advantage

            # return = Q(s,a) = Advantage + V(s)
            returns = advantages + state_values

            advantages = advantages.squeeze(-1)
            returns = returns.squeeze(-1)

            # Advantage standardization (성능차이 큼)
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        # PPO update
        dataset = TensorDataset(states, actions, action_log_probs_old, returns, advantages)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        for epoch in range(1, self.n_epochs+1):
            for batch in dataloader:
                states_b, actions_b, old_action_log_probs_b, returns_b, advantages_b = batch

                # ValueNet Loss
                pred_values = self.value_net(states_b).squeeze(-1)
                value_loss = nn.functional.mse_loss(pred_values, returns_b)

                # PolciyNet Loss
                logits = self.policy_net(states_b)
                distributions = Categorical(logits=logits)
                chosen_action_log_probs = distributions.log_prob(actions_b)

                ratio = torch.exp(chosen_action_log_probs - old_action_log_probs_b)
                surrogate_1 = advantages_b*ratio
                surrogate_2 = advantages_b*torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio)
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                # Entropy
                entropy = distributions.entropy().mean()

                # Total Loss
                total_loss = policy_loss + self.vf_coef*value_loss - self.entropy_coef*entropy

                self.optim_value_net.zero_grad()
                self.optim_policy_net.zero_grad()

                total_loss.backward()

                self.optim_value_net.step()
                self.optim_policy_net.step()


def train_agent(agent:PpoAgent, num_steps=10000000):
    episode_reward_history = []
    episode_reward = 0.0
    episode_count = 0

    state, info = env.reset()

    for num_step in tqdm(range(1, num_steps+1)):
        action, action_log_prob = agent.get_action_log_prob(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.save_buffer(state, action, action_log_prob, next_state, reward, done)

        episode_reward += reward
        state = next_state

        if done:
            episode_reward_history.append(episode_reward)
            recent_reward = np.mean(episode_reward_history[-10:])
            episode_count += 1

            if recent_reward>=800:
                print('Early Stopped!!')
                print(f'Episode: {episode_count}, Recent reward: {recent_reward:.4f}')
                break

            if episode_count%10==0:
                print(f'Episode: {episode_count}, Recent reward: {recent_reward:.4f}')

            state, info = env.reset()
            episode_reward = 0.0

        if agent.is_buffer_full():
            agent.update_policy_value_net()

    env.close()


def render_agent(agent:PpoAgent, num_episodes=20):
    render_env = gym.make(
        id="CarRacing-v3",
        render_mode='human',
        domain_randomize=False,
        continuous=False,
    )

    _ = str(input('Press ENTER to start rendering : '))

    for num_episode in range(1, num_episodes+1):
        state, info = render_env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.get_action_determistic(state)
            next_state, reward, terminated, truncated, info = render_env.step(action)
            done = terminated or truncated

            total_reward += reward
            state = next_state

        print(f'Episode: {num_episode}  Reward: {total_reward:.4f}')
    render_env.close()


def save_policy_net_params(agent: PpoAgent):
    is_save = str(input('Do you want to save this Policy Net parameters? [Yes/No] : '))
    if is_save.strip().lower() == 'yes':
        now = datetime.now()
        datetime_str = now.strftime("%Y-%m-%d_%H:%M:%S")
        state_dict_saved_path = f'./model/ppo_policy_net_state_dict_{datetime_str}.pt'

        torch.save(agent.policy_net.state_dict(), state_dict_saved_path)
        print('Model state dict saved!!')
    else:
        print('Model doesn\'t saved!!')


if __name__=="__main__":
    agent = PpoAgent(
        observation_shape = env.observation_space.shape,
        action_size = env.action_space.n,
        gamma = 0.99,
        gae_lmda = 0.95,
        n_steps = 2048,
        n_epochs = 10,
        lr = 2e-4,
        batch_size = 128,
        clip_ratio = 0.2,
        vf_coef = 0.7,
        entropy_coef = 0.01
    )

    train_agent(agent, num_steps=1000000)
    render_agent(agent, num_episodes=20)
    save_policy_net_params(agent)
