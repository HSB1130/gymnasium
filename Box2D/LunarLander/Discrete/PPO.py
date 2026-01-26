from tqdm import tqdm
import numpy as np
import gymnasium as gym
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.categorical import Categorical


env = gym.make(
    id='LunarLander-v3',
    render_mode=None
)


class RolloutBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def is_full(self):
        return len(self.buffer) >= self.buffer_size

    def save(self, trajectory):
        self.buffer.append(trajectory)

    def sample(self):
        states, actions, action_log_probs, next_states, rewards, dones = map(torch.stack, zip(*self.buffer))
        rewards = rewards.unsqueeze(dim=1)
        dones = dones.unsqueeze(dim=1)
        self.buffer.clear()
        return states, actions, action_log_probs, next_states, rewards, dones


class PolicyNet(nn.Module):
    def __init__(self, observation_size, action_size, hidden_size=64):
        super().__init__()

        self.FC = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, state):
        action_logit = self.FC(state)
        return action_logit


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
        state_value = self.FC(state)
        return state_value


class PpoAgent:
    def __init__(self, observation_size, action_size, n_steps, n_epochs, batch_size, lr, gae_lmda, clip_ratio, vf_coef, entropy_coef):
        self.observation_size = observation_size
        self.action_size = action_size

        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.gamma = 0.99
        self.lr = lr

        self.gae_lmda = gae_lmda
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef

        self.rollout_buffer = RolloutBuffer(buffer_size=n_steps)

        self.policy_net = PolicyNet(self.observation_size, self.action_size)
        self.optimizer_policy_net = optim.Adam(params=self.policy_net.parameters(), lr=self.lr)

        self.value_net = ValueNet(self.observation_size)
        self.optimizer_value_net = optim.Adam(params=self.value_net.parameters(), lr=self.lr)

    def is_buffer_full(self):
        return self.rollout_buffer.is_full()

    def save_buffer(self, state, action, action_log_prob, next_state, reward, done):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        action_log_prob = torch.tensor(action_log_prob, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        trajactory = (state, action, action_log_prob, next_state, reward, done)
        self.rollout_buffer.save(trajactory)

    def get_action_deterministic(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action_logit = self.policy_net(state)
        max_prob_action = torch.argmax(action_logit)
        return max_prob_action.detach().numpy()

    def get_action_log_prob(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action_logit = self.policy_net(state)
            distribution = Categorical(logits=action_logit)
            chosen_action = distribution.sample()
            chosen_action_log_prob = distribution.log_prob(chosen_action)
        return chosen_action.detach().numpy(), chosen_action_log_prob.item()

    def update_policy_value_net(self):
        states, actions, old_action_log_probs, next_states, rewards, dones = self.rollout_buffer.sample()

        # GAE
        with torch.no_grad():
            state_values = self.value_net(states)
            next_state_values = self.value_net(next_states)

            # delta  = {Rt+gamma*Vw(St+1)} - Vw(St)
            deltas = rewards + (1-dones)*self.gamma*next_state_values - state_values

            advantages = torch.zeros_like(rewards)
            last_advantage = 0.0

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
        dataset = TensorDataset(states, actions, old_action_log_probs, returns, advantages)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        for epoch in range(1, self.n_epochs+1):
            for batch in dataloader:
                states_b, actions_b, old_action_log_probs_b, returns_b, advantages_b = batch

                # PolciyNet Loss
                logits = self.policy_net(states_b)
                distributions = Categorical(logits=logits)
                chosen_action_log_probs = distributions.log_prob(actions_b)

                # ValueNet Loss
                pred_values = self.value_net(states_b).squeeze(-1)
                value_loss = nn.functional.mse_loss(pred_values, returns_b)

                ratio = torch.exp(chosen_action_log_probs - old_action_log_probs_b)
                surrogate_1 = advantages_b*ratio
                surrogate_2 = advantages_b*torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio)
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

                # Entropy
                entropy = distributions.entropy().mean()

                # Total Loss
                total_loss = policy_loss + self.vf_coef*value_loss - self.entropy_coef*entropy

                self.optimizer_value_net.zero_grad()
                self.optimizer_policy_net.zero_grad()

                total_loss.backward()

                self.optimizer_value_net.step()
                self.optimizer_policy_net.step()


def train_agent(agent: PpoAgent, num_steps):
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
            recent_reward = np.mean(episode_reward_history[-100:])
            episode_count += 1

            if recent_reward>=260:
                print('Early Stopped!!')
                print(f'Episode: {episode_count}, Recent reward: {recent_reward:.3f}')
                break

            if episode_count%100==0:
                print(f'Episode: {episode_count}, Recent reward: {recent_reward:.3f}')

            state, info = env.reset()
            episode_reward = 0.0

        if agent.is_buffer_full():
            agent.update_policy_value_net()


def render_agent(agent:PpoAgent, num_episodes):
    render_env = gym.make(
        id='LunarLander-v3',
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
        action_size=env.action_space.n,
        n_steps=2048,
        n_epochs=10,
        batch_size=64,
        lr=3e-4,
        gae_lmda=0.95,
        clip_ratio=0.2,
        vf_coef=0.7,
        entropy_coef=0.01
    )

    train_agent(agent, num_steps=2000000)
    render_agent(agent, num_episodes=20)
    env.close()
