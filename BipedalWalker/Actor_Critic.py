import numpy as np
import gymnasium as gym
import torch
from torch import nn, optim
from torch.distributions import TransformedDistribution
from torch.distributions.normal import Normal
from torch.distributions.transforms import TanhTransform

env = gym.make(
    id = "BipedalWalker-v3",
    render_mode=None
)


class PolicyNet(nn.Module):
    def __init__(self, observation_size=24, action_size=4, act_low=-1.0, act_high=1.0, hidden_size=128):
        super().__init__()
        self.log_std_min = -5.0
        self.log_std_max = 2.0

        self.FC = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.avg_head = nn.Linear(hidden_size, action_size)
        self.log_std_head = nn.Linear(hidden_size, action_size)

        # 행동 범위: 스칼라/벡터 모두 허용
        act_low_t  = torch.as_tensor(act_low,  dtype=torch.get_default_dtype())
        act_high_t = torch.as_tensor(act_high, dtype=torch.get_default_dtype())
        if act_low_t.ndim == 0:
            act_low_t  = act_low_t.repeat(action_size)
        if act_high_t.ndim == 0:
            act_high_t = act_high_t.repeat(action_size)
        assert act_low_t.shape == (action_size,) and act_high_t.shape == (action_size,)
        self.register_buffer("act_low",  act_low_t)
        self.register_buffer("act_high", act_high_t)

        # 파라미터 초기화
        nn.init.zeros_(self.avg_head.weight)
        nn.init.zeros_(self.avg_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, -0.5)

    def forward(self, state):
        hidden = self.FC(state)
        avg = self.avg_head(hidden)
        log_std = self.log_std_head(hidden)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return avg, std

    def get_action_log_prob(self, state):
        avg, std = self.forward(state)
        base_dist = Normal(avg, std)
        tanh_dist = TransformedDistribution(base_dist, [TanhTransform(cache_size=1)])

        u = tanh_dist.rsample()                        # (-1, 1), (B, action_dim)
        logp_u = tanh_dist.log_prob(u).sum(-1, keepdim=True)  # (B, 1)

        # [low, high] 스케일
        action = (self.act_high + self.act_low)/2 + (self.act_high - self.act_low)/2 * u
        return action, logp_u


class ValueNet(nn.Module):
    def __init__(self, observation_size=24, hidden_size=128):
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
    def __init__(self, observation_size=24, action_size=4, lr=1e-4):
        self.gamma = 0.99
        self.lr = lr

        self.policy_net = PolicyNet(observation_size, action_size)
        self.optimizer_policy_net = optim.Adam(params=self.policy_net.parameters(), lr=self.lr)

        self.value_net = ValueNet(observation_size)
        self.optimizer_value_net = optim.Adam(params=self.value_net.parameters(), lr=lr)

    def get_action_log_prob_from_policy_net(self, state):
        state = torch.as_tensor(state, dtype=torch.float32)
        action, action_log_prob = self.policy_net.get_action_log_prob(state)
        return action, action_log_prob

    def update_policy_value_net(self, state, action_log_prob, next_state, reward, done):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.long)

        pred_state_value = self.value_net(state)
        with torch.no_grad():
            next_state_value = self.value_net(next_state)
            target_state_value = reward + (1-done)*self.gamma*next_state_value

        target_state_value = target_state_value.detach()
        loss_value = nn.functional.mse_loss(pred_state_value, target_state_value)

        delta = target_state_value - pred_state_value
        delta = delta.detach()
        loss_policy = -delta*action_log_prob

        self.optimizer_value_net.zero_grad()
        self.optimizer_policy_net.zero_grad()

        loss_value.backward()
        loss_policy.backward()

        self.optimizer_value_net.step()
        self.optimizer_policy_net.step()


def train_agent(agent:AcAgent, num_episodes):
    reward_history = []

    for episode in range(1, num_episodes+1):
        state, info = env.reset()
        done = False

        total_reward = 0.0

        while not done:
            action, action_log_prob = agent.get_action_log_prob_from_policy_net(state)
            next_state, reward, terminated, truncated, info = env.step(action.detach().numpy())
            done = terminated or truncated

            agent.update_policy_value_net(state, action_log_prob, next_state, reward, done)

            total_reward += reward
            state = next_state

        print(f'Episode : {episode}  Total Reward : {total_reward:.4f}')



def render_agent(agent:AcAgent, num_episodes):
    render_env = gym.make(
        id = "BipedalWalker-v3",
        render_mode='human'
    )

    for episode in range(1, num_episodes+1):
        state, info = render_env.reset()
        done = False

        total_reward = 0.0

        while not done:
            action, log_prob = agent.get_action_log_prob_from_policy_net(state)
            next_state, reward, terminated, truncated, info = render_env.step(action.detach().numpy())
            done = terminated or truncated

            total_reward += reward
            state = next_state

        print(f'Episode : {episode}  Total Reward : {total_reward:.4f}')

    render_env.close()


if __name__=='__main__':
    agent = AcAgent(
        observation_size=env.observation_space.shape[0],
        action_size=env.action_space.shape[0]
    )

    train_agent(agent, num_episodes=10000)
    render_agent(agent, num_episodes=10)