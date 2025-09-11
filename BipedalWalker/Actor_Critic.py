import numpy as np
import gymnasium as gym
import torch
from torch import nn, optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


env = gym.make(
    id = "BipedalWalker-v3",
    render_mode=None
)


class PolicyNet(nn.Module):
    def __init__(self, observation_size=24, action_size=4, hidden_size=128):
        super().__init__()

        self.FC = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.avg_head = nn.Linear(hidden_size, action_size)
        self.log_std_head = nn.Linear(hidden_size, action_size)

        nn.init.zeros_(self.avg_head.weight)
        nn.init.zeros_(self.avg_head.bias)

        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, -0.5)

    def forward(self, state):
        hidden = self.FC(state)
        avg = self.avg_head(hidden)
        log_std = self.log_std_head(hidden)
        std = torch.exp(log_std)
        return avg, std

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



    def get_action_from_policy_net(self, state):
        state = torch.tensor(state, dtype=torch.float32)

        avg, std = self.policy_net(state)
        distribution = Normal(loc=avg, scale=std)


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
            action = agent.get_action_from_policy_net(state)
            next_state, reward, terminated, truncated, info = render_env.step(action)
            done = terminated or truncated

            total_reward += reward
            state = next_state

        print(f'Episode : {episode}  Total Reward : {total_reward:.4f}')

    render_env.close()


class SquashedGaussianPolicy(nn.Module):
    def __init__(self,
            state_dim=10,
            action_dim=4,
            hidden=(128, 128),
            act_low=-1.0, act_high=1.0
        ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden[1], action_dim)
        self.log_std_head = nn.Linear(hidden[1], action_dim)

        # 행동 범위 (Box space일 경우 벡터로도 OK)
        self.register_buffer("act_low", torch.as_tensor(act_low).repeat(action_dim) if isinstance(act_low, (int,float)) else torch.as_tensor(act_low))
        self.register_buffer("act_high", torch.as_tensor(act_high).repeat(action_dim) if isinstance(act_high, (int,float)) else torch.as_tensor(act_high))

        # 합리적 초기값
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)

        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, -0.5)

        self.LOG_STD_MIN, self.LOG_STD_MAX = -5.0, 2.0  # 안정화용 클램프

    def forward(self, s):
        h = self.net(s)
        mu = self.mu_head(h)
        log_std = torch.clamp(self.log_std_head(h), self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std

    @torch.no_grad()
    def act(self, s, deterministic=False):
        """환경 스텝 시 사용 (탐험 off 시 deterministic=True)"""
        mu, std = self.forward(s)
        if deterministic:
            z = mu
        else:
            z = mu + std * torch.randn_like(std)
        a, _ = self._squash_and_scale(z, mu, std)
        return a

    def sample_and_log_prob(self, s):
        """학습 시 사용: reparameterization + log_prob (tanh 보정 포함)"""
        mu, std = self.forward(s)
        dist = Normal(mu, std)
        # reparameterization trick
        z = dist.rsample()
        a, logp = self._squash_and_scale(z, mu, std, return_logp=True)
        return a, logp, mu, std

    def _squash_and_scale(self, z, mu, std, return_logp=False):
        # tanh squashing
        u = torch.tanh(z)  # (-1,1)
        # scale to [low, high]
        a = (self.act_high + self.act_low)/2 + (self.act_high - self.act_low)/2 * u

        if not return_logp:
            return a, None

        # log-prob with tanh correction (SAC/PPO에서 중요)
        # base log_prob
        dist = Normal(mu, std)
        logp_z = dist.log_prob(z).sum(dim=-1, keepdim=True)
        # tanh change-of-variables: sum log(1 - tanh(z)^2)
        # add small epsilon for numerical stability
        eps = 1e-6
        log_det_jac = torch.log(1 - u.pow(2) + eps).sum(dim=-1, keepdim=True)
        logp_u = logp_z - log_det_jac

        # 선형 스케일링의 야코비안은 상수이므로 로그확률에 더해져도 학습엔 상수항(무시 가능).
        return a, logp_u


if __name__=='__main__':
    agent = AcAgent(
        observation_size=env.observation_space.shape[0],
        action_size=env.action_space.shape[0]
    )

    # render_agent(agent)