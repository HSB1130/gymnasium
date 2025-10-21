import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import gymnasium as gym

env = gym.make(
    id = 'FrozenLake-v1',
    desc=None,
    map_name="8x8",
    is_slippery=False,
    render_mode=None
)

class MonteCarloOffAgent:
    def __init__(self):
        self.gamma = 0.99
        self.epsilon = 0.15
        self.alpha = 0.1
        self.action_size = env.action_space.n

        self.Q = defaultdict(lambda: 0)
        self.memory = []

    def add_memory(self, state, action, reward):
        self.memory.append((state, action, reward))

    def reset_memory(self):
        self.memory.clear()

    def get_action_from_Q(self, state, is_greedy=False):
        epsilon = 0 if is_greedy else self.epsilon

        if np.random.rand() < epsilon:
            return np.random.choice(self.action_size)
        else:
            q_values = [self.Q[(state, action)] for action in range(self.action_size)]
            max_action_value = max(q_values)

            max_action_candidates = [idx for idx, action_value in enumerate(q_values) if action_value==max_action_value]
            return np.random.choice(max_action_candidates)

    def get_policy_prob(self, state, action, is_greedy=False):
        q_values = [self.Q[(state, a)] for a in range(self.action_size)]
        max_q_value = max(q_values)

        optimal_actions = [a for a in range(self.action_size) if q_values[a] == max_q_value]
        num_optimal_actions = len(optimal_actions)

        if is_greedy:
            if action in optimal_actions:
                return 1.0 / num_optimal_actions
            else:
                return 0.0
        else:
            if action in optimal_actions:
                return (1 - self.epsilon)/num_optimal_actions + self.epsilon/self.action_size
            else:
                return self.epsilon / self.action_size

    def update_Q(self):
        G = 0
        rho = 1

        for data in reversed(self.memory):
            state, action, reward = data
            key = (state, action)

            G = reward + self.gamma*G

            target_prob = self.get_policy_prob(state, action, is_greedy=True)
            behavior_prob = self.get_policy_prob(state, action, is_greedy=False)

            rho *= target_prob / behavior_prob

            if rho == 0:
                break

            self.Q[key] += self.alpha*rho*(G - self.Q[key])


def train_agent(agent:MonteCarloOffAgent, num_episodes):
    for i in tqdm(range(num_episodes)):
        state, info = env.reset()
        agent.reset_memory()

        while True:
            action = agent.get_action_from_Q(state, is_greedy=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.add_memory(state, action, reward)

            if terminated or truncated:
                agent.update_Q()
                break

            state = next_state


def get_optimal_policy(agent:MonteCarloOffAgent):
    optimal_policy = {}

    for state in range(env.observation_space.n):
        q_values = []
        for action in range(env.action_space.n):
            q_values.append(agent.Q[(state, action)])
        max_action = np.argmax(q_values)

        optimal_policy[state] = max_action
    return optimal_policy


def render_optimal_policy(optimal_policy, num_render=1):
    _ = str(input('Press ENTER to start rendering : '))

    for _ in range(num_render):
        tmp_env = gym.make(
            id = 'FrozenLake-v1',
            desc=None,
            map_name="8x8",
            is_slippery=False,
            render_mode='human'
        )

        state, info = tmp_env.reset()

        while True:
            action = optimal_policy[state]
            next_state, reward, terminated, truncated, info = tmp_env.step(action)

            print(f'State     : {state}')
            print(f'Action    : {action}')
            print(f'Reward    : {reward}')
            print(f'Next Sate : {next_state}', end='\n\n')
            time.sleep(1)

            if terminated or truncated:
                tmp_env.close()
                break

            state = next_state


if __name__=='__main__':
    agent = MonteCarloOffAgent()
    train_agent(agent, num_episodes=200000)

    optimal_policy = get_optimal_policy(agent)
    render_optimal_policy(optimal_policy)
