import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import gymnasium as gym

'''
    0 : 왼쪽
    1 : 아래
    2 : 오른쪽
    3 : 위
'''

env = gym.make(
    id = 'FrozenLake-v1',
    desc=None,
    map_name="8x8",
    is_slippery=False,
    render_mode=None
)

class MonteCarloAgent:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0.1
        self.alpha = 0.1
        self.action_size = env.action_space.n

        self.Q = defaultdict(lambda: 0)
        self.memory = []

    def add_memory(self, state, action, reward):
        '''
        memory에 1번의 time-step에서 (state, action, reward) 추가
        '''
        self.memory.append((state, action, reward))

    def reset_memory(self):
        '''
        memory 전체 초기화
        '''
        self.memory.clear()

    def get_action_from_Q(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            q_values = [self.Q[(state, action)] for action in range(self.action_size)]
            max_action_value = max(q_values)

            max_action_candidates = [idx for idx, action_value in enumerate(q_values) if action_value==max_action_value]
            return np.random.choice(max_action_candidates)

    def update_Q_Policy(self):
        G = 0.0

        for data in reversed(self.memory):
            state, action, reward = data
            key = (state, action)

            G = reward + self.gamma*G
            self.Q[key] += self.alpha*(G - self.Q[key])


def train_agent(agent:MonteCarloAgent, num_episodes:int):
    for i in tqdm(range(num_episodes)):
        state, info = env.reset()
        agent.reset_memory()

        terminated = False

        while True:
            action = agent.get_action_from_Q(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.add_memory(state, action, reward)

            if terminated or truncated:
                agent.update_Q_Policy()
                break

            state = next_state


def get_optimal_policy(agent:MonteCarloAgent):
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
    agent = MonteCarloAgent()
    train_agent(agent, num_episodes=100000)

    optimal_policy = get_optimal_policy(agent)
    render_optimal_policy(optimal_policy)
    env.close()
