import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict, deque
import gymnasium as gym

env = gym.make('Taxi-v3')

class QLearningAgent:
    def __init__(self, action_size):
        self.gamma = 0.9
        self.epsilon = 0.1
        self.alpha = 0.8
        self.action_size = action_size

        self.Q = defaultdict(lambda: 0)

    def get_action_from_Q(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            q_values = [self.Q[(state, action)] for action in range(self.action_size)]
            max_q_value = max(q_values)
            max_action_candidates = [idx for idx, q_value in enumerate(q_values) if q_value == max_q_value]
            return np.random.choice(max_action_candidates)

    def update_Q(self, state, action, reward, next_state, done):
        if done:
            max_next_q = 0
        else:
            next_qs = [self.Q[next_state, next_action] for next_action in range(self.action_size)]
            max_next_q = max(next_qs)

        target = reward + self.gamma*max_next_q
        self.Q[(state, action)] += self.alpha * (target - self.Q[(state, action)])


def train_agent(agent:QLearningAgent, num_episodes):
    for _ in tqdm(range(num_episodes)):
        state, info = env.reset()

        while True:
            action = agent.get_action_from_Q(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.update_Q(state, action, reward, next_state, terminated or truncated)

            if terminated or truncated:
                break

            state = next_state


def get_optimal_policy(agent:QLearningAgent):
    optimal_policy = {}

    for state in range(env.observation_space.n):
        q_values = []
        for action in range(env.action_space.n):
            q_values.append(agent.Q[(state, action)])
        max_action = np.argmax(q_values)

        optimal_policy[state] = max_action
    return optimal_policy


def render_optimal_policy(optimal_policy:dict, num_render=1):
    _ = str(input('Press ENTER to start rendering : '))

    for _ in range(num_render):
        tmp_env = gym.make(
            id='Taxi-v3',
            render_mode='human'
        )

        state, info = tmp_env.reset()
        step_count = 0

        while True:
            action = optimal_policy[state]
            next_state, reward, terminated, truncated, info = tmp_env.step(action)

            print(f'Step      : {step_count}')
            print(f'State     : {state}')
            print(f'Action    : {action}')
            print(f'Next State: {next_state}')
            print(f'Reward    : {reward}', end='\n\n')
            time.sleep(1)

            step_count += 1

            if terminated or truncated:
                break

            state = next_state

        tmp_env.close()


if __name__=='__main__':
    agent = QLearningAgent(action_size=env.action_space.n)
    train_agent(agent, num_episodes=10000)

    optimal_policy = get_optimal_policy(agent)
    render_optimal_policy(optimal_policy)
