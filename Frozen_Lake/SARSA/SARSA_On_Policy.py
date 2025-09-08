import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict, deque
import gymnasium as gym

env = gym.make(
    id = 'FrozenLake-v1',
    desc=None,
    map_name="8x8",
    is_slippery=False,
    render_mode=None
)


class SarsaAgent:
    def __init__(self):
        self.gamma = 0.9
        self.epsiloin = 0.1
        self.alpha = 0.1

        self.Q = defaultdict(lambda: 0)
        self.policy = defaultdict(lambda: {0:0.25, 1:0.25, 2:0.25, 3:0.25})
        self.memory = deque(maxlen=2)

    def reset_memory(self):
        self.memory.clear()

    def get_action_from_policy(self, state):
        action_probs = self.policy[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def get_greedy_policy(self, state, epsilon=0.0, action_size=4):
        q_values = [self.Q[(state, a)] for a in range(action_size)]
        max_q = max(q_values)

        candidates = [a for a, q in enumerate(q_values) if q == max_q]
        max_action = np.random.choice(candidates)

        prob = {a: epsilon / action_size for a in range(action_size)}
        prob[max_action] += (1 - epsilon)
        return prob

    def update_Q_Policy(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return

        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]

        if done:
            next_Q = 0
        else:
            next_Q = self.Q[(next_state, next_action)]

        target = reward + self.gamma*next_Q
        self.Q[(state, action)] += self.alpha*(target-self.Q[(state, action)])
        self.policy[state] = self.get_greedy_policy(state, epsilon=self.epsiloin)


def run_episodes(agent:SarsaAgent, num_episodes):
    for _ in tqdm(range(num_episodes)):
        state, info = env.reset()
        agent.reset_memory()

        while True:
            action = agent.get_action_from_policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.update_Q_Policy(state, action, reward, terminated or truncated)

            if terminated or truncated:
                agent.update_Q_Policy(next_state, None, None, None)
                break

            state = next_state


def get_optimal_policy(agent:SarsaAgent):
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

            if terminated:
                tmp_env.close()
                break

            state = next_state


if __name__=='__main__':
    agent = SarsaAgent()
    run_episodes(agent, num_episodes=1000000)

    optimal_policy = get_optimal_policy(agent)
    render_optimal_policy(optimal_policy)
    env.close()
