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


class SarsaOffAgent:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0.1
        self.alpha = 0.1

        self.Q = defaultdict(lambda: 0)
        self.policy = defaultdict(lambda: {0:0.25, 1:0.25, 2:0.25, 3:0.25})
        self.policy_b = defaultdict(lambda: {0:0.25, 1:0.25, 2:0.25, 3:0.25})
        self.memory = deque(maxlen=2)

    def reset_memory(self):
        self.memory.clear()

    def get_action_from_policy(self, state):
        action_probs = self.policy_b[state]
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

    def update_Q_policy(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return

        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]

        if done:
            next_q = 0
            rho = 1
        else:
            next_q = self.Q[(next_state, next_action)]
            rho = self.policy[next_state][next_action] / self.policy_b[next_state][next_action]

        target = reward + self.gamma*rho*next_q
        self.Q[(state, action)] += self.alpha*(target - self.Q[(state, action)])

        self.policy[state] = self.get_greedy_policy(state, epsilon=0.0)
        self.policy_b[state] = self.get_greedy_policy(state, epsilon=self.epsilon)


def train_agent(agent:SarsaOffAgent, num_episodes):
    for _ in tqdm(range(num_episodes)):
        state, info = env.reset()
        agent.reset_memory()

        while True:
            action = agent.get_action_from_policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.update_Q_policy(state, action, reward, terminated or truncated)

            if terminated or truncated:
                agent.update_Q_policy(next_state, None, None, None)
                break

            state = next_state


def get_optimal_policy(agent:SarsaOffAgent):
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
    agent = SarsaOffAgent()
    train_agent(agent, num_episodes=10000)

    optimal_policy = get_optimal_policy(agent)
    render_optimal_policy(optimal_policy)
