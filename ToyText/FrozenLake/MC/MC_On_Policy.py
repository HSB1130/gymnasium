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

        self.Q = defaultdict(lambda: 0)
        self.policy = defaultdict(lambda: {0:0.25, 1:0.25, 2:0.25, 3:0.25})
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

    def get_action_from_policy(self, state):
        '''
        현재 state에서 policy에 의해 하나의 action 반환
        '''
        action_probs = self.policy[state]
        actions = list(action_probs.keys())
        prob = list(action_probs.values())

        return np.random.choice(actions, p=prob)

    def get_greedy_policy(self, state, epsilon=0.0, action_size=4):
        '''
        Q-value를 보고, state 에서 가장 큰 값을 가지는 action을 선택
        대신 epsilon-greedy를 적용
        '''
        q_values = [self.Q[(state, a)] for a in range(action_size)]
        max_q = max(q_values)

        candidates = [a for a, q in enumerate(q_values) if q == max_q]
        max_action = np.random.choice(candidates)

        prob = {a: epsilon / action_size for a in range(action_size)}
        prob[max_action] += (1 - epsilon)
        return prob

    def update_Q_Policy(self):
        '''
        memory에 저장 된 1개의 episode를 통해 Q-value 갱신
        갱신 된 Q-value를 통해 policy 갱신
        '''
        G = 0.0
        visited_states = set()

        for data in reversed(self.memory):
            state, action, reward = data
            key = (state, action)

            visited_states.add(state)

            G = reward + self.gamma*G
            self.Q[key] += self.alpha*(G - self.Q[key])

        for visited_state in visited_states:
            self.policy[visited_state] = self.get_greedy_policy(visited_state, epsilon=self.epsilon)


def train_agent(agent:MonteCarloAgent, num_episodes:int):
    for i in tqdm(range(num_episodes)):
        state, info = env.reset()
        agent.reset_memory()

        terminated = False

        while True:
            action = agent.get_action_from_policy(state)
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
