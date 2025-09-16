import torch
import gymnasium as gym
from DQN import DqnAgent
from Actor_Critic import AcAgent


def render_agent(num_episodes, model_state_dict_saved_path):
    env = gym.make(
        'LunarLander-v3',
        render_mode='human'
    )

    agent = AcAgent(
        observation_size=env.observation_space.shape[0],
        action_size=env.action_space.n
    )

    state_dict = torch.load(model_state_dict_saved_path, map_location="cpu")
    agent.policy_net.load_state_dict(state_dict)

    for episode in range(1, num_episodes+1):
        state, info = env.reset()
        done = False

        total_reward = 0.0

        while not done:
            action = agent.get_action_by_determistic(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            state = next_state

        print(f'Episode {episode}\'s total Reard : {total_reward}')

    env.close()

if __name__=='__main__':
    # model_state_dict_saved_path = './model/actor_critic_policy_net_state_dict_2025-09-09_19:09:27.pt'
    model_state_dict_saved_path = '/Users/hsb/Desktop/gymnasium/LunarLander/model/actor_critic_policy_net_state_dict_2025-09-09_19:09:27.pt'
    model_state_dict_saved_path = './model/actor_critic_policy_net_state_dict_2025-09-15_02:07:23.pt'
    render_agent(
        num_episodes=20,
        model_state_dict_saved_path=model_state_dict_saved_path
    )