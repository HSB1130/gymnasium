import torch
import gymnasium as gym
from PPO import PolicyNet


def get_action_from_policy_net(policy_net:PolicyNet, state):
    state = torch.tensor(state, dtype=torch.uint8)
    logit = policy_net(state)
    action = torch.argmax(logit).item()
    return action


def render_agent(num_episodes, model_state_dict_saved_path):
    env = gym.make(
        id="CarRacing-v3",
        render_mode='human',
        domain_randomize=False,
        continuous=False,
    )

    policy_net = PolicyNet(env.observation_space.shape, env.action_space.n)
    state_dict = torch.load(model_state_dict_saved_path, map_location="cpu")
    policy_net.load_state_dict(state_dict)

    for episode in range(1, num_episodes+1):
        state, info = env.reset()
        done = False

        total_reward = 0.0

        while not done:
            action = get_action_from_policy_net(policy_net, state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            state = next_state

        print(f'Episode {episode}\'s total Reard : {total_reward:.4f}')

    env.close()

if __name__=='__main__':
    model_state_dict_saved_path = '/Users/hsb/Desktop/gymnasium/Box2D/CarRacing/model/ppo_policy_net_state_dict_2026-01-04_05:25:09.pt'

    render_agent(
        num_episodes=20,
        model_state_dict_saved_path=model_state_dict_saved_path
    )