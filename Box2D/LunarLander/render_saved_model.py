import torch
import gymnasium as gym
from Actor_Critic import PolicyNet


def get_action_from_policy_net(policy_net:PolicyNet, state):
    state = torch.tensor(state, dtype=torch.float32)
    logit = policy_net(state)
    action = torch.argmax(logit).item()
    return action


def render_agent(num_episodes, model_state_dict_saved_path):
    env = gym.make(
        'LunarLander-v3',
        render_mode='human'
    )

    policy_net = PolicyNet(env.observation_space.shape[0], env.action_space.n)
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
    model_state_dict_saved_path = './model/actor_critic_policy_net_state_dict_2025-09-16_11:19:26.pt'
    render_agent(
        num_episodes=20,
        model_state_dict_saved_path=model_state_dict_saved_path
    )