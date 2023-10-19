import traceback

import torch
import os

from gymnn.racingenv import RacingEnv
from gymnn.DDPG import Actor, Critic, state_dim, action_dim, device


def load_max_reward_from_file(filename):
    """Loads the max_reward value from a given .pth file."""
    checkpoint = torch.load(filename)
    return checkpoint['max_reward']

def find_top_n_models(directory=".", n=10):
    """Finds the top n .pth files with the highest max_reward in the given directory."""
    rewards = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".pth") and "cur_model" in filename:
            try:
                reward = load_max_reward_from_file(filename)
                rewards.append((filename, reward))
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    # Sort the rewards in descending order and take the top n
    sorted_rewards = sorted(rewards, key=lambda x: x[1], reverse=True)[:n]

    return sorted_rewards
def test_model(filepath, env, testing_episodes=10):
    # Load the checkpoint first to get the number of hidden layers
    checkpoint = torch.load(filepath)

    # Extract the number of hidden layers from the saved hyperparameters
    num_hidden_layers = checkpoint['hyperparameters'][0]

    # Initialize the models with the correct number of hidden layers
    actor = Actor(state_dim, action_dim, num_hidden_layers=num_hidden_layers).to(device)
    critic = Critic(state_dim, action_dim, num_hidden_layers=num_hidden_layers).to(device)

    # Load the state dicts
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])

    total_reward = 0
    for _ in range(testing_episodes):
        state = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                action = actor(state_tensor).cpu().numpy()
            state, reward, done, _ = env.step(action)
            env.render(mode='rgb')
            total_reward += reward

    average_reward = total_reward / testing_episodes

    return average_reward

def test_top_models(filepaths, env):
    results = []

    for i,filepath in enumerate(filepaths):
        try:
            average_reward = test_model(filepath[0], env)
            print(i,'test complete for',filepath[0],' reward: ',average_reward)
            results.append((average_reward, filepath[0]))
        except:
            traceback.print_exc()

    results.sort(key=lambda x: x[0], reverse=True)

    return results




if __name__ == '__main__':
    top_models = find_top_n_models(n=100)
    for idx, (filename, reward) in enumerate(top_models, start=1):
        print(f"{idx}. Model: {filename} with a reward of {reward}")
    env = RacingEnv(render=True,max_time=100)

    test_results = test_top_models(top_models, env)

    for reward, filepath in test_results:
        print(f"Model {filepath} achieved an average reward of {reward}")#Best Average Reward: -4873.5967768155515 with parameters: (3, 3000, 0.3, 0.1, 0.1, 0.997, 0.5, 3000)
