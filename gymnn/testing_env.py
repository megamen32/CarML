import math
import os.path
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from gymnn.racingenv import RacingEnv


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_layers=1, hidden_size=128):
        super(Actor, self).__init__()

        # Initial layer
        layers = [nn.Linear(input_dim, hidden_size), nn.ReLU()]

        # Add hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Final layer
        layers.append(nn.Linear(hidden_size, output_dim))
        layers.append(nn.Tanh())

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class Critic(nn.Module):
    def __init__(self, input_dim, action_dim, num_hidden_layers=1, hidden_size=128):
        super(Critic, self).__init__()

        # Initial layer
        layers = [nn.Linear(input_dim + action_dim, hidden_size), nn.ReLU()]

        # Add hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Final layer
        layers.append(nn.Linear(hidden_size, 1))

        self.fc = nn.Sequential(*layers)

    def forward(self, state, action):
        return self.fc(torch.cat([state, action], dim=1))


def save_model(actor_model, critic_model, params, reward, filename):
    """Saves the actor and critic models along with hyperparameters and max reward."""
    torch.save({
        'actor_state_dict': actor_model.state_dict(),
        'critic_state_dict': critic_model.state_dict(),
        'hyperparameters': params,
        'max_reward': reward
    }, filename)

def load_model(filename):
    """Loads the actor and critic models, their hyperparameters, and the max reward."""
    checkpoint = torch.load(filename)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    loaded_actor = Actor(state_dim, action_dim).to(device)
    loaded_critic = Critic(state_dim, action_dim).to(device)

    loaded_actor.load_state_dict(checkpoint['actor_state_dict'])
    loaded_critic.load_state_dict(checkpoint['critic_state_dict'])

    return loaded_actor, loaded_critic, checkpoint['hyperparameters'], checkpoint['max_reward']


def train(env, actor, critic, params,num_episodes=1000, gamma=0.99, actor_lr=0.1, critic_lr=0.01, tau=0.90, noise_std=0.5,
          best_reward=float('-inf'), patience=10, model__pth='best_cur_model.pth'):

    target_actor = Actor(state_dim, action_dim).to(device)
    target_critic = Critic(state_dim, action_dim).to(device)

    # Initialize target weights with source weights
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
    criterion = nn.MSELoss()
    no_improve_counter = 0  # Counter for episodes without improvement
    last_best_reward = best_reward

    replay_buffer = deque(maxlen=int(env.max_time/env.delta_time*3))
    last_100_rewards=[]
    no_training_frame=0
    frames_to_train=50
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)

                noise_cur=max(0.001,noise_std*(num_episodes*0.8-episode)/(num_episodes*0.8))
                if random.random()<noise_cur:
                    noise = np.random.normal(0, noise_cur, size=action_dim)
                    action = noise
                else:
                    action = actor(state_tensor).cpu().numpy()



            next_state, reward, done, _ = env.step(action)
            if total_reward > last_best_reward:
                last_best_reward = total_reward
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            if no_improve_counter >= patience:
                last_best_reward=-math.inf
                done=True
            replay_buffer.append((state, action, reward, next_state, done))

            total_reward += reward
            state = next_state
            env.render(mode='train')
            no_training_frame+=1

            if len(replay_buffer) > 32 and no_training_frame>frames_to_train:
                no_training_frame=0
                for _ in range(frames_to_train):
                    minibatch = random.sample(replay_buffer, 32)
                    states, actions, rewards, next_states, dones = zip(*minibatch)

                    states = torch.stack([torch.FloatTensor(s).to(device) for s in states])
                    actions = torch.stack([torch.FloatTensor(a).to(device) for a in actions])
                    rewards = torch.FloatTensor(rewards).to(device).unsqueeze(1)
                    next_states = torch.stack([torch.FloatTensor(ns).to(device) for ns in next_states])
                    not_dones = torch.FloatTensor([not d for d in dones]).to(device).unsqueeze(1)

                    target_actions = target_actor(next_states)
                    target_q_values = target_critic(next_states, target_actions)
                    expected_q_values = rewards + gamma * target_q_values * not_dones

                    predicted_q_values = critic(states, actions)
                    critic_loss = criterion(predicted_q_values, expected_q_values)

                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_optimizer.step()

                    actor_loss = -critic(states, actor(states)).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

                    for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


        print(f"Episode {episode + 1}: Total Reward: {total_reward}")
        last_100_rewards.append(total_reward)
        average_reward=  np.mean(last_100_rewards)
        if total_reward > best_reward:
            best_reward = total_reward
            best_params = params
            save_model(actor, critic, best_params,best_reward,model__pth)
    return actor,critic


device = 'cuda' if torch.cuda.is_available() else 'cpu'

from itertools import product

# Define hyperparameters to search over
hidden_layers_variants=[0,1,2,3,4]
num_episodes_options = [200,1000]
gamma_options = [0.30,0.50, 0.98]
actor_lr_options = [0.1, 0.01]
critic_lr_options = [0.1,0.01]
tau_options = [0.999,0.995]
noise_std_options = [10,5, 2]

# Create a list of all combinations
parameter_combinations = list(product(hidden_layers_variants,num_episodes_options, gamma_options, actor_lr_options, critic_lr_options, tau_options, noise_std_options))

best_reward = float('-inf')
env = RacingEnv()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]



def train_single():
    best_params = None
    best_reward_params=best_reward=-math.inf
    for params in parameter_combinations:
        model__pth = f'best_model_{params[0]}_128.pth'
        if os.path.exists(model__pth) and False:
            actor,critic,_,best_reward=load_model(model__pth)
        else:
            actor = Actor(state_dim, action_dim,params[0])
            critic = Critic(state_dim, action_dim,params[0])
        actor=actor.to(device)
        critic=critic.to(device)
        print(f"Training with parameters: layers={params[0]} num_episodes={params[1]}, gamma={params[2]}, actor_lr={params[3]}, critic_lr={params[4]}, tau={params[5]}, noise_std={params[6]}")
        env = RacingEnv()
        # Train the model with the current set of parameters
        actor,critic = train(env,actor,critic,params=params, num_episodes=params[1], gamma=params[2], actor_lr=params[3], critic_lr=params[4], tau=params[5], noise_std=params[6],best_reward=best_reward,model__pth=model__pth)
        env.close()
        # Evaluate the model (This can be your own evaluation metric, here I just use the total reward from the last episode as an example)
        total_reward = 0
        testing_episodes = 10
        for _ in range(testing_episodes):  # Run for 10 episodes to get an average performance
            state = env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).to(device)
                    action = actor(state_tensor).cpu().numpy()
                state, reward, done, _ = env.step(action)
                total_reward += reward

        average_reward = total_reward / testing_episodes
        print(f"Average reward: {average_reward} on {params}")

        try:
            if average_reward > best_reward_params:

                best_reward = average_reward
                best_params = params
                print(f"Best parameters: layers={best_params[0]} num_episodes={best_params[1]}, gamma={best_params[2]}, actor_lr={best_params[3]}, critic_lr={best_params[4]}, tau={best_params[5]}, noise_std={best_params[6]} with reward: {best_reward}")
                save_model(actor,critic,params,average_reward,'final_'+model__pth)
        except:
            traceback.print_exc()
    print(f"Best parameters: layers={best_params[0]} num_episodes={best_params[1]}, gamma={best_params[2]}, actor_lr={best_params[3]}, critic_lr={best_params[4]}, tau={best_params[5]}, noise_std={best_params[6]} with reward: {best_reward}")

def train_and_evaluate(params):
    env = RacingEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = Actor(state_dim, action_dim, params[0]).to(device)
    critic = Critic(state_dim, action_dim, params[0]).to(device)

    actor, critic = train(env, actor, critic,params=params, num_episodes=params[1], gamma=params[2], actor_lr=params[3], critic_lr=params[4], tau=params[5], noise_std=params[6],model__pth=f'best_cur_model{params}.pth')
    env.close()

    total_reward = 0
    testing_episodes = 10
    for _ in range(testing_episodes):
        state = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                action = actor(state_tensor).cpu().numpy()
            state, reward, done, _ = env.step(action)
            total_reward += reward

    average_reward = total_reward / testing_episodes

    return average_reward, params
def train_multiple():
    from multiprocessing import Pool, cpu_count

# This function wraps the training and evaluation for a given set of hyperparameters


    # Use the number of cores available in your machine
    num_cores = cpu_count()

    # Create a pool of worker processes
    with Pool(processes=num_cores) as pool:
        results = pool.map(train_and_evaluate, parameter_combinations)

    # Find the best result
    best_result = max(results, key=lambda x: x[0])

    print(f"Best Average Reward: {best_result[0]} with parameters: {best_result[1]}")


if __name__=='__main__':
    train_multiple()