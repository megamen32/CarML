import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from gymnn.racingenv import RacingEnv

class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, num_hidden_layers=1, hidden_size=128):
        super(Actor, self).__init__()

        # Initial layer
        layers = [nn.Linear(input_dim, hidden_size), nn.Tanh()]

        # Add hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())

        self.fc = nn.Sequential(*layers)

        # Output layers for mean and standard deviation
        self.mu = nn.Linear(hidden_size, action_dim)
        self.sigma = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = self.fc(x)
        mu = self.mu(x)
        sigma = torch.clamp(self.sigma(x), min=-20, max=2)  # Clamp for stability
        return mu, sigma

class Critic(nn.Module):
    def __init__(self, input_dim, num_hidden_layers=1, hidden_size=128):
        super(Critic, self).__init__()

        # Initial layer
        layers = [nn.Linear(input_dim, hidden_size), nn.Tanh()]

        # Add hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())

        # Final layer
        layers.append(nn.Linear(hidden_size, 1))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class PPO:
    def __init__(self, state_dim, action_dim, num_hidden_layers, hidden_size, lr=0.001, gamma=0.99, clip_epsilon=0.2, update_epochs=10, mini_batch_size=64):
        self.actor = Actor(state_dim, action_dim, num_hidden_layers, hidden_size).to(device)
        self.critic = Critic(state_dim, num_hidden_layers, hidden_size).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).to(device)
        mu, sigma = self.actor(state_tensor)
        action_dist = torch.distributions.Normal(mu, sigma.exp())
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).sum(dim=-1)
        return action.cpu().numpy(), log_prob

    def update(self, memory):
        for _ in range(self.update_epochs):
            # Sample mini batches
            state_batch, action_batch, old_log_prob_batch, reward_batch, done_batch = memory.sample(self.mini_batch_size)

            # Recompute expected returns
            returns = []
            R = 0
            for reward, done in zip(reversed(reward_batch), reversed(done_batch)):
                R = reward + self.gamma * R * (1-done)
                returns.insert(0, R)
            returns = torch.tensor(returns).to(device)
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)  # Normalize

            # Consolidate batches to tensors
            state = torch.FloatTensor(state_batch).to(device)
            action = torch.FloatTensor(action_batch).to(device)
            old_log_prob = torch.FloatTensor(old_log_prob_batch).to(device)
            R = torch.FloatTensor(returns).to(device)

            # Train critic
            value = self.critic(state)
            advantage = R - value.squeeze()
            critic_loss = (advantage ** 2).mean()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Recompute actor's output for current state
            mu, sigma = self.actor(state)
            action_dist = torch.distributions.Normal(mu, sigma.exp())
            new_log_prob = action_dist.log_prob(action).sum(dim=-1)
            ratio = (new_log_prob - old_log_prob).exp()
            surrogate1 = ratio * advantage
            surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()


class Memory:
    def __init__(self):
        self.memory = []

    def add(self, state, action, log_prob, reward, done):
        self.memory.append((state, action, log_prob, reward, done))

    def sample(self, batch_size):
        mini_batch = random.sample(self.memory, min(batch_size,len(self.memory)))
        state_batch, action_batch, log_prob_batch, reward_batch, done_batch = zip(*mini_batch)
        return state_batch, action_batch, log_prob_batch, reward_batch, done_batch

    def clear(self):
        self.memory = []

# PPO Training Loop
def train_ppo(env, agent, memory, episodes=1000):
    total_rewards = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            memory.add(state, action, log_prob, reward, done)
            state = next_state
            env.render('train')
        total_rewards.append(episode_reward)
        agent.update(memory)
        memory.clear()
        print(f"Episode {episode + 1}, Reward: {episode_reward}")
    return total_rewards

# Initialize environment and PPO agent
env = RacingEnv(render=True)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
memory = Memory()
ppo_agent = PPO(state_dim, action_dim, num_hidden_layers=3, hidden_size=64)

# Train PPO agent
train_ppo(env, ppo_agent, memory)