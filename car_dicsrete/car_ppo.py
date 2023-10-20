import math
import os.path
import random

import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
from gym.envs.box2d import lunar_lander
from torch import Tensor

from car_dicsrete.discreteracingenv import DiscreteRacingEnv
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.1)


# Параметры
GAMMA = 0.99
LR = 0.000007
CLIP_EPSILON = 0.2
EPOCHS = 20
BATCH_SIZE = 2048*32



# Определение модели
class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions, neurans=128):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_dim, neurans),nn.Tanh(),
            nn.Linear(neurans, neurans),nn.Tanh(),

            nn.Linear(neurans, n_actions),nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(input_dim, neurans),nn.Tanh(),
            nn.Linear(neurans, neurans),nn.Tanh(),

            nn.Linear(neurans, 1)
        )

    def forward(self, x):
        prob = self.actor(x)
        value = self.critic(x)
        return prob, value

# PPO update function
def ppo_update(optimizer, states, actions, old_probs, rewards, dones, model, clip_epsilon=CLIP_EPSILON):
    for _ in range(EPOCHS):
        for idx in range(0, len(states), BATCH_SIZE):
            state_batch = states[idx:idx+BATCH_SIZE]
            action_batch = actions[idx:idx+BATCH_SIZE]
            old_prob_batch = old_probs[idx:idx+BATCH_SIZE]
            reward_batch = rewards[idx:idx+BATCH_SIZE]
            done_batch = dones[idx:idx+BATCH_SIZE]
            state_batch = state_batch.to(device)
            action_batch = action_batch.to(device)
            old_prob_batch = old_prob_batch.to(device)
            reward_batch = reward_batch.to(device)
            done_batch = done_batch.to(device)
            prob, value = model(state_batch)
            prob = prob.gather(1, action_batch.unsqueeze(-1)).squeeze(-1)

            ratio = prob / old_prob_batch
            advantage = reward_batch - value.squeeze(-1)

            surrogate_1 = ratio * advantage
            surrogate_2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage
            actor_loss = -torch.min(surrogate_1, surrogate_2).mean()

            critic_loss = advantage.pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Main training loop

env = DiscreteRacingEnv(render=True)
#env.metadata['render_fps']=1000
model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)
model.actor.apply(init_weights)
model.critic.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)


episdo = 0
global_step = 0
# Load model and optimizer from checkpoint if exists
load_checkpoint = True  # Set to False if you want to train from scratch
if load_checkpoint and os.path.exists('model_checkpoint.pth'):
    checkpoint = torch.load("model_checkpoint.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    episdo = checkpoint['episode']
    global_step = checkpoint['global_step']
while True:
    step = 0
    state = env.reset()
    done = False
    episode_reward = 0

    prob = None
    states, actions, old_probs, rewards, dones = [], [], [], [], []

    while step < 1000 and not done:
        step += 1
        global_step += 1
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        state = state.to(device)


        prob, _ = model(state)
        action = torch.multinomial(prob, 1).item()

        old_prob = prob[0][action].item() if prob is not None else 1.0
        custom_info = f'{prob.data if not prob is None else "Random"}'
        next_state, reward, done, _ = env.step(action)

        env.render('train', custom_info=custom_info)
        episode_reward += reward

        states.append(state)
        actions.append(torch.tensor(action))
        old_probs.append(torch.tensor(old_prob))
        rewards.append(torch.tensor(reward))
        dones.append(torch.tensor(done))

        if done:
            episdo += 1
            print(f"Episode {episdo} Reward: {episode_reward}")
            ppo_update(optimizer,
                       torch.cat(states),
                       torch.stack(actions),
                       torch.stack(old_probs),
                       torch.stack(rewards),
                       torch.stack(dones),
                       model)
            # Save the model every 50 episodes
            if episdo % 50 == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode': episdo,
                    'global_step': global_step,
                }, "model_checkpoint.pth")

        state = next_state
