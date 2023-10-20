import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
from gym.envs.box2d import lunar_lander

# Параметры
GAMMA = 0.99
LR = 0.001
CLIP_EPSILON = 0.2
EPOCHS = 4
BATCH_SIZE = 64

# Определение модели
class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128),nn.ReLU(),
            nn.Linear(128, n_actions),nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(input_dim, 128),nn.ReLU(),
            nn.Linear(128, 1)
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

env = gym.make('LunarLander-v2',render_mode='human')
env.metadata['render_fps']=1000
model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
optimizer = optim.Adam(model.parameters(), lr=LR)



while True:
    step=0
    state,_ = env.reset()
    done = False
    episode_reward = 0
    states, actions, old_probs, rewards, dones = [], [], [], [], []
    while step<1000 and not done:
        step+=1
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        prob, _ = model(state)
        action = torch.multinomial(prob, 1).item()
        old_prob = prob[0][action].item()

        next_state, reward, done, _,_ = env.step(action)
        episode_reward += reward

        states.append(state)
        actions.append(torch.tensor(action))
        old_probs.append(torch.tensor(old_prob))
        rewards.append(torch.tensor(reward))
        dones.append(torch.tensor(done))

        if done:
            print(f"Episode Reward: {episode_reward}")
            ppo_update(optimizer,
                       torch.cat(states),
                       torch.stack(actions),
                       torch.stack(old_probs),
                       torch.stack(rewards),
                       torch.stack(dones),
                       model)



        state = next_state
