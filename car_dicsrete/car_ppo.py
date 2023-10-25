import os.path

import torch
import torch.nn as nn
import torch.optim as optim

from car_dicsrete.discreteracingenv import DiscreteRacingEnv
from ppo.shared_layers import ActorCritic


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.1)


# Параметры
GAMMA = 0.99
LR = 0.001
CLIP_EPSILON = 0.2
EPOCHS = 10
BATCH_SIZE = 2048*256



# Определение модели

# PPO update function

def get_model_signature(model):
    signature = []
    for param in model.parameters():
        signature.append(str(param.shape))
    return "_".join(signature).replace(",", "x").replace("(", "").replace(")", "").replace(" ", "")


# Main training loop

env = DiscreteRacingEnv(patience=1000)
#env.metadata['render_fps']=1000
model = ActorCritic(env.observation_space.shape[0], env.action_space.n)

MODEL_CHECKPOINT__PTH = f"model_checkpoint_{get_model_signature(model.shared_layers)}.pth"


model.to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)


episdo = 0
global_step = 0
# Load model and optimizer from checkpoint if exists
load_checkpoint = True  # Set to False if you want to train from scratch
if load_checkpoint and os.path.exists(MODEL_CHECKPOINT__PTH):
    checkpoint = torch.load(MODEL_CHECKPOINT__PTH)
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

    while step < 10000 and not done:
        step += 1
        global_step += 1
        state =state.unsqueeze(0)
        state = state.to(device)


        prob, _ = model(state)
        action = torch.multinomial(prob, 1).item()

        old_prob = prob[0][action].item() if prob is not None else 1.0
        env.custom_info = f'{prob.data if not prob is None else "Random"}'
        next_state, reward, done, _ = env.step(action)

        episode_reward += reward

        states.append(state)
        actions.append(torch.tensor(action))
        old_probs.append(torch.tensor(old_prob))
        rewards.append(torch.tensor(reward))
        dones.append(torch.tensor(done))
        if not done:
            state = next_state
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
        }, MODEL_CHECKPOINT__PTH)


