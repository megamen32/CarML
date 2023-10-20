## imprt
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
## model
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

## training
env = gym.make('MountainCar-v0',render_mode='human')
env.metadata['render_fps']=1000
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

policy = PolicyNetwork(input_dim, output_dim)
optimizer = optim.Adam(policy.parameters(), lr=0.0007)

num_episodes = 1000
gamma = 0.99
prev_max_step = 30000
max_wait_for_better_results = 1.2
for episode in range(num_episodes):
    state,_ = env.reset()
    rewards = []
    states = []
    log_probs=[]
    step=0
    done=False

    while not done and step<prev_max_step*max_wait_for_better_results:
        step+=1
        probs = policy(torch.FloatTensor(state))

        action = np.random.choice(output_dim,p=probs.detach().numpy())
        log_prob = torch.log(probs[action])

        states.append(state)
        log_probs.append(log_prob)

        state, reward, done, _,_ = env.step(action)
        rewards.append(reward)

    if step* max_wait_for_better_results <prev_max_step:
        prev_max_step=step
        max_wait_for_better_results*=0.90
    if prev_max_step<step:
        no_improve_counter+=1
    if no_improve_counter>20:
        max_wait_for_better_results*=1.1
        no_improve_counter=0

    # Compute returns
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.FloatTensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # normalize

    # Optimize policy
    optimizer.zero_grad()
    for log_prob, R in zip(log_probs, returns):
        loss = -log_prob * R  # negative because we want to maximize
        loss.backward()
    optimizer.step()


    print(f"Episode {episode}, Total Reward: {sum(rewards)}")

