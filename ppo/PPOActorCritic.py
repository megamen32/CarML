import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ActorCritic(nn.Module):
    def __init__(self, input_dim, n_actions, actor_hidden_size=32, critic_hidden_size=32, actor_layers=1, critic_layers=1):
        super(ActorCritic, self).__init__()

        # Слои для актера
        self.actor_layers = [nn.Linear(input_dim, actor_hidden_size), nn.ReLU()]
        for _ in range(actor_layers - 1):
            self.actor_layers.extend([nn.Linear(actor_hidden_size, actor_hidden_size), nn.ReLU()])
        self.actor_layers = nn.Sequential(*self.actor_layers)

        self.actor_head = nn.Sequential(
            nn.Linear(actor_hidden_size, n_actions),
            nn.Softmax(dim=-1)
        )

        # Слои для критика
        self.critic_layers = [nn.Linear(input_dim, critic_hidden_size), nn.ReLU()]
        for _ in range(critic_layers - 1):
            self.critic_layers.extend([nn.Linear(critic_hidden_size, critic_hidden_size), nn.ReLU()])
        self.critic_layers = nn.Sequential(*self.critic_layers)

        self.critic_head = nn.Linear(critic_hidden_size, 1)

    def forward(self, x):
        actor_representation = self.actor_layers(x)
        critic_representation = self.critic_layers(x)

        action_probs = self.actor_head(actor_representation)
        state_value = self.critic_head(critic_representation)

        return action_probs, state_value

def ppo_update(optimizer, states, actions, old_probs, rewards, dones, model, clip_epsilon=0.2,EPOCHS=10,BATCH_SIZE=512):
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
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