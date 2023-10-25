from torch import nn as nn


class ActorCriticShared(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_size=32):
        super(ActorCritic, self).__init__()

        # Общие слои
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size),nn.ReLU(),
        )

        # "Голова" для актера
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size, n_actions),
            nn.Softmax(dim=-1)
        )

        # "Голова" для критика
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        shared_representation = self.shared_layers(x)

        action_probs = self.actor_head(shared_representation)
        state_value = self.critic_head(shared_representation)

        return action_probs, state_value
