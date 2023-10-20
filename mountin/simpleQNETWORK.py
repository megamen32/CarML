import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
# Определение Q-сети
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Создаем нашу Q-сеть
env = gym.make('MountainCar-v0',render_mode='human')
env.metadata['render_fps']=1000
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
q_net = QNetwork(input_dim, output_dim)
target_net = QNetwork(input_dim, output_dim)

# Копируем веса из Q-сети в целевую сеть
target_net.load_state_dict(q_net.state_dict())



EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

# Функция для выбора действия с помощью ε-жадной политики
def select_action(state, eps_threshold):
    sample = random.random()
    if sample > eps_threshold:
        with torch.no_grad():
            q_values = q_net(torch.FloatTensor(state).unsqueeze(0))
            action = q_values.max(1)[1].item()
            return action
    else:
        return random.randrange(env.action_space.n)


# Определение буфера опыта
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Инициализация буфера опыта
replay_buffer = ReplayBuffer(10000)

# Параметры обучения
BATCH_SIZE = 512
GAMMA = 0.99
TARGET_UPDATE = 10  # Частота обновления целевой сети

# Определение функции потерь и оптимизатора
criterion = nn.MSELoss()
optimizer = optim.Adam(q_net.parameters(), lr=0.0007)


def update_model():
    if len(replay_buffer) < BATCH_SIZE:
        return

    # Извлечь мини-пакет из буфера опыта
    transitions = replay_buffer.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Преобразовать в тензоры PyTorch
    state_batch = torch.FloatTensor(batch.state)
    action_batch = torch.LongTensor(batch.action).unsqueeze(1)
    reward_batch = torch.FloatTensor(batch.reward)
    next_state_batch = torch.FloatTensor(batch.next_state)
    done_batch = torch.FloatTensor(batch.done)

    # Вычислить ожидаемые Q-значения
    q_values = q_net(state_batch).gather(1, action_batch)

    # Вычислить ожидаемые Q-значения для следующих состояний
    with torch.no_grad():
        next_q_values = target_net(next_state_batch).max(1)[0]
        target_q_values = reward_batch + GAMMA * next_q_values * (1 - done_batch)
        target_q_values = target_q_values.unsqueeze(1)

    # Вычислить потерю
    loss = criterion(q_values, target_q_values)

    # Оптимизировать модель
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

num_episodes=10000
# Основной цикл обучения
for episode in range(num_episodes):
    state,_ = env.reset()
    total_reward = 0
    done = False
    step=0

    while not done and step<6000:
        step+=1
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episode / EPS_DECAY)
        action = select_action(state, eps_threshold)

        next_state, reward, done, _ ,_= env.step(action)
        total_reward += reward

        # Сохраняем переход в буфере опыта
        replay_buffer.push(state, action, reward, next_state, done)

        # Обновляем модель
        update_model()

        state = next_state

    # Обновляем целевую сеть
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(q_net.state_dict())

    print(f"Episode {episode}, Total Reward: {total_reward}")

# После обучения
env.close()