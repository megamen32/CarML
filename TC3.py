import torch
import torch.nn as nn
import torch.optim as optim
import random
from main import RealisticCar2D,check_collision_with_track,track_2D,place_car_on_track,distance_to_centerline,compute_distance_central_line,compute_closest_point_idx,compute_track_length,compute_travel_distance
import pygame
import numpy as np
from replayplayer import draw_track,SCREEN_HEIGHT,SCREEN_WIDTH,scale_factor,init_screen,translation,draw_car,plot_track_and_cars
import os
from replaybuffer import ReplayBuffer

policy_model_path = 'best_policy_model.pth'
def save_model(model, optimizer, filename, learning_rate, noise_std):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'learning_rate': learning_rate,
        'noise_std': noise_std
    }
    torch.save(state, filename)
def load_model(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    learning_rate = checkpoint['learning_rate']
    noise_std = checkpoint['noise_std']
    return learning_rate, noise_std

class TwinQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TwinQNetwork, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, state):
        return self.q1(state), self.q2(state)
class TargetQNetwork(TwinQNetwork):
    def __init__(self, input_dim, output_dim):
        super(TargetQNetwork, self).__init__(input_dim, output_dim)

    def update_from_model(self, model, tau=0.1):
        for target_param, param in zip(self.parameters(), model.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.q1=nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_dim),nn.Tanh()
        )

    def forward(self, state):

        return self.q1(state)


def get_state(car, track_2D):
    _,fclosest_point = compute_distance_central_line(car, track_2D,10)  # Assuming you have this function implemented
    _,closest_point = compute_distance_central_line(car, track_2D,0)  # Assuming you have this function implemented
    fpoint_x,fpoint_y=fclosest_point
    point_x,point_y=closest_point
    position_x,position_y=car.position
    speed_x,spped_y = car.velocity
    return torch.FloatTensor([fpoint_x,fpoint_y,position_x,position_y ,speed_x,spped_y,point_x,point_y,car.current_steering_angle])

# Hyperparameters
input_dim = 9#len(get_state(RealisticCar2D(),track_2D))
output_dim = 2  # [acceleration, turn_angle]
initial_learning_rate = 0.01
min_learning_rate = 0.00001
learning_rate=initial_learning_rate
initial_noise_std=0.5
min_noise_std = 0.05
max_noise_std=0.5
gamma = 0.7  # Discount factor
batch_size=256
# Initialize Q-Network and optimizer
network = TwinQNetwork(input_dim, output_dim)
policy_network=PolicyNetwork(input_dim, output_dim)
optimizer = optim.Adam(network.parameters(), lr=learning_rate)
if os.path.exists(policy_model_path):
    learning_rate, initial_noise_std = load_model(policy_network, optimizer,policy_model_path)

target_q_network = TargetQNetwork(input_dim, output_dim)
twin_model_path = 'best_model_twin.pth'
if os.path.exists( twin_model_path):
    network.load_state_dict(torch.load(twin_model_path))



criterion = nn.MSELoss()


# Function to perform Q-Learning within the racing environment with corrected action selection
import math
import random


def advanced_reward_function(car, track_2D, collision, time, prev_closest_point_idx):
    collision_penalty=0
    if collision:
        collision_penalty= -100.0
        return collision_penalty,car.closest_point_idx

    min_distance, closest_point = compute_distance_central_line(car, track_2D, 0)
    speed = np.linalg.norm(car.velocity)

    total_track_length = compute_track_length(track_2D)
    distance = compute_travel_distance(track_2D, closest_point, car.closest_point_idx)
    #normalized_progress = distance_to_closest_point / total_track_length

    backward_penalty = 0
    if car.closest_point_idx < prev_closest_point_idx:
        backward_penalty = -50.0  # Выберите значение штрафа, которое вам подходит
        return -backward_penalty,car.closest_point_idx

    finish_reward = 0
    if car.closest_point_idx == len(track_2D) - 1:
        finish_reward = 5000 / time
        print('finish!', finish_reward, time)

    reward = distance + finish_reward + backward_penalty +collision_penalty+(1-min_distance)*0.1+speed*0.1

    return reward, car.closest_point_idx


# Modified batch training function with enhancements
import pygame

def enhanced_q_learning(q_network, target_q_network, optimizer, criterion, num_cars=2, n_episodes=10000):
    global learning_rate
    # Pygame initialization
    pygame.init()

    screen = init_screen()
    clock = pygame.time.Clock()

    episode_rewards = []
    last_100_rewards=[]
    losses = []
    max_episode_reward = -math.inf
    max_time = 5000
    delta_time = 1
    previous_avg_reward=-math.inf
    noise_std = initial_noise_std
    road_width = 30
    replay_buffer = ReplayBuffer(5000)






    for episode in range(n_episodes):
        cars = [RealisticCar2D() for _ in range(num_cars)]
        spacing = len(track_2D) // num_cars
        for i, car in enumerate(cars):
            place_car_on_track(car, track_2D, 0)
        #plot_track_and_cars(track_2D, [car.position for car in cars])

        states = [get_state(car, track_2D) for car in cars]
        done_flags = [False] * num_cars
        episode_reward = 0
        time = 0
        episode_loss = 0
        episode_trainig_count=0
        prev_closest_point_idx = [compute_closest_point_idx(cars[_],track_2D) for _ in range(num_cars)]



        while not all(done_flags) and time < max_time:
            time += delta_time

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return


            # Your track drawing code here
            draw_track(screen,track_2D,road_width)

            for i in range(num_cars):
                if done_flags[i]:
                    continue

                state = states[i]
                action = policy_network(state.unsqueeze(0)).squeeze().detach().numpy()
                noise = np.random.normal(0, noise_std, size=action.shape)
                action += noise

                car = cars[i]
                car.update_position(delta_time=delta_time, acceleration=action[0], steering_angle=action[1], road_width=road_width)
                next_state = get_state(car, track_2D)
                collision, _ = check_collision_with_track(car.position, track_2D, road_width=road_width)
                reward,new_closest_point_idx = advanced_reward_function(car, track_2D, collision,time,prev_closest_point_idx[i])
                prev_closest_point_idx[i] = new_closest_point_idx
                # Store this experience into the Replay Buffer
                state_array = np.array(state)
                replay_buffer.push(state, action, reward, next_state, collision)

                # Update done flags and states
                done_flags[i] = collision
                states[i] = next_state
                episode_reward += reward

                # Draw the car
                draw_car(car, screen,track_2D)

            pygame.display.update()
            if len(replay_buffer) >= batch_size and int(time)%5==0:
                episode_trainig_count+=1
                episode_loss = train_batch(criterion, episode_loss, optimizer, q_network, replay_buffer, target_q_network)

        # Decay epsilon

        losses.append(episode_loss / time)  # averaging over the episode
        episode_rewards.append(episode_reward)
        last_100_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: trained {episode_trainig_count} t:{time} Total Reward: {episode_reward},  Avg Loss: {episode_loss / time}, noise {noise_std} lr {learning_rate}")
        if len(last_100_rewards) > 100:
            last_100_rewards.pop(0)
            avg_reward = np.mean(last_100_rewards)
            # Обновление learning rate (здесь используется простейшая логика, вы можете выбрать более сложный метод)


            if avg_reward > previous_avg_reward:
                noise_std *= 0.95  # Уменьшаем шум на 10%
                learning_rate*=0.995
            else:
                noise_std *= 1.05  # Увеличиваем шум на 10%
                learning_rate *= 1.001
            learning_rate = max(min_learning_rate, learning_rate)
            previous_avg_reward=avg_reward

            # Ограничиваем noise_std между минимальным и максимальным значениями
            noise_std = min(max_noise_std, max(min_noise_std, noise_std))

            # Обновление learning rate и noise_std в оптимизаторе и шумовом генераторе
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            last_100_rewards=last_100_rewards[-50:]

        if episode_reward > max_episode_reward:

            max_episode_reward = episode_reward
            torch.save(q_network.state_dict(), twin_model_path)

            save_model(policy_network, optimizer, policy_model_path, learning_rate, noise_std)

    return episode_rewards


def train_batch(criterion, episode_loss, optimizer, q_network, replay_buffer, target_q_network):
    sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = replay_buffer.sample(batch_size)
    sampled_next_states_array = np.vstack(sampled_next_states)
    # Then convert it to a PyTorch tensor
    sampled_next_states_tensor = torch.FloatTensor(sampled_next_states_array)
    # Compute target Q-values using both the target Q-Networks
    with torch.no_grad():
        next_actions = policy_network(sampled_next_states_tensor)
        q1_target, q2_target = target_q_network(sampled_next_states_tensor)
        min_q_target = torch.min(q1_target, q2_target)

        expanded_sampled_rewards = torch.FloatTensor(sampled_rewards).unsqueeze(1)
        target_value = torch.FloatTensor(expanded_sampled_rewards) + gamma * min_q_target.squeeze()
    # Compute current Q-values
    q1_values, q2_values = q_network(sampled_next_states_tensor)
    loss_q1 = criterion(q1_values.squeeze(), target_value)
    loss_q2 = criterion(q2_values.squeeze(), target_value)
    loss_q = loss_q1 + loss_q2
    episode_loss += loss_q.item()
    optimizer.zero_grad()
    loss_q.backward()
    optimizer.step()
    # Update the target network
    target_q_network.update_from_model(q_network)
    return episode_loss


# Training the Q-Network within the racing environment
episode_rewards_fixed = enhanced_q_learning(network, target_q_network,optimizer, criterion)

# Displaying the first 10 episode rewards
episode_rewards_fixed[:10]
