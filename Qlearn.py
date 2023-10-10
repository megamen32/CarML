import torch
import torch.nn as nn
import torch.optim as optim
import random
from main import RealisticCar2D,check_collision_with_track,track_2D,place_car_on_track
import pygame
import numpy as np
import matplotlib.pyplot as plt
import os


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
class TargetQNetwork(QNetwork):
    def __init__(self, input_dim, output_dim):
        super(TargetQNetwork, self).__init__(input_dim, output_dim)

    def update_from_model(self, model, tau=0.1):
        for target_param, param in zip(self.parameters(), model.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)



# Initialize epsilon decay parameters
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 500


# Hyperparameters
input_dim = 4  # [x_position, y_position, x_velocity, y_velocity]
output_dim = 2  # [acceleration, turn_angle]
learning_rate = 0.001
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate

# Initialize Q-Network and optimizer
q_network = QNetwork(input_dim, output_dim)
if os.path.exists('best_model.pth'):
    q_network.load_state_dict(torch.load('best_model.pth'))

target_q_network = TargetQNetwork(input_dim, output_dim)
target_q_network.eval()
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

def get_state(car):
    return torch.FloatTensor(np.concatenate([car.position, car.velocity]))

# Function to perform Q-Learning within the racing environment with corrected action selection
import math
import random


def distance_to_centerline(car_position, segment_start, segment_end):
    """
    Compute the distance of the car to the centerline of the track segment.

    Parameters:
        car_position (array): The [x, y] position of the car.
        segment_start (array): The [x1, y1] coordinates representing the start of the segment.
        segment_end (array): The [x2, y2] coordinates representing the end of the segment.

    Returns:
        distance (float): The distance of the car to the centerline of the segment.
    """
    x1, y1 = segment_start
    x2, y2 = segment_end
    x0, y0 = car_position

    # Compute the distance from point to line (in 2D)
    distance = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return distance


# Update the advanced_reward_function to work with the new track representation
def advanced_reward_function(car, track_2D, collision):
    """
    Computes an advanced reward based on the car's state and track information.

    Parameters:
        car (RealisticCar2D): The car object containing its current state.
        track_2D (list): The list of points representing the track.
        collision (bool): Whether the car has collided with the track boundaries.

    Returns:
        reward (float): The computed reward.
    """
    if collision:
        return -100.0  # Large negative reward for collision

    # Compute the distance to the centerline of the closest track segment
    min_distance = float('inf')
    for i in range(len(track_2D) - 1):
        segment_start = track_2D[i]
        segment_end = track_2D[i + 1]
        distance = distance_to_centerline(car.position, segment_start, segment_end)
        min_distance = min(min_distance, distance)

    # Compute the speed of the car
    speed = np.linalg.norm(car.velocity)

    # Reward based on staying close to the centerline and maintaining a reasonable speed
    reward = 1.0 - 0.1 * min_distance + 0.2 * speed

    return reward
# Modified batch training function with enhancements
def enhanced_batch_q_learning(q_network, target_q_network, optimizer, criterion, num_cars=50, batch_size=32, n_episodes=100):
    episode_rewards = []
    replay_buffer = []
    buffer_limit = 5000
    epsilon = 1.0  # Initial epsilon
    epsilon_decay = 0.995  # Decay rate
    min_epsilon = 0.01  # Minimum epsilon
    max_episode_reward=-math.inf
    max_time=5000
    delta_time=1
    # Initialize Pygame screen and clock
    SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
    CAR_RADIUS=5
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    scale_factor = 2  # Scaling factor for visualization
    translation = [SCREEN_WIDTH // 4, SCREEN_HEIGHT // 4]  # Translation for visualization

    for episode in range(n_episodes):
        # Initialize multiple cars and their states
        cars = [RealisticCar2D() for _ in range(num_cars)]
        for i, car in enumerate(cars):
            place_car_on_track(car, track_2D, 0)
        print("Initial positions of cars:", [car.position for car in cars])

        states = [get_state(car) for car in cars]
        done_flags = [False] * num_cars
        episode_reward = 0
        episode_replay = []
        time=0
        road_width=10


        while not all(done_flags) and time<max_time:
            time+=delta_time
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            screen.fill((0, 0, 0))  # Fill the screen with black

            draw_track(road_width, scale_factor, screen, translation)
            for i in range(num_cars):
                if done_flags[i]:
                    continue  # Skip if this car is done

                state = states[i]
                episode_replay.append(state)

                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = np.random.uniform(-1, 1, size=output_dim)  # Random action
                else:
                    with torch.no_grad():
                        q_values = q_network(state.unsqueeze(0))
                    action_idx = q_values.argmax().item()  # Greedy action index
                    action = np.zeros(output_dim)
                    action[action_idx] = 1  # One-hot encoded action

                # Step the environment
                car = cars[i]
                car.update_position(delta_time=delta_time, acceleration=action[0], steering_angle=action[1])
                next_state = get_state(car)
                collision, _ = check_collision_with_track(car.position, track_2D, road_width=road_width)
                reward = advanced_reward_function(car, track_2D, collision)  # Placeholder reward function
                done_flags[i] = collision  # Episode ends for this car if there's a collision
                draw_car(CAR_RADIUS, car, scale_factor, screen, translation)

                # Store transition in replay buffer
                replay_buffer.append((state, action, reward, next_state))
                if len(replay_buffer) > buffer_limit:
                    replay_buffer.pop(0)  # Remove the oldest experience if the buffer is full

                # Batch update Q-values if enough samples are available
                if len(replay_buffer) >= batch_size:
                    batch = random.sample(replay_buffer, batch_size)
                    states_batch, actions_batch, rewards_batch, next_states_batch = zip(*batch)
                    states_batch = torch.stack(states_batch)
                    actions_batch = torch.FloatTensor(np.array(actions_batch))

                    rewards_batch = torch.FloatTensor(np.array(rewards_batch))
                    next_states_batch = torch.stack(next_states_batch)

                    with torch.no_grad():
                        target_q_values = target_q_network(next_states_batch)
                    targets = rewards_batch + gamma * target_q_values.max(dim=1)[0]

                    q_values = q_network(states_batch).gather(1, torch.max(actions_batch, 1)[1].unsqueeze(-1)).squeeze()
                    loss = criterion(q_values, targets)
                    #print('Loss:',loss[0])

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Update the target network, copying all weights and biases in DQN
                    target_q_network.update_from_model(q_network)

                # Update state and episode reward
                states[i] = next_state
                episode_reward += reward
            pygame.display.update()
            clock.tick(30)

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)


        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total Reward: {episode_reward}, Epsilon: {epsilon}")
        if episode_reward > max_episode_reward:
            max_episode_reward = episode_reward
            torch.save(q_network.state_dict(), 'best_model.pth')
            #visualize_game_with_pygame(episode_replay, track_2D)

    return episode_rewards


def draw_car(CAR_RADIUS, car, scale_factor, screen, translation):
    car_position = [int(coord * scale_factor + translation[idx % 2]) for idx, coord in enumerate(car.position)]
    pygame.draw.circle(screen, (255, 0, 0), car_position, CAR_RADIUS)


# The function is not run here but can be executed in your local environment.


# Training the Q-Network within the racing environment
episode_rewards_fixed = enhanced_batch_q_learning(q_network, target_q_network,optimizer, criterion)

# Displaying the first 10 episode rewards
episode_rewards_fixed[:10]
