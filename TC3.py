import torch
import torch.nn as nn
import torch.optim as optim
import random
from main import RealisticCar2D,check_collision_with_track,track_2D,place_car_on_track,distance_to_centerline,compute_distance_central_line
import pygame
import numpy as np
from replayplayer import draw_track,SCREEN_HEIGHT,SCREEN_WIDTH,scale_factor,init_screen,translation,draw_car,plot_track_and_cars
import os
from replaybuffer import ReplayBuffer

class TwinQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TwinQNetwork, self).__init__()
        # Q1 architecture
        self.q1_fc1 = nn.Linear(input_dim, 64)
        self.q1_fc2 = nn.Linear(64, 32)
        self.q1_fc3 = nn.Linear(32, output_dim)

        # Q2 architecture
        self.q2_fc1 = nn.Linear(input_dim, 64)
        self.q2_fc2 = nn.Linear(64, 32)
        self.q2_fc3 = nn.Linear(32, output_dim)

    def forward(self, state):
        # Q1 forward pass
        q1 = torch.relu(self.q1_fc1(state))
        q1 = torch.relu(self.q1_fc2(q1))
        q1 = self.q1_fc3(q1)

        # Q2 forward pass
        q2 = torch.relu(self.q2_fc1(state))
        q2 = torch.relu(self.q2_fc2(q2))
        q2 = self.q2_fc3(q2)

        return q1, q2
class TargetQNetwork(TwinQNetwork):
    def __init__(self, input_dim, output_dim):
        super(TargetQNetwork, self).__init__(input_dim, output_dim)

    def update_from_model(self, model, tau=0.1):
        for target_param, param in zip(self.parameters(), model.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))



# Hyperparameters
input_dim = 7  # [ x_velocity, y_velocity,closest_point_x,closest_point_y,current_steering_angle]
output_dim = 2  # [acceleration, turn_angle]
learning_rate = 0.001
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate
batch_size=64
# Initialize Q-Network and optimizer
network = TwinQNetwork(input_dim, output_dim)
policy_network=PolicyNetwork(input_dim, output_dim)
target_q_network = TargetQNetwork(input_dim, output_dim)
twin_model_path = 'best_model_twin.pth'
if os.path.exists( twin_model_path):
    network.load_state_dict(torch.load(twin_model_path))


optimizer = optim.Adam(network.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

def get_state(car, track_2D):
    _,closest_point = compute_distance_central_line(car, track_2D)  # Assuming you have this function implemented
    point_x,point_y=closest_point
    position_x,position_y=car.position
    speed_x,spped_y = car.velocity
    return torch.FloatTensor([position_x,position_y ,speed_x,spped_y,point_x,point_y,car.current_steering_angle])

# Function to perform Q-Learning within the racing environment with corrected action selection
import math
import random





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
    min_distance,_ = compute_distance_central_line(car, track_2D)

    # Compute the speed of the car
    speed = np.linalg.norm(car.velocity)

    # Reward based on staying close to the centerline and maintaining a reasonable speed
    reward = 1.0 - 0.1 * min_distance + 0.2 * speed

    return reward





# Modified batch training function with enhancements
import pygame

def enhanced_q_learning(q_network, target_q_network, optimizer, criterion, num_cars=16, n_episodes=10000):
    # Pygame initialization
    pygame.init()

    screen = init_screen()
    clock = pygame.time.Clock()

    episode_rewards = []
    max_episode_reward = -math.inf
    max_time = 5000
    delta_time = 1
    epsilon = 0.9  # Initial epsilon
    epsilon_decay = 0.995  # Decay rate
    min_epsilon = 0.0001  # Minimum epsilon
    road_width = 10
    replay_buffer = ReplayBuffer(10000)






    for episode in range(n_episodes):
        cars = [RealisticCar2D() for _ in range(num_cars)]
        spacing = len(track_2D) // num_cars
        for i, car in enumerate(cars):
            place_car_on_track(car, track_2D, i* spacing)
        #plot_track_and_cars(track_2D, [car.position for car in cars])

        states = [get_state(car, track_2D) for car in cars]
        done_flags = [False] * num_cars
        episode_reward = 0
        time = 0



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
                if random.random() < epsilon:
                    action = np.random.uniform(-1, 1, size=(output_dim,))
                else:
                    with torch.no_grad():
                        action = policy_network(state.unsqueeze(0)).squeeze().numpy()

                car = cars[i]
                car.update_position(delta_time=delta_time, acceleration=action[0], steering_angle=action[1], road_width=road_width)
                next_state = get_state(car, track_2D)
                collision, _ = check_collision_with_track(car.position, track_2D, road_width=road_width)
                reward = advanced_reward_function(car, track_2D, collision)

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

                optimizer.zero_grad()
                loss_q.backward()
                optimizer.step()

                # Update the target network
                target_q_network.update_from_model(q_network)

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Total Reward: {episode_reward}, Epsilon: {epsilon}")
        if episode_reward > max_episode_reward:
            max_episode_reward = episode_reward
            torch.save(q_network.state_dict(), twin_model_path)

    return episode_rewards


# Training the Q-Network within the racing environment
episode_rewards_fixed = enhanced_q_learning(network, target_q_network,optimizer, criterion)

# Displaying the first 10 episode rewards
episode_rewards_fixed[:10]
