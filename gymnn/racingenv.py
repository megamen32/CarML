import math
import random

import gym
import numpy as np

import gymnn
import pygame
import torch
from gym import spaces

import main
from main import RealisticCar2D, place_car_on_track, check_collision_with_track, compute_travel_distance
from replayplayer import init_screen, draw_track, draw_car


def get_state(car, track_2D, road_width):
    # Lateral velocity (velocity component perpendicular to the car's orientation)
    lateral_velocity = np.dot(car.velocity, np.array([-car.orientation[1], car.orientation[0]]))

    # Next curve direction (simple version; more accurate methods might require analyzing multiple future points)
    max_idx = min(car.closest_point_idx + 2, len(track_2D) - 1)
    next_dir = np.array(track_2D[max_idx]) - np.array(track_2D[min(car.closest_point_idx + 1, len(track_2D) - 1)])
    curve_direction = 1 if np.cross(next_dir, car.orientation) > 0 else -1  # 1 for left turn, -1 for right turn

    # Next curve distance
    next_curve_distance = np.linalg.norm(np.array(track_2D[max_idx]) - car.position)

    return torch.FloatTensor([
        road_width,
        car.speed,
        car.acceleration,
        car.turning_rate,
        lateral_velocity,
        car.distance_to_central,
        car.angle,
        next_curve_distance,
        curve_direction
    ])


class RacingEnv(gym.Env):
    def advanced_reward_function(self,car, collision):
        COLLISION_PENALTY = -1000.0
        BACKWARD_PENALTY = -50.0
        FINISH_MULTIPLIER = 50000
        # Check for collisions
        if collision:
            return COLLISION_PENALTY, False

        # Check for backward movement
        if car.closest_point_idx < car.lastest_point_idx:
            return BACKWARD_PENALTY, False

        # Calculate distance traveled
        distance_reward = 0
        if car.closest_point_idx > car.lastest_point_idx:
            # Normalize by dividing by the total track length
            total_track_length = sum(np.linalg.norm(np.array(self.track_2D[i]) - np.array(self.track_2D[i - 1])) for i in range(1, len(self.track_2D)))
            distance = compute_travel_distance(self.track_2D, car.position, car.closest_point_idx)/total_track_length

        finish_reward = 0
        if car.closest_point_idx == self.total_segments - 2:
            finish_reward = FINISH_MULTIPLIER / self.time
            return finish_reward, True

        # Calculate alignment reward
        alignment_reward = car.alignment * car.speed

        # Calculate the total reward
        total_reward = distance_reward + alignment_reward + finish_reward

        return total_reward, finish_reward > 0
    def __init__(self, road_width=100, delta_time=1,max_time=100,render=True,patience=10):
        super(RacingEnv, self).__init__()
        self._road_width=road_width
        self.patience=patience
        self.max_time = max_time
        self.time=0
        self.delta_time=delta_time
        # Action space: [acceleration, turn_angle]
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        # State space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.cumulative_reward = 0  # To store cumulative rewards
        self._render=render
        if render:
            pygame.init()
            pygame.font.init()
            self.font = pygame.font.SysFont(None, 36)  # Use default font, size 36
            self.screen, self.manager = init_screen()

        self.reset()

    def reset(self, seed=0, options=None):
        if options is None:
            options = {}
        self.road_width =random.uniform(0.2,1.5)* self._road_width
        self.segments_length = self.road_width // 10
        self.track_2D=main.make_strange_trace(random.random() > 0.5, self.segments_length)
        self.total_segments=len(self.track_2D)
        self.car = RealisticCar2D()

        place_car_on_track(self.car, self.track_2D, 0)
        self.time=0
        self._step=0
        self.reward=0
        self.last_best_reward=-math.inf
        self.no_improve_counter=0
        self.cumulative_reward=0
        self.finish=False
        state = get_state(self.car, self.track_2D,self.road_width)  # Assuming single car for simplicity
        return state


    def step(self, action):
        self.time+=self.delta_time
        self._step+=1
        self.car.update_position(track_2D=self.track_2D,acceleration=action[0], steering_angle=action[1], road_width=self.road_width,
                                 delta_time=self.delta_time)

        next_state = get_state(self.car, self.track_2D,self.road_width)
        collision, _ = check_collision_with_track(self.car.position, self.track_2D, road_width=self.road_width)

        self.reward, self.finish = self.advanced_reward_function(self.car, collision)
        self.cumulative_reward += self.reward
        done = collision or self.max_time <self.time or self.finish
        if self.cumulative_reward > self.last_best_reward:
            self.last_best_reward = self.cumulative_reward
            self.no_improve_counter = 0
        else:
            self.no_improve_counter += 1
        if self.no_improve_counter >= self.patience:
            done=True

        return next_state, self.reward, done, {}

    def render(self, mode='human',noise_std=None,custom_info=None):
        # Clear the screen
        self.screen.fill((0, 0, 0))

        # Draw the track
        draw_track(self.screen, self.track_2D, self.road_width)
        speed_text = self.font.render(f"Speed: {self.car.speed:.2f} dist_central={self.car.distance_to_central:.2f}  t:{self.time:.2f}/{self.max_time}", True, (255, 255, 255))
        current_reward_text = self.font.render(f"Current Reward: {self.reward:.2f}", True, (255, 255, 255))
        cumulative_reward_text = self.font.render(f"Cumulative Reward: {self.cumulative_reward:.2f}", True, (255, 255, 255))
        if noise_std:
                noise = self.font.render(f"noise: {noise_std:.2f}", True, (255, 255, 255))
                self.screen.blit(noise, (10, 130))  # Below the cumulative_reward_text
        if custom_info:
            noise = self.font.render(custom_info, True, (255, 255, 255))
            self.screen.blit(noise, (10, 170))  # Below the cumulative_reward_text

        # Draw these texts on the screen
        self.screen.blit(speed_text, (10, 10))  # Top left corner
        self.screen.blit(current_reward_text, (10, 50))  # Below the speed text
        self.screen.blit(cumulative_reward_text, (10, 90))  # Below the current reward text

    # Draw the car
        draw_car(self.car, self.screen, self.track_2D)

        # Refresh the display
        pygame.display.flip()

        if mode=='human':
            # Optionally add delay for better visualization
            pygame.time.wait(50)

    def close(self):
        # Cleanup logic
        pygame.quit()

if __name__ == '__main__':
    env = RacingEnv()
    human_action=[0,0]
    num_to_info=10
    c_infp=0
    rewards=[]
    while True:
        c_infp+=1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    human_action[0] = 1.0  # forward
                elif event.key == pygame.K_s:
                    human_action[0] = -1.0  # backward
                elif event.key == pygame.K_a:
                    human_action[1] = -1.0  # turn left
                elif event.key == pygame.K_d:
                    human_action[1] = 1.0  # turn right
            elif event.type == pygame.KEYUP:
                if event.key in [pygame.K_w, pygame.K_s]:
                    human_action[0] = 0.0
                elif event.key in [pygame.K_a, pygame.K_d]:
                    human_action[1] = 0.0
        env.render(mode='human')
        next_state, reward, done, _ = env.step(action=human_action)
        rewards.append(reward)
        if c_infp>num_to_info:
            print('cur reward',np.sum(rewards))
            rewards=[]
            c_infp=0
        if done:
            env.reset()
