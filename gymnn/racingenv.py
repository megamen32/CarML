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


def get_state(car, track_2D):
    fclosest_point = track_2D[min((car.closest_point_idx+5),len(track_2D)-1)]  # Assuming you have this function implemented
    distance =car.distance_to_central  # Assuming you have this function implemented
    fpoint_x,fpoint_y=fclosest_point
    point_x,point_y= track_2D[(car.closest_point_idx)]
    accelration=car.acceleration
    speed_x,spped_y = car.velocity
    posx,posy=car.position
    return torch.FloatTensor([distance,posx,posy,fpoint_x,fpoint_y,accelration ,speed_x,spped_y,point_x,point_y,car.current_steering_angle])


class RacingEnv(gym.Env):
    def advanced_reward_function(self,car, collision):
        collision_penalty=0
        if collision:
            collision_penalty= -10000.0
            return collision_penalty,car.closest_point_idx

        distance=-0.5

        backward_penalty = 0
        if car.closest_point_idx < car.lastest_point_idx:
            backward_penalty = -50.0  # Выберите значение штрафа, которое вам подходит
            return backward_penalty,car.closest_point_idx
        elif  car.closest_point_idx > car.lastest_point_idx:
            distance = compute_travel_distance(self.track_2D, self.track_2D[car.closest_point_idx], car.closest_point_idx)

        finish_reward = 0
        if car.closest_point_idx == len(self.track_2D) - 2:
            finish_reward = 5000000000 / self.time
            print('finish!', finish_reward, self.time)
        segment_dir = np.array(self.track_2D[car.closest_point_idx + 1]) - np.array(self.track_2D[car.closest_point_idx])
        segment_dir = segment_dir / np.linalg.norm(segment_dir)  # Normalize to get a unit vector


        # Use car's velocity to get the direction of movement
        car_dir = np.array(car.velocity)  # Assuming car.velocity is a 2D vector
        car_dir = car_dir / np.linalg.norm(car_dir)  # Normalize to get a unit vector

        # Calculate the alignment of the car's velocity with the segment's direction
        alignment = np.dot(car_dir, segment_dir)

        #center_reward = (self.road_width - car.distance_to_central) / self.road_width
        reward = distance * 2 + finish_reward + backward_penalty + collision_penalty + car.speed*alignment
        return reward,finish_reward>0
    def __init__(self, road_width=100, delta_time=1,max_time=100):
        super(RacingEnv, self).__init__()
        self.road_width = road_width
        self.max_time = max_time
        self.time=0
        self.delta_time=delta_time
        # Action space: [acceleration, turn_angle]
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        # State space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        pygame.init()
        self.screen, self.manager = init_screen()

        self.reset()

    def reset(self, seed=0, options=None):
        if options is None:
            options = {}
        self.track_2D=main.make_strange_trace(random.random()>0.5,self.road_width//10)

        self.car = RealisticCar2D()

        place_car_on_track(self.car, self.track_2D, 0)
        self.time=0
        self._step=0

        state = get_state(self.car, self.track_2D)  # Assuming single car for simplicity
        return state


    def step(self, action):
        self.time+=self.delta_time
        self._step+=1
        self.car.update_position(track_2D=self.track_2D,acceleration=action[0], steering_angle=action[1], road_width=self.road_width,
                                 delta_time=self.delta_time)

        next_state = get_state(self.car, self.track_2D)
        collision, _ = check_collision_with_track(self.car.position, self.track_2D, road_width=self.road_width)

        reward, finish = self.advanced_reward_function(self.car, collision)

        done = collision or self.max_time <self.time or finish

        return next_state, reward, done, {}

    def render(self, mode='human'):
        # Clear the screen
        self.screen.fill((0, 0, 0))

        # Draw the track
        draw_track(self.screen, self.track_2D, self.road_width)

        # Draw the car
        draw_car(self.car, self.screen, self.track_2D)

        # Refresh the display
        pygame.display.flip()

        # Optionally add delay for better visualization
        #pygame.time.wait(50)

    def close(self):
        # Cleanup logic
        pygame.quit()
