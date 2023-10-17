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


def get_state(car, track_2D,road_width):
    fclosest_point = track_2D[min((car.closest_point_idx+5),len(track_2D)-1)]  # Assuming you have this function implemented
    fdistance =car.distance_to_central  # Assuming you have this function implemented
    distance =car.distance_to_curr  # Assuming you have this function implemented
    fpoint_x,fpoint_y=fclosest_point
    point_x,point_y= track_2D[(car.closest_point_idx)]
    accelration=car.acceleration
    speed_x,spped_y = car.velocity
    posx,posy=car.position
    angle=car.angle
    return torch.FloatTensor([road_width,distance,angle,fdistance,posx,posy,fpoint_x,fpoint_y,accelration ,speed_x,spped_y,point_x,point_y,car.current_steering_angle])


class RacingEnv(gym.Env):
    def advanced_reward_function(self,car, collision):
        collision_penalty=0
        if collision:
            collision_penalty= -10000.0
            return collision_penalty,False

        distance=-1

        backward_penalty = 0
        if car.closest_point_idx < car.lastest_point_idx:
            backward_penalty = -50.0  # Выберите значение штрафа, которое вам подходит
            return backward_penalty,False
        elif  car.closest_point_idx > car.lastest_point_idx:
            distance = compute_travel_distance(self.track_2D, car.position, car.closest_point_idx)

        finish_reward = 0
        total_segments =self.total_segments
        if car.closest_point_idx == total_segments - 2:
            finish_reward = 500000 / self.time
            return finish_reward,True


        #center_reward = (self.road_width - car.distance_to_central) / self.road_width
        reward = distance * 2 + finish_reward + backward_penalty + collision_penalty + car.alignment*car.speed
        return reward,finish_reward>0
    def __init__(self, road_width=100, delta_time=1,max_time=100,render=True):
        super(RacingEnv, self).__init__()
        self._road_width=road_width

        self.max_time = max_time
        self.time=0
        self.delta_time=delta_time
        # Action space: [acceleration, turn_angle]
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        # State space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
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

        return next_state, self.reward, done, {}

    def render(self, mode='human',noise_std=None):
        # Clear the screen
        self.screen.fill((0, 0, 0))

        # Draw the track
        draw_track(self.screen, self.track_2D, self.road_width)
        speed_text = self.font.render(f"Speed: {self.car.speed:.2f} dist_central={self.car.distance_to_central:.2f}", True, (255, 255, 255))
        current_reward_text = self.font.render(f"Current Reward: {self.reward:.2f}", True, (255, 255, 255))
        cumulative_reward_text = self.font.render(f"Cumulative Reward: {self.cumulative_reward:.2f}", True, (255, 255, 255))
        if noise_std:
                noise = self.font.render(f"noise: {noise_std:.2f}", True, (255, 255, 255))
                self.screen.blit(noise, (10, 130))  # Below the cumulative_reward_text

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
