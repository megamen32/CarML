import math
import random

import gym
import numpy as np

import gymnn
import pygame
import torch
from gym import spaces

import carmodel
import trackgenerator
from carmodel import RealisticCar2D, place_car_on_track, check_collision_with_track, compute_travel_distance
from replayplayer import init_screen, draw_track, draw_car

player_idle = 500


def get_state(car, track_2D,total_track_length, n=1):
    # Lateral velocity (velocity component perpendicular to the car's orientation)

    # Values for next n points




    # Flatten the lists to add to the state
    curve_directions = np.array(car.curve_directions).flatten()
    curve_distances = np.array(car.curve_distances).flatten()
    return torch.FloatTensor([
        car.speed_normilize,
        car.acceleration,
        car.turning_rate,
        car.lateral_velocity,
        car.distance_to_central,
        car.angle,
        car.alignment,
        *curve_distances,
        *curve_directions
    ])

ACTIONS = {
    0: [0.0, 0.0],  # Do nothing
    1: [1.0, 0.0],  # Accelerate forward
    2: [-1.0, 0.0], # Accelerate backward
    3: [0.0, -1.0], # Turn left
    4: [0.0, 1.0]   # Turn right
}
class DiscreteRacingEnv(gym.Env):
    def advanced_reward_function(self,car, collision):
        COLLISION_PENALTY = -2
        BACKWARD_PENALTY = -0.2
        FINISH_MULTIPLIER = 100
        # Check for collisions
        if collision:
            return COLLISION_PENALTY, False

        # Check for backward movement
        if car.closest_point_idx < car.lastest_point_idx:
            return BACKWARD_PENALTY, False

        # Calculate distance traveled
        distance_reward = -0.05
        if car.closest_point_idx > car.lastest_point_idx:
            # Normalize by dividing by the total track length


            distance_reward = 0.5

        finish_reward = 0
        if car.closest_point_idx == self.total_segments - 2:
            finish_reward = FINISH_MULTIPLIER * self.time/self.max_time
            return finish_reward, True

        # Calculate alignment reward

        alignment_reward =car.distance_to_central*0.05
        if abs(car.turning_rate)>0.8 and car.speed_normilize<0.9:
            alignment_reward=-0.05

        # Calculate the total reward
        total_reward = distance_reward + alignment_reward + finish_reward

        return total_reward, finish_reward > 0
    def __init__(self, road_width=100, delta_time=1,max_time=20000,render_mode='human',patience=500):
        super(DiscreteRacingEnv, self).__init__()
        self.metadata['render_fps'] =1000
        self.render_mode=render_mode
        render=True if render_mode=='human' else None
        self._road_width=road_width
        self.patience=patience
        self.max_time = max_time
        self.time=0
        self.delta_time=delta_time
        # Action space: [acceleration, turn_angle]
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.CHECK_IMPROVEMENT_INTERVAL= 5
        self.REWARD_HISTORY_SIZE=100
        self.custom_info=None
        # State space


        self._render=render
        if render:
            pygame.init()
            pygame.font.init()
            self.font = pygame.font.SysFont(None, 36)  # Use default font, size 36
            self.screen, self.manager,self.clock = init_screen()

        state=self.reset()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(state.shape[0],), dtype=np.float32)

    def reset(self, seed=0, options=None):
        if options is None:
            options = {}
        self.road_width = self._road_width#*random.uniform(0.2,1.5)
        self.segments_length = self.road_width *0.8
        self.track_2D= trackgenerator.create_complex_track_v2(road_width= self.segments_length)#,radius=200)
        self.total_segments=len(self.track_2D)
        self.car = RealisticCar2D()

        place_car_on_track(self.car, self.track_2D, 0)
        self.total_track_length = sum(np.linalg.norm(np.array(self.track_2D[i]) - np.array(self.track_2D[i - 1])) for i in range(1, len(self.track_2D)))
        self.time=0
        self._step=0
        self.reward=0
        self.last_best_reward=-math.inf
        self.avg_recent_reward = 0
        self.last_avg_recent_reward = -math.inf
        self.no_improve_counter=0
        self.cumulative_reward=0
        self.spin_change=0
        #self.reward_history = [0] * self.REWARD_HISTORY_SIZE
        self.current_reward_index = 0
        self.finish=False
        state = get_state(self.car, self.track_2D,self.total_track_length)  # Assuming single car for simplicity
        return state


    def step(self, action):
        continuous_action = ACTIONS[action]
        self.time+=self.delta_time
        self._step+=1
        self.car.update_position(track_2D=self.track_2D,acceleration=continuous_action[0], steering_angle=continuous_action[1], road_width=self.road_width,
                                 delta_time=self.delta_time)

        collision, closest_point = check_collision_with_track(self.car.position, self.track_2D, road_width=self.road_width)

        if collision:
            # 1. Вектор вдоль сегмента трассы
            segment_dir = np.array(self.track_2D[self.car.closest_point_idx+1]) - np.array(self.track_2D[self.car.closest_point_idx])
            segment_dir_normalized = segment_dir / np.linalg.norm(segment_dir)

            # 2. Нормаль к этому вектору
            normal = np.array([-segment_dir_normalized[1], segment_dir_normalized[0]])

            # Отразим скорость машины относительно нормали
            reflection = self.car.velocity - 2 * np.dot(self.car.velocity, normal) * normal

            # 3. Применяем отражение
            self.car.velocity = 0.5 * reflection  # учитываем потерю энергии при столкновении
            self.car.acceleration = 0
            self.car.orientation = self.car.velocity / np.linalg.norm(self.car.velocity)
            # Возвращаем машину обратно на трассу
            direction_to_closest_point = np.array(closest_point) - self.car.position
            direction_normalized = direction_to_closest_point / np.linalg.norm(direction_to_closest_point)
            self.car.position += 0.2 * self.road_width * direction_normalized


        self.car.cache_state(self.track_2D,self.delta_time,self.road_width,self.segments_length)
        next_state = get_state(self.car, self.track_2D,self.total_track_length)

        self.reward, self.finish = self.advanced_reward_function(self.car, collision)
        self.cumulative_reward += self.reward
        done = self.max_time <self.time or self.finish
        if self.finish:
            print('finish cum_reward=',self.cumulative_reward,'reward=',self.reward,'time=',self.time,'speed=',self.car.speed_normilize)
        #self.reward_history[self.current_reward_index] = self.reward
        done = self.is_no_improve(done,collision)
        if self.render_mode=='human':
            self.render()
        return next_state, self.reward, done, {}

    def is_no_improve(self, done,collision):
        alpha = 0.5  # Smoothing coefficient
        self.avg_recent_reward = alpha * self.reward + (1 - alpha) * self.avg_recent_reward
        position_change = np.linalg.norm(self.car.position - self.car.prev_position)


        if position_change < self.car.max_speed//2 and abs(self.car.current_steering_angle) > 0.8 and self.car.speed_normilize>0.05:
                self.spin_change += 1
        else:
                self.spin_change = 0
        if self.spin_change> 5:
            self.no_improve_counter+= 5
            self.spin_change=0
        if collision:
            self.no_improve_counter+=10
        if self.car.closest_point_idx <= self.car.lastest_point_idx:
           self.no_improve_counter += 1
        else:
            if self.car.closest_point_idx >= self.car.lastest_point_idx:
                self.no_improve_counter =0

            if self.avg_recent_reward > self.last_best_reward:
                self.last_best_reward = self.avg_recent_reward
                self.no_improve_counter = 0
            elif self.avg_recent_reward < self.last_avg_recent_reward:
                self.no_improve_counter += 1

        if self.no_improve_counter >= self.patience:
            done = True

        self.last_avg_recent_reward = self.avg_recent_reward

        return done

    def render(self,noise_std=None,custom_info=None):
        # Clear the screen
        if not custom_info:
            custom_info=self.custom_info
        self.screen.fill((0, 0, 0))

        # Draw the track
        draw_track(self.screen, self.track_2D, self.road_width,self.car.position)
        speed_text = self.font.render(f"Speed: {self.car.speed_normilize:.2f}", True, (255, 255, 255))
        distance_to_central_text = self.font.render(f"Distance to Central: {self.car.distance_to_central:.2f}", True, (255, 255, 255))
        acceleration_text = self.font.render(f"Acceleration: {self.car.acceleration:.2f}", True, (255, 255, 255))
        turning_rate_text = self.font.render(f"Turning Rate: {self.car.turning_rate:.2f}", True, (255, 255, 255))
        lateral_velocity_text = self.font.render(f"Lateral Velocity: {self.car.lateral_velocity:.2f}", True, (255, 255, 255))
        angle_text = self.font.render(f"Angle: {self.car.angle:.2f}", True, (255, 255, 255))
        alignment_text = self.font.render(f"Alignment: {self.car.alignment:.2f}", True, (255, 255, 255))

        # Curve information (assuming you have n=2 for now, can be extended for more)
        curve_1_distance_str = ', '.join([f"{item:.2f}" for item in self.car.curve_distances])
        curve_1_distance_text = self.font.render(f"Curve 1 Distance: {curve_1_distance_str}", True, (255, 255, 255))
        curve_1_dirs_str = ', '.join([f"{item:.2f}" for item in self.car.curve_directions])
        curve_1_direction_text = self.font.render(f"Curve 1 Direction: {curve_1_dirs_str}", True, (255, 255, 255))

        # Drawing them on the screen
        y_offset = 210  # Starting y-coordinate
        y_gap = 40  # Gap between each text

        self.screen.blit(speed_text, (10, y_offset))
        y_offset += y_gap
        self.screen.blit(distance_to_central_text, (10, y_offset))
        y_offset += y_gap
        self.screen.blit(acceleration_text, (10, y_offset))
        y_offset += y_gap
        self.screen.blit(turning_rate_text, (10, y_offset))
        y_offset += y_gap
        self.screen.blit(lateral_velocity_text, (10, y_offset))
        y_offset += y_gap
        self.screen.blit(angle_text, (10, y_offset))
        y_offset += y_gap
        self.screen.blit(alignment_text, (10, y_offset))
        y_offset += y_gap
        self.screen.blit(curve_1_distance_text, (10, y_offset))
        y_offset += y_gap
        self.screen.blit(curve_1_direction_text, (10, y_offset))
        y_offset += y_gap
        speed_text = self.font.render(f"t:{self.time:.2f}/{self.max_time}", True, (255, 255, 255))
        current_reward_text = self.font.render(f"Current Reward: {self.reward:.2f}", True, (255, 255, 255))
        cumulative_reward_text = self.font.render(f"Cumulative Reward: {self.cumulative_reward:.2f}", True, (255, 255, 255))
        if noise_std:
            if isinstance(noise_std,float):
                noise = self.font.render(f"noise: {noise_std:.2f}", True, (255, 255, 255))
            else:
                noise = self.font.render(f"noise: {noise_std}", True, (255, 255, 255))
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

        pygame.event.pump()
        # Refresh the display
        pygame.display.flip()

        if self.render_mode=='human' and self.metadata['render_fps'] < 1000:
            # Calculate delay to achieve the required FPS
            delay = self.metadata['render_fps']
            self.clock.tick(delay)


    def close(self):
        # Cleanup logic
        pygame.display.quit()
        pygame.quit()

if __name__ == '__main__':
    env = DiscreteRacingEnv()
    env.metadata['render_fps']=30
    human_action=0
    num_to_info=10
    c_infp=0
    rewards=[]
    player= player_idle
    done=False
    while True:
        c_infp+=1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()
            elif event.type == pygame.KEYDOWN:
                player= player_idle
                if event.key == pygame.K_w:
                    human_action = 1  # Accelerate forward
                elif event.key == pygame.K_s:
                    human_action = 2  # Accelerate backward
                elif event.key == pygame.K_a:
                    human_action = 3  # Turn left
                elif event.key == pygame.K_d:
                    human_action = 4  # Turn right
            elif event.type == pygame.KEYUP:
                player= player_idle
                if event.key in [pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d]:
                    human_action = 0  # Do nothing

        if human_action == 0:
            player-=1
        if  player<=0:
            distance_to_central = 1-env.car.distance_to_central
            if env.car.curve_directions[-1] > 0:  # Если следующий поворот налево
                if distance_to_central > 0.5:  # Если машина слишком справа
                    human_action = 2 if env.car.acceleration>0 else 1 # Slow
                elif distance_to_central < 0.5 and distance_to_central > 0.3:
                    human_action = 3 # Переместите влево
                else:
                    human_action=1
            elif env.car.curve_directions[-1] < 0:  # Если следующий поворот направо
                if distance_to_central > 0.5:  # Если машина слишком справа
                    human_action = 2 if env.car.acceleration>0 else 1  # Slow
                elif distance_to_central < 0.5 and distance_to_central > 0.3:
                    human_action = 4 # Переместите право
                else:
                    human_action=1

            else:
                human_action = 2 if distance_to_central > 1 - 0.5 else 1  # Ускорить, если машина далеко от центра; иначе замедлить

# Ускорить или замедлить в зависимости от расстояния до центра
        next_state, reward, done, _ = env.step(action=human_action)
        rewards.append(reward)
        if c_infp>num_to_info:
            print('cur reward',np.sum(rewards))
            rewards=[]
            c_infp=0
        if done:
            player= player_idle
            human_action=0
            env.reset()
