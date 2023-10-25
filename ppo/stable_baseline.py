import os.path
import traceback

import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv

from car_dicsrete.discreteracingenv import DiscreteRacingEnv

PATH = "ppo_discreteracingenv4"
def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = DiscreteRacingEnv()
        return env
    return _init

vec_env=make_vec_env(DiscreteRacingEnv)
class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[128, 128],  # Пример архитектуры

                                         )

# Создаем модель PPO
model = PPO(CustomPolicy, vec_env, verbose=1)
if os.path.exists(f'{PATH}.zip'):
    try:
        model = PPO.load(PATH, vec_env)
    except:
        traceback.print_exc()
# Обучаем модель
if __name__ == '__main__':
# Определяем или загружаем вашу среду
    vec_env = SubprocVecEnv([make_env("DiscreteRacingEnv",i) for i in range(7)])# Ваша среда
    model.learn(total_timesteps=1000000)
    model.save(PATH)
    # Оцениваем модель
    env2=DiscreteRacingEnv()
    obs,_ = env2.reset()
    for _ in range(10000):
        actions, _states = model.predict(obs)
        action=np.max(actions)
        obs, rewards,  truncated,_,_ = env2.step(action)
        if truncated:
            env2.reset()
        #env.render()
