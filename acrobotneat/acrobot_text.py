import sys


from gym.envs.classic_control import AcrobotEnv,CartPoleEnv
import pygame
from pygame.locals import QUIT, KEYDOWN, K_UP, K_DOWN, K_RIGHT, K_LEFT

if __name__ == "__main__":
    pygame.init()  # Инициализируйте pygame
    env = CartPoleEnv(render_mode="human")

    while True:
        env.reset()
        done=False
        while not done:
            #action=1
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN:
                    if event.key == K_RIGHT:
                        action=1  # Предположим, что действие "2" соответствует движению вверх
                    elif event.key == K_LEFT:
                        action=0  # Предположим, что действие "0" соответствует движению влево
                    elif event.key == K_UP:
                        pass

            _,_,done,_,_=env.step(action)
            env.render()
            # добавьте здесь логику, если есть действие, соответствующее движению вправо
