# First, let's import the required libraries
import os
import traceback

import pygame
import numpy as np
from carmodel import compute_distance_central_line, RealisticCar2D

# Constants
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
translation = [WINDOW_WIDTH // 4 + 100, WINDOW_HEIGHT // 4 + 100]
CAR_RADIUS = 5
scale_factor=1
# Import additional pygame modules
import pygame_gui
manager=None
from screeninfo import get_monitors
from multiprocessing import Lock

POSITION_FILE = 'last_position.txt'
lock = Lock()

def get_screen_resolution():
    monitor = get_monitors()[0]
    return monitor.width, monitor.height

SCREEN_WIDTH, SCREEN_HEIGHT = get_screen_resolution()

def get_next_position():
    with lock:
        if not os.path.exists(POSITION_FILE):
            with open(POSITION_FILE, 'w') as f:
                f.write("0,0")

        x,y=0,0
        try:
            with open(POSITION_FILE, 'r') as f:
                x, y = map(int, f.read().split(','))
        except:
            pass

        new_x = x + WINDOW_HEIGHT  # Здесь 800 - это ширина окна приложения
        new_y = y

        if new_x+WINDOW_WIDTH > SCREEN_WIDTH:
            new_x = 0
            new_y += WINDOW_HEIGHT  # Здесь 600 - это высота окна приложения

        if new_y+WINDOW_HEIGHT > SCREEN_HEIGHT:
            new_x, new_y = 0, 0

        with open(POSITION_FILE, 'w') as f:
            f.write(f"{new_x},{new_y}")

    return new_x, new_y
def init_screen():
    global manager

    # Получаем следующую позицию для окна
    x_position, y_position = get_next_position()

    # Инициализируем экран на указанной позиции
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x_position,y_position)
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.NOFRAME, pygame.RESIZABLE)

    clock = pygame.time.Clock()
    pygame.display.set_caption('Racing Game Visualization')
    manager = pygame_gui.UIManager((WINDOW_WIDTH, WINDOW_HEIGHT))

    return screen,manager,clock




def draw_track(  screen,track_2D,road_width,car_position):
    global manager
    screen.fill((0, 0, 0))  # Fill the screen with black
    # Draw the track
    offset_x = screen.get_width() // 2 - int(car_position[0] * scale_factor)
    offset_y = screen.get_height() // 2 - int(car_position[1] * scale_factor)

    # Apply the offset to the track's points
    scaled_track = [(int(x * scale_factor + offset_x), int(y * scale_factor + offset_y)) for x, y in track_2D]
    for i in range(len(scaled_track) - 1):
        x1, y1 = scaled_track[i]
        x2, y2 = scaled_track[i + 1]
        dx, dy = x2 - x1, y2 - y1

        # Perpendicular vector for the border
        px, py = -dy, dx
        len_p = np.sqrt(px ** 2 + py ** 2)
        px, py = px / len_p, py / len_p

        # Draw the two borders
        border_width = road_width // 2
        pygame.draw.line(screen, (200, 200, 200), (x1 + px * border_width, y1 + py * border_width), (x2 + px * border_width, y2 + py * border_width), 2)
        pygame.draw.line(screen, (200, 200, 200), (x1 - px * border_width, y1 - py * border_width), (x2 - px * border_width, y2 - py * border_width), 2)
    # Additional code to display current metrics



def draw_car(car:RealisticCar2D, screen, track_2D):
    translation= screen.get_width() // 2 - int(car.position[0] * scale_factor), screen.get_height() // 2 - int(car.position[1] * scale_factor)

    car_position = [screen.get_width() // 2, screen.get_height() // 2]
    orientation = [int(coord * scale_factor*5 + translation[idx % 2]) for idx, coord in enumerate(car.orientation)]
    car_forward=car.position+orientation
    pygame.draw.circle(screen, (255, 0, 0), car_position, CAR_RADIUS)
    pygame.draw.circle(screen, (255, 0, 255), car_forward, CAR_RADIUS//2)
    closest_central_point = [int(coord * scale_factor + translation[idx % 2]) for idx, coord in enumerate(car.closest_central_point)]
    pygame.draw.circle(screen, (128, 128, 255), closest_central_point, 3)
    # Compute distance to central line and closest point on central line

    # Apply scaling and translation to the closest_point_on_central_line
    goal = [int(coord * scale_factor + translation[idx % 2]) for idx, coord in enumerate(track_2D[car.lastest_point_idx])]
    pygame.draw.circle(screen, (0,255, 0), goal, 3)

    for i in range(1,car.n+1):
        ngoal = [int(coord * scale_factor + translation[idx % 2]) for idx, coord in enumerate(track_2D[min(car.closest_point_idx+i*car.curve_step,len(track_2D)-1)])]
        pygame.draw.circle(screen, (0,255-i/(car.n+1)*255, 255-i/(car.n+1)*255), ngoal, 3)

    #pygame.draw.line(screen, (0, 255, 0), car_position, closest_point_scaled, 1)
    cur_segment = [int(coord * scale_factor + translation[idx % 2]) for idx, coord in enumerate(track_2D[car.closest_point_idx])]
    pygame.draw.circle(screen, (0, 0, 255), cur_segment, 4)
    scaled_velocity = [int(v * 3) for v in car.velocity]  # Assuming car.velocity is a 2D vector
    velocity_endpoint = [car_position[i] + scaled_velocity[i] for i in range(2)]
    pygame.draw.line(screen, (255, 255, 0), car_position, velocity_endpoint, 1)

