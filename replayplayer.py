# First, let's import the required libraries
import os
import traceback

import pygame
import numpy as np
from main import compute_distance_central_line

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
            traceback.print_exc()

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
    pygame.display.set_caption('Racing Game Visualization')
    manager = pygame_gui.UIManager((WINDOW_WIDTH, WINDOW_HEIGHT))

    return screen,manager




def draw_track(  screen,track_2D,road_width):
    global manager
    screen.fill((0, 0, 0))  # Fill the screen with black
    # Draw the track
    scaled_track = [(int(x * scale_factor + translation[0]), int(y * scale_factor + translation[1])) for x, y in track_2D]
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



def draw_car(car, screen, track_2D):
    car_position = [int(coord * scale_factor + translation[idx % 2]) for idx, coord in enumerate(car.position)]
    pygame.draw.circle(screen, (255, 0, 0), car_position, CAR_RADIUS)

    # Compute distance to central line and closest point on central line

    # Apply scaling and translation to the closest_point_on_central_line
    goal = [int(coord * scale_factor + translation[idx % 2]) for idx, coord in enumerate(track_2D[car.lastest_point_idx])]
    pygame.draw.circle(screen, (0,255, 0), goal, 3)

    ngoal = [int(coord * scale_factor + translation[idx % 2]) for idx, coord in enumerate(track_2D[min(car.closest_point_idx+5,len(track_2D)-1)])]
    pygame.draw.circle(screen, (0,255, 255), ngoal, 3)

    #pygame.draw.line(screen, (0, 255, 0), car_position, closest_point_scaled, 1)
    cur_segment = [int(coord * scale_factor + translation[idx % 2]) for idx, coord in enumerate(track_2D[car.closest_point_idx])]
    pygame.draw.circle(screen, (0, 0, 255), cur_segment, 4)
    scaled_velocity = [int(v * 3) for v in car.velocity]  # Assuming car.velocity is a 2D vector
    velocity_endpoint = [car_position[i] + scaled_velocity[i] for i in range(2)]
    pygame.draw.line(screen, (255, 255, 0), car_position, velocity_endpoint, 1)
    if car.closest_point_idx + 1 < len(track_2D):
        # Direction of the track segment
        dir_vector = np.array(track_2D[car.closest_point_idx + 1]) - np.array(track_2D[car.closest_point_idx])
        # Normalize the direction
        dir_vector /= np.linalg.norm(dir_vector)

        # Compute the two potential perpendicular directions
        perp_vector1 = np.array([-dir_vector[1], dir_vector[0]])
        perp_vector2 = -perp_vector1  # The opposite direction

        # Compute points along these vectors
        point1 = car.position + 0.1 * perp_vector1
        point2 = car.position + 0.1 * perp_vector2

        # Determine which point is closer to the track segment's center
        segment_center = 0.5 * (np.array(track_2D[car.closest_point_idx]) + np.array(track_2D[car.closest_point_idx + 1]))
        if np.linalg.norm(point1 - segment_center) < np.linalg.norm(point2 - segment_center):
            perp_vector = perp_vector1
        else:
            perp_vector = perp_vector2

        # Find the intersection point on the track
        intersection_point = car.position + car.distance_to_central * perp_vector

        car_position = [int(coord * scale_factor + translation[idx % 2]) for idx, coord in enumerate(car.position)]
        intersection_point_scaled = [int(coord * scale_factor + translation[idx % 2]) for idx, coord in enumerate(intersection_point)]

        # Draw the perpendicular line
        pygame.draw.line(screen, (0, 100, 100), tuple(car_position), tuple(intersection_point_scaled), 3)


