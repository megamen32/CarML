# First, let's import the required libraries
import traceback

import pygame
import numpy as np
from main import compute_distance_central_line

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
translation = [SCREEN_WIDTH // 4+100, SCREEN_HEIGHT // 4+100]
CAR_RADIUS = 5
scale_factor=1
# Import additional pygame modules
import pygame_gui
manager=None
def init_screen():
    global manager
    # Initialize screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Racing Game Visualization')
    manager = pygame_gui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT))


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


