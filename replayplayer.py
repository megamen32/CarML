# First, let's import the required libraries
import pygame
import numpy as np
from main import compute_distance_central_line

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
translation = [SCREEN_WIDTH // 4+100, SCREEN_HEIGHT // 4+100]
CAR_RADIUS = 5
scale_factor=1


def init_screen():
    # Initialize screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Racing Game Visualization')
    return screen

# Function to visualize the game using Pygame
def visualize_game_with_pygame(replay, track_2D):
    """
    Visualize a replay of the game using Pygame.

    Parameters:
        replay (list): A list of states representing the replay of an episode.
        track_2D (list): The list of [x, y] coordinates representing the loop track.

    Returns:
        None: The function will produce a Pygame window to visualize the replay.
    """
    clock = pygame.time.Clock()
    scale_factor = 0.1  # Adjust this value as needed
    translation = [SCREEN_WIDTH // 4, SCREEN_HEIGHT // 4]  # Adjust these values as needed

    for i, state in enumerate(replay):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        screen.fill((0, 0, 0))  # Fill the screen with black

        # Draw the track
        scaled_track = [(int(x * scale_factor + translation[0]), int(y * scale_factor + translation[1])) for x, y in track_2D]
        pygame.draw.lines(screen, (255, 255, 255), False, scaled_track, 2)

        # Draw the car
        car_position = [int(coord * scale_factor + translation[idx % 2]) for idx, coord in enumerate(state[:2])]
        pygame.draw.circle(screen, (255, 0, 0), car_position, CAR_RADIUS)

        pygame.display.update()
        clock.tick(30)


def plot_track_and_cars(track, car_positions):
    import matplotlib.pyplot as plt
    track_np = np.array(track)
    car_positions_np = np.array(car_positions)

    plt.scatter(track_np[:, 0], track_np[:, 1], c='blue', label='Track')
    plt.scatter(car_positions_np[:, 0], car_positions_np[:, 1], c='red', label='Cars')

    plt.axis('equal')
    plt.legend()
    plt.show()


# Please note that you'll need to adapt this function into your existing code
# and call it with the appropriate replay and track_2D parameters to visualize the game.
def visualize_replay(replay, track_2D):
    """
    Visualize a replay of an episode using Matplotlib.

    Parameters:
        replay (list): A list of states representing the replay of an episode.
        track_2D (list): The list of [x, y] coordinates representing the loop track.

    Returns:
        None: The function will produce a Matplotlib plot to visualize the replay.
    """


    # Extract track coordinates
    track_x, track_y = zip(*track_2D)

    for i, state in enumerate(replay):
        plt.figure(figsize=(10, 10))

        # Plot the track
        plt.plot(track_x, track_y, 'k-', linewidth=2)

        # Extract car position from the state
        car_position = state[:2]  # Assuming the first two elements represent car position
        plt.scatter(*car_position, c='r', s=100, label='Car')

        plt.title(f"Frame {i + 1}")
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()

        # Show or save the frame
        plt.show()  # Replace with plt.savefig() to save frames


def save_replay(replay, track_2D, episode_num, folder='replays'):

    """
    Save a replay of an episode as a series of images.

    Parameters:
        replay (list): A list of states representing the replay of an episode.
        track_2D (list): The list of [x, y] coordinates representing the loop track.
        episode_num (int): The episode number.
        folder (str): The folder where to save the replays.

    Returns:
        None: The function will save images to visualize the replay.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Extract track coordinates
    track_x, track_y = zip(*track_2D)

    for i, state in enumerate(replay):
        plt.figure(figsize=(10, 10))

        # Plot the track
        plt.plot(track_x, track_y, 'k-', linewidth=2)

        # Extract car position from the state
        car_position = state[:2]  # Assuming the first two elements represent car position
        plt.scatter(*car_position, c='r', s=100, label='Car')

        plt.title(f"Frame {i + 1}")
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()

        # Save the frame
        plt.savefig(f"{folder}/episode_{episode_num}_frame_{i + 1}.png")
        plt.close()
def draw_track(  screen,track_2D,road_width):
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
def draw_car(car, screen, track_2D):
    car_position = [int(coord * scale_factor + translation[idx % 2]) for idx, coord in enumerate(car.position)]
    pygame.draw.circle(screen, (255, 0, 0), car_position, CAR_RADIUS)

    # Compute distance to central line and closest point on central line
    distance_to_central_line, closest_point_on_central_line = compute_distance_central_line(car, track_2D)

    # Apply scaling and translation to the closest_point_on_central_line
    closest_point_scaled = [int(coord * scale_factor + translation[idx % 2]) for idx, coord in enumerate(closest_point_on_central_line)]

    pygame.draw.line(screen, (0, 255, 0), car_position, closest_point_scaled, 1)
    goal = [int(coord * scale_factor + translation[idx % 2]) for idx, coord in enumerate(track_2D[car.closest_point_idx])]
    pygame.draw.circle(screen, (0, 0, 255), goal, 4)
    #pygame.draw.line(screen, (0, 0, 255), car_position, goal, 1)


