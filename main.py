# Importing required libraries
import numpy as np

# Improved Track Representation
# Using a list of dictionaries to represent the track, encoding curvature and width
track = [
    {'curvature': 0, 'width': 10},
    {'curvature': 0.1, 'width': 10},
    {'curvature': 0, 'width': 10},
    {'curvature': -0.1, 'width': 10},
    {'curvature': 0, 'width': 10},
]


def create_loop_track(radius=100, num_segments=100):
    """
    Create a loop track with the given radius and number of segments.

    Parameters:
        radius (int): The radius of the loop.
        num_segments (int): The number of segments to divide the loop into.

    Returns:
        track_2D (list): A list of [x, y] coordinates representing the loop track.
    """
    track_2D = []
    for i in range(num_segments):
        angle = i * 2 * np.pi / num_segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        track_2D.append([x, y])
    track_2D.append(track_2D[0])  # Close the loop
    return track_2D
# Decode track into 2D coordinates for visualization (or for more advanced physics)
def decode_track_to_2D(track, segment_length=1.0):
    x, y = 0, 0  # Initialize coordinates
    angle = 0  # Initialize angle
    track_2D = [(x, y)]  # List to hold 2D coordinates of the track

    for segment in track:
        dx = segment_length * np.cos(angle)
        dy = segment_length * np.sin(angle)
        x += dx
        y += dy
        track_2D.append((x, y))

        # Update the angle based on the curvature
        angle += segment['curvature']

    return track_2D

#track_2D=decode_track_to_2D(track)
track_2D=create_loop_track()
# Importing required libraries
import math


# More Realistic Car2D class with advanced physics features
class RealisticCar2D:
    def __init__(self, position=np.array([0.0, 0.0]), velocity=np.array([0.0, 0.0]), mass=1.0, max_speed=10.0, max_acceleration=1,drag_coefficient=0.001,max_steering_rate=0.1):
        self.position = position  # Position vector [x, y]
        self.velocity = velocity  # Velocity vector [vx, vy]
        self.mass = mass  # Mass of the car
        self.max_speed = max_speed  # Maximum speed
        self.max_acceleration = max_acceleration # Maximum acceleration
        self.drag_coefficient = drag_coefficient  # Aerodynamic drag coefficient
        self.current_steering_angle = 0.0  # New attribute
        self.max_steering_rate = 0.1  # Maximum rate of change of steering angle per second
        self.distance_covered=0

    def update_position(self, delta_time, acceleration=0.0, steering_angle=0.0, road_width=10.0):
        old_position = self.position.copy()
        # 1. Tire Friction and Acceleration
        acceleration = min(acceleration, self.max_acceleration)

        # 2. Mass and Inertia

        # 3. Aerodynamic Drag
        speed = np.linalg.norm(self.velocity)
        direction = self.velocity / speed if speed != 0 else np.array([1.0, 0.0])
        acceleration_vector = acceleration * direction
        drag_force = -self.drag_coefficient * self.velocity * speed

        # 4. Steering and Turning Radius
        steering_diff = steering_angle - self.current_steering_angle
        steering_diff = np.clip(steering_diff, -self.max_steering_rate * delta_time, self.max_steering_rate * delta_time)
        self.current_steering_angle += steering_diff

        rotation_matrix = np.array([[math.cos(self.current_steering_angle), -math.sin(self.current_steering_angle)],
                                    [math.sin(self.current_steering_angle), math.cos(self.current_steering_angle)]])

        self.velocity = np.dot(rotation_matrix, self.velocity)

        # 5. Update Velocity and Position
        self.velocity += (acceleration_vector + drag_force / self.mass) * delta_time
        self.position += self.velocity * delta_time + 0.5 * acceleration_vector * delta_time ** 2

        # 7. Speed Caps
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = (self.velocity / speed) * self.max_speed
        distance_moved = np.linalg.norm(np.array(self.position) - np.array(old_position))
        self.distance_covered += distance_moved


def check_collision_with_track(car_position, track_2D, road_width):
    closest_point = None
    min_distance = float('inf')

    for point in track_2D:
        distance = np.linalg.norm(car_position - np.array(point))
        if distance < min_distance:
            min_distance = distance
            closest_point = point

    if min_distance > road_width / 2:
        return True, closest_point  # Collision detected
    else:
        return False, closest_point  # No collision


# Function to check collision between two cars
def check_collision_between_cars(car1_position, car2_position, car_size=1.0):
    distance_between_cars = np.linalg.norm(car1_position - car2_position)
    if distance_between_cars < car_size:
        return True  # Collision detected
    else:
        return False  # No collision

def place_car_on_track(car, track_2D, start_index):
    """
    Place the car on a specific segment of the track.

    Parameters:
        car (RealisticCar2D): The car object.
        track_2D (list): The list of [x, y] coordinates representing the loop track.
        segment_index (int): The index of the track segment where the car should be placed.

    Returns:
        None: The car's position is updated in place.
    """
    car.position = np.array(track_2D[start_index % len(track_2D)])

    # Compute the direction towards the next point in the track
    next_point = np.array(track_2D[(start_index + 1) % len(track_2D)])
    direction_vector = next_point - car.position

    # Normalize the direction vector
    direction_vector /= np.linalg.norm(direction_vector)

    # Set the car's velocity to make it head towards the centerline
    car.velocity = direction_vector


def distance_to_centerline(point, segment_start, segment_end):
    A = np.array(segment_start)
    B = np.array(segment_end)
    C = np.array(point)

    AB = B - A
    AC = C - A

    t = np.dot(AC, AB) / np.dot(AB, AB)

    # Clamp t in the range [0, 1]
    t = max(0, min(1, t))

    closest_point = A + t * AB

    distance = np.linalg.norm(closest_point - C)

    return distance, closest_point


def compute_distance_central_line(car, track_2D,time_step=1):
    future_position = car.position + np.array(car.velocity) * time_step
    min_distance = float('inf')
    closest_point_on_central_line = None
    for i in range(len(track_2D) - 1):
        segment_start = track_2D[i]
        segment_end = track_2D[i + 1]
        distance, closest_point = distance_to_centerline(future_position, segment_start, segment_end)
        if distance < min_distance:
            min_distance = distance
            closest_point_on_central_line = closest_point
    return min_distance, closest_point_on_central_line

