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
    def __init__(self, position=np.array([0.0, 0.0]), velocity=np.array([0.0, 0.0]), mass=1.0, max_speed=10.0, max_acceleration=1,drag_coefficient=0.1,max_steering_rate=0.1):
        self.position = position  # Position vector [x, y]
        self.velocity = velocity  # Velocity vector [vx, vy]
        self.mass = mass  # Mass of the car
        self.max_speed = max_speed  # Maximum speed
        self.max_acceleration = max_acceleration # Maximum acceleration
        self.drag_coefficient = drag_coefficient  # Aerodynamic drag coefficient
        self.current_steering_angle = 0.0  # New attribute
        self.max_steering_rate = 0.1  # Maximum rate of change of steering angle per second

    def update_position(self, delta_time, acceleration=0.0, steering_angle=0.0, road_width=10.0):
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

def distance_to_centerline(car_position, segment_start, segment_end):
    """
    Compute the distance of the car to the centerline of the track segment.

    Parameters:
        car_position (array): The [x, y] position of the car.
        segment_start (array): The [x1, y1] coordinates representing the start of the segment.
        segment_end (array): The [x2, y2] coordinates representing the end of the segment.

    Returns:
        distance (float): The distance of the car to the centerline of the segment.
    """
    x1, y1 = segment_start
    x2, y2 = segment_end
    x0, y0 = car_position

    # Compute the distance from point to line (in 2D)
    distance = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return distance
def compute_distance_central_line(car, track_2D):
    min_distance = float('inf')
    for i in range(len(track_2D) - 1):
        segment_start = track_2D[i]
        segment_end = track_2D[i + 1]
        distance = distance_to_centerline(car.position, segment_start, segment_end)
        min_distance = min(min_distance, distance)
    return min_distance
