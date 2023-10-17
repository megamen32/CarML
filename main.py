# Importing required libraries
import random

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
import numpy as np

def create_double_loop_track(radius1=100, radius2=50, num_segments=100):
    """
    Create a double loop track with two different radii and number of segments.

    Parameters:
        radius1 (int): The radius of the first loop.
        radius2 (int): The radius of the second loop.
        num_segments (int): The number of segments to divide each loop into.

    Returns:
        track_2D (list): A list of [x, y] coordinates representing the double loop track.
    """
    track_2D = []
    # Create the first loop
    for i in range(num_segments):
        angle = i * 2 * np.pi / num_segments
        x = radius1 * np.cos(angle)
        y = radius1 * np.sin(angle)
        track_2D.append([x, y])

    # Create the second loop with an offset to avoid overlap
    offset_x, offset_y = radius1 * 2 + radius2, 0
    for i in range(num_segments):
        angle = i * 2 * np.pi / num_segments
        x = radius2 * np.cos(angle) + offset_x
        y = radius2 * np.sin(angle) + offset_y
        track_2D.append([x, y])

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

def create_circular_track(radius, road_width, start_angle=0, end_angle=3/2 * np.pi):
    # Calculate the circumference of the circle
    circumference = 2 * np.pi * radius

    # Determine the number of segments based on the circumference and road width
    num_segments = int(circumference / road_width)

    # Generate the circular track
    angles = np.linspace(start_angle, end_angle, num_segments)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    return list(zip(x, y))

def make_strange_trace(forward=True,road_width = 10):
    # Create the first circle
    first_circle_radius =random.randint(59,200)
    first_circle_segments = 100
    if forward:
        first_circle = create_circular_track(first_circle_radius,  road_width)
    else:
        first_circle = create_circular_track(first_circle_radius,  road_width,0,-3/2*np.pi)
# Remove last 25% to make space for the second circle

    final_track = first_circle
    return final_track

def compute_track_length(track_2D):
    return sum(np.linalg.norm(np.array(track_2D[i]) - np.array(track_2D[i - 1])) for i in range(1, len(track_2D)))

def compute_travel_distance(track_2D, closest_point, closest_point_idx):
    distance_to_segment_start = sum(np.linalg.norm(np.array(track_2D[i]) - np.array(track_2D[i - 1])) for i in range(1, closest_point_idx))
    distance_within_segment = np.linalg.norm(np.array(closest_point) - np.array(track_2D[closest_point_idx]))
    return distance_to_segment_start + distance_within_segment


#track_2D=decode_track_to_2D(track)

# Importing required libraries
import math


# More Realistic Car2D class with advanced physics features
class RealisticCar2D:
    def __init__(self, position=np.array([0.0, 0.0]), velocity=np.array([0.0, 0.0]), mass=1.0, max_speed=10, max_acceleration=1,drag_coefficient=0,max_steering_rate=0.01):
        self.position = position  # Position vector [x, y]
        self.velocity = velocity  # Velocity vector [vx, vy]
        self.mass = mass  # Mass of the car
        self.max_speed = max_speed  # Maximum speed
        self.max_acceleration = max_acceleration # Maximum acceleration
        self.drag_coefficient = drag_coefficient  # Aerodynamic drag coefficient
        self.current_steering_angle = 0.0  # New attribute
        self.max_steering_rate = max_steering_rate # Maximum rate of change of steering angle per second
        self.max_acceleration_rate = 0.1  # Maximum rate of change of steering angle per second
        self.acceleration = 0 # Acceleration
        self.orientation = np.array([1.0, 0.0])
        self.closest_point_idx = 0#read-only
        self.distance_to_central=0#read-only
        self.speed=0#read-only
        self.lastest_point_idx=0#read-only
        self.distance_to_curr=0#read-only
        self.angle=0#read-only
        self.alignment=0#read-only


    def update_position(self, delta_time,track_2D, acceleration=0.0, steering_angle=0.0, road_width=10.0):
        old_position = self.position.copy()
        # 1. Tire Friction and Acceleration

        # 2. Mass and Inertia
        acc_diff = acceleration - self.acceleration
        acc_diff = np.clip(acc_diff, -self.max_acceleration_rate * delta_time, self.max_acceleration_rate * delta_time)
        self.acceleration += acc_diff
        self.acceleration = min(max(-self.max_acceleration, self.acceleration), self.max_acceleration)

        # 3. Aerodynamic Drag
        direction = self.orientation
        acceleration_vector = self.acceleration * direction
        drag_force = -self.drag_coefficient * self.velocity * self.speed

        # 4. Steering and Turning Radius
        steering_diff = steering_angle - self.current_steering_angle
        steering_diff = np.clip(steering_diff, -self.max_steering_rate * delta_time, self.max_steering_rate * delta_time)
        self.current_steering_angle += steering_diff

        rotation_matrix = np.array([[math.cos(self.current_steering_angle), -math.sin(self.current_steering_angle)],
                                    [math.sin(self.current_steering_angle), math.cos(self.current_steering_angle)]])

        self.orientation = np.dot(rotation_matrix, self.orientation)
        self.velocity = np.dot(rotation_matrix, self.velocity)

        # 5. Update Velocity and Position
        self.velocity += (acceleration_vector + drag_force / self.mass) * delta_time
        self.position += self.velocity * delta_time + 0.5 * acceleration_vector * delta_time ** 2

        # 7. Speed Caps
        self.speed = np.linalg.norm(self.velocity)
        if self.speed > self.max_speed:
            self.velocity = (self.velocity / self.speed) * self.max_speed
        self.lastest_point_idx=self.closest_point_idx
        self.closest_point_idx,self.distance_to_curr = compute_closest_point_idx(self, track_2D)
        self.distance_to_central= self.compute_future_point_idx( track_2D)
        if self.closest_point_idx + 1 < len(track_2D):
            # Direction of the track segment
            track_dir = np.array(track_2D[self.closest_point_idx + 1]) - np.array(track_2D[self.closest_point_idx])
            track_dir /= np.linalg.norm(track_dir)  # Normalize the track direction

            # Car's direction
            car_dir = self.orientation

            # Compute the dot product between the car direction and track direction
            self.alignment = np.dot(track_dir, car_dir)

            # Compute the angle using the arccosine function
            self.angle = np.arccos(np.clip(self.alignment, -1.0, 1.0))

    def compute_future_point_idx(car, track_2D):
        if car.closest_point_idx + 1 >= len(track_2D):
            return 0.0  # If it's the last point, return distance 0

        p1 = track_2D[car.closest_point_idx]
        p2 = track_2D[car.closest_point_idx + 1]
        p = car.position

        if np.array_equal(p1, p2):
            return np.linalg.norm(p - p1)

        # Calculate the line segment distance
        num = abs((p2[1] - p1[1]) * p[0] + (p1[0] - p2[0]) * p[1] + (p2[0] * p1[1] - p1[0] * p2[1]))
        den = np.linalg.norm(np.array(p2) - np.array(p1))
        distance = num / den

        return distance



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
    car.orientation=direction_vector
    car.velocity = direction_vector
    car.acceleration=0.51
    car.closest_point_idx=start_index


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

def compute_closest_point_idx(car, track_2D):
    min_distance = float('inf')
    closest_idx = 0
    for i in range(len(track_2D)-1):
        distance,_ = distance_to_centerline(car.position, track_2D[i], track_2D[(i + 1) ])
        if distance < min_distance:
            min_distance = distance
            closest_idx = i
    return closest_idx,min_distance