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


def create_loop_track(radius=50, num_segments=100):
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
    def __init__(self, position=np.array([0.0, 0.0]), velocity=np.array([0.0, 0.0]), mass=1.0, max_speed=10.0, drag_coefficient=0.1):
        self.position = position  # Position vector [x, y]
        self.velocity = velocity  # Velocity vector [vx, vy]
        self.mass = mass  # Mass of the car
        self.max_speed = max_speed  # Maximum speed
        self.drag_coefficient = drag_coefficient  # Aerodynamic drag coefficient

    def update_position(self, delta_time, acceleration=np.array([0.0, 0.0]), steering_angle=0.0, road_width=10.0):
        # 1. Tire Friction and Acceleration
        # Limit the acceleration based on tire friction (simplified model)
        max_accel = 2.0  # Arbitrary max acceleration due to tire friction
        acceleration_magnitude = np.linalg.norm(acceleration)
        if acceleration_magnitude > max_accel:
            acceleration = (acceleration / acceleration_magnitude) * max_accel

        # 2. Mass and Inertia
        # F = m*a (Force is directly applied as acceleration; mass is considered in collision dynamics)

        # 3. Aerodynamic Drag
        speed = np.linalg.norm(self.velocity)
        drag_force = -self.drag_coefficient * self.velocity * speed

        # 4. Steering and Turning Radius
        # Change in heading direction due to steering angle (simplified to directly affect velocity)
        rotation_matrix = np.array([[math.cos(steering_angle), -math.sin(steering_angle)],
                                    [math.sin(steering_angle), math.cos(steering_angle)]])
        self.velocity = np.dot(rotation_matrix, self.velocity)

        # 5. Update Velocity and Position
        # v = u + a*t and s = u*t + 0.5*a*t^2
        self.velocity += (acceleration + drag_force / self.mass) * delta_time
        self.position += self.velocity * delta_time + 0.5 * acceleration * delta_time ** 2

        # 6. Collision Physics
        # Simple collision detection and elastic collision response
        if abs(self.position[1]) > road_width / 2:
            self.position[1] = np.sign(self.position[1]) * road_width / 2  # Reset position to boundary
            self.velocity[1] = -0.5 * self.velocity[1]  # Simple elastic collision (50% energy loss)

        # 7. Speed Caps
        # Limit speed to max_speed
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

def place_car_on_track(car, track_2D, segment_index):
    """
    Place the car on a specific segment of the track.

    Parameters:
        car (RealisticCar2D): The car object.
        track_2D (list): The list of [x, y] coordinates representing the loop track.
        segment_index (int): The index of the track segment where the car should be placed.

    Returns:
        None: The car's position is updated in place.
    """
    segment_start = track_2D[segment_index]
    segment_end = track_2D[(segment_index + 1) % len(track_2D)]  # Loop back to start if at end

    # Place the car at the midpoint of the segment
    midpoint = [(segment_start[0] + segment_end[0]) / 2, (segment_start[1] + segment_end[1]) / 2]
    car.position = np.array(midpoint)