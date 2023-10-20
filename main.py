# Importing required libraries

# Improved Track Representation
# Using a list of dictionaries to represent the track, encoding curvature and width


import numpy as np


# Decode track into 2D coordinates for visualization (or for more advanced physics)


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
    def __init__(self, position=np.array([0.0, 0.0]), velocity=np.array([0.0, 0.0]), mass=1.0, max_speed=10, max_acceleration=1, drag_coefficient=0.2, max_steering_rate=0.05,n=1,curve_step=2):
        self.position = position
        self.velocity = velocity
        self.orientation = np.array([1.0, 0.0])
        self.turning_rate = 0.0
        self.mass = mass
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.drag_coefficient = drag_coefficient
        self.current_steering_angle = 0.0
        self.max_steering_rate = max_steering_rate
        self.acceleration = 0
        self.closest_point_idx = 0
        self.speed = np.linalg.norm(self.velocity)
        self.previous_angle = np.arctan2(self.orientation[1], self.orientation[0])
        self.distance_to_central=0
        self.angle=0
        self.lastest_point_idx=0
        self.drift_factor=0.1
        self.current_angle=0
        self.prev_position=self.position
        self.prev_orientation=self.orientation
        self.drifting=False
        self.lateral_velocity=0
        self.n=n
        self.alignment=0
        self.curve_step=curve_step
        self.curve_directions = [0 for _ in range(n)]
        self.curve_distances =  [0 for _ in range(n)]
        #print('start',len(self.curve_directions))

    def update_position(self, delta_time, track_2D,road_width, acceleration=0.0, steering_angle=0.0):
        # Update acceleration
        self.prev_position=self.position.copy()
        self.prev_orientation=self.prev_orientation.copy()
        acc_diff = acceleration - self.acceleration
        acc_diff = np.clip(acc_diff, -self.max_acceleration * delta_time, self.max_acceleration * delta_time)
        self.acceleration += acc_diff


        acceleration_vector = self.acceleration * self.orientation

        # Update steering angle
        steering_diff = steering_angle - self.current_steering_angle
        steering_diff = np.clip(steering_diff, -self.max_steering_rate * delta_time, self.max_steering_rate * delta_time)
        self.current_steering_angle += steering_diff
        if self.speed > 0.9 and (steering_angle > 0.5 or steering_angle < -0.5):
            self.drifting=True
            # Modify velocity to add drift effect
            drift_vector = self.velocity * self.drift_factor
            self.velocity = (1 - self.drift_factor) * (self.velocity + (acceleration_vector / self.mass) * delta_time) + drift_vector
            self.current_steering_angle += steering_diff*self.drift_factor

            # Rotation matrix
            rotation_matrix = np.array([[math.cos(self.current_steering_angle), -math.sin(self.current_steering_angle)],
                                        [math.sin(self.current_steering_angle), math.cos(self.current_steering_angle)]])
            self.orientation = np.dot(rotation_matrix, self.orientation)
        else:

            if self.speed > 0.1 and not (self.speed > 0.9):
                self.drifting=False
                self.current_steering_angle += steering_diff
                # Rotation matrix
                rotation_matrix = np.array([[math.cos(self.current_steering_angle), -math.sin(self.current_steering_angle)],
                                            [math.sin(self.current_steering_angle), math.cos(self.current_steering_angle)]])
                self.orientation = np.dot(rotation_matrix, self.orientation)
                self.velocity = np.dot(rotation_matrix, self.velocity)
            self.velocity+= (acceleration_vector / self.mass - self.speed*self.speed**self.max_speed*self.drag_coefficient) * delta_time

        # Update position
        self.position += self.velocity * delta_time

        # Update speed and cache state
        self.speed = np.linalg.norm(self.velocity)/self.max_speed
        if self.speed>1:
            self.velocity=self.velocity/self.speed
            self.speed = np.linalg.norm(self.velocity)/self.max_speed

        self.cache_state(track_2D,delta_time,road_width)

    def cache_state(self, track_2D,delta_time,road_width):
        self.previous_angle = self.current_angle
        self.current_angle = np.arctan2(self.orientation[1], self.orientation[0])
        #self.angle=current_angle
        self.turning_rate = (self.current_angle - self.previous_angle) / delta_time
        self.lateral_velocity = np.dot(self.velocity, np.array([-self.orientation[1], self.orientation[0]]))

        self.lastest_point_idx=self.closest_point_idx
        self.closest_point_idx, _ = compute_closest_point_idx(self, track_2D)
        self.distance_to_central=self.compute_distance_to_central_line(track_2D, road_width)
        if self.closest_point_idx + 1 < len(track_2D):
            # Direction of the track segment
            track_dir = np.array(track_2D[self.closest_point_idx + 1]) - np.array(track_2D[self.closest_point_idx])
            track_dir /= np.linalg.norm(track_dir)  # Normalize the track direction

            # Car's direction
            # Compute the dot product between the car direction and track direction
            self.alignment = np.dot(track_dir, self.orientation)
            self.angle=self.compute_angle_to_central_line(track_2D)
            self.curve_directions = []
            self.curve_distances =  []
            for i in range(1, (self.n +1)):
                i=i*self.curve_step
                idx = min(self.closest_point_idx + i, len(track_2D) - 1)
                next_dir = np.array(track_2D[idx]) - self.position

                curve_dir =  np.cross(next_dir, self.orientation)  # 1 for left turn, -1 for right turn
                curve_distance = np.linalg.norm(next_dir)

                self.curve_directions.append(curve_dir)
                self.curve_distances.append(curve_distance/(road_width*i))
                #print(len(self.curve_directions))

    def compute_distance_to_central_line(car, track_2D, road_width):
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

        return distance*2/road_width
    def compute_angle_to_central_line(car, track_2D):
        # Вычисляем вектор сегмента
        if car.closest_point_idx + 1 >= len(track_2D):
            segment_direction = np.array(track_2D[0]) - np.array(track_2D[car.closest_point_idx])
        else:
            segment_direction = np.array(track_2D[car.closest_point_idx+1]) - np.array(track_2D[car.closest_point_idx])

        # Вычисляем косинус угла между векторами
        cos_angle = np.dot(car.orientation, segment_direction) / (np.linalg.norm(car.orientation) * np.linalg.norm(segment_direction))
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))

        # Определяем направление поворота (положительное или отрицательное)
        cross_product = np.cross(car.orientation, segment_direction)
        if cross_product < 0:
            angle_rad = -angle_rad

        #angle_deg = np.degrees(angle_rad)
        return angle_rad









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
    car.position = np.array(track_2D[start_index % len(track_2D)], dtype=np.float32)

    # Compute the direction towards the next point in the track
    next_point = np.array(track_2D[(start_index + 1) % len(track_2D)])
    direction_vector = next_point - car.position

    # Normalize the direction vector
    direction_vector /= np.linalg.norm(direction_vector)

    # Set the car's velocity to make it head towards the centerline
    car.orientation=direction_vector
    car.velocity = direction_vector*0.1
    car.acceleration=1
    car.closest_point_idx=start_index
    #car.closest_point_idx=start_index


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