import random
import time

import numpy as np
import pygame

import replayplayer

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


def create_circular_segment(radius, road_width, start_angle, end_angle, forward=True):
    num_segments = int(radius * abs(end_angle - start_angle) / road_width)

    if forward:
        angles = np.linspace(start_angle, end_angle, num_segments)
    else:
        angles = np.linspace(end_angle, start_angle, num_segments)

    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    return list(zip(x, y))


def line_intersection(line1, line2):
    """ Check if two lines (segments) intersect """
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(xdiff, ydiff)
    if div == 0:
        return False

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    # Check if the intersection point is within both the segments
    if ((min(line1[0][0], line1[1][0]) <= x <= max(line1[0][0], line1[1][0])) and
            (min(line1[0][1], line1[1][1]) <= y <= max(line1[0][1], line1[1][1])) and
            (min(line2[0][0], line2[1][0]) <= x <= max(line2[0][0], line2[1][0])) and
            (min(line2[0][1], line2[1][1]) <= y <= max(line2[0][1], line2[1][1]))):
        return True
    return False



def is_self_intersecting(new_track, old_track):
    """ Check if the new segment intersects with the old track """
    for i in range(len(new_track) - 1):
        for j in range(len(old_track) - 1):
            if line_intersection(new_track[i:i+2], old_track[j:j+2]):
                return True
    return False


def segment_direction(segment):
    """ Computes the direction of the segment """
    if len(segment) < 2:
        raise ValueError("The segment must have at least two points to determine direction.")

    dx = segment[-1][0] - segment[-2][0]
    dy = segment[-1][1] - segment[-2][1]
    angle = np.arctan2(dy, dx)
    return angle


def is_segment_intersecting(track, segment, min_distance):
    for point in segment:
        for track_point in track:
            if np.linalg.norm(np.array(point) - np.array(track_point)) < min_distance:
                return True
    return False


def create_straight_track_v2(start_point, end_point, road_width):
    """
    Generates a straight track segment between two points.
    Args:
    - start_point (tuple): The starting point of the segment.
    - end_point (tuple): The ending point of the segment.
    - road_width (float): The width of the road.
    Returns:
    - List of points representing the track segment.
    """
    # Calculate the distance between the two points
    distance = np.sqrt((start_point[0] - end_point[0])**2 + (start_point[1] - end_point[1])**2)

    # Calculate the number of segments based on the distance and road_width
    num_segments = int(distance / road_width) + 1

    x_points = np.linspace(start_point[0], end_point[0], num_segments)
    y_points = np.linspace(start_point[1], end_point[1], num_segments)

    return list(zip(x_points, y_points))

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

def make_strange_trace(forward=True,road_width = 10,radius=None):
    # Create the first circle

    first_circle_radius =random.randint(59,200) if not radius else radius
    first_circle_segments = 100
    if forward:
        first_circle = create_circular_track(first_circle_radius,  road_width)
    else:
        first_circle = create_circular_track(first_circle_radius,  road_width,0,-3/2*np.pi)
    # Remove last 25% to make space for the second circle

    final_track = first_circle
    return final_track
def create_complex_track_v2(num_parts=20, road_width=30, max_angle=np.pi/6, max_attempts=10):
    if random.random()<0.9:
        return make_strange_trace(random.random()<0.5,road_width//2)
    track = [(0, 0)]  # Starting point
    segment_length=road_width

    last_angle = random.uniform(0, 2*np.pi)  # Starting direction

    total_segment=0
    for _ in range(num_parts):
        added_segment = False  # Flag to check if we've added a segment in this iteration
        attempt = 0
        while attempt < max_attempts and not added_segment:
            last_point = track[-1]

            # Adjust the direction and length
            angle_offset = random.uniform(-max_angle, max_angle)
            length_offset = random.uniform(-road_width / 2, road_width / 2)

            last_angle += angle_offset
            end_point = (last_point[0] + (segment_length + length_offset) * np.cos(last_angle),
                         last_point[1] + (segment_length + length_offset) * np.sin(last_angle))

            segment = create_straight_track_v2(last_point, end_point,road_width/2*0.9)

            if not is_self_intersecting(segment, track):
                track.extend(segment[1:])  # Exclude the first point to avoid repetition
                added_segment = True
                attempt =0
                total_segment+=1
            else:
                attempt += 1

        if not added_segment:  # If we couldn't add a segment after max_attempts
            print(f"Failed to add a segment after {max_attempts} attempts.")
            break
    if total_segment<num_parts/2:
        return make_strange_trace(random.random()<0.5,road_width//2)

    return track


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    while True:

        track=create_complex_track_v2()

                # Unzip the x and y coordinates
        x, y = zip(*track)

        # Plot the track
        plt.figure(figsize=(100, 100))
        plt.plot(x, y, '-o')
        plt.title("Complex Track")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.savefig('img')
        time.sleep(10)