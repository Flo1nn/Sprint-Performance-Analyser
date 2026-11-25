import math

def calculate_distance(point_a,point_b):
    """Calculate the distance between two points"""


    x1, y1 = point_a
    x2, y2 = point_b
    
    distance = math.sqrt(
        (x2- x1) ** 2 + (y2-y1) ** 2
    )

    return distance


def get_athlete_center(points):
    """Return the midpoint between an athlete's hips."""

    left_hip = points[23]
    right_hip = points[24]


    center_x = (left_hip[0] + right_hip[0]) / 2
    center_y = (left_hip[1] + right_hip[1]) / 2

    return (center_x,center_y)


def find_closest_athlete(athletes, click_position, max_distance=100):
    """Find the closest athlete within the allowed distance."""

    if not athletes:
        return None

    distances = []

    for athlete in athletes:
        distance = calculate_distance(athlete, click_position)
        distances.append(distance)

    closest_index = distances.index(min(distances))

    if distances[closest_index] > max_distance:
        return None

    return closest_index

def point_inside_box(click_position, box):
    """Checking whether a click is inside a bounding box"""

    x, y = click_position
    x1, y1, x2, y2 = box

    return x1 <= x <= x2 and y1 <= y <= y2