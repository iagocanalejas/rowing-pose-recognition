import math
from typing import List, Tuple


class Join:
    LEFT_ELBOW = (11, 13, 15)
    RIGHT_ELBOW = (12, 14, 16)
    LEFT_HIP = (25, 23, 11)
    RIGHT_HIP = (26, 24, 12)
    LEFT_KNEE = (23, 25, 27)
    RIGHT_KNEE = (24, 26, 28)

    @staticmethod
    def compute_angle(points: List[Tuple[int, int, int]], markers: Tuple[int, int, int], normalize=False) -> int:
        x1, y1 = points[markers[0]][1], points[markers[0]][2]
        x2, y2 = points[markers[1]][1], points[markers[1]][2]
        x3, y3 = points[markers[2]][1], points[markers[2]][2]

        angle = int(math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)))

        if angle < 0:
            angle += 360

        if normalize:
            angle = angle % 180

        return angle
