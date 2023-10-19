import time
from typing import Any, Generator, List, Optional, Tuple

import cv2
from cv2.typing import MatLike
from mediapipe.python.solutions.pose import Pose

from rowing.joins import Join


class PoseDetector:
    previous_time = 0

    _LINES = {
        11: [12, 13, 23],
        12: [14, 24],
        13: [15],
        14: [16],
        15: [],
        16: [],
        23: [24, 25],
        24: [26],
        25: [27],
        26: [28],
        27: [],
        28: [],
    }

    LEFT_MARKERS = [11, 13, 15, 23, 25, 27]
    TOP_MARKERS = [11, 12, 13, 14, 15, 16, 23, 24]
    RIGHT_MARKERS = [12, 14, 16, 24, 26, 28]
    BOTTOM_MARKERS = [23, 24, 25, 26, 27, 28]

    ALL_MARKERS = list(set(RIGHT_MARKERS + LEFT_MARKERS + TOP_MARKERS + BOTTOM_MARKERS))

    def __init__(self, display_points=False, display_angles=False, display_framerate=False):
        self.pose = Pose()
        self.display_points = display_points
        self.display_angles = display_angles
        self.display_framerate = display_framerate

    def detect(
        self,
        capture: cv2.VideoCapture,
        markers: List[int],
    ) -> Generator[Tuple[MatLike, Optional[List[Tuple[int, int, int]]]], Any, Any]:
        while True:
            _, img = capture.read()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = self.pose.process(img_rgb)
            if not results.pose_landmarks:
                yield img, None
                continue

            points = self._get_points(img, results.pose_landmarks)

            self._draw_points_if_needed(img, points=points, markers=markers)

            self._draw_angle_if_needed(img, points=points, markers=Join.LEFT_ELBOW, normalize=True)
            self._draw_angle_if_needed(img, points=points, markers=Join.LEFT_KNEE, normalize=True, show_below=True)
            self._draw_angle_if_needed(img, points=points, markers=Join.LEFT_HIP)

            self._show_fps_if_needed(img)

            yield img, points

    def _draw_points_if_needed(self, image: MatLike, points: List[Tuple[int, int, int]], markers: List[int]):
        if not self.display_points:
            return

        for marker in markers:
            cv2.circle(image, (points[marker][1], points[marker][2]), 5, (255, 0, 0), cv2.FILLED)
            self._draw_lines(image, points=points, markers=markers, current_marker=marker)

    def _draw_lines(self, image: MatLike, points: List[Tuple[int, int, int]], markers: List[int], current_marker: int):
        lines = [line for line in self._LINES[current_marker] if line in markers]
        for line in lines:
            cv2.line(
                image,
                pt1=(points[current_marker][1], points[current_marker][2]),
                pt2=(points[line][1], points[line][2]),
                color=(255, 255, 255),
            )

    def _draw_angle_if_needed(
        self,
        image: MatLike,
        points: List[Tuple[int, int, int]],
        markers: Tuple[int, int, int],
        normalize=False,
        show_below=False,
    ):
        if not self.display_angles:
            return

        x2, y2 = points[markers[1]][1], points[markers[1]][2]
        angle = Join.compute_angle(points, markers, normalize=normalize)

        cv2.putText(
            image,
            text=str(angle),
            org=(x2 - 20, y2 + 30) if show_below else (x2 - 20, y2 - 20),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=2,
            color=(255, 255, 255),
            thickness=2,
        )

    def _show_fps_if_needed(self, image: MatLike):
        if not self.display_framerate:
            return

        current_time = time.time()
        cv2.putText(
            image,
            text=str(int(1 / (current_time - self.previous_time))),
            org=(70, 50),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=3,
            color=(255, 0, 0),
            thickness=3,
        )
        self.previous_time = current_time

    @staticmethod
    def _get_points(img: MatLike, landmarks) -> List[Tuple[int, int, int]]:
        lm_list = []

        for id, lm in enumerate(landmarks.landmark):
            h, w, _ = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_list.append([id, cx, cy])

        return lm_list
