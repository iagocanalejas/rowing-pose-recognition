import time
from typing import Any, Generator, List, Tuple

import cv2
from cv2.typing import MatLike
from mediapipe.python.solutions.pose import Pose


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
    RIGHT_MARKERS = [12, 14, 16, 24, 26, 28]
    ALL_MARKERS = RIGHT_MARKERS + LEFT_MARKERS

    def __init__(self, display_framerate: bool = False):
        self.pose = Pose()
        self.display_framerate = display_framerate

    def detect(self, capture: cv2.VideoCapture, markers: List[int]) -> Generator[List[Tuple[int, int, int]], Any, Any]:
        while True:
            _, img = capture.read()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = self.pose.process(img_rgb)
            points = self._get_points(img, results.pose_landmarks)

            # draw required points and lines
            for marker in markers:
                cv2.circle(img, (points[marker][1], points[marker][2]), 5, (255, 0, 0), cv2.FILLED)
                self._draw_lines(img, points=points, markers=markers, current_marker=marker)

            self._show_fps_if_needed(img)

            cv2.imshow("Image", img)
            cv2.waitKey(1)

            yield points

    def _draw_lines(self, image: MatLike, points: List[Tuple[int, int, int]], markers: List[int], current_marker: int):
        lines = [line for line in self._LINES[current_marker] if line in markers]
        for line in lines:
            cv2.line(
                image,
                pt1=(points[current_marker][1], points[current_marker][2]),
                pt2=(points[line][1], points[line][2]),
                color=(255, 255, 255),
            )

    def _show_fps_if_needed(self, image: MatLike):
        if self.display_framerate:
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
