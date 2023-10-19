from typing import List, Tuple

import cv2
from cv2.typing import MatLike

from rowing.joins import Join


# TODO: each stroke:
#   - check arms at the catch
#   - check legs at the end
class ErgometerTrainer:
    stroke_count = 0
    previous_hip_angle = 0

    def detect_stroke(self, image: MatLike, points: List[Tuple[int, int, int]]):
        hip_angle = Join.compute_angle(points, Join.LEFT_HIP)
        elbow_angle = Join.compute_angle(points, Join.LEFT_ELBOW)

        if self.previous_hip_angle < 90 and hip_angle > 90:
            self.stroke_count += 1

        if hip_angle < 35:
            self._show_mark(image, text="CATCH", y=80)

        if hip_angle > 140 and elbow_angle > 100:
            self._show_mark(image, text="FINISH", y=80)

        self._show_stroke_counter(image)
        self.previous_hip_angle = hip_angle

    def _show_stroke_counter(self, image: MatLike):
        cv2.putText(
            image,
            text=str(self.stroke_count),
            org=(50, 50),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=2,
            color=(255, 255, 255),
            thickness=3,
        )

    def _show_mark(self, image: MatLike, text: str, y: int):
        cv2.putText(
            image,
            text=text,
            org=(50, y),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=2,
            color=(255, 255, 255),
            thickness=3,
        )
