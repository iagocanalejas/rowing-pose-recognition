#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from typing import List

import cv2
from rowing.pose import PoseDetector
from rowing.trainers.erg import ErgometerTrainer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def main(file_path: str, markers: List[int]):
    detector = PoseDetector(display_points=True, display_angles=True)
    trainer = ErgometerTrainer()

    for img, points in detector.detect(cv2.VideoCapture(file_path), markers=markers):
        if not points:
            cv2.imshow("Image", img)
            cv2.waitKey(1)
            continue

        trainer.detect_stroke(img, points=points)

        cv2.imshow("Image", img)
        cv2.waitKey(5)


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="File to process.")
    parser.add_argument("--mode", type=str, default="all", help="'left' | 'right' | 'top' | 'bottom' | 'all'")
    return parser.parse_args()


def _chose_markers(mode: str) -> List[int]:
    match mode:
        case "left":
            return PoseDetector.LEFT_MARKERS
        case "top":
            return PoseDetector.TOP_MARKERS
        case "right":
            return PoseDetector.RIGHT_MARKERS
        case "bottom":
            return PoseDetector.BOTTOM_MARKERS
        case "all":
            return PoseDetector.ALL_MARKERS
        case _:
            raise NotImplementedError(mode)


if __name__ == "__main__":
    args = _parse_arguments()
    logger.info(f"{os.path.basename(__file__)}:: args -> {args.__dict__}")

    main(args.file, _chose_markers(args.mode))
