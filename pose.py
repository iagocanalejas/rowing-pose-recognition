#!/usr/bin/env python3

import cv2
from rowing.pose import PoseDetector

if __name__ == "__main__":
    PoseDetector(display_framerate=True).detect(cv2.VideoCapture("videos/ergo.mp4"), PoseDetector.ALL_MARKERS)
