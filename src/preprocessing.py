import cv2
import numpy as np


def simulate_low_resolution(image):

    h, w = image.shape

    small = cv2.resize(image, (w//2, h//2))
    upscaled = cv2.resize(small, (w, h))

    return upscaled
