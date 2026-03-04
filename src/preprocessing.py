import numpy as np
import cv2


def simulate_low_resolution(image):

    h, w = image.shape

    # Downscale
    small = cv2.resize(image, (w//2, h//2), interpolation=cv2.INTER_LINEAR)

    # Upscale
    upscaled = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    return upscaled
