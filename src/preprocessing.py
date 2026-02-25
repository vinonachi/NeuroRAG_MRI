import numpy as np
from skimage.transform import resize


def simulate_low_resolution(image, scale=2):
    h, w = image.shape
    small = resize(image, (h // scale, w // scale), anti_aliasing=True)
    restored = resize(small, (h, w), anti_aliasing=True)
    return restored
