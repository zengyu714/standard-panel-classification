import cv2
from numpy import random
from bluntools.image_ops import random_gamma, random_sigmoid, normalize, random_hflip


def random_mirror(image, boxes):
    if random.randint(2):
        height, width = image.shape[:2]
        image = image[:, ::-1]

        whwh = [width, height] * 2
        boxes *= whwh  # to absolute coords
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
        boxes /= whwh  # to relative coords (percentage)

    return image, boxes


def resize(image, size):
    lose_dim = image.shape[-1] == 1
    image = cv2.resize(image, (size, size))
    if lose_dim:
        image = image[..., None]
    return image


class BaseTransform(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = resize(image, self.size)
        image = normalize(image)
        return image, boxes, labels


class Augmentation(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes, labels):
        image = resize(image, self.size)
        image = normalize(image)
        image = random_gamma(image, low=0.8, high=1.2)
        image, boxes = random_mirror(image, boxes)
        return image, boxes, labels
