import albumentations as albu
import cv2
import kvt
import kvt.augmentation
import numpy as np


def sample_transform(split, **params):
    def transform(image, mask=None):
        def _transform(image):
            return image

        image = _transform(image)
        return image

    return transform
