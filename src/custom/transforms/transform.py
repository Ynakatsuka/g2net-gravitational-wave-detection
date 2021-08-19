import albumentations as albu
import cv2
import numpy as np

import kvt
import kvt.augmentation


def sample_transform(split, **params):
    def transform(image, mask=None):
        def _transform(image):
            return image

        image = _transform(image)
        return image

    return transform
