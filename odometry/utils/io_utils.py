import cv2
import numpy as np


def resize_image(image, target_size):
    return cv2.resize(image, target_size, cv2.INTER_LINEAR)


def save_image(image, image_filepath):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(image_filepath, image)


def load_image(image_filepath, target_size=None):
    image = cv2.imread(image_filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if target_size is not None:
        image = resize_image(image, target_size)
    return image


def convert_hwc_to_chw(image):
    return image.transpose((2, 0, 1))


def convert_chw_to_hwc(image):
    return image.transpose((1, 2, 0))
