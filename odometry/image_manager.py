import os
import cv2
import numpy as np


class ImageManager:
    TRAIN = 'train'
    def __init__(self,
                 directory,
                 image_filenames=None,
                 height=None,
                 width=None,
                 stride=1,
                 sample=False):
        self.directory = directory
        self.selected_image_filenames = image_filenames
        self.height = height
        self.width = width
        self.stride = stride
        self.sample = sample
        self.step = self.stride if self.sample else 1

        os.makedirs(self.directory, exist_ok=True)
        self.reset()

    def reset(self):
        self._image_filenames = sorted(os.listdir(self.directory))
        if self.selected_image_filenames is not None:
            self._image_filenames = [image_filename for image_filename in self._image_filenames \
                                     if image_filename in self.selected_image_filenames]

        self.num_images = len(self.image_filenames)
        if self.num_images == 0:
            return
        if self.height is not None and self.width is not None:
            return
        image = self.load_image(self.image_filenames[0])
        self.height, self.width = image.shape[:2]

    def resize_image(self, image, target_size):
        return cv2.resize(image, target_size, cv2.INTER_LINEAR)

    def save_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        if self.height is None and self.width is None:
            self.height = height
            self.width = width
        elif self.height is None:
            ratio = self.width / float(width)
            self.height = int(height * ratio)
        elif self.width is None:
            ratio = self.height / float(height)
            self.width = int(width * ratio)
        elif not np.array_equal((self.height, self.width), image.shape[:2]):
            image = self.resize_image(image, (self.width, self.height))

        if self.num_images > 1e9:
            raise ValueError('Too many images')
        image_filename = '{}.jpg'.format(str(self.num_images).zfill(10))
        cv2.imwrite(os.path.join(self.directory, image_filename), image)
        self.reset()

    def load_image(self, image_filename, target_size=None):
        image = cv2.imread(os.path.join(self.directory, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if target_size is not None:
            image = self.resize_image(image, target_size)
        return image

    @property
    def image_filenames(self):
        return self._image_filenames[::self.step]

    @property
    def image_filepaths(self):
        return [os.path.join(self.directory, image_filename) for image_filename in self.image_filenames]

    @property
    def next_image_filenames(self):
        return self._image_filenames[self.stride::self.step] + [None] * self.stride

    @staticmethod
    def convert_hwc_to_chw(image):
        return image.transpose((2, 0, 1))

    @staticmethod
    def convert_chw_to_hwc(image):
        return image.transpose((1, 2, 0))

    def __repr__(self):
        return 'ImageManager(dir={}, image_height={}, image_width={}, stride={}, sample={}, step={})'.format(
            self.directory, self.height, self.width, self.stride, self.sample, self.step)
