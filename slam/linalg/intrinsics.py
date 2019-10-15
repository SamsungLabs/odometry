import numpy as np


class Intrinsics:
    def __init__(self, f_x, f_y, c_x, c_y, width, height):
        self.f_x = f_x
        self.f_y = f_y
        self.c_x = c_x
        self.c_y = c_y

        self.f_x_scaled = f_x * width
        self.f_y_scaled = f_y * height
        self.c_x_scaled = c_x * width
        self.c_y_scaled = c_y * height

        self.width = width
        self.height = height

    def forward(self, xy):
        xy_processed = xy.copy()
        xy_processed[0] = (xy[0] - self.c_x_scaled) / self.f_x_scaled
        xy_processed[1] = (xy[1] - self.c_y_scaled) / self.f_y_scaled
        return xy_processed

    def backward(self, xy):
        xy_processed = xy.copy()
        xy_processed[0] = xy[0] * self.f_x_scaled + self.c_x_scaled
        xy_processed[1] = xy[1] * self.f_y_scaled + self.c_y_scaled
        return xy_processed

    def create_frustrum(self, x_pixels, y_pixels, depth):
        xy_pixels = np.c_[[x_pixels, y_pixels]]
        xy_frustrum = self.forward(xy_pixels) * depth
        return np.concatenate([xy_frustrum, np.expand_dims(depth, axis=0)])
