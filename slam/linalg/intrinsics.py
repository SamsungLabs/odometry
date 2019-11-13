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

        self.pixels = np.c_[np.meshgrid(np.arange(0., self.width), np.arange(0., self.height))]

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

    def to_pixels(self, points):
        xy_points = points[:2]
        z_points = points[2]
        return self.backward(xy_points / z_points)

    def to_points(self, depth):
        xy_points = self.forward(self.pixels) * depth
        z_points = depth[None]
        return np.concatenate([xy_points, z_points])

    def __repr__(self):
        s = [f'f_x={self.f_x}, f_y={self.f_y}',
             f'c_x={self.c_x}, c_y={self.c_y}',
             f'f_x_scaled={self.f_x_scaled}, f_y_scaled={self.f_y_scaled}',
             f'c_x_scaled={self.c_x_scaled}, c_y_scaled={self.c_y_scaled}',
             f'width={self.width}, height={self.height}']
        return '\n'.join(s)
