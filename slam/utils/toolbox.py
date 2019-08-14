import numpy as np
import torch


class Toolbox:
    def __init__(self, backend='numpy', cuda=False):
        if not backend in ('numpy', 'torch'):
            raise ValueError(f'Unknown backend: "{backend}"')

        self.backend = backend
        self.cuda = cuda

    def bmm(self, first_matrix, second_matrix):
        if self.backend == 'torch':
            return torch.bmm(first_matrix, second_matrix)
        elif self.backend == 'numpy':
            return first_matrix @ second_matrix

    def clip(self, x, amin=-np.inf, amax=np.inf):
        return torch.clamp(x, amin, amax) if self.backend == 'torch' else np.clip(x, amin, amax)

    def acos(self, x):
        return torch.acos(x) if self.backend == 'torch' else np.arccos(x)

    def btrace(self, x):
        if self.backend == 'torch':
            return x.diagonal(dim1=1, dim2=2).sum(1)
        elif self.backend == 'numpy':
            return np.trace(x, axis1=1, axis2=2)

    def btranspose(self, x):
        return x.transpose(2, 1) if self.backend == 'torch' else x.transpose((0, 2, 1))

    def from_numpy(self, x):
        return self.to_gpu(torch.from_numpy(x)) if self.backend == 'torch' else x

    def item(self, x):
        return self.to_cpu(x).item() if self.backend == 'torch' else x

    def to_gpu(self, x):
        return x.cuda() if self.backend == 'torch' and self.cuda else x

    def to_cpu(self, x):
        return x.cpu() if self.backend == 'torch' and self.cuda else x
