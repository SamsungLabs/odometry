from .basic_ops import Abs
from .basic_ops import Min
from .basic_ops import Max
from .basic_ops import Mean
from .basic_ops import Std
from .basic_ops import Percentile

from .functions import Affine
from .functions import Clip
from .functions import Inverse
from .functions import Divide
from .functions import Repeat
from .functions import affine
from .functions import clip
from .functions import inverse
from .functions import divide
from .functions import repeat
from .functions import expand_as
from .functions import chunk
from .functions import concat

from .wrappers import activ
from .wrappers import conv2d
from .wrappers import conv2d_transpose
from .wrappers import gated_conv2d
from .wrappers import gated_conv2d_transpose
from .wrappers import dense

from .special_layers import AddGrid
from .special_layers import DepthFlow
from .special_layers import add_grid
from .special_layers import depth_flow
from .special_layers import construct_outputs

from .transforms import transform_inputs


__all__ = [
    'Abs',
    'Min',
    'Max',
    'Mean',
    'Std',
    'Percentile',
    'Affine',
    'Clip',
    'Inverse',
    'Divide',
    'Repeat',
    'affine',
    'clip',
    'inverse',
    'divide',
    'repeat',
    'expand_as',
    'chunk',
    'concat',
    'activ',
    'conv2d',
    'conv2d_transpose',
    'gated_conv2d',
    'gated_conv2d_transpose',
    'dense',
    'AddGrid',
    'DepthFlow',
    'add_grid',
    'depth_flow',
    'construct_outputs',
    'transform_inputs'
]

CUSTOM_LAYERS = {
     'Abs': Abs,
     'Min': Min,
     'Max': Max,
     'Mean': Mean,
     'Std': Std,
     'Percentile': Percentile,
     'Affine': Affine,
     'Clip': Clip,
     'Inverse': Inverse,
     'Divide': Divide, 
     'Repeat': Repeat,
     'affine': affine,
     'clip': clip,
     'inverse': inverse,
     'divide': divide,
     'repeat': repeat,
     'expand_as': expand_as,
     'chunk': chunk,
     'concat': concat,
     'activ': activ,
     'conv2d': conv2d,
     'conv2d_transpose': conv2d_transpose,
     'gated_conv2d': gated_conv2d,
     'gated_conv2d_transpose': gated_conv2d_transpose,
     'dense': dense,
     'AddGrid': AddGrid,
     'DepthFlow': DepthFlow,
     'add_grid': add_grid,
     'depth_flow': depth_flow,
     'construct_outputs': construct_outputs,
     'transform_inputs': transform_inputs
}
