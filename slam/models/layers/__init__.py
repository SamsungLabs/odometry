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
from .wrappers import construct_fc
from .wrappers import construct_double_fc

from .special_layers import AddGrid
from .special_layers import DepthFlow
from .special_layers import add_grid
from .special_layers import depth_flow
from .special_layers import construct_outputs


__all__ = [
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
    'construct_fc',
    'construct_double_fc',
    'AddGrid',
    'DepthFlow',
    'add_grid',
    'depth_flow',
    'construct_outputs'
]

CUSTOM_LAYERS = {
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
     'construct_fc': construct_fc,
     'construct_double_fc': construct_double_fc,
     'AddGrid': AddGrid,
     'DepthFlow': DepthFlow,
     'add_grid': add_grid,
     'depth_flow': depth_flow,
     'construct_outputs': construct_outputs
}
