import numpy as np
import torch as pt

from .RootInit import RootInit as Root

class OrthoInit(Root):
    defaults = {
        **Root.defaults,
        'xavier': True
    }
    
    def __init__(self, path_name = None, verbose = None, xavier = False):
        self.xavier = self.defaults['xavier'] if xavier is None else xavier
        
        super().__init__(path_name = path_name, verbose = verbose)
        
    def init(self, shape):
        seed = np.random.randn(*shape)
        
        u, _, v = np.linalg.svd(seed, full_matrices = False)
        init_tensor = (u * (2 / np.sum(shape))) @ v if self.xavier else u @ v
        
        init_tensor = pt.from_numpy(init_tensor).type(pt.Tensor)
        
        self._v_msg('Tensor shape %s created with orthogonal init, Xavier %s.' % (shape, self.xavier))
        
        return init_tensor