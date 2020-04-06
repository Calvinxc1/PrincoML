import numpy as np
import torch as pt

from .RootInit import RootInit as Root

class FlatInit(Root):
    defaults = {
        **Root.defaults,
        'value': 0
    }
    
    def __init__(self, path_name = None, verbose = None, value = None):
        self.value = self.defaults['value'] if value is None else value
        
        super().__init__(path_name = path_name, verbose = verbose)
        
    def init(self, shape):
        init_tensor = pt.from_numpy(np.full(shape, self.value)).type(pt.Tensor)
        
        self._v_msg('Tensor shape %s created with flat value %s.' % (shape, self.value))
        
        return init_tensor