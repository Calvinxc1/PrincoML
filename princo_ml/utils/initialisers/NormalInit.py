import numpy as np
import torch as pt

from .RootInit import RootInit as Root

class NormalInit(Root):
    defaults = {
        **Root.defaults,
        'mean': 0,
        'stdev': 1
    }
    
    def __init__(self, path_name = None, verbose = None, mean = None, stdev = None):
        self.mean = self.defaults['mean'] if mean is None else mean
        self.stdev = self.defaults['stdev'] if stdev is None else stdev
        
        super().__init__(path_name = path_name, verbose = verbose)
        
    def init(self, shape):
        init_tensor = (pt.randn(shape) * self.stdev) + self.mean
        
        self._v_msg('Tensor shape %s created with normal mean %s & stdev %s.' % (shape, self.mean, self.stdev))
        
        return init_tensor