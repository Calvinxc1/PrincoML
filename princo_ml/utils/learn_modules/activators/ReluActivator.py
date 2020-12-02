import numpy as np
import torch as pt

from .RootActivator import RootActivator as Root

class ReluActivator(Root):
    defaults = {
        **Root.defaults,
        'leak': 0
    }
    
    def __init__(self, path_name = None, verbose = None,
                 leak = None
                ):
        super().__init__(path_name = path_name, verbose = verbose)
        self.leak = self.defaults['leak'] if leak is None else leak
    
    def activate(self, input_tensor):
        activated_tensor = pt.max(input_tensor, input_tensor * self.leak)
        self._v_msg('ReLU Activation on %s shape tensor' % (tuple([dim for dim in activated_tensor.size()]),))
        return activated_tensor