import torch as pt

from .RootInit import RootInit as Root

class HingeInit(Root):
    defaults = {
        **Root.defaults,
        'value': 0
    }
    
    def __init__(self, path_name = None, verbose = None, value = None):
        self.value = self.defaults['value'] if value is None else value
        
        super().__init__(path_name = path_name, verbose = verbose)
        
    def init(self, input_tensor, hinge_count):
        input_min, input_max = [func(input_tensor.detach(), dim=0)[0] for func in (pt.min, pt.max)]
        
        init_tensor = pt.stack([
            input_min + (((input_max - input_min) / (hinge_count+1)) * i) 
            for i in range(1, hinge_count+1)
        ], dim = 0)
        
        #self._v_msg('Tensor shape %s created with hinge-ready values.' % (init_tensor.size()))
        
        return init_tensor