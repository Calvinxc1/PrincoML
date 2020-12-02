import torch as pt

from .RootActivator import RootActivator as Root

class SoftReluActivator(Root):
    defaults = {
        **Root.defaults,
        'clamp': 32,
        'leak': 0
    }
    
    def __init__(self, path_name = None, verbose = None,
                 clamp = None, leak = None
                ):
        super().__init__(path_name = path_name, verbose = verbose)
        self.clamp = self.defaults['clamp'] if clamp is None else clamp
        self.leak = pt.exp(pt.Tensor([self.defaults['leak'] if leak is None else leak]))
    
    def activate(self, input_tensor):
        activated_tensor = input_tensor.clone()
        activate_mask = pt.abs(activated_tensor) <= self.clamp
        for col_idx in range(activate_mask.size()[1]):
            mask = activate_mask[:,col_idx]
            activated_tensor[mask,col_idx] = pt.log(self.leak + pt.exp(activated_tensor[mask,col_idx]))
        
        self._v_msg('Soft ReLU Activation on %s shape tensor' % (tuple([dim for dim in activated_tensor.size()]),))
        
        return activated_tensor