import torch as pt

from ml_lib.utils.learn_modules.activators.RootActivate import RootActivate as Root

class SoftReluActivator(Root):
    defaults = {
        **Root.defaults,
        'clamp': 32
    }
    
    def __init__(self, path_name = None, verbose = None,
                 clamp = None
                ):
        super().__init__(path_name = path_name, verbose = verbose)
        self.clamp = self.defaults['clamp'] if clamp is None else clamp
    
    def activate(self, input_tensor):
        activated_tensor = input_tensor.copy()
        activate_mask = pt.abs(activated_tensor) <= self.clamp
        for col_idx in range(activate_mask.size()[1]):
            mask = activate_mask[:,col_idx]
            activated_tensor[mask,col_idx] = pt.log(1 + pt.exp(activated_tensor[mask,col_idx]))
        
        self._v_msg('Soft ReLU Activation on %s shape tensor' % (tuple([dim for dim in activated_tensor.size()]),))
        
        return activated_tensor