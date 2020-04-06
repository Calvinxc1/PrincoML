import torch as pt

from .RootCombiner import RootCombiner as Root

class HingeCombiner(Root):
    defaults = {
        **Root.defaults,
        'hinge_func': pt.tanh
    }
    
    def __init__(self, path_name = None, verbose = None, hinge_func = None):
        super().__init__(path_name = path_name, verbose = verbose)
        self.hinge_func = self.defaults['hinge_func'] if hinge_func is None else hinge_func
    
    def combine(self, input_tensor, coefs):        
        hinge_groups = self.build_hinge_groups(input_tensor, coefs['hinge'])
        
        combine_tensor = coefs['bias'].unsqueeze(dim = 0) + (
            pt.einsum('ij,jk->ik', (input_tensor, coefs['weights']))
        ) + (
            pt.einsum('ijl,jkl->ik', (hinge_groups, coefs['hinge_bias']))
        ) + (
            pt.einsum('ijl,jkl->ik', ((input_tensor.unsqueeze(dim = 2) * hinge_groups), coefs['hinge_weights']))
        )
        
        return combine_tensor
            
    def build_hinge_groups(self, input_tensor, hinge_coefs):
        input_min, input_max = [func(input_tensor, dim=0, keepdim=True)[0] for func in (pt.min, pt.max)]
        
        hinge_up = [
            input_tensor - input_min,
            *[input_tensor - coef.unsqueeze(dim = 0) for coef in hinge_coefs]
        ]

        hinge_down = [
            *[coef.unsqueeze(dim = 0) - input_tensor for coef in hinge_coefs],
            input_max - input_tensor
        ]

        hinge_groups = pt.stack([
            self.hinge_func(up) * self.hinge_func(down)
            for up, down in zip(hinge_up, hinge_down)
        ], dim = 2)
        
        return hinge_groups