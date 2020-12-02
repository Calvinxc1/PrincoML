import torch as pt

from .RootCombiner import RootCombiner as Root

class SimpleCombiner(Root):
    defaults = {
        **Root.defaults,
        'bias_active': True
    }
    
    def __init__(self, path_name = None, verbose = None, bias_active = None):
        super().__init__(path_name = path_name, verbose = verbose)
        self.bias_active = self.defaults['bias_active'] if bias_active is None else bias_active
    
    def combine(self, input_tensor, coefs):
        combined_tensor = pt.einsum('ij,jk->ik', (input_tensor, coefs['weight']))
        if self.bias_active: combined_tensor += coefs['bias'].unsqueeze(dim=0)
        
        #self._v_msg('Combined to produce %s shape tensor' % (tuple([dim for dim in combined_tensor.size()]),))
        
        return combined_tensor