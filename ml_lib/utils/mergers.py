import torch as pt

from ml_lib.utils.inits import Constant

class MergerRoot():
    def __init__(self):
        self.init = False
        self.primed = False
        self.coefs = {}
        
    def init_merger(self, col_count, input_count, reinit = False):
        if self.init & (not reinit):
            raise Exception('Recursor is initialized and reinit is False')
    
    def prime_merger(self, reprime = False):
        if self.primed & (not reprime):
            raise Exception('Recursor is primed and reprime is False')
            
        for coef in self.coefs.values():
            coef.requires_grad = True
    
    def deprime_merger(self):
        ## Placeholder, define in children
        pass
        
    def merge(self, input_tensor):
        ## Placeholder, define in children
        pass
    
    def get_coefs(self):
        coefs = [(coef_label, coef_value) for coef_label, coef_value in self.coefs.items()]
        return coefs
    
    def update_coefs(self, new_coefs):
        for coef_label, coef_value in new_coefs:
            self.coefs[coef_label] = coef_value

class Sum(MergerRoot):
    def merge(self, input_tensor):
        merged_tensor = input_tensor.sum(dim = 2)
        return merged_tensor
    
class Product(MergerRoot):
    def merge(self, input_tensor):
        merged_tensor = input_tensor.prod(dim = 2)
        return merged_tensor
    
class Smooth(MergerRoot):
    def __init__(self,
                 init = Constant, init_params = {}
                ):
        super().__init__()
        self.Init = init(**init_params)
        
    def init_merger(self, col_count, input_count, reinit = False):
        super().init_merger(col_count, input_count, reinit = reinit)
        self.coefs['smoother'] = self.Init.init((1, col_count, input_count))
        
    def merge(self, input_tensor):
        input_weights = pt.exp(self.coefs['smoother'])
        input_weights = input_weights / input_weights.sum(dim = 2, keepdim = True)
        
        merged_tensor = (input_tensor * input_weights).sum(dim = 2)
        return merged_tensor