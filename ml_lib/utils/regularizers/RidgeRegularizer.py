import torch as pt

from ml_lib.utils.regularizers.RootRegularizer import RootRegularizer as Root

class RidgeRegularizer(Root):
    defaults = {
        **Root.defaults,
        'lamb': 1e-3,
        'mean': True
    }
    
    def __init__(self, path_name = None, verbose = None, lamb = None, mean = None):
        self.lamb = self.defaults['lamb'] if lamb is None else lamb
        self.mean = self.defaults['mean'] if mean is None else mean
        
        super().__init__(path_name = path_name, verbose = verbose)
        
    def regularize(self, coefs):
        reg_loss = pt.cat([coef.reshape(1,-1).squeeze() for coef in coefs], dim = 0) ** 2
        reg_loss = reg_loss.mean() if self.mean else reg_loss.sum()
        reg_loss = reg_loss * self.lamb
        return reg_loss