import torch as pt

from .RootRegularizer import RootRegularizer as Root

class NormRegularizer(Root):
    defaults = {
        **Root.defaults,
        'l1': 1e-3,
        'l2': 1e-3,
        'mean': True
    }
    
    def __init__(self, path_name = None, verbose = None, l1 = None, l2 = None, mean = None):
        self.l1 = self.defaults['l1'] if l1 is None else l1
        self.l2 = self.defaults['l2'] if l2 is None else l2
        self.mean = self.defaults['mean'] if mean is None else mean
        
        super().__init__(path_name = path_name, verbose = verbose)
        
    def regularize(self, coefs):
        reg_loss = 0
        
        if self.l1 != 0:
            reg_loss_lasso = pt.abs(pt.cat([coef.reshape(1,-1).squeeze() for coef in coefs], dim = 0))
            reg_loss_lasso = reg_loss_lasso.mean() if self.mean else reg_loss_lasso.sum()
            reg_loss_lasso = reg_loss_lasso * self.l1
            reg_loss += reg_loss_lasso
            
        if self.l2 != 0:
            reg_loss_ridge = pt.cat([coef.reshape(1,-1).squeeze() for coef in coefs], dim = 0) ** 2
            reg_loss_ridge = pt.sqrt(reg_loss_ridge.mean() if self.mean else reg_loss_ridge.sum())
            reg_loss_ridge = reg_loss_ridge * self.l2
            reg_loss += reg_loss_ridge
        
        return reg_loss