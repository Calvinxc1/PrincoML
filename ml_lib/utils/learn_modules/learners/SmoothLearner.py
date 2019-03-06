import numpy as np
import torch as pt

from ml_lib.utils.learn_modules.learners.RootLearner import RootLearner as Root

class SmoothLearner(Root):
    defaults = {
        **Root.defaults,
        'alpha': 0.2,
        'beta': 0.1,
        'clamper': 1e-16
    }
    
    def __init__(self, path_name = None, verbose = None, coef_scale = None,
                 alpha = None, beta = None, clamper = None, scalers = None
                ):
        super().__init__(path_name = path_name, verbose = verbose, coef_scale = coef_scale)
        self.alpha = self.defaults['alpha'] if alpha is None else alpha
        self.beta = self.defaults['beta'] if beta is None else beta
        self.clamper = self.defaults['clamper'] if clamper is None else clamper
        self.gradient = None
        self.grad_sq = None
        self.iter_count = 1
    
    def learn(self, loss, coefs):
        gradient = pt.autograd.grad(loss, coefs, create_graph = True)[0]
        
        with pt.no_grad():
            if self.gradient is None:
                self.gradient = gradient
            else:
                self.gradient = (self.alpha * gradient) + ((1 - self.alpha) * self.gradient)
            
            if self.grad_sq is None:
                self.grad_sq = gradient ** 2
            else:
                self.grad_sq = (self.beta * (gradient ** 2)) + ((1 - self.beta) * self.grad_sq)
            
            learn_step = self.gradient / pt.clamp(pt.sqrt(self.grad_sq), self.clamper, np.inf)
            
            if self.coef_scale: learn_step /= coefs.size()[0]
            
            self.iter_count += 1
            
        self._v_msg('Smooth updated %s coefs.' % coefs.numel())
            
        return learn_step
    