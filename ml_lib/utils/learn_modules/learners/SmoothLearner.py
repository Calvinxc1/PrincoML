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
    
    def __init__(self, path_name = None, verbose = None, learn_rate = None, coef_scale = None,
                 alpha = None, beta = None, clamper = None, scalers = None
                ):
        super().__init__(path_name = path_name, verbose = verbose, learn_rate = learn_rate, coef_scale = coef_scale)
        self.alpha = self.defaults['alpha'] if alpha is None else alpha
        self.beta = self.defaults['beta'] if beta is None else beta
        self.clamper = self.defaults['clamper'] if clamper is None else clamper
        self.gradient = 0
        self.grad_sq = 0
        self.iter_count = 1
    
    def learn(self, loss, coefs):
        gradient = pt.autograd.grad(loss, coefs, create_graph = True)[0]
        
        with pt.no_grad():
            self.gradient = (self.alpha * gradient) + ((1 - self.alpha) * self.gradient)
            self.grad_sq = (self.beta * (gradient ** 2)) + ((1 - self.beta) * self.grad_sq)
            
            learn_step = self.gradient / pt.clamp(pt.sqrt(self.grad_sq), self.clamper, np.inf)
            
            learn_step *= self.learn_rate
            if self.coef_scale: learn_step /= coefs.size()[0]
            
            self.iter_count += 1
            
        self._v_msg('Smooth updated %s coefs, learn_rate %s.' % (coefs.numel(), self.learn_rate))
            
        return learn_step
    