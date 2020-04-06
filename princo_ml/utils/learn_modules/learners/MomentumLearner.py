import torch as pt

from .RootLearner import RootLearner as Root

class MomentumLearner(Root):
    defaults = {
        **Root.defaults,
        'alpha': 0.8
    }
    
    def __init__(self, path_name = None, verbose = None, coef_scale = None,
                 alpha = None
                ):
        super().__init__(path_name = path_name, verbose = verbose, coef_scale = coef_scale)
        self.alpha = self.defaults['alpha'] if alpha is None else alpha
        self.prior_grad = 0
    
    def learn(self, loss, coefs):
        (gradient,) = pt.autograd.grad(loss, coefs, create_graph = True)
        
        with pt.no_grad():
            learn_step = (self.alpha * gradient) + ((1 - self.alpha) * self.prior_grad)
            self.prior_grad = gradient
            
            if self.coef_scale: learn_step /= coefs.size()[0]
            
        self._v_msg('Momentum updated %s coefs.' % new_coefs.numel())
            
        return learn_step
    