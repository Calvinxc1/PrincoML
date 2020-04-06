import numpy as np
import torch as pt

from .RootLearner import RootLearner as Root

class SmoothLearner(Root):
    defaults = {
        **Root.defaults,
        'alpha': 0.2,
        'beta': 0.1,
        'clamper': 1e-16
    }
    
    def __init__(self, path_name = None, verbose = None, coef_scale = None,
                 alpha = None, beta = None, clamper = None
                ):
        super().__init__(path_name = path_name, verbose = verbose, coef_scale = coef_scale)
        self.alpha = self.defaults['alpha'] if alpha is None else alpha
        self.beta = self.defaults['beta'] if beta is None else beta
        self.clamper = self.defaults['clamper'] if clamper is None else clamper
        self.gradient = {}
        self.grad_sq = {}
    
    def learn(self, loss, coefs):
        gradient = self.order_grads(loss, coefs)
        coef_count = sum([coef.numel() for coef in coefs.values()])
        
        learn_step = {}
        with pt.no_grad():
            for coef in gradient.keys():
                self.gradient[coef] = (self.alpha * gradient[coef]) + ((1-self.alpha) * self.gradient.get(coef, gradient[coef]))
                self.grad_sq[coef] = (self.beta * (gradient[coef]**2)) + ((1-self.beta) * self.grad_sq.get(coef, gradient[coef]**2))
                learn_step[coef] = self.gradient[coef] / pt.clamp(pt.sqrt(self.grad_sq[coef]), self.clamper, np.inf)
                if self.coef_scale: learn_step[coef] /= coef_count

        #self._v_msg('Smooth updated %s coefs.' % coefs.numel())
            
        return learn_step
    
    def order_grads(self, loss, coefs):
        coef_names = []
        coef_vals = []
        for coef_name, coef_val in coefs.items():
            coef_names.append(coef_name)
            coef_vals.append(coef_val)

        new_grad = pt.autograd.grad(loss, coef_vals, retain_graph = True)

        gradients = {}
        for coef_name, coef_grad in zip(coef_names, new_grad):
            gradients[coef_name] = coef_grad

        return gradients