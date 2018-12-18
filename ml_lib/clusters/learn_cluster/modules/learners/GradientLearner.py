import torch as pt

from ml_lib.clusters.learn_cluster.modules.learners.RootLearner import RootLearner as Root

class GradientLearner(Root):
    defaults = {
        **Root.defaults,
        'learn_rate': 1e-3
    }
    
    def __init__(self, path_name = None, verbose = None, learn_rate = None):
        super().__init__(path_name = path_name, verbose = verbose)
        self.learn_rate = self.defaults['learn_rate'] if learn_rate is None else learn_rate
        
    def learn(self, loss, coefs):
        (gradient,) = pt.autograd.grad(loss, coefs)
        with pt.no_grad():
            new_coefs = coefs - (gradient * self.learn_rate)
            
        self._v_msg('Gradient Descent updated %s coefs, learn_rate %s.' % (new_coefs.numel(), self.learn_rate))
            
        return new_coefs