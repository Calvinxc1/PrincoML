import torch as pt

from ml_lib.clusters.learn_cluster.modules.learners.RootLearner import RootLearner as Root

class MomentumLearner(Root):
    defaults = {
        **Root.defaults,
        'alpha': 0.8
    }
    
    def __init__(self, path_name = None, verbose = None, learn_rate = None, coef_scale = None,
                 alpha = None
                ):
        super().__init__(path_name = path_name, verbose = verbose, learn_rate = learn_rate, coef_scale = coef_scale)
        self.alpha = self.defaults['alpha'] if alpha is None else alpha
        self.prior_grad = 0
    
    def learn(self, loss, coefs):
        (gradient,) = pt.autograd.grad(loss, coefs, create_graph = True)
        
        with pt.no_grad():
            learn_step = (self.alpha * gradient) + ((1 - self.alpha) * self.prior_grad)
            self.prior_grad = gradient
            
            learn_step *= self.learn_rate
            if self.coef_scale: learn_step /= coefs.size()[0]
            
        self._v_msg('Momentum updated %s coefs, learn_rate %s.' % (new_coefs.numel(), self.learn_rate))
            
        return learn_step
    