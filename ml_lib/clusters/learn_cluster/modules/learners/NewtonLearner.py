import torch as pt

from ml_lib.clusters.learn_cluster.modules.learners.RootLearner import RootLearner as Root

class NewtonLearner(Root):
    defaults = {
        **Root.defaults,
        'full_hessian': False
    }
    
    def __init__(self, path_name = None, verbose = None, learn_rate = None, coef_scale = None,
                 full_hessian = None
                ):
        super().__init__(path_name = path_name, verbose = verbose, learn_rate = learn_rate, coef_scale = coef_scale)
        self.full_hessian = self.defaults['full_hessian'] if full_hessian is None else full_hessian
    
    def learn(self, loss, coefs):
        (gradient,) = pt.autograd.grad(loss, coefs, create_graph = True)
        flat_grad = gradient.view(gradient.numel())
        hessian = self._calc_hessian(flat_grad, coefs)
        with pt.no_grad():
            if self.full_hessian:
                inv_hess = pt.inverse(hessian)
                learn_step = flat_grad @ inv_hess
            else:
                learn_step = flat_grad / pt.diag(hessian)
            
            learn_step *= self.learn_rate
            if self.coef_scale: learn_step /= coefs.size()[0]
            learn_step[learn_step != learn_step] = 0
            
            learn_step = learn_step.view(coefs.size())
            
        self._v_msg('Newton Method updated %s coefs, learn_rate %s.' % (new_coefs.numel(), self.learn_rate))
            
        return learn_step
    
    def _calc_hessian(self, gradient, coefs):
        hessian = pt.stack([
            pt.autograd.grad(deriv, coefs, retain_graph = True)[0].view(gradient.size())
            for deriv in gradient
        ], dim = 1)
        return hessian