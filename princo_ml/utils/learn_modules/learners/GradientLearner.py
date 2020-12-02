import torch as pt

from .RootLearner import RootLearner as Root

class GradientLearner(Root):
    def learn(self, loss, coefs):
        (gradient,) = pt.autograd.grad(loss, coefs)
        with pt.no_grad():
            learn_step = gradient
            if self.coef_scale: gradient /= coefs.size()[0]
            
        self._v_msg('Gradient Descent updated %s coefs.' % new_coefs.numel())
            
        return learn_step